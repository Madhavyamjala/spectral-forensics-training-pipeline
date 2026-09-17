"""
Adapter contract and the persistent-worker protocol.

An *adapter* binds one manipulation model from the spec to (a) the environment it needs and
(b) a standalone worker script that actually renders videos. The worker runs inside the
adapter's own venv, so nothing about its dependencies leaks into the driver process.

Spawning a process per video would be hopeless - most of these models take 10-60 s to load and
we have 33,333 videos - so workers are **persistent** and speak a line-delimited JSON protocol
over stdin/stdout:

    worker -> {"event": "ready", "model": "...", "device": "cuda:2"}
    driver -> {"job_id": "...", "source_path": "...", "output_path": "...", ...}
    worker -> {"event": "result", "job_id": "...", "ok": true, "metadata": {...}}
             or {"event": "result", "job_id": "...", "ok": false, "error": "..."}
    driver -> {"cmd": "shutdown"}

Anything a worker writes to stderr is captured into the run log, so a model's own progress
output never corrupts the protocol stream. A worker that dies is restarted by the scheduler and
the job it was holding is retried once before being recorded as failed.

Input : a `Job` row plus the GPU index to run on.
Output: an mp4 at `output_path` and a metadata dict merged into the manifest row.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from csf.generation.envs import EnvSpec, ReadyEnv, build_env
from csf.logging_utils import get_logger

log = get_logger("generation.adapter")

WORKER_DIR = Path(__file__).resolve().parent / "workers"


class AdapterError(RuntimeError):
    pass


class NotImplementedAdapter(AdapterError):
    """Raised for models in the spec that have no working public implementation wired up yet."""


@dataclass
class Adapter:
    """One manipulation model: which env it needs, which worker renders it, how it is configured.

    `key` is the *slot* in the specification document. `actual_model` is what really renders the
    video, which differs whenever the document names a model with no runnable public release and
    a substitute stands in for it (VideoReTalking -> LatentSync, for example).

    Those two are kept apart deliberately. The slot preserves the document's allocation - the
    source-content mix and per-family balance it designed - while `actual_model` is what lands in
    the manifest, so the per-method accuracy breakdown reports the model that genuinely produced
    the artifacts. Recording the slot name instead would quietly attribute LatentSync's
    fingerprint to VideoReTalking, and any attribution study built on it would be wrong.
    """
    key: str                                   # matches spec.Pipeline.key (the document's slot)
    family: str                                # the family it belongs to (informational)
    env_name: str
    worker: str                                # file name under adapters/workers/
    implemented: bool = True
    tier: int = 1                              # 1 = wired up and expected to run; 2 = scaffold
    actual_model: str = ""                     # empty => the slot runs its own named model
    cost_s: float = 60.0                       # measured/estimated seconds per video, for planning
    vram_gb: float = 8.0                       # peak VRAM per worker, for per-GPU concurrency
    options: Dict[str, object] = field(default_factory=dict)
    note: str = ""

    @property
    def runs(self) -> str:
        """The model that actually renders - what the manifest records."""
        return self.actual_model or self.key

    @property
    def substituted(self) -> bool:
        """Return whether another model renders this specification slot."""
        return bool(self.actual_model) and self.actual_model != self.key

    @property
    def worker_path(self) -> Path:
        """Return the filesystem path of this adapter worker."""
        return WORKER_DIR / self.worker

    def payload(self, job, output_path: Path, gpu: int) -> Dict[str, object]:
        """The JSON message sent to the worker for one video."""
        meta = {}
        if getattr(job, "metadata", None):
            try:
                meta = json.loads(job.metadata)
            except (TypeError, json.JSONDecodeError):
                meta = {}
        return {
            "job_id": job.job_id, "video_id": job.video_id, "model": self.runs,
            "spec_model": self.key, "family": job.family, "source_path": job.source_path,
            "driving_path": job.driving_path, "audio_path": job.audio_path,
            "output_path": str(output_path), "variant": job.variant, "operation": job.operation,
            "mask_size": job.mask_size, "mask_motion": job.mask_motion, "prompt": job.prompt,
            "seed": job.seed, "gpu": gpu, "options": dict(self.options), "metadata": meta,
        }


# --------------------------------------------------------------------------------------
# persistent worker process
# --------------------------------------------------------------------------------------


class WorkerProcess:
    """A long-lived model process pinned to one GPU."""

    def __init__(self, adapter: Adapter, env: ReadyEnv, gpu: int, log_dir: Path,
                 startup_timeout: int = 1800, job_timeout: int = 1800):
        """Initialize a persistent worker subprocess for one adapter and GPU."""
        self.adapter = adapter
        self.env = env
        self.gpu = gpu
        self.log_dir = Path(log_dir)
        self.startup_timeout = startup_timeout
        self.job_timeout = job_timeout
        self.proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_tail: "queue.Queue[str]" = queue.Queue(maxsize=200)
        self._stdout_lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self.jobs_done = 0

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        """Start the worker process and wait for its ready message."""
        if not self.adapter.worker_path.exists():
            raise AdapterError(f"Worker script missing for adapter '{self.adapter.key}': "
                               f"{self.adapter.worker_path}")
        environ = self.env.environ()
        environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        environ["CSF_GPU"] = str(self.gpu)
        environ["PYTHONUNBUFFERED"] = "1"
        # workers that serve several spec models (background modes, the three inpainting
        # baselines) pick their variant up from the environment at load() time
        environ["CSF_ADAPTER"] = self.adapter.key
        environ["CSF_SPEC_MODEL"] = self.adapter.key
        environ["CSF_ACTUAL_MODEL"] = self.adapter.runs
        for name, value in self.adapter.options.items():
            environ[f"CSF_{name.upper()}"] = str(value)
        # the shared worker helpers live next to the worker scripts
        environ["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(WORKER_DIR), environ.get("PYTHONPATH", ""))))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log.info("Starting worker %s on GPU %d (env %s)", self.adapter.key, self.gpu, self.env.spec.name)
        self.proc = subprocess.Popen(
            [str(self.env.python), "-u", str(self.adapter.worker_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environ, text=True, bufsize=1, cwd=str(self.env.root))
        self._stdout_lines = queue.Queue()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stdout_thread.start()

        ready = self._read_json(self.startup_timeout)
        if not ready or ready.get("event") != "ready":
            tail = self.stderr_tail()
            self.stop()
            raise AdapterError(f"Worker '{self.adapter.key}' did not become ready "
                               f"(got {ready!r}).\nstderr tail:\n{tail}")
        log.info("Worker %s ready on GPU %d", self.adapter.key, self.gpu)

    def _drain_stderr(self) -> None:
        """Continuously copy worker stderr into the diagnostic log."""
        assert self.proc and self.proc.stderr
        path = self.log_dir / f"worker_{self.adapter.key}_gpu{self.gpu}.log"
        with open(path, "a", encoding="utf-8") as fh:
            for line in self.proc.stderr:
                fh.write(line)
                if self._stderr_tail.full():
                    try:
                        self._stderr_tail.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._stderr_tail.put_nowait(line.rstrip())
                except queue.Full:
                    pass

    def stderr_tail(self, lines: int = 25) -> str:
        """Return the most recent worker stderr lines."""
        buf: List[str] = []
        while not self._stderr_tail.empty():
            try:
                buf.append(self._stderr_tail.get_nowait())
            except queue.Empty:
                break
        return "\n".join(buf[-lines:])

    def alive(self) -> bool:
        """Return whether the worker subprocess is still running."""
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        """Stop the worker subprocess and release its streams."""
        if self.proc is None:
            return
        try:
            if self.alive() and self.proc.stdin:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        finally:
            if self.alive():
                self.proc.kill()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            self.proc = None

    # ---------------- protocol ----------------

    def _drain_stdout(self) -> None:
        """Feed every stdout line into a queue.

        A dedicated reader thread rather than select(): stdout is a buffered text stream, so a
        JSON line can already sit in Python's buffer while the underlying fd reports "not
        readable". Gating readline() on select() therefore stalls until the job timeout whenever
        a worker prints anything non-protocol to stdout - which upstream repos and tqdm do all
        the time. Blocking readline in a thread has no such blind spot, and works on Windows.
        """
        assert self.proc and self.proc.stdout
        try:
            for line in self.proc.stdout:
                self._stdout_lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._stdout_lines.put(None)        # sentinel: the stream is closed

    def _read_json(self, timeout: float) -> Optional[Dict[str, object]]:
        """Next protocol message, skipping any non-JSON the worker printed."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._stdout_lines.get(timeout=min(5.0, remaining))
            except queue.Empty:
                if not self.alive() and self._stdout_lines.empty():
                    return None
                continue
            if line is None:                    # stdout closed - the worker is gone
                return None
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.debug("[%s] non-protocol stdout: %s", self.adapter.key, line[:200])
                continue
            if isinstance(msg, dict):
                return msg

    def run(self, payload: Dict[str, object]) -> Dict[str, object]:
        """Send one job and wait for its result. Raises `AdapterError` if the worker dies."""
        if not self.alive():
            raise AdapterError(f"Worker '{self.adapter.key}' is not running")
        assert self.proc and self.proc.stdin
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise AdapterError(f"Worker '{self.adapter.key}' stdin closed: {exc}") from exc

        msg = self._read_json(self.job_timeout)
        if msg is None:
            tail = self.stderr_tail()
            raise AdapterError(f"Worker '{self.adapter.key}' produced no result for "
                               f"{payload.get('job_id')} (timeout {self.job_timeout}s or crash)."
                               f"\nstderr tail:\n{tail}")
        if msg.get("event") != "result":
            raise AdapterError(f"Worker '{self.adapter.key}' sent an unexpected message: {msg!r}")
        self.jobs_done += 1
        return msg


# --------------------------------------------------------------------------------------
# worker pool per (adapter, gpu)
# --------------------------------------------------------------------------------------


class WorkerPool:
    """Caches one live worker per (adapter, gpu), restarting it if it dies.

    `max_resident` bounds how many model processes stay loaded on a GPU at once; the least
    recently used worker is retired when the bound is hit, which is what lets a single GPU work
    through several families without running out of VRAM.
    """

    def __init__(self, envs_root: Path, log_dir: Path, max_resident: int = 1,
                 job_timeout: int = 1800, offline: bool = False):
        """Initialize the resident worker pool and environment builder."""
        self.envs_root = Path(envs_root)
        self.log_dir = Path(log_dir)
        self.max_resident = max(1, max_resident)
        self.job_timeout = job_timeout
        self.offline = offline
        self._workers: Dict[tuple, WorkerProcess] = {}
        self._order: List[tuple] = []
        self._envs: Dict[str, ReadyEnv] = {}
        self._lock = threading.RLock()

    def _env(self, spec: EnvSpec) -> ReadyEnv:
        """Build or reuse the environment required by an adapter."""
        with self._lock:
            if spec.name not in self._envs:
                self._envs[spec.name] = build_env(spec, self.envs_root, offline=self.offline)
            return self._envs[spec.name]

    def get(self, adapter: Adapter, spec: EnvSpec, gpu: int, slot: int = 0) -> WorkerProcess:
        """A live worker for (adapter, gpu, slot). `slot` lets one GPU hold several workers of
        the same model, which is what fills a 143 GB card running a 3 GB net."""
        key = (adapter.key, gpu, slot)
        with self._lock:
            return self._get_locked(adapter, spec, gpu, slot, key)

    def _get_locked(self, adapter: Adapter, spec: EnvSpec, gpu: int, slot: int,
                    key: tuple) -> WorkerProcess:
        """Return a live worker while the pool lock is held."""
        worker = self._workers.get(key)
        if worker is not None and worker.alive():
            self._touch(key)
            return worker
        if worker is not None:
            log.warning("Worker %s on GPU %d (slot %d) died -> restarting", adapter.key, gpu, slot)
            worker.stop()
            self._workers.pop(key, None)
            if key in self._order:
                self._order.remove(key)

        resident = [k for k in self._order if k[1] == gpu and k[0] != adapter.key]
        while len(resident) >= self.max_resident:
            victim = resident.pop(0)
            log.info("Retiring worker %s on GPU %d slot %d to free VRAM", *victim)
            self._workers.pop(victim).stop()
            self._order.remove(victim)

        env = self._env(spec)
        if not Path(env.python).exists():
            raise AdapterError(
                f"The interpreter for env '{spec.name}' is missing at {env.python}. Rebuild it "
                f"with: python -m csf.generation.envs --build {spec.name} --force")
        worker = WorkerProcess(adapter, env, gpu, self.log_dir, job_timeout=self.job_timeout)
        worker.start()
        self._workers[key] = worker
        self._order.append(key)
        return worker

    def _touch(self, key: tuple) -> None:
        """Mark a worker as the most recently used pool entry."""
        if key in self._order:
            self._order.remove(key)
            self._order.append(key)

    def shutdown(self) -> None:
        """Stop every resident worker in the pool."""
        with self._lock:
            self._shutdown_locked()

    def _shutdown_locked(self) -> None:
        """Stop all workers while the pool lock is held."""
        for key, worker in list(self._workers.items()):
            log.debug("Stopping worker %s", key)
            worker.stop()
        self._workers.clear()
        self._order.clear()
