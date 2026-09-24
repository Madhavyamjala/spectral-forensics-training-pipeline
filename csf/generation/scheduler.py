"""
Multi-GPU generation scheduler.

Drives the 33,333 jobs across a bounded set of GPUs (three, on the target machine - the other
three are occupied by vLLM workers). The scheduling rule that matters is **batch by model**:
loading SAM2, FLUX or Stable Video Diffusion costs 10-60 s, so jobs are grouped by adapter and a
worker renders its whole group before the GPU moves on. Model loads therefore cost O(models),
not O(videos).

Work is split across GPUs by *estimated cost*, not job count, because a TokenFlow job takes
minutes while an INSwapper job takes seconds. Each GPU owns whole model groups, so two GPUs never
hold the same model resident at once.

Resumability is per job: every outcome is appended to `ledger.jsonl` as soon as it happens, so a
run that dies after 20,000 videos resumes at 20,001. Re-running is always safe.

Guards that stop a bad run from wasting the week:
  * a job whose worker dies is retried once, then recorded as failed,
  * a model group whose first `fail_fast` jobs all fail is abandoned (usually a missing weight),
  * generation pauses if free disk falls below `min_free_gb`,
  * `--deadline` stops scheduling new groups once the wall-clock budget is spent, so the run
    always ends with a usable, manifest-able set of videos rather than being killed mid-write.

Input : jobs.csv, GenerationConfig.
Output: mp4s under `<video_root>/AI Edited/<family>/`, ledger.jsonl, failures.csv, summary.json.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import threading
import time
from collections import defaultdict
from queue import Empty, Queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from csf.generation.adapters import ADAPTERS, WorkerPool, adapter_for, env_specs
from csf.generation.adapters.base import AdapterError
from csf.generation.diskcheck import write_probe
from csf.generation.envs import EnvBuildError
from csf.generation import progress as progress_ui
from csf.generation.jobs import Job
from csf.logging_utils import get_logger

log = get_logger("generation.scheduler")

DEFAULT_COST = 60.0
#: Fraction of a card's VRAM we are willing to commit to resident workers.
VRAM_HEADROOM = 0.85


def concurrency_for(model: str, gpu_vram_gb: float, cap: int) -> int:
    """How many workers of `model` to run on one GPU.

    An H200 has 143 GB and INSwapper needs ~3 GB, so one worker per card wastes almost all of
    it. The small nets are latency-bound anyway - they spend most of their time decoding video
    and detecting faces on the CPU - so several in parallel scale nearly linearly. Diffusion
    samplers already saturate the SMs and gain far less, but they are also the ones whose VRAM
    footprint limits the count, so the same formula handles both.
    """
    a = ADAPTERS.get(model)
    vram = a.vram_gb if a is not None else 8.0
    fit = int((gpu_vram_gb * VRAM_HEADROOM) // max(1.0, vram))
    return max(1, min(cap, fit))


def cost_of(model: str) -> float:
    """Seconds per video for `model`, from its adapter. Used to balance GPUs and estimate ETA."""
    a = ADAPTERS.get(model)
    return a.cost_s if a is not None else DEFAULT_COST


@dataclass
class Outcome:
    job_id: str
    ok: bool
    output_path: str = ""
    error: str = ""
    seconds: float = 0.0
    metadata: dict = field(default_factory=dict)
    #: The job never ran - its environment could not be built, or the group ended early. It is
    #: recorded so the failure is visible, but it does not count against the job's attempts:
    #: nothing about *this job* failed, and burning its retries on a broken env (or an offline
    #: node) strands work that would succeed the moment the env is fixed.
    env_error: bool = False


class Ledger:
    """Append-only record of every job outcome; the source of truth for resuming."""

    def __init__(self, path: Path):
        """Load prior outcomes and open the append-only generation ledger."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.done: Dict[str, bool] = {}
        self.attempts: Dict[str, int] = {}
        self.errors: Dict[str, str] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    job_id = rec["job_id"]
                    ok = bool(rec.get("ok"))
                    self.done[job_id] = ok
                    # ledgers written before env_error existed: a "worker unavailable"
                    # error is an env failure by any other name, so do not charge it as an
                    # attempt now that the distinction exists.
                    env_error = bool(rec.get("env_error")) or \
                        str(rec.get("error") or "").startswith("worker unavailable:")
                    if not env_error:
                        self.attempts[job_id] = self.attempts.get(job_id, 0) + 1
                    self.attempts.setdefault(job_id, 0)
                    if ok:
                        self.errors.pop(job_id, None)
                    else:
                        self.errors[job_id] = str(rec.get("error") or "")
            log.info("Ledger %s: %d job(s) already recorded (%d ok, %d failed)", self.path,
                     len(self.done), sum(1 for v in self.done.values() if v),
                     sum(1 for v in self.done.values() if not v))

    def record(self, outcome: Outcome) -> None:
        """Append an outcome unless its job has already been recorded."""
        with self._lock:
            self.done[outcome.job_id] = outcome.ok
            if not outcome.env_error:
                self.attempts[outcome.job_id] = self.attempts.get(outcome.job_id, 0) + 1
            self.attempts.setdefault(outcome.job_id, 0)
            if outcome.ok:
                self.errors.pop(outcome.job_id, None)
            else:
                self.errors[outcome.job_id] = outcome.error[:400]
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"job_id": outcome.job_id, "ok": outcome.ok,
                                     "output_path": outcome.output_path,
                                     "error": outcome.error[:400],
                                     "seconds": round(outcome.seconds, 2),
                                     "env_error": outcome.env_error,
                                     "metadata": outcome.metadata}) + "\n")

    def succeeded(self, job_id: str) -> bool:
        """Return whether a job has a recorded successful outcome."""
        return self.done.get(job_id) is True

    def seen(self, job_id: str) -> bool:
        """Return whether a job has any recorded outcome."""
        return job_id in self.done

    def attempt_count(self, job_id: str) -> int:
        """How many outcomes - successful or not - this job has recorded."""
        return self.attempts.get(job_id, 0)

    def failure_reasons(self, limit: int = 5) -> List[Tuple[str, int]]:
        """The most common failure messages, for diagnosing a run that produced nothing."""
        tally: Dict[str, int] = {}
        for job_id, ok in self.done.items():
            if ok:
                continue
            reason = (self.errors.get(job_id) or "unknown error").strip().splitlines()
            key = reason[-1][:160] if reason else "unknown error"
            tally[key] = tally.get(key, 0) + 1
        return sorted(tally.items(), key=lambda kv: -kv[1])[:limit]


# --------------------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------------------


def group_jobs(jobs: Sequence[Job], ledger: Ledger, retry_failed: bool = True,
               max_attempts: int = 3, output_for=None) -> Dict[str, List[Job]]:
    """Pending jobs, grouped by model.

    A job is dropped once it has succeeded - that is the whole point of the ledger. A job that
    *failed* is retried on the next run until it has used up `max_attempts`, because most
    failures here are environmental rather than intrinsic to the job: an adapter environment
    that had not finished building, a checkpoint that had not been staged, a GPU that was busy.
    Treating the first failure as final turned those into a permanently dead run whose only
    symptom was "nothing to do" followed by "produced no videos at all".

    `retry_failed=False` restores the old behaviour (one attempt, ever); `max_attempts` caps how
    many times a genuinely broken job is allowed to burn a worker slot.
    """
    groups: Dict[str, List[Job]] = defaultdict(list)
    exhausted = 0
    orphaned = 0
    for job in jobs:
        if ledger.succeeded(job.job_id):
            # "succeeded" is only true while the video is still there. A recorded success whose
            # file is gone (deleted, or written somewhere the driver could not see) would
            # otherwise be skipped forever, leaving a hole nothing ever fills.
            if output_for is not None:
                out = Path(output_for(job))
                if not (out.exists() and out.stat().st_size > 0):
                    orphaned += 1
                    groups[job.model].append(job)
            continue
        if ledger.seen(job.job_id):
            if not retry_failed:
                continue
            if ledger.attempt_count(job.job_id) >= max(1, max_attempts):
                exhausted += 1
                continue
        groups[job.model].append(job)
    if orphaned:
        log.warning("%d job(s) are recorded as successful but their video is missing from the "
                    "video root -> re-generating them", orphaned)
    if exhausted:
        log.warning("%d job(s) have failed %d time(s) and will not be retried again. Fix the "
                    "underlying error and re-run with generation.max_attempts raised, or delete "
                    "their rows from the ledger.", exhausted, max_attempts)
    return dict(groups)


def balance(groups: Dict[str, List[Job]], gpus: Sequence[int]) -> Dict[int, List[str]]:
    """Assign whole model groups to GPUs, greedily levelling estimated cost."""
    load = {g: 0.0 for g in gpus}
    plan: Dict[int, List[str]] = {g: [] for g in gpus}
    ordered = sorted(groups, key=lambda m: -len(groups[m]) * cost_of(m))
    for model in ordered:
        gpu = min(load, key=lambda g: load[g])
        plan[gpu].append(model)
        load[gpu] += len(groups[model]) * cost_of(model)
    for gpu in gpus:
        log.info("GPU %d: %d model group(s), %d video(s), est %.1f h", gpu, len(plan[gpu]),
                 sum(len(groups[m]) for m in plan[gpu]), load[gpu] / 3600.0)
    return plan


# --------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------


class Progress:
    """Live progress across every GPU, as a bar when attached to a terminal.

    All GPU threads share one counter, so the bar reflects the whole run rather than any single
    worker - which is what matters when the ETA is measured in days.
    """

    def __init__(self, total: int):
        """Initialize thread-safe counters and the shared progress bar."""
        self.total = total
        self.ok = 0
        self.failed = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._last_log = 0.0
        self._bar_cm = progress_ui.bar(total, "generating", "video", log_every=50,
                                       log_seconds=300)
        self._bar = self._bar_cm.__enter__()

    def close(self) -> None:
        """Tear the bar down; never let display trouble mask a run's result."""
        try:
            self._bar_cm.__exit__(None, None, None)
        except Exception:                                    # noqa: BLE001
            pass

    def update(self, ok: bool) -> None:
        """Record one outcome and periodically report progress."""
        with self._lock:
            if ok:
                self.ok += 1
            else:
                self.failed += 1
            done = self.ok + self.failed
            self._bar.update(1)
            self._bar.set_postfix_str(f"ok {self.ok:,} failed {self.failed:,}")
            now = time.monotonic()
            if done % 200 == 0 or now - self._last_log > 900:
                self._last_log = now
                elapsed = now - self.started
                rate = done / max(1e-6, elapsed)
                remaining = (self.total - done) / rate if rate > 0 else float("inf")
                log.info("Progress %d/%d (%.1f%%) | ok %d failed %d | %.1f videos/min | ETA %.1f h",
                         done, self.total, 100.0 * done / max(1, self.total), self.ok, self.failed,
                         rate * 60.0, remaining / 3600.0)


class GenerationScheduler:
    def __init__(self, video_root: Path, envs_root: Path, log_dir: Path, gpus: Sequence[int],
                 job_timeout: int = 1800, fail_fast: int = 8, min_free_gb: float = 50.0,
                 offline: bool = False, deadline_hours: Optional[float] = None,
                 gpu_vram_gb: float = 143.0, max_workers_per_gpu: int = 1,
                 staged_dir: Optional[Path] = None):
        """Configure generation paths, limits, GPUs, and worker concurrency."""
        # Absolute, always. Workers are launched with cwd set to their env root, so a relative
        # output path would land under cache/.../envs/<env>/ instead of the video root - the
        # driver then records "ok" for a file it cannot find. Same reasoning as the interpreter.
        self.video_root = Path(os.path.abspath(video_root))
        self.envs_root = Path(envs_root)
        self.staged_dir = Path(os.path.abspath(staged_dir)) if staged_dir else None
        self.log_dir = Path(log_dir)
        self.gpus = list(gpus)
        self.job_timeout = job_timeout
        self.fail_fast = fail_fast
        self.min_free_gb = min_free_gb
        self.offline = offline
        self.deadline = (time.monotonic() + deadline_hours * 3600.0) if deadline_hours else None
        self.gpu_vram_gb = gpu_vram_gb
        self.max_workers_per_gpu = max(1, max_workers_per_gpu)
        self.failures: List[Tuple[str, str, str]] = []
        self._fail_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._probed_at = 0.0
        self._probe_error: Optional[str] = None
        self.probe_interval_s = 60.0

    def output_path(self, job: Job) -> Path:
        """Return the destination video path for a job."""
        return self.video_root / "AI Edited" / job.family / f"{job.video_id}.mp4"

    def _disk_ok(self) -> bool:
        """Return whether generation can still write videos: free space *and* quota."""
        try:
            free_gb = shutil.disk_usage(self.video_root).free / 2 ** 30
        except OSError:
            return True
        if free_gb < self.min_free_gb:
            log.error("Only %.1f GiB free under %s (min_free_gb=%.1f) - pausing generation",
                      free_gb, self.video_root, self.min_free_gb)
            return False
        # A quota refuses writes while disk_usage still reports the filesystem's free blocks,
        # so the space check above cannot see it. Probe rather than trust, but throttled: this
        # runs before every job and the answer does not change by the second.
        now = time.monotonic()
        with self._probe_lock:
            due = now - self._probed_at >= self.probe_interval_s
            if due:
                self._probed_at = now
        if due:
            problem = write_probe(self.video_root, mib=8)
            self._probe_error = problem
        if self._probe_error:
            log.error("Cannot write under %s - pausing generation. %s",
                      self.video_root, self._probe_error)
            return False
        return True

    def run(self, jobs: Sequence[Job], ledger: Ledger, retry_failed: bool = True,
            max_attempts: int = 3) -> Dict[str, object]:
        """Schedule all pending jobs and return an execution summary."""
        groups = group_jobs(jobs, ledger, retry_failed, max_attempts, self.output_path)
        if not groups:
            ok = sum(1 for j in jobs if ledger.succeeded(j.job_id)
                      and self.output_path(j).exists())
            if ok == len(jobs):
                log.info("Nothing to do: all %d job(s) already succeeded", len(jobs))
            else:
                log.error("Nothing to do: %d of %d job(s) failed previously and have no retries "
                          "left. Most common failures:", len(jobs) - ok, len(jobs))
                for reason, count in ledger.failure_reasons():
                    log.error("  %4dx %s", count, reason)
            return {"generated": 0, "failed": 0, "skipped": len(jobs),
                    "exhausted": len(jobs) - ok}

        unknown = [m for m in groups if m not in ADAPTERS]
        if unknown:
            raise AdapterError(f"jobs reference unregistered models: {sorted(unknown)}")

        pending = sum(len(v) for v in groups.values())
        log.info("Generation: %d pending job(s) across %d model group(s) on GPUs %s",
                 pending, len(groups), self.gpus)
        plan = balance(groups, self.gpus)
        progress = Progress(pending)
        specs = env_specs()

        threads = []
        for gpu in self.gpus:
            t = threading.Thread(target=self._gpu_loop, name=f"gpu{gpu}",
                                 args=(gpu, plan[gpu], groups, ledger, progress, specs),
                                 daemon=False)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        progress.close()

        summary = {"generated": progress.ok, "failed": progress.failed,
                   "pending_at_start": pending,
                   "elapsed_hours": round((time.monotonic() - progress.started) / 3600.0, 3)}
        log.info("Generation finished: %s", json.dumps(summary))
        return summary

    def _gpu_loop(self, gpu: int, models: Sequence[str], groups: Dict[str, List[Job]],
                  ledger: Ledger, progress: Progress, specs) -> None:
        """Run assigned model groups sequentially on one GPU."""
        pool = WorkerPool(self.envs_root, self.log_dir, max_resident=self.max_workers_per_gpu,
                          job_timeout=self.job_timeout, offline=self.offline,
                          staged_dir=self.staged_dir)
        try:
            for model in models:
                if self.deadline and time.monotonic() > self.deadline:
                    log.warning("GPU %d: wall-clock deadline reached -> skipping remaining groups "
                                "(%s)", gpu, ", ".join(models[models.index(model):]))
                    break
                self._run_group(gpu, model, groups[model], pool, ledger, progress, specs)
        finally:
            pool.shutdown()

    def _run_group(self, gpu: int, model: str, jobs: List[Job], pool: WorkerPool, ledger: Ledger,
                   progress: Progress, specs) -> None:
        """Run one model group across the allowed concurrent workers."""
        adapter = adapter_for(model)
        spec = specs[adapter.env_name]
        # never start more workers than there is work; six videos do not need four processes
        slots = min(len(jobs),
                    concurrency_for(model, self.gpu_vram_gb, self.max_workers_per_gpu))
        log.info("GPU %d: starting model '%s' (%d videos, env %s, %d worker(s) x %.0f GB)",
                 gpu, model, len(jobs), adapter.env_name, slots, adapter.vram_gb)

        if not adapter.implemented:
            reason = f"NotImplemented[{model}]: {adapter.note or 'no runnable public release wired up'}"
            log.warning("GPU %d: model '%s' is not wired up -> recording %d job(s) as failed",
                        gpu, model, len(jobs))
            for job in jobs:
                ledger.record(Outcome(job.job_id, False, error=reason))
                self._note_failure(job, reason)
                progress.update(False)
            return

        queue: "Queue[Optional[Job]]" = Queue()
        for job in jobs:
            queue.put(job)
        abandon = threading.Event()
        consecutive = itertools.count()
        fail_streak = [0]
        streak_lock = threading.Lock()

        def slot_loop(slot: int) -> None:
            """Consume jobs on one worker slot, recording every job whatever goes wrong.

            The wrapper matters: an unexpected exception used to kill the thread outright, and
            the videos still queued were then neither generated nor written to the ledger. They
            simply vanished, and the run reported a clean finish for a group that produced
            nothing.
            """
            try:
                _slot_loop(slot)
            except Exception as exc:                          # noqa: BLE001 - never lose jobs
                log.exception("GPU %d slot %d: '%s' failed unexpectedly", gpu, slot, model)
                if not abandon.is_set():
                    abandon.set()
                    self._drain(queue, ledger, progress,
                                f"worker slot crashed: {type(exc).__name__}: {exc}"[:500])

        def _slot_loop(slot: int) -> None:
            try:
                worker = pool.get(adapter, spec, gpu, slot)
            except (AdapterError, EnvBuildError, OSError) as exc:
                reason = f"worker/env unavailable: {exc}"[:500]
                log.error("GPU %d slot %d: cannot start '%s': %s", gpu, slot, model, reason)
                if slot == 0:                      # slot 0 failing means the model cannot run
                    abandon.set()
                    self._drain(queue, ledger, progress, reason)
                return

            while not abandon.is_set():
                if self.deadline and time.monotonic() > self.deadline:
                    log.warning("GPU %d slot %d: deadline reached inside '%s'", gpu, slot, model)
                    return
                if not self._disk_ok():
                    abandon.set()
                    return
                try:
                    job = queue.get_nowait()
                except Empty:
                    return

                out = self.output_path(job)
                if out.exists() and out.stat().st_size > 0 and not ledger.seen(job.job_id):
                    ledger.record(Outcome(job.job_id, True, str(out)))
                    progress.update(True)
                    continue

                started = time.monotonic()
                try:
                    worker = pool.get(adapter, spec, gpu, slot)
                except (AdapterError, EnvBuildError, OSError) as exc:
                    outcome = Outcome(job.job_id, False, error=f"worker unavailable: {exc}"[:500],
                                      env_error=True)
                else:
                    outcome = self._run_one(worker, pool, adapter, spec, gpu, job, out, slot)
                outcome.seconds = time.monotonic() - started
                ledger.record(outcome)
                progress.update(outcome.ok)

                with streak_lock:
                    if outcome.ok:
                        fail_streak[0] = 0
                    else:
                        fail_streak[0] += 1
                        self._note_failure(job, outcome.error)
                        if fail_streak[0] >= self.fail_fast and not abandon.is_set():
                            log.error("GPU %d: '%s' failed %d times in a row -> abandoning the "
                                      "group. Last error: %s", gpu, model, fail_streak[0],
                                      outcome.error[:300])
                            abandon.set()
                            self._drain(queue, ledger, progress,
                                        f"group abandoned after {fail_streak[0]} consecutive failures")

        threads = [threading.Thread(target=slot_loop, args=(i,), name=f"gpu{gpu}-{model}-{i}")
                   for i in range(slots)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # A group must account for every job it was given. If a thread still died in a way that
        # escaped both handlers, the queue holds work nobody recorded; sweep it rather than let
        # the run report success for videos that were never attempted.
        stranded = 0
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                break
            stranded += 1
            ledger.record(Outcome(job.job_id, False, env_error=True,
                                  error="job was never attempted (worker group ended early)"))
            self._note_failure(job, "never attempted")
            progress.update(False)
        if stranded:
            log.error("GPU %d: '%s' ended with %d job(s) unattempted - recorded as failed",
                      gpu, model, stranded)
        log.info("GPU %d: finished model '%s'", gpu, model)

    def _drain(self, queue: "Queue[Optional[Job]]", ledger: Ledger, progress: Progress,
               reason: str) -> None:
        """Record every remaining job in an abandoned group as failed."""
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return
            ledger.record(Outcome(job.job_id, False, error=reason))
            self._note_failure(job, reason)
            progress.update(False)

    def _run_one(self, worker, pool: WorkerPool, adapter, spec, gpu: int, job: Job,
                 out: Path, slot: int = 0) -> Outcome:
        """Run one job, restarting a failed worker once."""
        payload = adapter.payload(job, out, gpu)
        for attempt in (1, 2):
            try:
                msg = worker.run(payload)
            except AdapterError as exc:
                if attempt == 1:
                    log.warning("Worker %s died on %s (%s) -> restarting and retrying once",
                                adapter.key, job.job_id, str(exc)[:200])
                    try:
                        worker = pool.get(adapter, spec, gpu, slot)
                        continue
                    except (AdapterError, EnvBuildError) as exc2:
                        return Outcome(job.job_id, False, error=f"restart failed: {exc2}"[:500])
                return Outcome(job.job_id, False, error=str(exc)[:500])
            if msg.get("ok"):
                return Outcome(job.job_id, True, str(msg.get("output_path") or out),
                               metadata=msg.get("metadata") or {})
            return Outcome(job.job_id, False, error=str(msg.get("error") or "unknown")[:500])
        return Outcome(job.job_id, False, error="exhausted retries")

    def _note_failure(self, job: Job, error: str) -> None:
        """Add a job failure to the thread-safe diagnostic list."""
        with self._fail_lock:
            self.failures.append((job.job_id, job.model, error[:300]))

    def write_failures(self, path: Path) -> Optional[Path]:
        """Write recorded generation failures to CSV when present."""
        if not self.failures:
            return None
        import csv
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["job_id", "model", "error"])
            writer.writerows(self.failures)
        log.info("Wrote %d failure row(s) -> %s", len(self.failures), path)
        return path
