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

import json
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from csf.generation.adapters import ADAPTERS, WorkerPool, adapter_for, env_specs
from csf.generation.adapters.base import AdapterError
from csf.generation.envs import EnvBuildError
from csf.generation.jobs import Job
from csf.logging_utils import get_logger

log = get_logger("generation.scheduler")

#: Rough seconds per video, used only to balance GPUs and to estimate the ETA. Measured
#: throughput replaces these as soon as a model has produced a few videos.
COST_HINTS: Dict[str, float] = {
    "inswapper": 8.0, "bg_real_composite": 25.0, "bg_flux_image": 45.0, "bg_svd_video": 120.0,
    "bg_propainter_recon": 70.0, "propainter_inpaint": 70.0, "propainter_object": 70.0,
    "e2fgvi_hq": 45.0, "sttn": 35.0, "fuseformer": 40.0, "wav2lip": 30.0, "fomm": 25.0,
    "tokenflow": 240.0,
}
DEFAULT_COST = 60.0


@dataclass
class Outcome:
    job_id: str
    ok: bool
    output_path: str = ""
    error: str = ""
    seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


class Ledger:
    """Append-only record of every job outcome; the source of truth for resuming."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.done: Dict[str, bool] = {}
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
                    self.done[rec["job_id"]] = bool(rec.get("ok"))
            log.info("Ledger %s: %d job(s) already recorded (%d ok)", self.path, len(self.done),
                     sum(1 for v in self.done.values() if v))

    def record(self, outcome: Outcome) -> None:
        with self._lock:
            self.done[outcome.job_id] = outcome.ok
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"job_id": outcome.job_id, "ok": outcome.ok,
                                     "output_path": outcome.output_path,
                                     "error": outcome.error[:400],
                                     "seconds": round(outcome.seconds, 2),
                                     "metadata": outcome.metadata}) + "\n")

    def succeeded(self, job_id: str) -> bool:
        return self.done.get(job_id) is True

    def seen(self, job_id: str) -> bool:
        return job_id in self.done


# --------------------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------------------


def group_jobs(jobs: Sequence[Job], ledger: Ledger, retry_failed: bool = False
               ) -> Dict[str, List[Job]]:
    """Pending jobs, grouped by model. Already-succeeded jobs are dropped."""
    groups: Dict[str, List[Job]] = defaultdict(list)
    for job in jobs:
        if ledger.succeeded(job.job_id):
            continue
        if ledger.seen(job.job_id) and not retry_failed:
            continue
        groups[job.model].append(job)
    return dict(groups)


def balance(groups: Dict[str, List[Job]], gpus: Sequence[int]) -> Dict[int, List[str]]:
    """Assign whole model groups to GPUs, greedily levelling estimated cost."""
    load = {g: 0.0 for g in gpus}
    plan: Dict[int, List[str]] = {g: [] for g in gpus}
    ordered = sorted(groups, key=lambda m: -len(groups[m]) * COST_HINTS.get(m, DEFAULT_COST))
    for model in ordered:
        gpu = min(load, key=lambda g: load[g])
        plan[gpu].append(model)
        load[gpu] += len(groups[model]) * COST_HINTS.get(model, DEFAULT_COST)
    for gpu in gpus:
        log.info("GPU %d: %d model group(s), %d video(s), est %.1f h", gpu, len(plan[gpu]),
                 sum(len(groups[m]) for m in plan[gpu]), load[gpu] / 3600.0)
    return plan


# --------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------


class Progress:
    def __init__(self, total: int):
        self.total = total
        self.ok = 0
        self.failed = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._last_log = 0.0

    def update(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self.ok += 1
            else:
                self.failed += 1
            done = self.ok + self.failed
            now = time.monotonic()
            if done % 50 == 0 or now - self._last_log > 300:
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
                 offline: bool = False, deadline_hours: Optional[float] = None):
        self.video_root = Path(video_root)
        self.envs_root = Path(envs_root)
        self.log_dir = Path(log_dir)
        self.gpus = list(gpus)
        self.job_timeout = job_timeout
        self.fail_fast = fail_fast
        self.min_free_gb = min_free_gb
        self.offline = offline
        self.deadline = (time.monotonic() + deadline_hours * 3600.0) if deadline_hours else None
        self.failures: List[Tuple[str, str, str]] = []
        self._fail_lock = threading.Lock()

    def output_path(self, job: Job) -> Path:
        return self.video_root / "AI Edited" / job.family / f"{job.video_id}.mp4"

    def _disk_ok(self) -> bool:
        try:
            free_gb = shutil.disk_usage(self.video_root).free / 2 ** 30
        except OSError:
            return True
        if free_gb < self.min_free_gb:
            log.error("Only %.1f GiB free under %s (min_free_gb=%.1f) - pausing generation",
                      free_gb, self.video_root, self.min_free_gb)
            return False
        return True

    def run(self, jobs: Sequence[Job], ledger: Ledger, retry_failed: bool = False) -> Dict[str, object]:
        groups = group_jobs(jobs, ledger, retry_failed)
        if not groups:
            log.info("Nothing to do: every job is already recorded in the ledger")
            return {"generated": 0, "failed": 0, "skipped": len(jobs)}

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

        summary = {"generated": progress.ok, "failed": progress.failed,
                   "pending_at_start": pending,
                   "elapsed_hours": round((time.monotonic() - progress.started) / 3600.0, 3)}
        log.info("Generation finished: %s", json.dumps(summary))
        return summary

    def _gpu_loop(self, gpu: int, models: Sequence[str], groups: Dict[str, List[Job]],
                  ledger: Ledger, progress: Progress, specs) -> None:
        pool = WorkerPool(self.envs_root, self.log_dir, max_resident=1,
                          job_timeout=self.job_timeout, offline=self.offline)
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
        adapter = adapter_for(model)
        spec = specs[adapter.env_name]
        log.info("GPU %d: starting model '%s' (%d videos, env %s)", gpu, model, len(jobs),
                 adapter.env_name)

        if not adapter.implemented:
            reason = f"NotImplemented[{model}]: {adapter.note or 'no runnable public release wired up'}"
            log.warning("GPU %d: model '%s' is not wired up -> recording %d job(s) as failed",
                        gpu, model, len(jobs))
            for job in jobs:
                ledger.record(Outcome(job.job_id, False, error=reason))
                self._note_failure(job, reason)
                progress.update(False)
            return

        try:
            worker = pool.get(adapter, spec, gpu)
        except (AdapterError, EnvBuildError) as exc:
            reason = f"worker/env unavailable: {exc}"[:500]
            log.error("GPU %d: cannot start '%s': %s", gpu, model, reason)
            for job in jobs:
                ledger.record(Outcome(job.job_id, False, error=reason))
                self._note_failure(job, reason)
                progress.update(False)
            return

        consecutive = 0
        for i, job in enumerate(jobs):
            if self.deadline and time.monotonic() > self.deadline:
                log.warning("GPU %d: deadline reached inside '%s' after %d/%d videos",
                            gpu, model, i, len(jobs))
                return
            if not self._disk_ok():
                return

            out = self.output_path(job)
            if out.exists() and out.stat().st_size > 0 and not ledger.seen(job.job_id):
                ledger.record(Outcome(job.job_id, True, str(out)))
                progress.update(True)
                continue

            started = time.monotonic()
            # always ask the pool: a worker that died on the previous job has been replaced,
            # and holding a stale reference here would burn a retry on every single job
            try:
                worker = pool.get(adapter, spec, gpu)
            except (AdapterError, EnvBuildError) as exc:
                outcome = Outcome(job.job_id, False, error=f"worker unavailable: {exc}"[:500])
                ledger.record(outcome)
                progress.update(False)
                self._note_failure(job, outcome.error)
                consecutive += 1
                if consecutive >= self.fail_fast:
                    log.error("GPU %d: '%s' cannot start a worker -> abandoning the group", gpu, model)
                    return
                continue
            outcome = self._run_one(worker, pool, adapter, spec, gpu, job, out)
            outcome.seconds = time.monotonic() - started
            ledger.record(outcome)
            progress.update(outcome.ok)
            if outcome.ok:
                consecutive = 0
            else:
                consecutive += 1
                self._note_failure(job, outcome.error)
                if consecutive >= self.fail_fast:
                    log.error("GPU %d: '%s' failed %d times in a row -> abandoning the group. "
                              "Last error: %s", gpu, model, consecutive, outcome.error[:300])
                    for rest in jobs[i + 1:]:
                        reason = f"group abandoned after {consecutive} consecutive failures"
                        ledger.record(Outcome(rest.job_id, False, error=reason))
                        self._note_failure(rest, reason)
                        progress.update(False)
                    return
        log.info("GPU %d: finished model '%s'", gpu, model)

    def _run_one(self, worker, pool: WorkerPool, adapter, spec, gpu: int, job: Job,
                 out: Path) -> Outcome:
        payload = adapter.payload(job, out, gpu)
        for attempt in (1, 2):
            try:
                msg = worker.run(payload)
            except AdapterError as exc:
                if attempt == 1:
                    log.warning("Worker %s died on %s (%s) -> restarting and retrying once",
                                adapter.key, job.job_id, str(exc)[:200])
                    try:
                        worker = pool.get(adapter, spec, gpu)
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
        with self._fail_lock:
            self.failures.append((job.job_id, job.model, error[:300]))

    def write_failures(self, path: Path) -> Optional[Path]:
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
