"""
Logging, crash forensics and stage bookkeeping.

Every rank writes a full DEBUG log to <work_dir>/logs/rank{R}.log; rank 0 also prints INFO
to the console. Failures are captured at three levels so a crash can always be pinpointed:

  * `stage(...)` context manager  - logs start / end / duration of a pipeline stage and, on an
                                    exception, writes <work_dir>/logs/crash_rank{R}.json with the
                                    stage name, the last reported step/context (e.g. the video ids in
                                    the failing batch), full traceback and a CUDA memory summary.
  * `sys.excepthook`/thread hook   - anything escaping a stage is still logged with traceback.
  * `faulthandler`                 - native crashes (segfaults in CUDA / bitsandbytes / OpenCV) dump
                                    the Python stack of every thread to <work_dir>/logs/fault_rank{R}.log.

`RunState` persists which stages finished (<work_dir>/state.json) so re-running the same command
resumes after the last completed stage.

Input : work_dir, rank, debug flag.
Output: configured `logging` hierarchy under the "csf" logger, crash/fault files on failure.
"""

from __future__ import annotations

import contextlib
import faulthandler
import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER_NAME = "csf"
_CONTEXT: Dict[str, Any] = {}
_FAULT_FILE = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


class _RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        record.stage = _CONTEXT.get("stage", "-")
        return True


def setup_logging(work_dir: Path, rank: int = 0, debug: bool = False) -> logging.Logger:
    global _FAULT_FILE
    log_dir = Path(work_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    rank_filter = _RankFilter(rank)

    fmt = logging.Formatter(
        "%(asctime)s | r%(rank)d | %(levelname)-7s | %(stage)s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_dir / f"rank{rank}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh.addFilter(rank_filter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug else (logging.INFO if rank == 0 else logging.WARNING))
    ch.setFormatter(logging.Formatter("%(asctime)s | r%(rank)d | %(levelname)-7s | %(stage)s | %(message)s",
                                      datefmt="%H:%M:%S"))
    ch.addFilter(rank_filter)
    logger.addHandler(ch)

    for noisy in ("transformers", "peft", "accelerate", "huggingface_hub", "diffusers"):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.WARNING)
        lg.addHandler(fh)

    _FAULT_FILE = open(log_dir / f"fault_rank{rank}.log", "a", encoding="utf-8")
    faulthandler.enable(file=_FAULT_FILE, all_threads=True)

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.warning("Interrupted by user (KeyboardInterrupt).")
            return
        if getattr(exc, "_csf_reported", False):
            logger.critical("Run aborted: %s: %s (details above and in logs/crash_rank%d.json)", exc_type.__name__, exc, rank)
            return
        logger.critical("Uncaught exception:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
        write_crash_report(work_dir, rank, exc)

    def _thread_excepthook(args):
        logger.critical("Uncaught exception in thread %s:\n%s", args.thread.name if args.thread else "?",
                        "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    _CONTEXT["work_dir"] = str(work_dir)
    _CONTEXT["rank"] = rank
    return logger


def set_context(**kwargs: Any) -> None:
    """Record the most recent step-level context (step, batch ids, ...) for crash reports."""
    _CONTEXT.update(kwargs)


def _cuda_memory_summary() -> Optional[str]:
    try:
        import torch
        if torch.cuda.is_available():
            return (f"allocated={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
                    f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB "
                    f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB")
    except Exception:
        pass
    return None


def write_crash_report(work_dir: Path, rank: int, exc: BaseException) -> Path:
    path = Path(work_dir) / "logs" / f"crash_rank{rank}.json"
    report = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "rank": rank,
        "exception_type": type(exc).__name__,
        "exception": str(exc)[:4000],
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
        "context": {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else repr(v))
                    for k, v in _CONTEXT.items()},
        "cuda_memory": _cuda_memory_summary(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    return path


@contextlib.contextmanager
def stage(name: str, work_dir: Path, rank: int = 0):
    """Wrap one pipeline stage: logs timing, attaches context to any failure and re-raises."""
    log = get_logger()
    previous = _CONTEXT.get("stage")
    _CONTEXT["stage"] = name
    for k in [k for k in _CONTEXT if k not in ("stage", "work_dir", "rank")]:
        _CONTEXT.pop(k, None)
    t0 = time.time()
    log.info("========== STAGE START: %s ==========", name)
    try:
        yield
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            log.warning("Stage %s interrupted after %.1fs", name, time.time() - t0)
            raise
        report = write_crash_report(work_dir, rank, exc)
        ctx = {k: v for k, v in _CONTEXT.items() if k not in ("work_dir", "rank")}
        log.critical("STAGE FAILED: %s after %.1fs | context=%s | %s: %s\n%s\nCrash report: %s",
                     name, time.time() - t0, ctx, type(exc).__name__, exc, traceback.format_exc(), report)
        if _cuda_memory_summary():
            log.critical("CUDA memory at failure: %s", _cuda_memory_summary())
        try:
            exc._csf_reported = True
        except AttributeError:
            pass
        raise
    else:
        log.info("========== STAGE DONE : %s (%.1fs) ==========", name, time.time() - t0)
    finally:
        _CONTEXT["stage"] = previous or "-"


class RunState:
    """Tiny JSON-backed record of completed stages, written atomically by rank 0."""

    def __init__(self, work_dir: Path):
        self.path = Path(work_dir) / "state.json"
        self.data: Dict[str, Any] = {"completed": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def reload(self) -> None:
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def done(self, stage_name: str) -> bool:
        return stage_name in self.data.get("completed", {})

    def mark(self, stage_name: str, info: Optional[Dict[str, Any]] = None) -> None:
        self.data.setdefault("completed", {})[stage_name] = {
            "time": datetime.now().isoformat(timespec="seconds"), **(info or {})}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def clear(self, stage_name: str) -> None:
        self.data.get("completed", {}).pop(stage_name, None)
        if self.path.parent.exists():
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


class Throughput:
    """Rate / ETA helper for progress log lines (logs stay informative even without tqdm)."""

    def __init__(self, total: int):
        self.total = max(total, 1)
        self.start = time.time()

    def line(self, done: int) -> str:
        elapsed = time.time() - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float("inf")
        eta_s = f"{eta / 60:.1f}min" if eta != float("inf") else "?"
        return f"{done}/{self.total} ({100 * done / self.total:.1f}%) | {rate:.2f}/s | ETA {eta_s}"
