"""
Progress reporting for the long-running stages.

Two different problems, and they need different treatment:

* **Countable work** - 33,333 videos, 53,510 clips to fetch, 21 repositories to download. A bar
  with a rate and an ETA is the right answer.
* **Opaque work** - `pip install torch` inside an adapter environment. There is nothing to count,
  it takes ten to twenty minutes, and the previous implementation captured the subprocess output
  and printed nothing until it finished, so the terminal looked frozen. A heartbeat showing
  elapsed time and the most recent line of output is what tells you it is alive.

Both degrade for non-interactive output. Under `nohup`, in a CI log or through a scheduler, a
carriage-return bar is useless noise, so they fall back to periodic log lines instead. Console
logging writes to stdout, so everything here writes to stderr and the two never interleave.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from csf.logging_utils import get_logger

log = get_logger("generation.progress")

#: Set CSF_NO_PROGRESS=1 to silence bars and heartbeats entirely (logs still happen).
DISABLED = os.environ.get("CSF_NO_PROGRESS", "").lower() in ("1", "true", "yes")


def is_interactive() -> bool:
    if DISABLED:
        return False
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class _LoggingBar:
    """Bar substitute for non-interactive output: an occasional log line with rate and ETA."""

    def __init__(self, total: Optional[int], desc: str, unit: str, log_every: int,
                 log_seconds: float):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.log_every = max(1, log_every)
        self.log_seconds = log_seconds
        self.n = 0
        self._started = time.monotonic()
        self._last = 0.0
        self._postfix = ""

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.monotonic()
        due = (self.n % self.log_every == 0) or (now - self._last >= self.log_seconds)
        if not due:
            return
        self._last = now
        elapsed = now - self._started
        rate = self.n / elapsed if elapsed > 0 else 0.0
        parts = [f"{self.desc}: {self.n:,}"]
        if self.total:
            parts[0] += f"/{self.total:,} ({100.0 * self.n / self.total:.1f}%)"
        parts.append(f"{rate:.2f} {self.unit}/s")
        if self.total and rate > 0:
            parts.append(f"ETA {_fmt_duration((self.total - self.n) / rate)}")
        parts.append(f"elapsed {_fmt_duration(elapsed)}")
        if self._postfix:
            parts.append(self._postfix)
        log.info(" | ".join(parts))

    def set_postfix_str(self, text: str) -> None:
        self._postfix = text

    def set_postfix(self, **kwargs) -> None:
        self.set_postfix_str(", ".join(f"{k}={v}" for k, v in kwargs.items()))

    def close(self) -> None:
        elapsed = time.monotonic() - self._started
        log.info("%s: finished %s%s in %s", self.desc, f"{self.n:,}",
                 f"/{self.total:,}" if self.total else "", _fmt_duration(elapsed))


@contextmanager
def bar(total: Optional[int], desc: str, unit: str = "it", log_every: int = 200,
        log_seconds: float = 60.0) -> Iterator:
    """A tqdm bar when a terminal is attached, periodic log lines otherwise."""
    if is_interactive():
        try:
            from tqdm.auto import tqdm
            handle = tqdm(total=total, desc=desc, unit=unit, file=sys.stderr,
                          dynamic_ncols=True, smoothing=0.1)
            try:
                yield handle
            finally:
                handle.close()
            return
        except ImportError:
            pass
    handle = _LoggingBar(total, desc, unit, log_every, log_seconds)
    try:
        yield handle
    finally:
        handle.close()


def track(iterable: Iterable, desc: str, total: Optional[int] = None, unit: str = "it",
          log_every: int = 200) -> Iterator:
    """Iterate with a progress bar."""
    if total is None:
        try:
            total = len(iterable)                            # type: ignore[arg-type]
        except TypeError:
            total = None
    with bar(total, desc, unit, log_every) as handle:
        for item in iterable:
            yield item
            handle.update(1)


class Heartbeat:
    """Liveness for an operation with nothing to count.

    Prints elapsed time (and the latest status line, if the caller sets one) every `interval`
    seconds, so an environment build that spends fifteen minutes inside pip is visibly working
    rather than apparently hung.
    """

    def __init__(self, desc: str, interval: float = 10.0):
        self.desc = desc
        self.interval = interval
        self.status = ""
        self._started = time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._printed = False

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def set_status(self, text: str) -> None:
        self.status = (text or "").strip()[:100]

    def _loop(self) -> None:
        interactive = is_interactive()
        while not self._stop.wait(self.interval):
            line = f"  [{_fmt_duration(self.elapsed):>7}] {self.desc}"
            if self.status:
                line += f" - {self.status}"
            if interactive:
                width = shutil.get_terminal_size((100, 24)).columns
                sys.stderr.write("\r" + line[:width - 1].ljust(width - 1))
                sys.stderr.flush()
                self._printed = True
            else:
                log.info(line.strip())

    def __enter__(self) -> "Heartbeat":
        if not DISABLED:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval)
        if self._printed:
            sys.stderr.write("\r" + " " * (shutil.get_terminal_size((100, 24)).columns - 1) + "\r")
            sys.stderr.flush()


def run_streaming(cmd: Sequence[str], cwd: Optional[Path] = None,
                  env: Optional[Dict[str, str]] = None, timeout: Optional[float] = None,
                  desc: str = "", tail_lines: int = 60) -> "subprocess.CompletedProcess":
    """Run a command, streaming its output to the debug log with a live heartbeat.

    `subprocess.run(capture_output=True)` buffers everything until the process exits, which is
    what made a long pip install look like a hang. Here each line goes to the log as it arrives,
    the most recent one drives the heartbeat, and the last `tail_lines` are kept so a failure can
    still report what went wrong.
    """
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, errors="replace")
    tail: List[str] = []
    with Heartbeat(desc or Path(str(cmd[0])).name) as beat:
        deadline = time.monotonic() + timeout if timeout else None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    log.debug("  %s", line[:400])
                    tail.append(line)
                    if len(tail) > tail_lines:
                        tail.pop(0)
                    beat.set_status(line)
                if deadline and time.monotonic() > deadline:
                    proc.kill()
                    raise subprocess.TimeoutExpired(cmd, timeout or 0)
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
            raise
        finally:
            if proc.poll() is None:
                proc.kill()
    return subprocess.CompletedProcess(list(cmd), proc.returncode, "\n".join(tail), "")
