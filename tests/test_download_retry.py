"""
Dependency-free checks for download retry classification and resumability messaging.

The feature-extraction stage downloads ~100k videos from the Hub, which is fronted by a CDN.
A 503 from that CDN means "ask again", but it was classified as permanent: the download raised
on the first attempt, ten such failures landed consecutively across the worker pool, and a run
aborted 98% of the way through 97,774 videos. These checks pin the classification and the
wording that tells an operator what survived.

Run: python tests/test_download_retry.py
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one assertion, printing its outcome without stopping the suite."""
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f": {detail}" if not condition else ""))
    if not condition:
        FAILURES.append(name)


def _load_classifier() -> types.ModuleType:
    """Load the retry helpers without importing cv2/torch, which tests must not need."""
    src = (ROOT / "csf" / "data" / "video_io.py").read_text(encoding="utf-8")
    body = src[src.index("def _status_code"):src.index("def download_video")]
    mod = types.ModuleType("video_io_retry")
    mod.__dict__.update({"re": re, "Optional": Optional})
    exec(compile(body, "video_io.py", "exec"), mod.__dict__)
    return mod


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _HubError(Exception):
    """Stands in for huggingface_hub.HfHubHTTPError, with or without the response attached."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        if status is not None:
            self.response = _Response(status)


def test_retry_classification() -> None:
    """Server-side failures retry; client-side ones do not."""
    print("download retry classification")
    mod = _load_classifier()

    # the exact error that aborted the run, which carries no response object
    real = _HubError("Server error '503 Service Unavailable' for url "
                     "'https://us.aws.cdn.hf.co/xet-bridge-us/6a95fe7c98cbb8825c478453/8767dd40'")
    check("the 503 that aborted the run is retried", mod.is_transient(real))

    for status in (500, 502, 503, 504):
        check(f"{status} is retried (response object)", mod.is_transient(_HubError("err", status)))
        check(f"{status} is retried (message only)",
              mod.is_transient(_HubError(f"Server error '{status}' for url ...")))

    for status in (400, 401, 403, 404, 416):
        check(f"{status} fails immediately - retrying cannot fix it",
              not mod.is_transient(_HubError(f"Client Error: {status} for url ...", status)))

    check("429 is retried", mod.is_transient(_HubError("429 Too Many Requests", 429)))
    check("a rate-limit phrase is retried", mod.is_transient(_HubError("Rate limit reached")))
    check("connection errors are retried", mod.is_transient(ConnectionError("reset by peer")))
    check("timeouts are retried", mod.is_transient(TimeoutError("Read timed out")))
    check("a truncated transfer is retried",
          mod.is_transient(Exception("IncompleteRead: remote end closed connection")))
    check("a programming error is not retried", not mod.is_transient(ValueError("bad row")))
    check("a decode error is not retried", not mod.is_transient(RuntimeError("codec not supported")))

    check("status is read from the response when present",
          mod._status_code(_HubError("no digits here", 503)) == 503)
    check("status is parsed from the message otherwise",
          mod._status_code(_HubError("Server error '503 Service Unavailable'")) == 503)
    check("a message with no status yields None", mod._status_code(Exception("boom")) is None)

    src = (ROOT / "csf" / "data" / "video_io.py").read_text(encoding="utf-8")
    check("download_video routes every failure through the classifier",
          "if not is_transient(exc) or attempt > max_retries:" in src)
    check("the retry loop still backs off exponentially",
          "base_backoff * 2 ** (attempt - 1)" in src)


def test_abort_message() -> None:
    """A run that stops must say what survived, in numbers."""
    print("abort message")
    src = (ROOT / "csf" / "data" / "feature_cache.py").read_text(encoding="utf-8")
    check("the abort reports how many items are already cached",
          "Nothing cached is lost:" in src and "item(s) for this" in src)
    check("it says a re-run resumes rather than starting over",
          "it does not start over" in src)
    check("it names the knob that widens the tolerance",
          "data.download_fail_fast" in src and "data.extract_fail_fast" in src)
    check("the cached count includes this run's successes",
          "cached = len(mine) - len(todo) + n_done" in src)


def main() -> int:
    for fn in (test_retry_classification, test_abort_message):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all download-retry checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
