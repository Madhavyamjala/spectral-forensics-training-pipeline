"""
Whether a directory can actually be written to - which free space does not answer.

`shutil.disk_usage` reports the filesystem's free blocks. On a shared cluster the limit that
bites first is usually a per-user or per-group quota, and no amount of free space reflects it:
preflight can report 30 TiB available and every write still fail with EDQUOT. That is not a
hypothetical - a full smoke sweep once failed on 26 of 27 models with "Disk quota exceeded",
each one wearing the costume of a different bug (a Hub download, an ffmpeg mux, a worker log,
`mkdir 'runs'`), because nothing upstream had asked the one question that mattered.

So ask it directly: write some bytes and see. A probe is worth a few MiB of IO because the
alternative is a two-day run that fails on its first write and spends the rest of the time
recording the consequences.
"""

from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path
from typing import Optional

#: EDQUOT on Linux. Python has no errno constant for it on every platform, hence the literal.
EDQUOT = getattr(errno, "EDQUOT", 122)

QUOTA_ADVICE = (
    "This is a quota, not free space - the filesystem may report terabytes available while "
    "your account cannot write another byte. Check it with `quota -s`, or `lfs quota -h -u "
    "$(whoami) <path>` on Lustre, and free space or ask for more before running again."
)


def write_probe(path, mib: int = 16) -> Optional[str]:
    """Return None when `mib` MiB can be written under `path`, or why it cannot.

    The probe file is written, flushed to disk (a buffered write can succeed against a quota
    that the flush then refuses) and removed, whatever happens.
    """
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"{target} could not be created: {exc}" + (
            "\n" + QUOTA_ADVICE if exc.errno in (EDQUOT, errno.ENOSPC) else "")

    probe: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(target), prefix=".csf_write_probe_",
                                         delete=False) as fh:
            probe = Path(fh.name)
            chunk = b"\0" * (1024 * 1024)
            for _ in range(max(1, mib)):
                fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        detail = f"writing {mib} MiB under {target} failed: {exc}"
        if exc.errno in (EDQUOT, errno.ENOSPC):
            detail += "\n" + QUOTA_ADVICE
        return detail
    finally:
        if probe is not None:
            try:
                probe.unlink()
            except OSError:                          # noqa: BLE001 - nothing left to do
                pass
    return None


def require_writable(path, mib: int = 16, label: str = "") -> None:
    """Raise with the quota advice attached when `path` cannot take `mib` MiB."""
    problem = write_probe(path, mib)
    if problem:
        where = f" for {label}" if label else ""
        raise RuntimeError(f"The directory{where} is not writable: {problem}")
