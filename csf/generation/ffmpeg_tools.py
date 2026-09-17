"""
Locating ffmpeg and ffprobe.

The pipeline needs ffmpeg to encode every generated clip and to lift audio for the lip-sync
family, and ffprobe to fingerprint the container of each produced file. On a managed cluster you
often cannot install either system-wide: no root, and a conda base environment that may be
unusable (`NoBaseEnvironmentError`).

So rather than requiring a system install, the binaries are resolved in this order:

    1. `CSF_FFMPEG` / `CSF_FFPROBE`   - an explicit path, which always wins
    2. `PATH`                         - a normal system or conda install
    3. `imageio-ffmpeg`               - a static ffmpeg wheel, `pip install imageio-ffmpeg`,
                                        which needs no root and no conda

`imageio-ffmpeg` bundles ffmpeg but **not** ffprobe, so when only that is available the container
probe falls back to `ffmpeg -i`, whose stderr carries the same stream information. Between that
and the OpenCV fallback in `kinetics.probe_video`, a machine with neither binary still builds a
usable source pool - it just records less metadata.

Results are cached: resolution runs once per process, not once per video.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from csf.logging_utils import get_logger

log = get_logger("generation.ffmpeg")

INSTALL_HINT = (
    "Install ffmpeg by whichever of these works on your machine:\n"
    "    pip install imageio-ffmpeg          # static build, no root, no conda\n"
    "    sudo apt install ffmpeg             # Debian/Ubuntu with root\n"
    "    conda install -c conda-forge ffmpeg # inside an activated conda env\n"
    "    module load ffmpeg                  # many HPC clusters\n"
    "or point CSF_FFMPEG / CSF_FFPROBE at existing binaries."
)


def _usable(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.is_file() and os.access(p, os.X_OK):
        return str(p)
    return shutil.which(path)


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> Optional[str]:
    """Path to an ffmpeg binary, or None."""
    for candidate in (os.environ.get("CSF_FFMPEG"), shutil.which("ffmpeg")):
        found = _usable(candidate)
        if found:
            return found
    try:
        import imageio_ffmpeg
        found = _usable(imageio_ffmpeg.get_ffmpeg_exe())
        if found:
            log.info("Using the ffmpeg bundled with imageio-ffmpeg: %s", found)
            return found
    except Exception:                          # noqa: BLE001 - optional dependency
        pass
    return None


@functools.lru_cache(maxsize=1)
def ffprobe_exe() -> Optional[str]:
    """Path to an ffprobe binary, or None.

    imageio-ffmpeg does not ship ffprobe, so this is often absent even when ffmpeg is present;
    callers should fall back to `probe_with_ffmpeg` or OpenCV rather than treating it as fatal.
    """
    for candidate in (os.environ.get("CSF_FFPROBE"), shutil.which("ffprobe")):
        found = _usable(candidate)
        if found:
            return found
    ffmpeg = ffmpeg_exe()                      # a system install keeps them side by side
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe" + (".exe" if os.name == "nt" else ""))
        found = _usable(str(sibling))
        if found:
            return found
    return None


def have_ffmpeg() -> bool:
    return ffmpeg_exe() is not None


_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")
_VIDEO = re.compile(r"Stream #\d+:\d+.*?Video:\s*([A-Za-z0-9_]+).*?(\d{2,5})x(\d{2,5})")
_FPS = re.compile(r"(\d+\.?\d*)\s*fps")
_BITRATE = re.compile(r"bitrate:\s*(\d+)\s*kb/s")


def probe_with_ffmpeg(path: Path) -> Optional[Dict[str, object]]:
    """Container metadata parsed from `ffmpeg -i`, for when ffprobe is unavailable.

    ffmpeg writes stream details to stderr and exits non-zero because no output was requested;
    both are expected here.
    """
    exe = ffmpeg_exe()
    if not exe:
        return None
    try:
        proc = subprocess.run([exe, "-hide_banner", "-i", str(path)],
                              capture_output=True, timeout=60, text=True)
    except (subprocess.SubprocessError, OSError):
        return None
    text = (proc.stderr or "") + (proc.stdout or "")
    vm = _VIDEO.search(text)
    if not vm:
        return None

    duration = 0.0
    dm = _DURATION.search(text)
    if dm:
        duration = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3))
    line = text[vm.start():vm.start() + 400]
    fm = _FPS.search(line)
    bm = _BITRATE.search(text)
    return {"duration_sec": round(duration, 3), "width": int(vm.group(2)),
            "height": int(vm.group(3)), "fps": round(float(fm.group(1)), 3) if fm else 0.0,
            "codec": vm.group(1), "bitrate": int(bm.group(1)) * 1000 if bm else 0,
            "has_audio": "Audio:" in text, "probe": "ffmpeg"}


def describe() -> Dict[str, object]:
    """What was found, for the preflight report."""
    return {"ffmpeg": ffmpeg_exe() or "MISSING", "ffprobe": ffprobe_exe() or "MISSING"}
