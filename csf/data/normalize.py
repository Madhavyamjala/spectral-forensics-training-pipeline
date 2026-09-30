"""
Format normalisation used by the SAFER evaluation protocol.

Real and generated videos usually differ in container properties (resolution, aspect ratio, frame rate,
duration, codec, bitrate, audio), and a classifier on those alone can separate them. Every video is
therefore re-encoded onto one fixed canvas before evaluation:

    letterboxed to 854x480 (aspect ratio kept, black bars, never cropped), resampled to 24 fps,
    cut to the first 10 s, audio removed, metadata stripped, H.264 CRF 23, yuv420p.

Scaling only the short side, or refusing to upscale, would leave resolution and aspect ratio as a class
signal; letterboxing keeps content that sits near the border (an edit there would otherwise be cropped
away). Duration is only capped, so where real and fake clip lengths differ it survives; the
metadata-shortcut classifier on the normalised files measures what is left.

Library: `normalize_video(src, dst)`.
CLI:     python -m csf.data.normalize SRC_DIR DST_DIR [--size 854x480 --fps 24 --max-seconds 10 --crf 23]
         mirrors SRC_DIR's tree into DST_DIR as .mp4; existing outputs are kept, so it resumes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_SIZE = (854, 480)
DEFAULT_FPS = 24.0
DEFAULT_MAX_SECONDS = 10.0
DEFAULT_CRF = 23
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".gif"}


def parse_size(spec: str) -> Tuple[int, int]:
    m = re.match(r"^(\d+)x(\d+)$", spec)
    if not m or int(m.group(1)) % 2 or int(m.group(2)) % 2:
        raise ValueError(f"size must be even WxH such as 854x480, got {spec!r}")
    return int(m.group(1)), int(m.group(2))


def _ffmpeg() -> str:
    from csf.generation.ffmpeg_tools import ffmpeg_exe
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg not found; install it (e.g. `pip install imageio-ffmpeg`) or set CSF_FFMPEG")
    return exe


def normalize_video(src: str, dst: Path, size: Tuple[int, int] = DEFAULT_SIZE, fps: float = DEFAULT_FPS,
                    max_seconds: float = DEFAULT_MAX_SECONDS, crf: int = DEFAULT_CRF,
                    ffmpeg: Optional[str] = None) -> Tuple[bool, str]:
    """Re-encode `src` onto the fixed canvas at `dst`. Returns (ok, note); never raises for a bad video."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return True, "cached"
    from csf.generation.kinetics import probe_video
    try:
        info = probe_video(Path(src))
    except Exception:
        info = None
    if not info or not info.get("width") or not info.get("height"):
        return False, "unreadable"
    w, h = size
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=bicubic,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps:g}")
    cmd = [ffmpeg or _ffmpeg(), "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-t", f"{max_seconds:g}",
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an",
           "-map_metadata", "-1", "-movflags", "+faststart", str(tmp)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False, "ffmpeg timeout"
    if res.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False, (res.stderr or "ffmpeg failed").strip().splitlines()[-1][:200]
    tmp.replace(dst)
    return True, "ok"


def normalize_tree(src_dir: Path, dst_dir: Path, size: Tuple[int, int] = DEFAULT_SIZE, fps: float = DEFAULT_FPS,
                   max_seconds: float = DEFAULT_MAX_SECONDS, crf: int = DEFAULT_CRF, workers: int = 8) -> dict:
    """Mirror every video under `src_dir` into `dst_dir` (same relative path, .mp4)."""
    ffmpeg = _ffmpeg()
    srcs = sorted(p for p in Path(src_dir).rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    counts = {"ok": 0, "cached": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(normalize_video, str(p), Path(dst_dir) / p.relative_to(src_dir).with_suffix(".mp4"),
                            size, fps, max_seconds, crf, ffmpeg): p for p in srcs}
        for i, fut in enumerate(as_completed(futs), 1):
            ok, note = fut.result()
            counts["cached" if note == "cached" else ("ok" if ok else "failed")] += 1
            if not ok:
                failures.append((str(futs[fut]), note))
            if i % 500 == 0:
                print(f"{i}/{len(srcs)} {counts}", flush=True)
    return {**counts, "total": len(srcs), "failures": failures[:20]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Normalise a folder of videos onto one fixed canvas (SAFER protocol).")
    ap.add_argument("src_dir")
    ap.add_argument("dst_dir")
    ap.add_argument("--size", default=f"{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    ap.add_argument("--crf", type=int, default=DEFAULT_CRF)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    res = normalize_tree(Path(a.src_dir), Path(a.dst_dir), parse_size(a.size), a.fps, a.max_seconds, a.crf, a.workers)
    print(res)
    return 0 if res["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
