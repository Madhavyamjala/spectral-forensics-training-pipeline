"""
Shared helpers for the manipulation workers.

Workers run inside their model's own virtual environment and must NOT import `csf` - that
package's dependencies are not installed there. This module is put on PYTHONPATH by the driver
and is the only thing workers share. It deliberately depends on nothing beyond numpy, OpenCV and
ffmpeg, all of which every adapter env installs anyway.

It provides:
  * `serve()`      - the line-delimited JSON protocol loop described in `adapters/base.py`
  * video io       - decode to frames, write frames back to H.264 mp4, remux the original audio
  * mask synthesis - the inpainting family's mask size / motion classes from the spec
  * small utilities - deterministic RNG, largest-face box, safe resize
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------------------


def emit(msg: Dict[str, object]) -> None:
    """Write one protocol message to stdout. Never write anything else to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def note(*parts: object) -> None:
    """Progress/diagnostics go to stderr, which the driver captures into the run log."""
    print(*parts, file=sys.stderr, flush=True)


def serve(load: Callable[[], object], render: Callable[[object, Dict[str, object]], Dict[str, object]],
          model_name: str = "") -> int:
    """Load the model once, then answer jobs until the driver says shutdown.

    `render` returns a metadata dict and must leave an mp4 at `payload["output_path"]`.
    Any exception is reported as a failed job; the worker stays alive for the next one.
    """
    try:
        state = load()
    except Exception as exc:                                  # noqa: BLE001 - report and exit
        emit({"event": "error", "fatal": True,
              "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        note(traceback.format_exc())
        return 1

    emit({"event": "ready", "model": model_name or os.path.basename(sys.argv[0]),
          "device": os.environ.get("CSF_GPU", "?")})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"event": "result", "job_id": None, "ok": False, "error": f"bad json: {exc}"})
            continue
        if payload.get("cmd") == "shutdown":
            break
        job_id = payload.get("job_id")
        try:
            out = Path(payload["output_path"])
            out.parent.mkdir(parents=True, exist_ok=True)
            meta = render(state, payload) or {}
            if not out.exists() or out.stat().st_size == 0:
                raise RuntimeError(f"worker finished but produced no video at {out}")
            emit({"event": "result", "job_id": job_id, "ok": True,
                  "output_path": str(out), "metadata": meta})
        except Exception as exc:                              # noqa: BLE001 - one bad job is not fatal
            note(traceback.format_exc())
            emit({"event": "result", "job_id": job_id, "ok": False,
                  "error": f"{type(exc).__name__}: {exc}"[:600]})
    return 0


# --------------------------------------------------------------------------------------
# video io
# --------------------------------------------------------------------------------------


def read_video(path: str, max_frames: int = 0, max_side: int = 0) -> Tuple[List[np.ndarray], float]:
    """Decode a clip to a list of RGB frames plus its fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if max_side:
                h, w = frame.shape[:2]
                scale = min(1.0, max_side / max(h, w))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                                       interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {path}")
    return frames, float(fps if fps > 0 else 25.0)


def _even(n: int) -> int:
    """Round a dimension down to the nearest positive even integer."""
    return n if n % 2 == 0 else n - 1


def write_video(frames: Sequence[np.ndarray], path: str, fps: float = 25.0,
                audio_from: Optional[str] = None, crf: int = 18) -> str:
    """Encode RGB frames to H.264 mp4, optionally remuxing audio from the source clip.

    Uses ffmpeg when available (better rate control and yuv420p output, which every decoder in
    the training pipeline can read) and falls back to OpenCV's mp4v writer otherwise.
    """
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    h, w = max(2, _even(h)), max(2, _even(w))
    prepared = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) if f.shape[:2] != (h, w) else f
                for f in frames]

    if _have_ffmpeg():
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", f"{fps:.4f}", "-i", "pipe:0"]
        if audio_from and Path(audio_from).exists():
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac",
                    "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        try:
            for frame in prepared:
                proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0 or not Path(path).exists():
            raise RuntimeError(f"ffmpeg encoding failed: {err[-800:]}")
        return path

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path}")
    for frame in prepared:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


_FFMPEG: Optional[bool] = None


def _have_ffmpeg() -> bool:
    """Return whether the ffmpeg executable is available."""
    global _FFMPEG
    if _FFMPEG is None:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=20, check=True)
            _FFMPEG = True
        except (OSError, subprocess.SubprocessError):
            _FFMPEG = False
    return _FFMPEG


def extract_audio(video_path: str, out_wav: str, sample_rate: int = 16000) -> Optional[str]:
    """Pull a mono wav out of a clip; returns None when the clip has no audio track."""
    if not _have_ffmpeg():
        return None
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-vn",
           "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", str(out_wav)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not Path(out_wav).exists() or Path(out_wav).stat().st_size < 1024:
        return None
    return str(out_wav)


def has_audio(video_path: str) -> bool:
    """Return whether the video contains an audio stream."""
    if not _have_ffmpeg():
        return False
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                           "stream=index", "-of", "csv=p=0", str(video_path)],
                          capture_output=True, text=True)
    return bool(proc.stdout.strip())


def run_cmd(cmd: Sequence[str], cwd: Optional[str] = None, timeout: int = 1800) -> str:
    """Run an upstream repo's CLI, raising with its stderr tail on failure."""
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise RuntimeError(f"command failed ({' '.join(map(str, cmd[:4]))}...): {tail}")
    return proc.stdout


# --------------------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------------------

MASK_AREA = {"small": (0.02, 0.06), "medium": (0.06, 0.15),
             "large": (0.15, 0.30), "very_large_irregular": (0.30, 0.45)}


def synth_masks(n_frames: int, height: int, width: int, size_class: str = "medium",
                motion: str = "slow", seed: int = 0) -> List[np.ndarray]:
    """Binary masks (255 = inpaint here) following the spec's size and motion classes.

    Used when no real object track is available - the video-inpainting family's masks are
    synthetic by design ("mask control, independent of which model runs").
    """
    rng = np.random.default_rng(seed)
    lo, hi = MASK_AREA.get(size_class, MASK_AREA["medium"])
    area = float(rng.uniform(lo, hi))
    # blob roughly square, area as a fraction of the frame
    side = int(round(np.sqrt(area * height * width)))
    bw = max(8, min(width - 4, int(side * float(rng.uniform(0.8, 1.4)))))
    bh = max(8, min(height - 4, int(round(area * height * width / max(1, bw)))))

    cx, cy = float(rng.uniform(bw / 2, width - bw / 2)), float(rng.uniform(bh / 2, height - bh / 2))
    speed = {"static": 0.0, "slow": 0.6, "fast": 3.0, "intermittent": 1.5,
             "partially_occluded": 1.0}.get(motion, 0.6)
    angle = float(rng.uniform(0, 2 * np.pi))
    vx, vy = speed * np.cos(angle), speed * np.sin(angle)

    irregular = size_class == "very_large_irregular"
    masks: List[np.ndarray] = []
    for t in range(n_frames):
        mask = np.zeros((height, width), dtype=np.uint8)
        if motion == "intermittent" and (t // max(1, n_frames // 6)) % 2 == 1:
            masks.append(mask)                     # mask disappears for stretches
            continue
        cx = float(np.clip(cx + vx, bw / 2, max(bw / 2, width - bw / 2)))
        cy = float(np.clip(cy + vy, bh / 2, max(bh / 2, height - bh / 2)))
        if irregular:
            pts = []
            for k in range(10):
                ang = 2 * np.pi * k / 10
                r = float(rng.uniform(0.6, 1.25))
                pts.append([int(cx + np.cos(ang) * bw / 2 * r), int(cy + np.sin(ang) * bh / 2 * r)])
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        else:
            cv2.ellipse(mask, (int(cx), int(cy)), (max(4, bw // 2), max(4, bh // 2)),
                        0, 0, 360, 255, -1)
        if motion == "partially_occluded" and t % 3 == 0:
            cut = mask.copy()
            cut[:, : width // 2] = 0
            mask = cut if cut.any() else mask
        masks.append(mask)
    if not any(m.any() for m in masks):
        masks[0][height // 3: 2 * height // 3, width // 3: 2 * width // 3] = 255
    return masks


def mask_from_boxes(boxes: Sequence[Tuple[int, int, int, int]], height: int, width: int,
                    dilate: int = 9) -> np.ndarray:
    """Rasterize bounding boxes into per-frame binary masks."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
    if dilate > 0:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8), iterations=1)
    return mask


def feather(mask: np.ndarray, radius: int = 9) -> np.ndarray:
    """Soft alpha in [0,1] from a binary mask, for seam-free compositing."""
    k = max(1, radius | 1)
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), 0)[..., None]


def write_masks(masks: Sequence[np.ndarray], folder: str) -> str:
    """Write binary masks to a numbered PNG sequence."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(masks):
        cv2.imwrite(str(Path(folder) / f"{i:05d}.png"), m)
    return folder


def write_frames(frames: Sequence[np.ndarray], folder: str) -> str:
    """Write video frames to a numbered PNG sequence."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(Path(folder) / f"{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    return folder


def read_frames(folder: str) -> List[np.ndarray]:
    """Read a numbered image sequence from a directory."""
    files = sorted(Path(folder).glob("*.png")) + sorted(Path(folder).glob("*.jpg"))
    out = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is not None:
            out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out


def scratch(job_id: str) -> tempfile.TemporaryDirectory:
    """Create and return a clean scratch directory for a job."""
    safe = "".join(c if c.isalnum() else "_" for c in str(job_id))[:60]
    return tempfile.TemporaryDirectory(prefix=f"csfgen_{safe}_")


def env_root() -> Path:
    """Return the configured root of the model environment."""
    return Path(os.environ.get("CSF_ENV_ROOT", "."))


def repo_path(name: str) -> Path:
    """Locate a cloned upstream repo inside this adapter's env."""
    root = env_root() / "repos" / name
    if root.exists():
        return root
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and Path(entry).name == name and Path(entry).exists():
            return Path(entry)
    raise RuntimeError(f"upstream repo '{name}' is not present under {env_root()/'repos'}; "
                       f"rebuild the env with: python -m csf.generation.envs --build <env>")


def require(path: Path, what: str) -> Path:
    """Return a required path or raise when it does not exist."""
    if not Path(path).exists():
        raise RuntimeError(f"{what} is missing at {path}. Stage it before running this adapter "
                           f"(see docs/REGENERATION.md).")
    return Path(path)
