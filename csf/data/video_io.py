"""
Video download (Hugging Face Hub) and frame decoding.

Input : repo id / repo path of one video; local file path for decoding.
Output: `download_video` -> local Path (retries rate limits with exponential backoff);
        `decode_frames`  -> list of `num_frames` uniformly spaced RGB uint8 frames at native
                            resolution (longest side capped to `max_side`);
        `to_vlm_frames`  -> (T, S, S, 3) uint8 array resized for the VLMs;
        `make_mosaic`    -> single square RGB mosaic image for Llama-3.2-Vision;
        `encode_jpegs` / `decode_jpegs` -> compact frame storage for the feature cache.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

from csf.logging_utils import get_logger

log = get_logger("data.video")


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    resp = getattr(exc, "response", None)
    return ("429" in msg or "rate limit" in msg or "too many requests" in msg
            or getattr(resp, "status_code", None) == 429)


def download_video(repo_id: str, repo_path: str, video_dir: Path, revision: Optional[str] = None,
                   max_retries: int = 8, base_backoff: float = 10.0) -> Path:
    target = video_dir / repo_path
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        try:
            return Path(hf_hub_download(repo_id, repo_path, repo_type="dataset", revision=revision,
                                        local_dir=str(video_dir)))
        except Exception as exc:
            attempt += 1
            transient = _is_rate_limit(exc) or isinstance(exc, (ConnectionError, TimeoutError)) \
                or "timed out" in str(exc).lower() or "connection" in str(exc).lower()
            if not transient or attempt > max_retries:
                raise RuntimeError(f"download failed for {repo_path} after {attempt} attempt(s): "
                                   f"{type(exc).__name__}: {exc}") from exc
            wait = min(base_backoff * 2 ** (attempt - 1), 300)
            log.warning("Transient download error on %s (attempt %d/%d): %s -> retry in %.0fs",
                        repo_path, attempt, max_retries, str(exc)[:200], wait)
            time.sleep(wait)


def decode_frames(path: Path, num_frames: int, max_side: int = 1024) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 0
            while cap.grab():
                total += 1
            cap.release()
            cap = cv2.VideoCapture(str(path))
        if total <= 0:
            raise RuntimeError(f"Video {path} has no decodable frames")
        wanted = np.linspace(0, total - 1, num_frames).round().astype(int)
        wanted_set = set(wanted.tolist())
        grabbed = {}
        idx = 0
        last = wanted.max()
        while idx <= last:
            if not cap.grab():
                break
            if idx in wanted_set:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    grabbed[idx] = frame
            idx += 1
    finally:
        cap.release()

    if not grabbed:
        raise RuntimeError(f"Decoded 0 frames from {path} (reported frame count {total})")
    frames = []
    available = sorted(grabbed)
    for w in wanted:
        nearest = min(available, key=lambda a: abs(a - w))
        frames.append(grabbed[nearest])

    h, w = frames[0].shape[:2]
    scale = min(1.0, max_side / max(h, w))
    out = []
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        if scale < 1.0:
            f = cv2.resize(f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


def to_vlm_frames(frames: List[np.ndarray], size: int) -> np.ndarray:
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])


def make_mosaic(frames: np.ndarray, n_tiles: int, size: int) -> np.ndarray:
    """Grid of `n_tiles` evenly spaced frames (e.g. 2x2) as one square image of side `size`."""
    grid = int(math.ceil(math.sqrt(n_tiles)))
    cell = size // grid
    idx = np.linspace(0, len(frames) - 1, n_tiles).round().astype(int)
    canvas = np.zeros((cell * grid, cell * grid, 3), dtype=np.uint8)
    for k, i in enumerate(idx):
        r, c = divmod(k, grid)
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cv2.resize(frames[i], (cell, cell),
                                                                              interpolation=cv2.INTER_AREA)
    if canvas.shape[0] != size:
        canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return canvas


def encode_jpegs(frames: np.ndarray, quality: int) -> tuple:
    blobs = []
    for f in frames:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        blobs.append(buf.ravel())
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    return np.concatenate(blobs).astype(np.uint8), offsets


def decode_jpegs(buffer: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    frames = []
    for i in range(len(offsets) - 1):
        img = cv2.imdecode(buffer[offsets[i]:offsets[i + 1]], cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames)
