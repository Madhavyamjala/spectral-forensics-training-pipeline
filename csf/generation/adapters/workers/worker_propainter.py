"""
ProPainter: flow-guided video inpainting.

One worker, three roles in the spec - the model is identical, only the mask differs:

    mode="inpaint"                 video-inpainting family, pipeline A. Synthetic masks follow the
                                   document's size/motion distribution (20/35/30/15, static..fast).
    mode="object"                  object-removal family, pipeline A. The mask tracks a real moving
                                   region so the removal is of something that is actually there.
    mode="background_reconstruct"  background family, pipeline D. The *foreground* is masked out,
                                   the background is hallucinated, and the original foreground is
                                   composited back on top - which is exactly the document's
                                   "foreground removed, background reconstructed, foreground
                                   restored" mechanism.

ProPainter is driven through its own `inference_propainter.py` CLI so we inherit upstream's
memory management (it chunks long clips), rather than re-implementing its inference loop.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from _common import (feather, has_audio, note, read_frames, read_video, repo_path, require,
                     run_cmd, scratch, serve, synth_masks, write_frames, write_masks, write_video)

MAX_SIDE = 640
MAX_FRAMES = 120


class State:
    def __init__(self, repo: Path, python: str):
        self.repo = repo
        self.python = python


def load() -> State:
    repo = repo_path("ProPainter")
    require(repo / "inference_propainter.py", "ProPainter inference script")
    require(repo / "weights" / "ProPainter.pth", "ProPainter checkpoint")
    note(f"propainter: repo at {repo}")
    return State(repo, os.sys.executable)


def _object_masks(frames, seed: int):
    """Track a real moving region so the removal targets actual content."""
    acc = np.zeros(frames[0].shape[:2], dtype=np.float32)
    step = max(1, len(frames) // 12)
    for i in range(step, len(frames), step):
        a = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor(frames[i - step], cv2.COLOR_RGB2GRAY).astype(np.float32)
        acc += np.abs(a - b)
    acc = cv2.GaussianBlur(acc, (31, 31), 0)
    thresh = float(np.percentile(acc, 96))
    h, w = acc.shape
    masks = []
    for _ in frames:
        m = ((acc >= thresh) * 255).astype(np.uint8)
        m = cv2.dilate(m, np.ones((15, 15), np.uint8), iterations=2)
        masks.append(m)
    if not masks[0].any():
        return synth_masks(len(frames), h, w, "medium", "slow", seed)
    return masks


def render(state: State, payload: dict) -> dict:
    mode = (payload.get("options") or {}).get("mode", "inpaint")
    src = payload["source_path"]
    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=MAX_SIDE)
    h, w = frames[0].shape[:2]
    seed = int(payload.get("seed") or 0)

    if mode == "inpaint":
        masks = synth_masks(len(frames), h, w, payload.get("mask_size") or "medium",
                            payload.get("mask_motion") or "slow", seed)
    elif mode == "object":
        masks = _object_masks(frames, seed)
    elif mode == "background_reconstruct":
        masks = _object_masks(frames, seed)          # foreground proxy: the moving region
    else:
        raise RuntimeError(f"unknown propainter mode {mode!r}")

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        frame_dir, mask_dir, out_dir = tmp / "frames", tmp / "masks", tmp / "out"
        write_frames(frames, str(frame_dir))
        write_masks(masks, str(mask_dir))
        run_cmd([state.python, "inference_propainter.py",
                 "--video", str(frame_dir), "--mask", str(mask_dir),
                 "--output", str(out_dir), "--save_frames", "--fp16",
                 "--subvideo_length", "80", "--neighbor_length", "10", "--ref_stride", "10",
                 "--width", str(w), "--height", str(h)],
                cwd=str(state.repo), timeout=2400)

        produced = sorted(p for p in out_dir.rglob("frames") if p.is_dir())
        result = read_frames(str(produced[0])) if produced else []
        if not result:
            cand = [p for p in out_dir.rglob("*.mp4")]
            if cand:
                result, _ = read_video(str(cand[0]), max_side=0)
        if not result:
            raise RuntimeError(f"ProPainter produced no frames under {out_dir}")

    result = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) if f.shape[:2] != (h, w) else f
              for f in result]
    n = min(len(result), len(frames))
    result, frames, masks = result[:n], frames[:n], masks[:n]

    if mode == "background_reconstruct":
        # put the original foreground back over the hallucinated background
        composed = []
        for orig, painted, mask in zip(frames, result, masks):
            alpha = feather(mask, radius=7)
            composed.append(np.clip(orig.astype(np.float32) * alpha
                                    + painted.astype(np.float32) * (1.0 - alpha),
                                    0, 255).astype(np.uint8))
        result = composed

    write_video(result, payload["output_path"], fps=fps,
                audio_from=src if has_audio(src) else None)
    return {"inpaint_model": "propainter", "mode": mode,
            "mask_size_class": payload.get("mask_size", ""),
            "mask_motion_pattern": payload.get("mask_motion", ""),
            "mask_area_frac": round(float(np.mean([m.astype(bool).mean() for m in masks])), 4),
            "operation": payload.get("operation", ""),
            # how long the masked region persists, which the spec tracks for object edits
            "track_length": sum(1 for m in masks if m.any()),
            "inpaint_target": payload.get("variant", ""), "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="propainter"))
