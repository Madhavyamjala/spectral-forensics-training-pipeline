"""
DiffuEraser - diffusion video object removal.

Fills the Object-WIPER slot, which has no public code release at all. The substitution is chosen
for artifact diversity, not just availability: ProPainter (already wired in the neighbouring
slot) propagates flow and features from visible frames, whereas DiffuEraser hallucinates the
occluded content with a Stable-Diffusion prior. Those leave different traces, which is the whole
point of having four pipelines in the family.

Recorded in the manifest as `diffueraser`. Apache-2.0, weights on the Hub.

Note on metrics: published PSNR comparisons against ProPainter disagree in direction, because
PSNR rewards conservative blur over plausible detail. Treat DiffuEraser as a different artifact
class rather than a strictly better inpainter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from _common import (has_audio, note, read_video, repo_path, require, run_cmd, scratch, serve,
                     synth_masks, write_video)

MAX_SIDE = 640
MAX_FRAMES = 100


class State:
    def __init__(self, repo: Path, weights: Path):
        """Store the reusable components required by this model worker."""
        self.repo, self.weights = repo, weights


def load() -> State:
    """Load the model and return its reusable worker state."""
    repo = repo_path("DiffuEraser")
    script = repo / "run_diffueraser.py"
    if not script.exists():
        alt = list(repo.glob("*.py"))
        raise RuntimeError(f"DiffuEraser entry script not found at {script}; repo has {alt[:6]}")
    weights = repo / "weights" / "diffuEraser"
    if not weights.exists() or not any(weights.iterdir()):
        raise RuntimeError(
            f"DiffuEraser weights missing at {weights}. Fetch them with:\n"
            f"    hf download lixiaowen/diffuEraser --local-dir {weights}")
    note(f"diffueraser: repo {repo}")
    return State(repo, weights)


def _moving_object_mask(frames, seed: int):
    """Build a deterministic moving mask for object removal."""
    acc = np.zeros(frames[0].shape[:2], np.float32)
    step = max(1, len(frames) // 12)
    for i in range(step, len(frames), step):
        a = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor(frames[i - step], cv2.COLOR_RGB2GRAY).astype(np.float32)
        acc += np.abs(a - b)
    acc = cv2.GaussianBlur(acc, (31, 31), 0)
    thresh = float(np.percentile(acc, 95))
    m = ((acc >= thresh) * 255).astype(np.uint8)
    m = cv2.dilate(m, np.ones((17, 17), np.uint8), iterations=2)
    if not m.any():
        return synth_masks(len(frames), *acc.shape, "medium", "slow", seed)
    return [m] * len(frames)


def render(state: State, payload: dict) -> dict:
    """Render one generation job with the loaded worker state."""
    src = payload["source_path"]
    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=MAX_SIDE)
    h, w = frames[0].shape[:2]
    masks = _moving_object_mask(frames, int(payload.get("seed") or 0))

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        in_mp4, mask_mp4, out_mp4 = tmp / "in.mp4", tmp / "mask.mp4", tmp / "out.mp4"
        write_video(frames, str(in_mp4), fps=fps)
        write_video([np.repeat(m[:, :, None], 3, axis=2) for m in masks], str(mask_mp4), fps=fps)

        run_cmd([sys.executable, "run_diffueraser.py",
                 "--input_video", str(in_mp4), "--input_mask", str(mask_mp4),
                 "--save_path", str(tmp / "results"),
                 "--video_length", str(min(len(frames), MAX_FRAMES)),
                 "--max_img_size", str(MAX_SIDE)],
                cwd=str(state.repo), timeout=2400)

        produced = sorted(tmp.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        produced = [p for p in produced if p.name not in ("in.mp4", "mask.mp4")]
        if not produced:
            raise RuntimeError(f"DiffuEraser produced no output for {payload['job_id']}")
        result, out_fps = read_video(str(produced[0]), max_side=0)

    write_video(result, payload["output_path"], fps=out_fps or fps,
                audio_from=src if has_audio(src) else None)
    return {"edit_model": "diffueraser", "operation": payload.get("operation", "object_removal"),
            "substitutes_for": payload.get("spec_model", "object_wiper"),
            "mask_area_frac": round(float(np.mean([m.astype(bool).mean() for m in masks])), 4),
            "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="diffueraser"))
