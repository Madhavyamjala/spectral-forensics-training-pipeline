"""
VACE (Wan2.1) - all-in-one masked video editing.

VACE fills four spec slots whose named models cannot be run unattended (AnyV2V x2,
VideoComposer, InsV2V), across two different tasks:

    task="inpainting"  mask-guided object insertion. The mask marks the region VACE regenerates
                       from the prompt, which is the conditioning VideoComposer and AnyV2V's
                       insertion path provide.
    task="depth"       whole-frame prompt-driven transformation for the video-to-video family.

All four slots are recorded in the manifest as `vace`, never under the slot names. That matters:
they share one VAE and one DiT, so their generative fingerprint is identical. Reporting them as
four distinct "methods" would show a single fingerprint under four labels and make any
per-method attribution result meaningless.

The 1.3B model is used (480p, Apache-2.0); 14B needs multi-GPU and does not fit the run budget.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from _common import (has_audio, note, read_video, repo_path, require, run_cmd, scratch, serve,
                     synth_masks, write_video)

#: VACE's Wan2.1 backbone generates 4n+1 frames; 49 keeps a job near ~3 min on an H200.
N_FRAMES = 49
WIDTH, HEIGHT = 832, 480


class State:
    def __init__(self, repo: Path, ckpt_dir: Path):
        """Store the reusable components required by this model worker."""
        self.repo, self.ckpt_dir = repo, ckpt_dir


def load() -> State:
    """Load the model and return its reusable worker state."""
    repo = repo_path("VACE")
    require(repo / "vace" / "vace_wan_inference.py", "VACE inference script")
    ckpt_dir = Path(os.environ.get("CSF_ENV_ROOT", ".")) / "weights" / "Wan2.1-VACE-1.3B"
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        raise RuntimeError(
            f"VACE weights missing at {ckpt_dir}. Fetch them with:\n"
            f"    hf download Wan-AI/Wan2.1-VACE-1.3B --local-dir {ckpt_dir}")
    note(f"vace: repo {repo}, ckpt {ckpt_dir}")
    return State(repo, ckpt_dir)


def _insertion_mask(frames, seed: int):
    """A plausible, mostly-static region to insert into: a low-motion patch of the frame."""
    h, w = frames[0].shape[:2]
    acc = np.zeros((h, w), np.float32)
    step = max(1, len(frames) // 8)
    for i in range(step, len(frames), step):
        a = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor(frames[i - step], cv2.COLOR_RGB2GRAY).astype(np.float32)
        acc += np.abs(a - b)
    acc = cv2.GaussianBlur(acc, (41, 41), 0)
    # insert where nothing is moving, so the inserted object does not fight the subject
    quiet = (acc <= np.percentile(acc, 35)).astype(np.uint8)
    quiet = cv2.erode(quiet, np.ones((25, 25), np.uint8), iterations=1)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(quiet, 8)
    if n <= 1:
        return synth_masks(len(frames), h, w, "medium", "static", seed)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cents[idx]
    side = int(min(h, w) * 0.28)
    mask = np.zeros((h, w), np.uint8)
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    cv2.rectangle(mask, (x0, y0), (x0 + side, y0 + side), 255, -1)
    return [mask] * len(frames)


def render(state: State, payload: dict) -> dict:
    """Render one generation job with the loaded worker state."""
    opts = payload.get("options") or {}
    task = opts.get("task", "inpainting")
    src = payload["source_path"]
    seed = int(payload.get("seed") or 0) % (2 ** 31)

    frames, fps = read_video(src, max_frames=N_FRAMES, max_side=max(WIDTH, HEIGHT))
    frames = [cv2.resize(f, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA) for f in frames]
    while len(frames) < N_FRAMES:                 # Wan2.1 wants exactly 4n+1 frames
        frames.append(frames[-1])
    frames = frames[:N_FRAMES]

    prompt = payload.get("prompt") or ""
    if task == "inpainting" and not prompt:
        prompt = "a natural object resting in the scene, photorealistic"
    if not prompt:
        prompt = "the same scene, restyled, photorealistic"

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        src_video, out_dir = tmp / "src.mp4", tmp / "out"
        write_video(frames, str(src_video), fps=fps)

        cmd = [sys.executable, "vace/vace_wan_inference.py",
               "--ckpt_dir", str(state.ckpt_dir), "--model_name", "vace-1.3B",
               "--size", f"{WIDTH}*{HEIGHT}", "--frame_num", str(N_FRAMES),
               "--src_video", str(src_video), "--prompt", prompt,
               "--base_seed", str(seed), "--save_dir", str(out_dir)]

        mask_frac = 0.0
        if task == "inpainting":
            masks = _insertion_mask(frames, seed)
            mask_mp4 = tmp / "mask.mp4"
            write_video([np.repeat(m[:, :, None], 3, axis=2) for m in masks], str(mask_mp4),
                        fps=fps)
            cmd += ["--src_mask", str(mask_mp4)]
            mask_frac = float(np.mean([m.astype(bool).mean() for m in masks]))

        run_cmd(cmd, cwd=str(state.repo), timeout=3600)

        produced = sorted(list(out_dir.rglob("*.mp4")) +
                          list((Path(state.repo) / "results").rglob("*.mp4")),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        produced = [p for p in produced if p.name not in ("src.mp4", "mask.mp4")]
        if not produced:
            raise RuntimeError(f"VACE produced no output for {payload['job_id']}")
        result, out_fps = read_video(str(produced[0]), max_side=0)

    write_video(result, payload["output_path"], fps=out_fps or fps,
                audio_from=src if has_audio(src) else None)
    return {"edit_model": "vace", "vace_backbone": "Wan2.1-VACE-1.3B", "vace_task": task,
            "substitutes_for": payload.get("spec_model", ""), "prompt": prompt,
            "operation": payload.get("operation", ""),
            "transformation_family": payload.get("variant", ""),
            "mask_area_frac": round(mask_frac, 4), "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="vace"))
