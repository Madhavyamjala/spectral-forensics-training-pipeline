"""
TokenFlow + Stable Diffusion - video-to-video transformation (pipeline A).

Training-free: the clip is DDIM-inverted, then edited with a prompt while diffusion features are
propagated across frames so the result stays temporally coherent. Applied to the whole frame,
not a localized region, which is what distinguishes this family from the inpainting ones.

The prompt comes from the job's transformation family (photorealistic->artistic, ->stylized/CG,
appearance/material, semantic/environmental), assigned by the planner to match the document's
transformation-type matrix.

TokenFlow's upstream driver is config-file based, so the worker writes a per-job YAML and calls
`run_tokenflow_pnp.py`, rather than reproducing its inversion loop.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from _common import (has_audio, note, read_frames, read_video, repo_path, require, run_cmd,
                     scratch, serve, write_frames, write_video)

MAX_SIDE = 512
N_FRAMES = 40                      # TokenFlow inverts every frame; 40 keeps a job near ~3 min


class State:
    def __init__(self, repo: Path, sd_id: str):
        """Store the reusable components required by this model worker."""
        self.repo, self.sd_id = repo, sd_id


def load() -> State:
    """Load the model and return its reusable worker state."""
    repo = repo_path("TokenFlow")
    require(repo / "run_tokenflow_pnp.py", "TokenFlow driver script")
    require(repo / "preprocess.py", "TokenFlow preprocess script")
    sd_id = os.environ.get("CSF_SD_ID", "stabilityai/stable-diffusion-2-1-base")
    note(f"tokenflow: repo {repo}, sd {sd_id}")
    return State(repo, sd_id)


def render(state: State, payload: dict) -> dict:
    """Render one generation job with the loaded worker state."""
    src = payload["source_path"]
    prompt = payload.get("prompt") or "in the style of an oil painting"
    seed = int(payload.get("seed") or 0)
    frames, fps = read_video(src, max_frames=N_FRAMES, max_side=MAX_SIDE)
    h, w = frames[0].shape[:2]
    w8, h8 = max(64, (w // 64) * 64), max(64, (h // 64) * 64)
    frames = [cv2.resize(f, (w8, h8), interpolation=cv2.INTER_AREA) for f in frames]

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        frame_dir = tmp / "frames"
        write_frames(frames, str(frame_dir))
        latents = tmp / "latents"

        # 1. DDIM inversion of the clip
        run_cmd([os.sys.executable, "preprocess.py", "--data_path", str(frame_dir),
                 "--sd_version", "2.1", "--inversion_prompt", "",
                 "--save_dir", str(latents), "--steps", "500",
                 "--n_frames", str(len(frames)), "--H", str(h8), "--W", str(w8)],
                cwd=str(state.repo), timeout=2400)

        # 2. prompt-guided edit with cross-frame feature propagation
        cfg = {"seed": seed, "device": "cuda", "output_path": str(tmp / "out"),
               "data_path": str(frame_dir), "latents_path": str(latents),
               "n_inversion_steps": 500, "n_frames": len(frames), "sd_version": "2.1",
               "guidance_scale": 7.5, "n_timesteps": 50, "prompt": prompt,
               "negative_prompt": "ugly, blurry, low res, watermark, text",
               "batch_size": 8, "pnp_attn_t": 0.5, "pnp_f_t": 0.8}
        cfg_path = tmp / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        run_cmd([os.sys.executable, "run_tokenflow_pnp.py", "--config_path", str(cfg_path)],
                cwd=str(state.repo), timeout=3600)

        result = []
        mp4s = sorted(tmp.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mp4s:
            result, _ = read_video(str(mp4s[0]), max_side=0)
        if not result:
            dirs = sorted((p for p in tmp.rglob("*") if p.is_dir() and any(p.glob("*.png"))),
                          key=lambda p: p.stat().st_mtime, reverse=True)
            for d in dirs:
                result = read_frames(str(d))
                if len(result) >= max(4, len(frames) // 2):
                    break
        if not result:
            raise RuntimeError(f"TokenFlow produced no frames for {payload['job_id']}")

    result = [cv2.resize(f, (w8, h8), interpolation=cv2.INTER_AREA) if f.shape[:2] != (h8, w8) else f
              for f in result]
    write_video(result, payload["output_path"], fps=fps,
                audio_from=src if has_audio(src) else None)
    return {"transform_model": "tokenflow", "transformation_family": payload.get("variant", ""),
            "prompt": prompt, "sd_version": state.sd_id, "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="tokenflow"))
