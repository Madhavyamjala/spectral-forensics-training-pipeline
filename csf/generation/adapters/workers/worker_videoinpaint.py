"""
E2FGVI-HQ / STTN / FuseFormer - the three transformer video-inpainting baselines.

These three share a lineage and a calling convention (frames folder + masks folder -> inpainted
clip), so one worker drives all of them, selected by `options["model"]`. Masks follow the
document's size and motion distribution, which is specified as independent of which model runs.

Upstream publishes checkpoints on Google Drive, which cannot be fetched unattended, so the env
build clones the repos but leaves the weights to be staged. `load()` checks for them and fails
with the exact path it wants rather than starting and dying on the first job.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from _common import (has_audio, note, read_frames, read_video, repo_path, require, run_cmd,
                     scratch, serve, synth_masks, write_frames, write_masks, write_video)

MAX_SIDE = 512
MAX_FRAMES = 100

SPECS = {
    "e2fgvi_hq": {"repo": "E2FGVI", "script": "test.py", "model": "e2fgvi_hq",
                  "ckpt": "release_model/E2FGVI-HQ-CVPR22.pth"},
    "sttn":      {"repo": "STTN", "script": "test.py", "model": "sttn",
                  "ckpt": "checkpoints/sttn.pth"},
    "fuseformer": {"repo": "FuseFormer", "script": "test.py", "model": "fuseformer",
                   "ckpt": "checkpoints/fuseformer.pth"},
}


class State:
    def __init__(self, name: str, repo: Path, ckpt: Path, spec: dict):
        """Store the reusable components required by this model worker."""
        self.name, self.repo, self.ckpt, self.spec = name, repo, ckpt, spec


def load() -> State:
    """Load the model and return its reusable worker state."""
    name = os.environ.get("CSF_ADAPTER", "")
    spec = SPECS.get(name)
    if spec is None:
        raise RuntimeError(f"worker_videoinpaint needs CSF_ADAPTER set to one of {sorted(SPECS)}, "
                           f"got {name!r}")
    repo = repo_path(spec["repo"])
    require(repo / spec["script"], f"{name} inference script")
    ckpt = require(repo / spec["ckpt"],
                   f"{name} checkpoint (upstream hosts it on Google Drive - stage it manually)")
    note(f"{name}: repo {repo}, checkpoint {ckpt}")
    return State(name, repo, ckpt, spec)


def render(state: State, payload: dict) -> dict:
    """Render one generation job with the loaded worker state."""
    src = payload["source_path"]
    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=MAX_SIDE)
    h, w = frames[0].shape[:2]
    # these models expect dimensions divisible by 8 (STTN/FuseFormer use fixed 432x240)
    w8, h8 = max(64, (w // 8) * 8), max(64, (h // 8) * 8)
    frames = [cv2.resize(f, (w8, h8), interpolation=cv2.INTER_AREA) for f in frames]
    masks = synth_masks(len(frames), h8, w8, payload.get("mask_size") or "medium",
                        payload.get("mask_motion") or "slow", int(payload.get("seed") or 0))

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        # FuseFormer names its output after the *basename* of --video and writes it into the
        # working directory, which is the clone every FuseFormer worker shares. With a fixed
        # name ("frames") two jobs in flight would overwrite each other, and the newest-mp4
        # search below could hand one job another's video - so the name carries the job's
        # unique scratch directory and the file is read by that exact name.
        tag = tmp.name
        frame_dir, mask_dir = tmp / f"{tag}_frames", tmp / f"{tag}_masks"
        write_frames(frames, str(frame_dir))
        write_masks(masks, str(mask_dir))
        cmd = [os.sys.executable, state.spec["script"], "--video", str(frame_dir),
               "--mask", str(mask_dir), "--ckpt", str(state.ckpt)]
        if state.name == "e2fgvi_hq":
            cmd += ["--model", "e2fgvi_hq", "--set_size", "--width", str(w8), "--height", str(h8),
                    "--savefps", str(int(round(fps)))]
        run_cmd(cmd, cwd=str(state.repo), timeout=1800)

        in_repo = Path(state.repo) / f"{frame_dir.name}_result.mp4"
        candidates = list(tmp.rglob("*.mp4")) + list(Path(state.repo, "results").glob("*.mp4"))
        if in_repo.exists():
            candidates.append(in_repo)
        produced = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        result = []
        if produced:
            result, _ = read_video(str(produced[0]), max_side=0)
        if in_repo.exists():
            in_repo.unlink()          # never leave a job's output in the shared clone
        if not result:
            dirs = [p for p in tmp.rglob("*") if p.is_dir() and p.name.startswith("result")]
            if dirs:
                result = read_frames(str(dirs[0]))
        if not result:
            raise RuntimeError(f"{state.name} produced no output for {payload['job_id']}")

    result = [cv2.resize(f, (w8, h8), interpolation=cv2.INTER_AREA) if f.shape[:2] != (h8, w8) else f
              for f in result]
    write_video(result, payload["output_path"], fps=fps,
                audio_from=src if has_audio(src) else None)
    return {"inpaint_model": state.name, "inpaint_target": payload.get("variant", ""),
            "mask_size_class": payload.get("mask_size", ""),
            "mask_motion_pattern": payload.get("mask_motion", ""),
            "mask_area_frac": round(float(np.mean([m.astype(bool).mean() for m in masks])), 4),
            "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="videoinpaint"))
