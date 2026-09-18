"""
DreamID-V - diffusion-transformer video face swapping.

Fills the SimSwap slot, whose weights are Google-Drive hosted. On identity preservation the
DreamID line reports 99.9% top-1 ID retrieval against SimSwap's 95.24%, and DreamID-V is the
video-native successor built on Wan2.1-1.3B, so temporal consistency comes from the backbone
rather than from post-hoc smoothing.

The trade is cost: this is a DiT sampling loop per clip, roughly 20x INSwapper's per-video time.
That is deliberate and priced into the run budget - it buys a genuinely different artifact class
(diffusion-synthesised faces) alongside INSwapper's ArcFace latent swap, which is what the family
needs if the detector is to generalise beyond one swap mechanism.

Recorded in the manifest as `dreamid_v`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from _common import (has_audio, note, read_video, repo_path, require, run_cmd, scratch, serve,
                     write_video)

N_FRAMES = 49
MAX_SIDE = 640


class State:
    def __init__(self, repo: Path, ckpt_dir: Path, script: Path):
        """Store the reusable components required by this model worker."""
        self.repo, self.ckpt_dir, self.script = repo, ckpt_dir, script


def load() -> State:
    """Load the model and return its reusable worker state."""
    repo = repo_path("DreamID-V")
    ckpt_dir = Path(os.environ.get("CSF_ENV_ROOT", ".")) / "weights" / "DreamID-V"
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        raise RuntimeError(
            f"DreamID-V weights missing at {ckpt_dir}. Fetch them with:\n"
            f"    hf download XuGuo699/DreamID-V --local-dir {ckpt_dir}")
    candidates = [repo / "inference.py", repo / "infer.py", repo / "scripts" / "inference.py"]
    script = next((c for c in candidates if c.exists()), None)
    if script is None:
        raise RuntimeError(
            f"DreamID-V inference entry point not found under {repo}. Looked for "
            f"{[str(c.relative_to(repo)) for c in candidates]}; the repo contains "
            f"{[p.name for p in repo.glob('*.py')][:8]}. Point the worker at the right script "
            f"before enabling this adapter.")
    note(f"dreamid: repo {repo}, script {script.name}")
    return State(repo, ckpt_dir, script)


def _best_face_frame(frames):
    """Sharpest frame of the identity donor - the swap is conditioned on a single source face."""
    best, score = frames[0], -1.0
    for f in frames[:: max(1, len(frames) // 12)]:
        v = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        if v > score:
            best, score = f, v
    return best


def render(state: State, payload: dict) -> dict:
    """Render one generation job with the loaded worker state."""
    src = payload["source_path"]
    donor = payload.get("driving_path") or ""
    if not donor or not Path(donor).exists() or donor == src:
        raise RuntimeError("face-swap job needs an identity donor distinct from the target")

    frames, fps = read_video(src, max_frames=N_FRAMES, max_side=MAX_SIDE)
    while len(frames) < N_FRAMES:
        frames.append(frames[-1])
    frames = frames[:N_FRAMES]
    donor_frames, _ = read_video(donor, max_frames=24, max_side=MAX_SIDE)

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        target_mp4, id_png, out_dir = tmp / "target.mp4", tmp / "identity.png", tmp / "out"
        write_video(frames, str(target_mp4), fps=fps)
        cv2.imwrite(str(id_png), cv2.cvtColor(_best_face_frame(donor_frames), cv2.COLOR_RGB2BGR))

        run_cmd([sys.executable, str(state.script),
                 "--ckpt_dir", str(state.ckpt_dir),
                 "--target_video", str(target_mp4), "--source_image", str(id_png),
                 "--output_dir", str(out_dir),
                 "--frame_num", str(N_FRAMES),
                 "--base_seed", str(int(payload.get("seed") or 0) % (2 ** 31))],
                cwd=str(state.repo), timeout=3600)

        produced = sorted(out_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not produced:
            raise RuntimeError(f"DreamID-V produced no output for {payload['job_id']}")
        result, out_fps = read_video(str(produced[0]), max_side=0)

    write_video(result, payload["output_path"], fps=out_fps or fps,
                audio_from=src if has_audio(src) else None)
    return {"face_swap_model": "dreamid_v", "backbone": "Wan2.1-1.3B-DiT",
            "substitutes_for": payload.get("spec_model", "simswap"),
            "source_identity_id": Path(donor).stem, "target_identity_id": Path(src).stem,
            "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="dreamid_v"))
