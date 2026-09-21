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

#: generate_dreamidv.py requires 4n+1 frames, and --size must be one of its SIZE_CONFIGS
#: keys. 832*480 is the 480p preset the README uses for single-GPU inference.
N_FRAMES = 49
MAX_SIDE = 640
SIZE = "832*480"
SAMPLE_STEPS = 20          # the README's single-GPU setting for the MediaPipe entry point


class State:
    def __init__(self, repo: Path, ckpt_dir: Path, dreamid_ckpt: Path, script: Path):
        """Store the reusable components required by this model worker."""
        self.repo, self.ckpt_dir = repo, ckpt_dir
        self.dreamid_ckpt, self.script = dreamid_ckpt, script


def load() -> State:
    """Load the model and return its reusable worker state."""
    repo = repo_path("DreamID-V")
    weights = Path(os.environ.get("CSF_ENV_ROOT", ".")) / "weights"
    # --ckpt_dir is the Wan2.1 1.3B release (VAE + T5 text encoder), --dreamidv_ckpt is the
    # DreamID-V DiT checkpoint itself. Upstream keeps them in separate Hub repos.
    ckpt_dir = weights / "Wan2.1-T2V-1.3B"
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        raise RuntimeError(
            f"The Wan2.1 backbone DreamID-V builds on is missing at {ckpt_dir}. Fetch it "
            f"with:\n    hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir {ckpt_dir}")
    dreamid_ckpt = require(weights / "DreamID-V" / "dreamidv.pth", "dreamidv.pth")
    script = require(repo / "generate_dreamidv.py", "generate_dreamidv.py")
    note(f"dreamid: repo {repo}, script {script.name}, backbone {ckpt_dir.name}")
    return State(repo, ckpt_dir, dreamid_ckpt, script)


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
        target_mp4, id_png, produced = tmp / "target.mp4", tmp / "identity.png", tmp / "out.mp4"
        write_video(frames, str(target_mp4), fps=fps)
        cv2.imwrite(str(id_png), cv2.cvtColor(_best_face_frame(donor_frames), cv2.COLOR_RGB2BGR))

        # --ref_video is the clip being manipulated and --ref_image the identity pasted onto
        # it; the swapface task takes no prompt.
        run_cmd([sys.executable, str(state.script),
                 "--task", "swapface", "--size", SIZE,
                 "--ckpt_dir", str(state.ckpt_dir),
                 "--dreamidv_ckpt", str(state.dreamid_ckpt),
                 "--ref_video", str(target_mp4), "--ref_image", str(id_png),
                 "--save_file", str(produced),
                 "--frame_num", str(N_FRAMES),
                 "--sample_steps", str(SAMPLE_STEPS),
                 "--base_seed", str(int(payload.get("seed") or 0) % (2 ** 31))],
                cwd=str(state.repo), timeout=3600)

        if not produced.exists():
            raise RuntimeError(f"DreamID-V produced no output for {payload['job_id']}")
        result, out_fps = read_video(str(produced), max_side=0)

    write_video(result, payload["output_path"], fps=out_fps or fps,
                audio_from=src if has_audio(src) else None)
    return {"face_swap_model": "dreamid_v", "backbone": "Wan2.1-1.3B-DiT",
            "substitutes_for": payload.get("spec_model", "simswap"),
            "source_identity_id": Path(donor).stem, "target_identity_id": Path(src).stem,
            "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="dreamid_v"))
