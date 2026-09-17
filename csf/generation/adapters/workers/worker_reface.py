"""
REFace - diffusion-based face swapping (WACV 2025).

Fills the FaceShifter slot: FaceShifter's public repository ships training code only, with no
released inference checkpoint, so the slot is otherwise unfillable.

LICENCE WARNING. REFace's code is MIT, but its checkpoint was trained on CelebAMask-HQ, whose
terms restrict use to non-commercial research. Videos produced here inherit that restriction. If
this dataset may ever be used commercially, skip the slot instead of shipping the videos:

    --set generation.skip_models='[faceshifter]'

The worker refuses to start unless the operator has acknowledged the restriction, either via
`generation.accept_noncommercial: true` (which sets CSF_ACCEPT_NONCOMMERCIAL) or by skipping it.
Recorded in the manifest as `reface`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

from _common import (has_audio, note, read_video, repo_path, require, run_cmd, scratch, serve,
                     write_video)

MAX_FRAMES = 120
MAX_SIDE = 640


class State:
    def __init__(self, repo: Path, ckpt: Path, script: Path):
        self.repo, self.ckpt, self.script = repo, ckpt, script


def load() -> State:
    if os.environ.get("CSF_ACCEPT_NONCOMMERCIAL", "").lower() not in ("1", "true", "yes"):
        raise RuntimeError(
            "REFace's checkpoint is trained on CelebAMask-HQ, which permits NON-COMMERCIAL "
            "RESEARCH USE ONLY, and videos it generates inherit that restriction. To use it, "
            "acknowledge explicitly with generation.accept_noncommercial: true. To leave the "
            "slot empty instead, use --set generation.skip_models='[faceshifter]'.")
    repo = repo_path("REFace")
    ckpt = require(repo / "checkpoints" / "last.ckpt", "REFace checkpoint")
    candidates = [repo / "scripts" / "inference.py", repo / "inference.py",
                  repo / "scripts" / "inference_test_bench.py"]
    script = next((c for c in candidates if c.exists()), None)
    if script is None:
        raise RuntimeError(
            f"REFace inference entry point not found under {repo}; looked for "
            f"{[str(c.relative_to(repo)) for c in candidates]}.")
    note(f"reface: repo {repo}, script {script.name} (NON-COMMERCIAL checkpoint)")
    return State(repo, ckpt, script)


def render(state: State, payload: dict) -> dict:
    src = payload["source_path"]
    donor = payload.get("driving_path") or ""
    if not donor or not Path(donor).exists() or donor == src:
        raise RuntimeError("face-swap job needs an identity donor distinct from the target")

    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=MAX_SIDE)
    donor_frames, _ = read_video(donor, max_frames=16, max_side=MAX_SIDE)

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        target_dir, out_dir = tmp / "target", tmp / "out"
        target_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(str(target_dir / f"{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        id_png = tmp / "identity.png"
        cv2.imwrite(str(id_png), cv2.cvtColor(donor_frames[0], cv2.COLOR_RGB2BGR))

        run_cmd([sys.executable, str(state.script),
                 "--ckpt", str(state.ckpt),
                 "--target_path", str(target_dir), "--source_path", str(id_png),
                 "--outdir", str(out_dir), "--n_samples", "1", "--scale", "3.5",
                 "--ddim_steps", "50",
                 "--seed", str(int(payload.get("seed") or 0) % (2 ** 31))],
                cwd=str(state.repo), timeout=3600)

        from _common import read_frames
        dirs = sorted((p for p in out_dir.rglob("*") if p.is_dir() and any(p.glob("*.png"))),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        result = read_frames(str(dirs[0])) if dirs else []
        if not result:
            raise RuntimeError(f"REFace produced no frames for {payload['job_id']}")

    write_video(result, payload["output_path"], fps=fps,
                audio_from=src if has_audio(src) else None)
    return {"face_swap_model": "reface", "licence": "non-commercial-research-only",
            "substitutes_for": payload.get("spec_model", "faceshifter"),
            "source_identity_id": Path(donor).stem, "target_identity_id": Path(src).stem,
            "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="reface"))
