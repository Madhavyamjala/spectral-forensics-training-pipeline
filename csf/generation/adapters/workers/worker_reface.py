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
    def __init__(self, repo: Path, ckpt: Path, script: Path, app=None):
        """Store the reusable components required by this model worker."""
        self.repo, self.ckpt, self.script = repo, ckpt, script
        self.app = app                      # face detector, for the region masks


def load() -> State:
    """Load the model and return its reusable worker state."""
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
    # the masks REFace inpaints through are ours to supply; detect once, reuse per frame
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    note(f"reface: repo {repo}, script {script.name} (NON-COMMERCIAL checkpoint)")
    return State(repo, ckpt, script, app)


def _face_mask(state: State, frame):
    """A filled ellipse over the detected face - the region REFace regenerates.

    Upstream masks come from a face-parsing network they do not ship with the inference
    script; an ellipse over the detector's box covers the same area more coarsely, which
    shows up as a softer blend boundary rather than as a different manipulation.
    """
    import numpy as np

    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    faces = state.app.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) if state.app else []
    if not faces:
        return mask
    biggest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = [int(v) for v in biggest.bbox]
    centre = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, int((x2 - x1) * 0.62)), max(1, int((y2 - y1) * 0.78)))
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, -1)
    return mask


def render(state: State, payload: dict) -> dict:
    """Render one generation job with the loaded worker state."""
    src = payload["source_path"]
    donor = payload.get("driving_path") or ""
    if not donor or not Path(donor).exists() or donor == src:
        raise RuntimeError("face-swap job needs an identity donor distinct from the target")

    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=MAX_SIDE)
    donor_frames, _ = read_video(donor, max_frames=16, max_side=MAX_SIDE)

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        # Upstream's per-index convention, which its loop reconstructs by string surgery:
        # target <dir>/<i>.jpg, reference <refdir>/<i+1>.jpg, masks <maskdir>/<i>_skin.png
        # and <i>_mouth.png. Passing index 0 of each is what tells it where the series is.
        target_dir, ref_dir = tmp / "target", tmp / "reference"
        mask_dir, out_dir = tmp / "masks", tmp / "out"
        for folder in (target_dir, ref_dir, mask_dir):
            folder.mkdir(parents=True, exist_ok=True)

        donor_bgr = cv2.cvtColor(donor_frames[0], cv2.COLOR_RGB2BGR)
        for i, frame in enumerate(frames):
            cv2.imwrite(str(target_dir / f"{i}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(ref_dir / f"{i + 1}.jpg"), donor_bgr)
            mask = _face_mask(state, frame)
            cv2.imwrite(str(mask_dir / f"{i}_skin.png"), mask)
            cv2.imwrite(str(mask_dir / f"{i}_mouth.png"), mask)

        env = dict(os.environ, CSF_REFACE_BATCH="1", CSF_REFACE_FRAMES=str(len(frames)))
        os.environ.update(env)
        run_cmd([sys.executable, str(state.script),
                 "--ckpt", str(state.ckpt),
                 "--image_path", str(target_dir / "0.jpg"),
                 "--reference_path", str(ref_dir / "1.jpg"),
                 "--mask_path", str(mask_dir / "0_skin.png"),
                 "--outdir", str(out_dir), "--n_samples", "1", "--scale", "3.5",
                 "--ddim_steps", "50",
                 "--seed", str(int(payload.get("seed") or 0) % (2 ** 31))],
                cwd=str(state.repo), timeout=7200)

        # results are <outdir>/results/<index>_<seed>.png - sort numerically, not
        # lexically, or frame 10 lands between frames 1 and 2
        produced = sorted(out_dir.rglob("results/*.png"),
                          key=lambda q: int(q.stem.split("_")[0]))
        result = []
        for path in produced:
            img = cv2.imread(str(path))
            if img is not None:
                result.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if len(result) < len(frames):
            raise RuntimeError(
                f"REFace returned {len(result)} of {len(frames)} frames for "
                f"{payload['job_id']}; the per-frame loop did not run to completion")

    write_video(result, payload["output_path"], fps=fps,
                audio_from=src if has_audio(src) else None)
    return {"face_swap_model": "reface", "licence": "non-commercial-research-only",
            "substitutes_for": payload.get("spec_model", "faceshifter"),
            "source_identity_id": Path(donor).stem, "target_identity_id": Path(src).stem,
            # recorded because it is a deviation: upstream masks come from a face-parsing
            # network, this one is an ellipse over the detector's box
            "mask_source": "insightface_bbox_ellipse",
            "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="reface"))
