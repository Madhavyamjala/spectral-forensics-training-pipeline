"""
StyleGANEX - StyleGAN latent/feature manipulation on unaligned faces and video.

The spec's own expression/attribute pipeline B. It gives the expression family a second
mechanism alongside LivePortrait's retargeting, which matters more here than anywhere else: with
only LivePortrait wired, reallocation would put all 4,750 of the family's videos through one
model, and the per-method breakdown for that family would be measuring a single artifact class.

StyleGAN inversion + latent editing leaves quite different traces from LivePortrait's warping
field, so the two together make the family's numbers mean something.

Weights are Drive-hosted upstream, so they are staged by hand into
`generation.staged_weights_dir` and the env build copies them in. Upstream publishes one
checkpoint per editing direction - and for video, only age and hair colour - so this worker
renders exactly those two variants; the planner routes the expression variants to LivePortrait.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from _common import (has_audio, note, read_video, repo_path, require, run_cmd, scratch,
                     serve, write_video)

MAX_FRAMES = 120
MAX_SIDE = 1024

#: variant -> the released checkpoint that performs it.
#:
#: StyleGANEX does not expose "editing directions" as a runtime flag: each published checkpoint
#: carries its own `editing_w` tensor (see the repo's video_editing.py), and `--scale_factor`
#: only scales that one direction. Upstream releases exactly two video-editing checkpoints -
#: age and hair colour - so those are the only variants this renderer can honestly produce.
#: Everything else in the expression family (smile, anger, surprise, gaze, mouth) is a
#: deformation LivePortrait performs, and the job planner routes those there instead.
VARIANT_CKPT = {
    "age": "styleganex_edit_age.pt",
    "hair_color": "styleganex_edit_hair.pt",
}


class State:
    def __init__(self, repo: Path, script: Path, ckpts: dict):
        """Store the reusable components required by this model worker."""
        self.repo, self.script, self.ckpts = repo, script, ckpts


def load() -> State:
    """Locate the repo's video-editing entry point and every checkpoint we may be asked for."""
    repo = repo_path("StyleGANEX")
    script = require(repo / "video_editing.py", "StyleGANEX video_editing.py")
    ckpts = {
        variant: require(repo / "pretrained_models" / name,
                         f"StyleGANEX checkpoint for '{variant}' (Drive-hosted upstream - stage "
                         f"{name} in generation.staged_weights_dir)")
        for variant, name in VARIANT_CKPT.items()
    }
    require(repo / "pretrained_models" / "shape_predictor_68_face_landmarks.dat",
            "dlib 68-point landmark predictor (the env build places it)")
    note(f"styleganex: repo {repo}, directions {sorted(ckpts)}")
    return State(repo, script, ckpts)


def _first_dlib_frame(frames):
    """Index of the first frame dlib's frontal detector accepts, or None.

    dlib is what StyleGANEX itself uses, so agreeing with it is the point: any other
    detector could nominate a frame that video_editing.py then rejects.
    """
    import dlib

    detector = dlib.get_frontal_face_detector()
    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if len(detector(gray, 1)) > 0:
            return index
    return None


def render(state: State, payload: dict) -> dict:
    """Render one generation job with the loaded worker state."""
    src = payload["source_path"]
    variant = payload.get("variant") or "age"
    ckpt = state.ckpts.get(variant)
    if ckpt is None:
        raise RuntimeError(
            f"StyleGANEX cannot produce the '{variant}' variant: the released video-editing "
            f"checkpoints cover {sorted(VARIANT_CKPT)} only. This job should have been routed "
            f"to a renderer that supports it.")

    magnitude = float((payload.get("metadata") or {}).get("edit_magnitude", 0.5) or 0.5)
    scale = round(max(0.2, magnitude * 3.0), 3)       # the repo's own range is roughly [0, 5]

    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=MAX_SIDE)
    h, w = frames[0].shape[:2]

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        out_dir = tmp / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        # video_editing.py reads only the FIRST frame to find its crop, with dlib's frontal
        # detector, and asserts if that frame has no detectable face. The clip pool was
        # qualified with SCRFD, which finds faces dlib does not, so a perfectly good clip
        # fails on its opening frame. Re-cut the clip to start where dlib agrees - using
        # dlib itself, so the answer here is the answer video_editing.py will get.
        start = _first_dlib_frame(frames)
        if start is None:
            raise RuntimeError(
                f"dlib found no frontal face in any of {len(frames)} frames; StyleGANEX "
                f"crops from the first frame and cannot run on this clip")
        src = str(tmp / "aligned.mp4")
        write_video(frames[start:], src, fps=fps)
        # video_editing.py reads the video itself and writes
        # <output_path>/<video stem>_<ckpt stem>.mp4 at 4x the cropped face resolution.
        run_cmd([sys.executable, str(state.script),
                 "--data_path", str(src), "--ckpt", str(ckpt),
                 "--output_path", str(out_dir), "--scale_factor", str(scale)],
                cwd=str(state.repo), timeout=2400)

        produced = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not produced:
            raise RuntimeError(f"StyleGANEX produced no video for {payload['job_id']}")
        result, _ = read_video(str(produced[0]), max_frames=MAX_FRAMES)

    if not result:
        raise RuntimeError(f"StyleGANEX output was unreadable for {payload['job_id']}")
    result = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) if f.shape[:2] != (h, w) else f
              for f in result]
    write_video(result, payload["output_path"], fps=fps,
                audio_from=src if has_audio(src) else None)
    return {"edit_model": "styleganex", "manipulation_type": variant,
            "latent_direction": variant, "edit_magnitude": magnitude, "scale_factor": scale,
            "checkpoint": Path(ckpt).name, "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="styleganex"))
