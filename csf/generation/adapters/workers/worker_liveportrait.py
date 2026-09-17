"""
LivePortrait - implicit-keypoint animation with stitching and retargeting.

Two spec slots, selected by `options["mode"]`:

    mode="reenact"     the Face2Face slot. Face2Face was never publicly released; LivePortrait
                       stands in with a genuinely different mechanism from the FOMM slot next to
                       it (implicit keypoints + stitching, versus FOMM's local affine warping),
                       so the family keeps two distinct artifact classes rather than two of one.
                       Driven video-to-video: the target clip is animated by a separate driver.

    mode="expression"  the GANimation slot. GANimation published no weights; LivePortrait's
                       retargeting ratios give the continuous, magnitude-controlled expression
                       edits that slot is defined by (eyes and lips, driven by scalar ratios),
                       which is the closest honest analogue to AU-conditioned control.

Recorded in the manifest as `liveportrait` / `liveportrait_expr`, never as the slot name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from _common import (has_audio, note, read_video, repo_path, require, run_cmd, scratch, serve,
                     write_video)

MAX_FRAMES = 160
MAX_SIDE = 720

#: (eye ratio, lip ratio) per expression variant - the magnitude the job asked for.
EXPRESSION_RETARGET = {
    "smile_happiness": (0.0, 0.5), "sadness_crying": (-0.2, -0.3), "anger": (-0.3, -0.2),
    "surprise": (0.6, 0.4), "eye_gaze_modification": (0.5, 0.0),
    "mouth_expression_modification": (0.0, 0.6), "age": (0.1, 0.1),
    "hair_color": (0.0, 0.0), "facial_attributes": (0.2, 0.2),
}


class State:
    def __init__(self, repo: Path, mode: str):
        self.repo, self.mode = repo, mode


def load() -> State:
    repo = repo_path("LivePortrait")
    require(repo / "inference.py", "LivePortrait inference script")
    weights = repo / "pretrained_weights"
    if not weights.exists() or not any(weights.iterdir()):
        raise RuntimeError(
            f"LivePortrait weights missing at {weights}. Fetch them with:\n"
            f"    cd {repo} && huggingface-cli download KlingTeam/LivePortrait "
            f"--local-dir pretrained_weights")
    note(f"liveportrait: repo {repo}")
    return State(repo, "")


def _collect(out_dir: Path, exclude_concat: bool = True):
    mp4s = sorted(out_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if exclude_concat:
        plain = [p for p in mp4s if "concat" not in p.name]
        mp4s = plain or mp4s
    return mp4s


def render(state: State, payload: dict) -> dict:
    opts = payload.get("options") or {}
    mode = opts.get("mode", "reenact")
    src = payload["source_path"]
    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=MAX_SIDE)

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        source_mp4, out_dir = tmp / "source.mp4", tmp / "animations"
        write_video(frames, str(source_mp4), fps=fps)
        cmd = [sys.executable, "inference.py", "-s", str(source_mp4), "-o", str(out_dir)]
        meta: dict = {}

        if mode == "reenact":
            driving = payload.get("driving_path") or ""
            if not driving or not Path(driving).exists() or driving == src:
                raise RuntimeError("reenactment job needs a driving clip distinct from the target")
            drive_frames, _ = read_video(driving, max_frames=MAX_FRAMES, max_side=512)
            driving_mp4 = tmp / "driving.mp4"
            write_video(drive_frames, str(driving_mp4), fps=fps)
            cmd += ["-d", str(driving_mp4), "--flag_relative_motion", "true"]
            meta.update(reenactment_model="liveportrait",
                        driving_video_id=Path(driving).stem, target_video_id=Path(src).stem)
        elif mode == "expression":
            variant = payload.get("variant") or "smile_happiness"
            eye, lip = EXPRESSION_RETARGET.get(variant, (0.2, 0.3))
            magnitude = float((payload.get("metadata") or {}).get("edit_magnitude", 0.5) or 0.5)
            # drive the clip with itself, then let retargeting supply the edit
            cmd += ["-d", str(source_mp4),
                    "--flag_eye_retargeting", "true", "--flag_lip_retargeting", "true",
                    "--eye_retargeting_multiplier", f"{1.0 + eye * magnitude:.3f}",
                    "--lip_retargeting_multiplier", f"{1.0 + lip * magnitude:.3f}"]
            meta.update(edit_model="liveportrait_expr", manipulation_type=variant,
                        edit_magnitude=magnitude, eye_ratio=eye, lip_ratio=lip)
        else:
            raise RuntimeError(f"unknown liveportrait mode {mode!r}")

        run_cmd(cmd, cwd=str(state.repo), timeout=1800)
        produced = _collect(out_dir)
        if not produced:
            raise RuntimeError(f"LivePortrait produced no output under {out_dir}")
        result, out_fps = read_video(str(produced[0]), max_side=0)

    write_video(result, payload["output_path"], fps=out_fps or fps,
                audio_from=src if has_audio(src) else None)
    meta.update(substitutes_for=payload.get("spec_model", ""), frames=len(result))
    return meta


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="liveportrait"))
