"""
Thin-Plate-Spline Motion Model - facial reenactment.

Fills the PIRenderer slot, whose checkpoints are Drive-hosted. TPSMM warps with thin-plate-spline
transformations rather than FOMM's local affine approximations, so the reenactment family keeps
three distinct motion representations across its wired slots (FOMM affine, TPSMM spline,
LivePortrait implicit keypoints) instead of three of the same.

MIT licensed. The vox checkpoint is hosted on Tsinghua Cloud / Drive / Yandex upstream, so it has
to be staged by hand; `load()` names the exact path it wants.

Recorded in the manifest as `tpsmm`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from _common import (has_audio, note, read_video, repo_path, require, run_cmd, scratch, serve,
                     write_video)

MAX_FRAMES = 120


class State:
    def __init__(self, repo: Path, ckpt: Path, config: Path):
        self.repo, self.ckpt, self.config = repo, ckpt, config


def load() -> State:
    repo = repo_path("TPSMM")
    require(repo / "demo.py", "TPSMM demo script")
    ckpt = require(repo / "checkpoints" / "vox.pth.tar",
                   "TPSMM vox checkpoint (Tsinghua Cloud / Drive hosted upstream - stage it "
                   "manually, see docs/REGENERATION.md)")
    config = require(repo / "config" / "vox-256.yaml", "TPSMM config")
    note(f"tpsmm: repo {repo}")
    return State(repo, ckpt, config)


def render(state: State, payload: dict) -> dict:
    src = payload["source_path"]
    driving = payload.get("driving_path") or ""
    if not driving or not Path(driving).exists() or driving == src:
        raise RuntimeError("reenactment job needs a driving clip distinct from the target")

    frames, fps = read_video(src, max_frames=1, max_side=1280)
    drive, _ = read_video(driving, max_frames=MAX_FRAMES, max_side=512)
    if len(drive) < 4:
        raise RuntimeError("driving clip is too short to reenact")

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        source_png, driving_mp4, out_mp4 = tmp / "source.png", tmp / "driving.mp4", tmp / "out.mp4"
        cv2.imwrite(str(source_png), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        write_video(drive, str(driving_mp4), fps=fps)

        run_cmd([sys.executable, "demo.py",
                 "--config", str(state.config), "--checkpoint", str(state.ckpt),
                 "--source_image", str(source_png), "--driving_video", str(driving_mp4),
                 "--result_video", str(out_mp4), "--mode", "relative", "--find_best_frame"],
                cwd=str(state.repo), timeout=1800)
        if not out_mp4.exists():
            raise RuntimeError("TPSMM produced no output file")
        result, out_fps = read_video(str(out_mp4), max_side=0)

    write_video(result, payload["output_path"], fps=out_fps or fps,
                audio_from=src if has_audio(src) else None)
    return {"reenactment_model": "tpsmm", "substitutes_for": payload.get("spec_model", "pirenderer"),
            "target_video_id": Path(src).stem, "driving_video_id": Path(driving).stem,
            "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="tpsmm"))
