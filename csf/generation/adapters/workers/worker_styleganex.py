"""
StyleGANEX - StyleGAN latent/feature manipulation on unaligned faces and video.

The spec's own expression/attribute pipeline B. It gives the expression family a second
mechanism alongside LivePortrait's retargeting, which matters more here than anywhere else: with
only LivePortrait wired, reallocation would put all 4,750 of the family's videos through one
model, and the per-method breakdown for that family would be measuring a single artifact class.

StyleGAN inversion + latent editing leaves quite different traces from LivePortrait's warping
field, so the two together make the family's numbers mean something.

Weights are Drive-hosted upstream, so they have to be staged by hand - `load()` names the exact
paths it wants. The attribute directions (smile, age, eyes, hair) are the repo's own pSp/e4e
latent directions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

from _common import (feather, has_audio, note, read_frames, read_video, repo_path, require,
                     run_cmd, scratch, serve, write_frames, write_video)

MAX_FRAMES = 120
MAX_SIDE = 1024

#: variant -> (StyleGANEX task, latent direction, sign). The repo exposes these as separate
#: inference tasks; `editing` covers the attribute directions.
VARIANT_TASK = {
    "smile_happiness":               ("editing", "smile", +1.0),
    "sadness_crying":                ("editing", "smile", -1.0),
    "anger":                         ("editing", "smile", -0.8),
    "surprise":                      ("editing", "eyes_open", +1.0),
    "eye_gaze_modification":         ("editing", "eyes_open", +0.8),
    "mouth_expression_modification": ("editing", "smile", +0.6),
    "age":                           ("editing", "age", +1.0),
    "hair_color":                    ("editing", "hair_color", +1.0),
    "facial_attributes":             ("editing", "glasses", +1.0),
}


class State:
    def __init__(self, repo: Path, ckpt: Path, script: Path):
        self.repo, self.ckpt, self.script = repo, ckpt, script


def load() -> State:
    repo = repo_path("StyleGANEX")
    candidates = [repo / "inference_playground.py", repo / "scripts" / "inference.py",
                  repo / "inference.py"]
    script = next((c for c in candidates if c.exists()), None)
    if script is None:
        raise RuntimeError(
            f"StyleGANEX inference entry point not found under {repo}; looked for "
            f"{[str(c.relative_to(repo)) for c in candidates]}.")
    ckpt = require(repo / "pretrained_models" / "styleganex_editing.pt",
                   "StyleGANEX editing checkpoint (Drive-hosted upstream - stage it manually, "
                   "see docs/REGENERATION.md)")
    note(f"styleganex: repo {repo}")
    return State(repo, ckpt, script)


def render(state: State, payload: dict) -> dict:
    src = payload["source_path"]
    variant = payload.get("variant") or "smile_happiness"
    task, direction, sign = VARIANT_TASK.get(variant, ("editing", "smile", 1.0))
    magnitude = float((payload.get("metadata") or {}).get("edit_magnitude", 0.5) or 0.5)
    scale = round(sign * magnitude * 3.0, 3)          # repo's factor range is roughly [-5, 5]

    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=MAX_SIDE)
    h, w = frames[0].shape[:2]

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        frame_dir, out_dir = tmp / "frames", tmp / "out"
        write_frames(frames, str(frame_dir))

        run_cmd([sys.executable, str(state.script),
                 "--ckpt", str(state.ckpt), "--data_path", str(frame_dir),
                 "--save_dir", str(out_dir), "--task", task,
                 "--editing_w_dir", direction, "--scale_factor", str(scale)],
                cwd=str(state.repo), timeout=2400)

        dirs = sorted((p for p in out_dir.rglob("*") if p.is_dir() and any(p.glob("*.png"))),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        result = read_frames(str(dirs[0])) if dirs else []
        if not result:
            result = read_frames(str(out_dir))
        if not result:
            raise RuntimeError(f"StyleGANEX produced no frames for {payload['job_id']}")

    result = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) if f.shape[:2] != (h, w) else f
              for f in result]
    write_video(result, payload["output_path"], fps=fps,
                audio_from=src if has_audio(src) else None)
    return {"edit_model": "styleganex", "manipulation_type": variant,
            "latent_direction": direction, "edit_magnitude": magnitude, "scale_factor": scale,
            "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="styleganex"))
