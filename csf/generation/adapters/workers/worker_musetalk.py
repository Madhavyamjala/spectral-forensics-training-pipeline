"""
MuseTalk 1.5 - latent-space audio-conditioned inpainting of the mouth region.

This is the spec's own pipeline B, not a substitute: MuseTalk really is available, with a
download script that pulls every component from the Hub (the earlier assessment that it needed
manual staging was wrong - only face-parse-bisent is Drive-hosted, and inference falls back to
MuseTalk's bbox path without it). MIT licensed, commercial use permitted.

Driven through `python -m scripts.inference` with a per-job YAML task file, matching upstream.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

from _common import (extract_audio, has_audio, note, read_video, repo_path, require, run_cmd,
                     scratch, serve, write_video)

MAX_SIDE = 720
MAX_FRAMES = 200


class State:
    def __init__(self, repo: Path, unet: Path, unet_cfg: Path):
        self.repo, self.unet, self.unet_cfg = repo, unet, unet_cfg


def load() -> State:
    repo = repo_path("MuseTalk")
    require(repo / "scripts" / "inference.py", "MuseTalk inference script")
    unet = require(repo / "models" / "musetalkV15" / "unet.pth", "MuseTalk v1.5 UNet")
    unet_cfg = require(repo / "models" / "musetalkV15" / "musetalk.json", "MuseTalk config")
    note(f"musetalk: repo {repo}")
    return State(repo, unet, unet_cfg)


def render(state: State, payload: dict) -> dict:
    src = payload["source_path"]
    donor = payload.get("audio_path") or ""
    if not donor or not Path(donor).exists() or donor == src:
        raise RuntimeError("lip-sync job needs an audio donor distinct from the target")
    if not has_audio(donor):
        raise RuntimeError(f"audio donor {Path(donor).name} has no audio track")

    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=MAX_SIDE)

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        face_mp4, wav, results = tmp / "face.mp4", tmp / "speech.wav", tmp / "results"
        write_video(frames, str(face_mp4), fps=fps)
        if extract_audio(donor, str(wav)) is None:
            raise RuntimeError(f"could not extract audio from donor {Path(donor).name}")

        cfg = {"task_0": {"video_path": str(face_mp4), "audio_path": str(wav), "bbox_shift": 0}}
        cfg_path = tmp / "task.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        run_cmd([sys.executable, "-m", "scripts.inference",
                 "--inference_config", str(cfg_path),
                 "--result_dir", str(results),
                 "--unet_model_path", str(state.unet),
                 "--unet_config", str(state.unet_cfg),
                 "--version", "v15"],
                cwd=str(state.repo), timeout=2400)

        produced = sorted(results.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not produced:
            raise RuntimeError(f"MuseTalk produced no output under {results}")
        result, out_fps = read_video(str(produced[0]), max_side=0)

    write_video(result, payload["output_path"], fps=out_fps or fps, audio_from=str(donor))
    return {"lip_sync_model": "musetalk", "musetalk_version": "1.5",
            "audio_source": Path(donor).stem, "speaker_id": Path(donor).stem,
            "speech_duration": round(len(result) / max(1.0, fps), 2), "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="musetalk"))
