"""
LatentSync 1.6 - audio-conditioned latent diffusion lip-sync.

Stands in for the VideoReTalking slot, whose checkpoint bundle is Google-Drive hosted. It is not
a downgrade: on HDTF the LatentSync paper reports FID 7.03 / SSIM 0.79 / SyncConf 8.9 / FVD 192.7
against VideoReTalking's 9.5 / 0.75 / 7.5 / 270.6 - better on every axis, and a higher sync
confidence than Wav2Lip (8.2) too.

The manifest records this as `latentsync`, not `videoretalking`, so the per-method breakdown
attributes the artifacts to the model that actually made them.

Driven through `python -m scripts.inference` with the stage2_512 UNet config, matching upstream's
own inference.sh.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from _common import (extract_audio, has_audio, note, read_video, repo_path, require, run_cmd,
                     scratch, serve, write_video)

MAX_SIDE = 720
MAX_FRAMES = 200


class State:
    def __init__(self, repo: Path, ckpt: Path, config: Path):
        """Store the reusable components required by this model worker."""
        self.repo, self.ckpt, self.config = repo, ckpt, config


def load() -> State:
    """Load the model and return its reusable worker state."""
    repo = repo_path("LatentSync")
    require(repo / "scripts" / "inference.py", "LatentSync inference script")
    ckpt = require(repo / "checkpoints" / "latentsync_unet.pt", "latentsync_unet.pt")
    require(repo / "checkpoints" / "whisper" / "tiny.pt", "Whisper tiny checkpoint")
    config = require(repo / "configs" / "unet" / "stage2_512.yaml", "LatentSync UNet config")
    note(f"latentsync: repo {repo}")
    return State(repo, ckpt, config)


def render(state: State, payload: dict) -> dict:
    """Render one generation job with the loaded worker state."""
    src = payload["source_path"]
    donor = payload.get("audio_path") or ""
    if not donor or not Path(donor).exists():
        raise RuntimeError("lip-sync job has no audio donor bound")
    if donor == src:
        raise RuntimeError("audio donor must differ from the target clip")
    if not has_audio(donor):
        raise RuntimeError(f"audio donor {Path(donor).name} has no audio track")

    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=MAX_SIDE)

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        face_mp4, wav, out_mp4 = tmp / "face.mp4", tmp / "speech.wav", tmp / "out.mp4"
        write_video(frames, str(face_mp4), fps=fps)
        if extract_audio(donor, str(wav)) is None:
            raise RuntimeError(f"could not extract audio from donor {Path(donor).name}")

        run_cmd([sys.executable, "-m", "scripts.inference",
                 "--unet_config_path", str(state.config),
                 "--inference_ckpt_path", str(state.ckpt),
                 "--inference_steps", str((payload.get("options") or {}).get("steps", 20)),
                 "--guidance_scale", str((payload.get("options") or {}).get("guidance", 1.5)),
                 "--seed", str(int(payload.get("seed") or 0) % (2 ** 31)),
                 "--enable_deepcache",
                 "--video_path", str(face_mp4),
                 "--audio_path", str(wav),
                 "--video_out_path", str(out_mp4)],
                cwd=str(state.repo), timeout=2400)
        if not out_mp4.exists():
            raise RuntimeError("LatentSync produced no output file")
        result, out_fps = read_video(str(out_mp4), max_side=0)

    write_video(result, payload["output_path"], fps=out_fps or fps, audio_from=str(donor))
    return {"lip_sync_model": "latentsync", "latentsync_version": "1.6",
            "substitutes_for": payload.get("spec_model", "videoretalking"),
            "audio_source": Path(donor).stem, "speaker_id": Path(donor).stem,
            "speech_duration": round(len(result) / max(1.0, fps), 2), "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="latentsync"))
