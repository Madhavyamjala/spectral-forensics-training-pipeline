"""
Wav2Lip - audio-driven mouth synthesis (lip-sync family, pipeline A).

The sync-expert-guided baseline. The spec pairs every target clip with a *different* clip as the
audio donor, so the mouth is driven by speech that does not belong to the video - which is what
makes the result a manipulation rather than a re-encode.

If the donor clip has no audio track (a large share of Kinetics does), the worker falls back to
the next donor it is given, and failing that reports the job as failed rather than emitting the
untouched video.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from _common import (extract_audio, has_audio, note, read_video, repo_path, require, run_cmd,
                     scratch, serve, write_video)

MAX_SIDE = 720
MAX_FRAMES = 200


class State:
    def __init__(self, repo: Path, ckpt: Path):
        self.repo, self.ckpt = repo, ckpt


def load() -> State:
    repo = repo_path("Wav2Lip")
    require(repo / "inference.py", "Wav2Lip inference script")
    ckpt = require(repo / "checkpoints" / "wav2lip_gan.pth", "wav2lip_gan.pth")
    require(repo / "face_detection" / "detection" / "sfd" / "s3fd.pth", "s3fd face detector")
    note(f"wav2lip: repo {repo}")
    return State(repo, ckpt)


def render(state: State, payload: dict) -> dict:
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
        face_mp4 = tmp / "face.mp4"
        write_video(frames, str(face_mp4), fps=fps)
        wav = tmp / "speech.wav"
        if extract_audio(donor, str(wav)) is None:
            raise RuntimeError(f"could not extract audio from donor {Path(donor).name}")
        out_mp4 = tmp / "result.mp4"
        run_cmd([os.sys.executable, "inference.py",
                 "--checkpoint_path", str(state.ckpt),
                 "--face", str(face_mp4), "--audio", str(wav),
                 "--outfile", str(out_mp4), "--pads", "0", "10", "0", "0",
                 "--resize_factor", "1", "--nosmooth"],
                cwd=str(state.repo), timeout=1800)
        if not out_mp4.exists():
            raise RuntimeError("Wav2Lip produced no output file")
        result, out_fps = read_video(str(out_mp4), max_side=0)

    write_video(result, payload["output_path"], fps=out_fps or fps, audio_from=str(donor))
    return {"lip_sync_model": "wav2lip", "audio_source": Path(donor).stem,
            "speaker_id": Path(donor).stem, "speech_duration": round(len(result) / max(1.0, fps), 2),
            "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="wav2lip"))
