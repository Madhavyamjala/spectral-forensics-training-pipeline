"""
SadTalker - audio -> 3D facial motion coefficients -> neural rendering.

The spec's own lip-sync pipeline D, and genuinely available: Apache-2.0 since upstream dropped
the non-commercial clause, with scripts/download_models.sh pulling every checkpoint from GitHub
Releases (the earlier assessment that it was Drive-only was wrong).

SadTalker animates a *still* portrait from audio, so the worker takes a representative frame of
the target clip as the source image and drives it with the donor's speech. `--still` keeps head
motion small, which is what makes the result read as an edit of the original shot rather than a
new one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from _common import (extract_audio, has_audio, note, read_video, repo_path, require, run_cmd,
                     scratch, serve, write_video)

MAX_FRAMES = 200


class State:
    def __init__(self, repo: Path, ckpt_dir: Path):
        self.repo, self.ckpt_dir = repo, ckpt_dir


def load() -> State:
    repo = repo_path("SadTalker")
    require(repo / "inference.py", "SadTalker inference script")
    ckpt_dir = repo / "checkpoints"
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        raise RuntimeError(
            f"SadTalker checkpoints missing at {ckpt_dir}. Run its own downloader:\n"
            f"    cd {repo} && bash scripts/download_models.sh")
    note(f"sadtalker: repo {repo}")
    return State(repo, ckpt_dir)


def _best_frame(frames):
    """The sharpest frame - SadTalker's 3DMM extractor is sensitive to motion blur."""
    best, score = frames[0], -1.0
    for f in frames[:: max(1, len(frames) // 12)]:
        v = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        if v > score:
            best, score = f, v
    return best


def render(state: State, payload: dict) -> dict:
    src = payload["source_path"]
    donor = payload.get("audio_path") or ""
    if not donor or not Path(donor).exists() or donor == src:
        raise RuntimeError("lip-sync job needs an audio donor distinct from the target")
    if not has_audio(donor):
        raise RuntimeError(f"audio donor {Path(donor).name} has no audio track")

    frames, fps = read_video(src, max_frames=MAX_FRAMES, max_side=720)

    with scratch(payload["job_id"]) as tmp:
        tmp = Path(tmp)
        img, wav, results = tmp / "source.png", tmp / "speech.wav", tmp / "results"
        cv2.imwrite(str(img), cv2.cvtColor(_best_frame(frames), cv2.COLOR_RGB2BGR))
        if extract_audio(donor, str(wav)) is None:
            raise RuntimeError(f"could not extract audio from donor {Path(donor).name}")

        run_cmd([sys.executable, "inference.py",
                 "--driven_audio", str(wav), "--source_image", str(img),
                 "--result_dir", str(results), "--checkpoint_dir", str(state.ckpt_dir),
                 "--preprocess", "full", "--still", "--size", "256"],
                cwd=str(state.repo), timeout=2400)

        produced = sorted(results.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        produced = [p for p in produced if "enhanced" not in p.name] or produced
        if not produced:
            raise RuntimeError(f"SadTalker produced no output under {results}")
        result, out_fps = read_video(str(produced[0]), max_side=0)

    write_video(result, payload["output_path"], fps=out_fps or fps, audio_from=str(donor))
    return {"lip_sync_model": "sadtalker", "audio_source": Path(donor).stem,
            "speaker_id": Path(donor).stem, "render_mode": "still_full",
            "speech_duration": round(len(result) / max(1.0, fps), 2), "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="sadtalker"))
