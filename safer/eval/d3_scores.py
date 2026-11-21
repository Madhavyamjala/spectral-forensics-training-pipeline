"""Rebuttal (rZyQ, t7Rq Q3): D3 (Zheng et al., ICCV 2025), a training-free detector, on the same
normalized videos SAFER sees.

Follows the official code (github.com/Zig-HS/D3, eval.py defaults):
  frames    ffmpeg `-ss START -t 3 -vf fps=8` to JPEG (as utils/video2frame.py), read with cv2 (BGR,
            as data/datasets.py), first 16 frames (8 if 8-15), crop 10% from each long side,
            resize 224x224, ImageNet mean/std
  encoder   XCLIPVisionModel microsoft/xclip-base-patch16, pooler_output per frame (--encoder XCLIP-16)
  score     d1 = ||f_t+1 - f_t||_2 (--loss l2); d2 = d1[1:] - d1[:-1]; std(d2) (unbiased)
One deviation: START = 0 instead of a random offset, so the run is deterministic.
The official eval treats real as the positive class with std as the score (higher std = more real),
so the CSV `score` is -std (higher = more likely fake); `d3_std` keeps the raw value.

  python -m safer.eval.d3_scores --videos LIST.csv --out OUT.csv [--limit N]
LIST.csv has columns video_id,path (the external_inputs/<ds>.videos.csv the harness writes, or a manifest).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
REPO = "microsoft/xclip-base-patch16"


def _ffmpeg() -> str:
    for cand in (os.environ.get("CSF_FFMPEG"), shutil.which("ffmpeg")):
        if cand:
            return cand
    raise SystemExit("ffmpeg not found (set CSF_FFMPEG or put it on PATH)")


def crop_center_by_percentage(img: np.ndarray, pct: float = 0.1) -> np.ndarray:
    h, w = img.shape[:2]
    if w > h:
        c = int(w * pct)
        return img[:, c:w - c]
    c = int(h * pct)
    return img[c:h - c, :]


def load_frames(path: str, ffmpeg: str, tmp: Path) -> Optional[np.ndarray]:
    import cv2
    for f in tmp.glob("*.jpg"):
        f.unlink()
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-ss", "0", "-t", "3", "-i", path, "-vf", "fps=8",
           str(tmp / "%d.jpg")]
    if subprocess.run(cmd, timeout=300).returncode != 0:
        return None
    files = sorted(tmp.glob("*.jpg"), key=lambda p: int(p.stem))
    if len(files) < 8:
        return None
    n = 8 if len(files) < 16 else 16
    out = []
    for f in files[:n]:
        img = cv2.imread(str(f))                                   # BGR, as the official loader
        img = crop_center_by_percentage(img, 0.1)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)   # albumentations Resize default
        out.append((img.astype(np.float32) / 255.0 - MEAN) / STD)
    return np.stack(out).transpose(0, 3, 1, 2)                       # T, C, H, W


class D3:
    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import XCLIPModel
        self.torch, self.device = torch, device
        # The official code uses XCLIPVisionModel.from_pretrained(REPO). With transformers 5.x that class
        # does not map the full checkpoint's `vision_model.*` keys and initializes the encoder randomly,
        # so load the full model and take its vision tower (the module XCLIPVisionModel wraps).
        full = XCLIPModel.from_pretrained(REPO)
        self.model = full.vision_model.to(device).eval()
        self._check_weights(full)
        del full

    @staticmethod
    def _check_weights(full) -> None:
        import glob
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        files = glob.glob(os.path.join(snapshot_download(REPO, allow_patterns=["*.safetensors"]), "*.safetensors"))
        if not files:
            return
        ck = {k[len("vision_model."):]: v for k, v in load_file(files[0]).items() if k.startswith("vision_model.")}
        sd = full.vision_model.state_dict()
        bad = [k for k in sd if k not in ck or not (sd[k].cpu() == ck[k]).all()]
        if bad:
            raise SystemExit(f"XCLIP vision weights not loaded from the checkpoint: {bad[:5]}")

    def __call__(self, frames: np.ndarray) -> float:
        torch = self.torch
        x = torch.from_numpy(frames).to(self.device)
        with torch.inference_mode():
            f = self.model(pixel_values=x, output_hidden_states=True).pooler_output[None]   # 1, T, 768
            assert f.shape[-1] == 768, f.shape
            d1 = torch.norm(f[:, 1:] - f[:, :-1], p=2, dim=-1)
            d2 = d1[:, 1:] - d1[:, :-1]
            return float(torch.std(d2, dim=1)[0])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    rows = list(csv.DictReader(open(a.videos, newline="", encoding="utf-8")))
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {r["video_id"] for r in csv.DictReader(open(out, newline="", encoding="utf-8"))}
    new = not out.exists()
    ff = _ffmpeg()
    det = D3(a.device)
    t0, n = time.time(), 0
    with open(out, "a", newline="", encoding="utf-8") as fh, tempfile.TemporaryDirectory() as td:
        w = csv.writer(fh)
        if new:
            w.writerow(["video_id", "score", "d3_std", "n_frames", "latency_ms"])
        for r in rows:
            if r["video_id"] in done:
                continue
            t = time.perf_counter()
            fr = load_frames(r["path"], ff, Path(td))
            if fr is None:
                w.writerow([r["video_id"], "", "", 0, ""])
                continue
            s = det(fr)
            w.writerow([r["video_id"], -s, s, len(fr), (time.perf_counter() - t) * 1000])
            n += 1
            if n in (1, 10) or n % 500 == 0:
                fh.flush()
                print(time.strftime("%H:%M:%S"), f"{n} videos, {(time.time() - t0) / n:.2f} s/video", flush=True)
    print(f"wrote {out} ({n} new)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
