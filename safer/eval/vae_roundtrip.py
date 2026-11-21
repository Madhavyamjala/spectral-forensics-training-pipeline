"""Rebuttal (4jrv): VAE round-trip vs. re-encoding intervention on real clips.

Each real clip is put on the detector's canvas (854x480, 24 fps, first N frames) and stored
losslessly as `src`. Variants are then made from `src` with content held fixed:

  identity          `src` itself (the detector's normalization re-encodes it at CRF 23, like every variant)
  vae_<name>        video-VAE encode -> decode (latent mode, no sampling), stored losslessly
  h264_crf<q>       libx264 at CRF q          (negative control: compression only)
  h265_crf<q>       libx265 at CRF q

The VAEs are not the SVD VAE that the latent (DIRE) tool uses, so the test is not circular.
For every variant a csv (path,label,category) is written for benchmark_external_outcomes.py
(`--dataset vae_<variant>=csv:<file>`), which then normalizes the videos and saves the 23 tool
features and every VLM score.

  python -m safer.eval.vae_roundtrip --list clips.txt --out OUT --vae wan --vae cog --crf264 18 28 35 --crf265 28
  python -m safer.eval.vae_roundtrip --list clips.txt --out OUT --stage encodes        # CPU part only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

W, H, FPS = 854, 480, 24

VAES = {
    # name: (diffusers class, repo, subfolder, dtype). Wan2.1 and CogVideoX are causal 3-D video VAEs
    # used by current text-to-video generators; neither is the SVD temporal VAE used by the DIRE tool.
    "wan": ("AutoencoderKLWan", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "vae", "float32"),
    "cog": ("AutoencoderKLCogVideoX", "zai-org/CogVideoX-2b", "vae", "bfloat16"),
}


def _ffmpeg() -> str:
    for cand in (os.environ.get("CSF_FFMPEG"), shutil.which("ffmpeg")):
        if cand:
            return cand
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit("ffmpeg not found (set CSF_FFMPEG or put it on PATH)")


def read_canvas(src: str, frames: int, ffmpeg: str) -> Optional[np.ndarray]:
    """First `frames` frames on the detector canvas (letterboxed, as in the normalization), RGB uint8."""
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=bicubic,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={FPS}")
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", src, "-vf", vf, "-frames:v", str(frames),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, capture_output=True, timeout=300)
    n = len(res.stdout) // (W * H * 3)
    if res.returncode != 0 or n < frames:
        return None
    return np.frombuffer(res.stdout[: frames * W * H * 3], np.uint8).reshape(frames, H, W, 3)


def write_lossless(frames: np.ndarray, dst: Path, ffmpeg: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    cmd = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264rgb", "-qp", "0", "-preset", "veryfast",
           str(tmp)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True, timeout=300)
    tmp.replace(dst)


def encode(src: Path, dst: Path, codec: str, crf: int, ffmpeg: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    lib = {"h264": "libx264", "h265": "libx265"}[codec]
    extra = ["-x265-params", "log-level=error"] if codec == "h265" else []
    cmd = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-c:v", lib, "-preset", "medium",
           "-crf", str(crf), "-pix_fmt", "yuv420p", *extra, str(tmp)]
    subprocess.run(cmd, check=True, timeout=600)
    tmp.replace(dst)


class Roundtrip:
    def __init__(self, name: str, device: str = "cuda", vae=None):
        import torch
        self.torch, self.name = torch, name
        cls, repo, sub, dt = VAES[name]
        self.dtype = getattr(torch, dt)
        if vae is None:
            import diffusers
            vae = getattr(diffusers, cls).from_pretrained(repo, subfolder=sub, torch_dtype=self.dtype)
        self.vae = vae
        self.vae.to(device=device, dtype=self.dtype).eval().requires_grad_(False)
        if hasattr(self.vae, "enable_tiling") and name == "cog":
            self.vae.enable_tiling()
        self.device = device

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        torch = self.torch
        t, h, w, _ = frames.shape
        ph, pw = (-h) % 16, (-w) % 16                     # both VAEs need H, W divisible by 8 (16 is safe)
        x = np.pad(frames, ((0, 0), (0, ph), (0, pw), (0, 0)), mode="edge")
        x = torch.from_numpy(x).to(self.device).permute(3, 0, 1, 2)[None].to(self.dtype) / 127.5 - 1.0
        with torch.inference_mode():
            z = self.vae.encode(x).latent_dist.mode()
            y = self.vae.decode(z).sample
        y = y[0, :, :t, :h, :w].float().clamp(-1, 1).permute(1, 2, 3, 0)
        return ((y + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="text file, one source video path per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=49, help="frames kept per clip (4k+1 for the video VAEs)")
    ap.add_argument("--vae", action="append", default=[], choices=sorted(VAES))
    ap.add_argument("--crf264", type=int, nargs="*", default=[18, 28, 35])
    ap.add_argument("--crf265", type=int, nargs="*", default=[28])
    ap.add_argument("--stage", choices=["all", "src", "encodes", "vae"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    ff = _ffmpeg()
    out = Path(a.out)
    srcs = [l.strip() for l in open(a.list) if l.strip()]
    if a.limit:
        srcs = srcs[: a.limit]
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

    status: Dict[str, str] = {}
    if a.stage in ("all", "src"):
        for s in srcs:
            cid = Path(s).stem
            dst = out / "identity" / f"{cid}.mp4"
            if dst.exists():
                status[cid] = "ok"
                continue
            fr = read_canvas(s, a.frames, ff)
            if fr is None:
                status[cid] = "too short or unreadable"
                continue
            write_lossless(fr, dst, ff)
            status[cid] = "ok"
        log(f"src: {sum(v == 'ok' for v in status.values())}/{len(srcs)} ok")
    clips = sorted(p for p in (out / "identity").glob("*.mp4") if not p.name.endswith(".part.mp4"))

    variants = ["identity"]
    if a.stage in ("all", "encodes"):
        for q in a.crf264:
            for p in clips:
                d = out / f"h264_crf{q}" / p.name
                if not d.exists():
                    encode(p, d, "h264", q, ff)
        for q in a.crf265:
            for p in clips:
                d = out / f"h265_crf{q}" / p.name
                if not d.exists():
                    encode(p, d, "h265", q, ff)
        log(f"encodes done for {len(clips)} clips")
    variants += [f"h264_crf{q}" for q in a.crf264] + [f"h265_crf{q}" for q in a.crf265]

    if a.stage in ("all", "vae"):
        for name in a.vae:
            rt = Roundtrip(name)
            t0, n = time.time(), 0
            for p in clips:
                d = out / f"vae_{name}" / p.name
                if d.exists():
                    continue
                fr = read_canvas(str(p), a.frames, ff)
                write_lossless(rt(fr), d, ff)
                n += 1
                if n in (1, 10) or n % 50 == 0:
                    log(f"vae_{name}: {n} clips, {(time.time() - t0) / n:.2f} s/clip")
            del rt
            import torch
            torch.cuda.empty_cache()
    variants += [f"vae_{n}" for n in a.vae]

    lists = out / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    for v in variants:
        rows = sorted(p for p in (out / v).glob("*.mp4") if not p.name.endswith(".part.mp4")) if (out / v).exists() else []
        with open(lists / f"vae_{v}.csv" if not v.startswith("vae_") else lists / f"{v}.csv", "w") as f:
            f.write("path,label,category\n")
            for p in rows:
                f.write(f"{p.resolve()},real,{v}\n")
    (out / "variants.json").write_text(json.dumps({"variants": variants, "frames": a.frames, "canvas": [W, H, FPS],
                                                   "n_src": len(srcs), "n_clips": len(clips),
                                                   "skipped": {k: v for k, v in status.items() if v != "ok"}}, indent=1))
    log(f"lists written to {lists}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
