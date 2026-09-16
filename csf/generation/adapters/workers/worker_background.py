"""
Background manipulation: SAM2 foreground segmentation + three background sources.

One worker covers pipelines A, B and C of the background family, because all three share the
expensive part - segmenting the moving foreground with SAM2 - and differ only in what gets put
behind it:

    mode="real_composite"  (A)  a *different real* Kinetics clip as the background. No generative
                                model at all, which is the point: it isolates compositing and
                                blending artifacts from generative ones.
    mode="flux_image"      (B)  a static FLUX.1-schnell still (4 steps, Apache-licensed).
    mode="svd_video"       (C)  a Stable Video Diffusion clip, so the background moves.

Pipeline D (reconstruct the background by inpainting) is ProPainter's job and lives in
worker_propainter.py.

The foreground mask is tracked across the whole clip with SAM2's video predictor when available,
falling back to per-frame prompting. Compositing is alpha-feathered so the seam is not a trivial
give-away - a detector that only learns "hard cut-out edge" would not generalise.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from _common import (emit, env_root, feather, has_audio, note, read_video, require, scratch,
                     serve, write_video)

MAX_SIDE = 768
MAX_FRAMES = 160


class State:
    def __init__(self, mode: str):
        self.mode = mode
        self.sam_video = None
        self.sam_image = None
        self.flux = None
        self.svd = None
        self._bg_cache = {}


# --------------------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------------------


def _load_sam(state: State) -> None:
    import torch
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt = require(env_root() / "weights" / "sam2.1_hiera_base_plus.pt", "SAM2 checkpoint")
    cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    try:
        state.sam_video = build_sam2_video_predictor(cfg, str(ckpt), device="cuda")
        note("sam2: video predictor loaded")
    except Exception as exc:                                  # noqa: BLE001 - fall back to image mode
        note(f"sam2: video predictor unavailable ({exc}); using per-frame prompting")
    model = build_sam2(cfg, str(ckpt), device="cuda")
    state.sam_image = SAM2ImagePredictor(model)


def _motion_prompt(frames) -> tuple:
    """Pick a point on the moving subject: the peak of the accumulated frame difference."""
    acc = np.zeros(frames[0].shape[:2], dtype=np.float32)
    step = max(1, len(frames) // 12)
    for i in range(step, len(frames), step):
        a = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor(frames[i - step], cv2.COLOR_RGB2GRAY).astype(np.float32)
        acc += np.abs(a - b)
    acc = cv2.GaussianBlur(acc, (31, 31), 0)
    if acc.max() <= 1e-6:
        h, w = acc.shape
        return w // 2, h // 2
    y, x = np.unravel_index(int(np.argmax(acc)), acc.shape)
    return int(x), int(y)


def _segment(state: State, frames) -> list:
    """Per-frame foreground masks (uint8 0/255) for the whole clip."""
    import torch

    px, py = _motion_prompt(frames)
    h, w = frames[0].shape[:2]

    if state.sam_video is not None:
        with scratch("sam2") as tmp:
            folder = Path(tmp)
            for i, f in enumerate(frames):
                cv2.imwrite(str(folder / f"{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    st = state.sam_video.init_state(video_path=str(folder))
                    state.sam_video.add_new_points_or_box(
                        inference_state=st, frame_idx=0, obj_id=1,
                        points=np.array([[px, py]], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32))
                    masks = [np.zeros((h, w), np.uint8) for _ in frames]
                    for idx, _ids, logits in state.sam_video.propagate_in_video(st):
                        if idx < len(masks):
                            m = (logits[0] > 0).cpu().numpy().astype(np.uint8)[0] * 255
                            masks[idx] = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                if sum(int(m.any()) for m in masks) >= max(2, len(frames) // 4):
                    return masks
                note("sam2: video propagation produced mostly-empty masks; falling back")
            except Exception as exc:                          # noqa: BLE001
                note(f"sam2: video propagation failed ({exc}); falling back to per-frame")

    masks = []
    for frame in frames:
        with torch.inference_mode():
            state.sam_image.set_image(frame)
            m, scores, _ = state.sam_image.predict(
                point_coords=np.array([[px, py]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32), multimask_output=True)
        best = m[int(np.argmax(scores))].astype(np.uint8) * 255
        masks.append(cv2.resize(best, (w, h), interpolation=cv2.INTER_NEAREST))
    return masks


# --------------------------------------------------------------------------------------
# backgrounds
# --------------------------------------------------------------------------------------


def _flux_background(state: State, prompt: str, size, seed: int):
    import torch
    from diffusers import FluxPipeline

    if state.flux is None:
        model_id = os.environ.get("CSF_FLUX_ID", "black-forest-labs/FLUX.1-schnell")
        state.flux = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        state.flux.enable_model_cpu_offload()
        note(f"flux: {model_id} loaded")
    w, h = size
    gen = torch.Generator("cpu").manual_seed(int(seed) % (2 ** 31))
    image = state.flux(prompt=prompt, num_inference_steps=4, guidance_scale=0.0,
                       height=(h // 16) * 16, width=(w // 16) * 16, generator=gen).images[0]
    return cv2.resize(np.array(image.convert("RGB")), (w, h), interpolation=cv2.INTER_LANCZOS4)


def _svd_background(state: State, prompt: str, size, seed: int, n_frames: int):
    """A moving background: FLUX makes the first frame, SVD animates it."""
    import torch
    from diffusers import StableVideoDiffusionPipeline
    from PIL import Image

    first = _flux_background(state, prompt, (1024, 576), seed)
    if state.svd is None:
        model_id = os.environ.get("CSF_SVD_ID", "stabilityai/stable-video-diffusion-img2vid-xt")
        state.svd = StableVideoDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16")
        state.svd.enable_model_cpu_offload()
        note(f"svd: {model_id} loaded")
    gen = torch.Generator("cpu").manual_seed(int(seed) % (2 ** 31))
    out = state.svd(Image.fromarray(first), decode_chunk_size=6, generator=gen,
                    num_frames=25, motion_bucket_id=90, noise_aug_strength=0.02).frames[0]
    clip = [cv2.resize(np.array(f.convert("RGB")), size, interpolation=cv2.INTER_LANCZOS4)
            for f in out]
    # loop (ping-pong) to cover the target clip length without a visible cut
    if len(clip) < n_frames:
        pong = clip + clip[::-1][1:-1]
        clip = [pong[i % len(pong)] for i in range(n_frames)]
    return clip[:n_frames]


def _real_background(state: State, payload: dict, size, n_frames: int):
    """A different real clip as the plate - pipeline A uses no generative model."""
    donor = payload.get("driving_path") or payload.get("audio_path") or ""
    if not donor or not Path(donor).exists() or donor == payload["source_path"]:
        raise RuntimeError("real_composite needs a second real clip as the background plate; "
                           "none was bound to this job")
    frames, _ = read_video(donor, max_frames=n_frames, max_side=MAX_SIDE)
    frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
    return [frames[i % len(frames)] for i in range(n_frames)]


# --------------------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------------------


def load() -> State:
    mode = os.environ.get("CSF_BG_MODE", "")
    state = State(mode)
    _load_sam(state)
    return state


def render(state: State, payload: dict) -> dict:
    mode = (payload.get("options") or {}).get("mode") or state.mode
    frames, fps = read_video(payload["source_path"], max_frames=MAX_FRAMES, max_side=MAX_SIDE)
    h, w = frames[0].shape[:2]
    masks = _segment(state, frames)
    covered = sum(1 for m in masks if m.any())
    if covered < max(2, len(frames) // 5):
        raise RuntimeError(f"foreground segmentation failed: only {covered}/{len(frames)} frames "
                           f"produced a mask")

    prompt = payload.get("prompt") or "a photorealistic outdoor scene"
    seed = int(payload.get("seed") or 0)
    if mode == "real_composite":
        background = _real_background(state, payload, (w, h), len(frames))
    elif mode == "flux_image":
        still = _flux_background(state, prompt, (w, h), seed)
        background = [still] * len(frames)
    elif mode == "svd_video":
        background = _svd_background(state, prompt, (w, h), seed, len(frames))
    else:
        raise RuntimeError(f"unknown background mode {mode!r}")

    out = []
    areas = []
    for frame, mask, bg in zip(frames, masks, background):
        alpha = feather(mask, radius=9)
        areas.append(float(mask.astype(bool).mean()))
        out.append(np.clip(frame.astype(np.float32) * alpha
                           + bg.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8))

    src = payload["source_path"]
    write_video(out, payload["output_path"], fps=fps,
                audio_from=src if has_audio(src) else None)
    return {"segmentation_model": "sam2.1_hiera_base_plus", "background_source": mode,
            "composite_mode": "alpha_feathered", "prompt": prompt if mode != "real_composite" else "",
            "foreground_area_frac": round(float(np.mean(areas)), 4),
            "mask_coverage": round(covered / max(1, len(frames)), 4), "frames": len(out)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="background"))
