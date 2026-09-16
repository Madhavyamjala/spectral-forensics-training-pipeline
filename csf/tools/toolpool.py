"""
Phase 3 - Tri-Domain Feature Extraction Toolpool.

All tools run on NATIVE-resolution frames (only capped to `tool_max_side`), never on the
224px VLM frames, because resizing destroys exactly the high-frequency evidence the spectral
and latent tools look for. Heavy tools operate on sparse 64x64 patches (the proposal's
"extreme patch sparsity") chosen by `propose_patches` from high-entropy / high-motion cells.

Tool groups and their features (fixed order = FEATURE_NAMES):
  spatial  : saturation statistics, luminance continuity, Laplacian edge artefacts,
             optical-flow temporal inconsistency, patch-vs-context colour discontinuity
  spectral : global / patch FFT high-frequency energy, radial spectrum slope, 3D-DCT
             spatio-temporal high-frequency energy, inter-frame phase-correlation spikes,
             noise-residual (PRNU proxy) inconsistency across patches
  latent   : Diffusion Reconstruction Error (DIRE, VAE formulation) of the global frame and of
             the patches. LOW error = sits on a generative latent manifold (AI-Generated signal),
             HIGH error = authentic camera content.

Input : list of T RGB uint8 frames (H, W, 3); ToolpoolConfig-like values from DataConfig.
Output: `ToolResult` with a float32 feature vector (NaN for tools that were not run), a
        per-group boolean mask and per-group wall-clock seconds (used as the dispatcher's cost).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.fft import dctn

from csf.logging_utils import get_logger

log = get_logger("tools")

TOOL_GROUPS = ["spatial", "spectral", "latent"]
FEATURE_NAMES: Dict[str, List[str]] = {
    "spatial": ["sat_mean", "sat_std", "sat_entropy", "lum_jump_std", "lum_jump_max",
                "edge_lapvar_global", "edge_lapvar_patch_ratio", "flow_inconsistency",
                "patch_context_color_gap"],
    "spectral": ["fft_hf_global", "spectrum_slope", "fft_hf_patch_mean", "fft_hf_patch_std",
                 "dct3d_hf_global", "dct3d_hf_patch_mean", "phase_corr_global",
                 "phase_shift_spike", "noise_residual_cv"],
    "latent": ["dire_global", "dire_patch_mean", "dire_patch_min", "dire_patch_std",
               "dire_patch_global_ratio"],
}
ALL_FEATURES: List[str] = [f for g in TOOL_GROUPS for f in FEATURE_NAMES[g]]
GROUP_SLICES: Dict[str, slice] = {}
_start = 0
for _g in TOOL_GROUPS:
    GROUP_SLICES[_g] = slice(_start, _start + len(FEATURE_NAMES[_g]))
    _start += len(FEATURE_NAMES[_g])

EPS = 1e-8


@dataclass
class ToolResult:
    features: np.ndarray
    mask: np.ndarray
    times: Dict[str, float] = field(default_factory=dict)
    patches: List[Tuple[int, int]] = field(default_factory=list)


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        img = cv2.resize(img, (max(size, int(round(w * scale))), max(size, int(round(h * scale)))),
                         interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return img[y:y + size, x:x + size]


def ensure_min_size(frames: List[np.ndarray], min_side: int) -> List[np.ndarray]:
    h, w = frames[0].shape[:2]
    if min(h, w) >= min_side:
        return frames
    scale = min_side / min(h, w)
    size = (int(math.ceil(w * scale)), int(math.ceil(h * scale)))
    return [cv2.resize(f, size, interpolation=cv2.INTER_CUBIC) for f in frames]


def propose_patches(frames: List[np.ndarray], patch: int, k: int) -> List[Tuple[int, int]]:
    """Top-k non-overlapping grid cells ranked by texture entropy (Laplacian energy) + temporal residual."""
    mid = len(frames) // 2
    g0 = _gray(frames[mid]).astype(np.float32)
    g1 = _gray(frames[min(mid + 1, len(frames) - 1)]).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g0, cv2.CV_32F))
    motion = np.abs(g1 - g0)
    h, w = g0.shape
    gh, gw = h // patch, w // patch
    if gh == 0 or gw == 0:
        return [(0, 0)]
    lap_c = lap[:gh * patch, :gw * patch].reshape(gh, patch, gw, patch).mean(axis=(1, 3))
    mot_c = motion[:gh * patch, :gw * patch].reshape(gh, patch, gw, patch).mean(axis=(1, 3))

    def z(a):
        return (a - a.mean()) / (a.std() + EPS)

    score = (z(np.log1p(lap_c)) + 0.5 * z(mot_c)).ravel()
    order = np.argsort(-score)[:max(1, k)]
    return [(int(i // gw) * patch, int(i % gw) * patch) for i in order]


def _patch_stack(frames: List[np.ndarray], yx: Tuple[int, int], patch: int) -> np.ndarray:
    y, x = yx
    return np.stack([f[y:y + patch, x:x + patch] for f in frames])


def _hf_ratio(gray: np.ndarray, cutoff: float = 0.5) -> float:
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    h, w = mag.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    return float(mag[r > cutoff * min(h, w) / 2].sum() / (mag.sum() + EPS))


def _spectrum_slope(gray: np.ndarray) -> float:
    """Slope of log radial power spectrum vs log frequency (natural images ~ -2)."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    power = np.abs(f) ** 2
    h, w = power.shape
    yy, xx = np.indices(power.shape)
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2).astype(np.int32)
    radial = np.bincount(r.ravel(), power.ravel()) / (np.bincount(r.ravel()) + EPS)
    freqs = np.arange(1, min(h, w) // 2)
    if len(freqs) < 4:
        return 0.0
    slope, _ = np.polyfit(np.log(freqs), np.log(radial[freqs] + EPS), 1)
    return float(slope)


def _dct3d_hf(cube: np.ndarray) -> float:
    if cube.size == 0 or cube.shape[0] < 2:
        return 0.0
    coeffs = np.abs(dctn(cube.astype(np.float32), norm="ortho"))
    t, h, w = coeffs.shape
    return float(coeffs[t // 2:, h // 2:, w // 2:].sum() / (coeffs.sum() + EPS))


def _phase_response(a: np.ndarray, b: np.ndarray) -> float:
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    _, response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
    return float(response)


def spatial_features(frames: List[np.ndarray], patches: List[Tuple[int, int]], patch: int) -> Dict[str, float]:
    sats = np.concatenate([cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[..., 1].ravel()[::7] for f in frames])
    hist = np.bincount(sats, minlength=256).astype(np.float64)
    p = hist / (hist.sum() + EPS)
    lum = np.array([f.mean() for f in frames], dtype=np.float32)
    jumps = np.abs(np.diff(lum)) if len(lum) > 1 else np.zeros(1, np.float32)

    grays = [_gray(f) for f in frames]
    lap_global = float(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]))
    lap_patch = float(np.mean([cv2.Laplacian(g[y:y + patch, x:x + patch], cv2.CV_64F).var()
                               for (y, x) in patches for g in grays[::max(1, len(grays) // 4)]]))

    small = [cv2.resize(g, (160, max(16, int(160 * g.shape[0] / g.shape[1])))) for g in grays]
    flow_incons = 0.0
    if len(small) >= 3:
        flows = [cv2.calcOpticalFlowFarneback(small[i], small[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                 for i in range(len(small) - 1)]
        mag = np.mean([np.linalg.norm(fl, axis=-1).mean() for fl in flows]) + 1e-3
        flow_incons = float(np.mean([np.abs(flows[i + 1] - flows[i]).mean() for i in range(len(flows) - 1)]) / mag)

    gaps = []
    mid = frames[len(frames) // 2].astype(np.float32)
    h, w = mid.shape[:2]
    for (y, x) in patches:
        inner = mid[y:y + patch, x:x + patch].reshape(-1, 3)
        y0, x0 = max(0, y - patch // 2), max(0, x - patch // 2)
        y1, x1 = min(h, y + patch + patch // 2), min(w, x + patch + patch // 2)
        ring_mask = np.ones((y1 - y0, x1 - x0), bool)
        ring_mask[y - y0:y - y0 + patch, x - x0:x - x0 + patch] = False
        ring = mid[y0:y1, x0:x1][ring_mask]
        if len(ring) == 0:
            continue
        gaps.append(float(np.abs(inner.mean(0) - ring.mean(0)).mean() + np.abs(inner.std(0) - ring.std(0)).mean()))

    return {
        "sat_mean": float((p * np.arange(256)).sum() / 255.0),
        "sat_std": float(np.sqrt((p * (np.arange(256) / 255.0 - (p * np.arange(256)).sum() / 255.0) ** 2).sum())),
        "sat_entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
        "lum_jump_std": float(jumps.std()),
        "lum_jump_max": float(jumps.max()),
        "edge_lapvar_global": float(np.log1p(lap_global)),
        "edge_lapvar_patch_ratio": float(np.log1p(lap_patch) - np.log1p(lap_global)),
        "flow_inconsistency": flow_incons,
        "patch_context_color_gap": float(max(gaps) if gaps else 0.0),
    }


def spectral_features(frames: List[np.ndarray], patches: List[Tuple[int, int]], patch: int) -> Dict[str, float]:
    grays = [_gray(f).astype(np.float32) for f in frames]
    crops = [_center_crop(g, 256) for g in grays]
    sub = crops[::max(1, len(crops) // 4)]
    fft_global = float(np.mean([_hf_ratio(c) for c in sub]))
    slope = float(np.mean([_spectrum_slope(c) for c in sub[:2]]))

    patch_hf, patch_dct, patch_phase, patch_resid = [], [], [], []
    for yx in patches:
        stack = np.stack([g[yx[0]:yx[0] + patch, yx[1]:yx[1] + patch] for g in grays])
        patch_hf.append(np.mean([_hf_ratio(s) for s in stack[::max(1, len(stack) // 4)]]))
        patch_dct.append(_dct3d_hf(stack[:8]))
        patch_phase.append(np.mean([_phase_response(stack[i], stack[i + 1]) for i in range(min(len(stack) - 1, 4))])
                           if len(stack) > 1 else 1.0)
        resid = stack[0] - cv2.GaussianBlur(stack[0], (3, 3), 0)
        patch_resid.append(resid.std())

    cube = np.stack([cv2.resize(c, (128, 128), interpolation=cv2.INTER_AREA) for c in crops[:8]])
    phase_global = (np.mean([_phase_response(crops[i], crops[i + 1]) for i in range(min(len(crops) - 1, 4))])
                    if len(crops) > 1 else 1.0)
    miss = 1.0 - np.asarray(patch_phase, dtype=np.float64)
    resid_arr = np.asarray(patch_resid, dtype=np.float64)
    return {
        "fft_hf_global": fft_global,
        "spectrum_slope": slope,
        "fft_hf_patch_mean": float(np.mean(patch_hf)),
        "fft_hf_patch_std": float(np.std(patch_hf)),
        "dct3d_hf_global": _dct3d_hf(cube),
        "dct3d_hf_patch_mean": float(np.mean(patch_dct)),
        "phase_corr_global": float(phase_global),
        "phase_shift_spike": float(miss.max() - np.median(miss)),
        "noise_residual_cv": float(resid_arr.std() / (resid_arr.mean() + EPS)),
    }


class LatentTool:
    """DIRE via a frozen video-diffusion VAE (SVD's temporal VAE; falls back to SD's VAE if unavailable)."""

    def __init__(self, model_id: str, subfolder: Optional[str], fallback_id: str, device, dtype):
        import torch
        self.torch = torch
        self.device = device
        self.dtype = dtype if device.type == "cuda" else torch.float32
        self.temporal = False
        try:
            from diffusers import AutoencoderKLTemporalDecoder
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(model_id, subfolder=subfolder, torch_dtype=self.dtype)
            self.temporal = True
            self.source = f"{model_id}/{subfolder}"
        except Exception as exc:
            log.warning("Could not load SVD VAE %s/%s (%s: %s) -> falling back to %s",
                        model_id, subfolder, type(exc).__name__, str(exc)[:200], fallback_id)
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(fallback_id, torch_dtype=self.dtype)
            self.source = fallback_id
        self.vae.to(device).eval().requires_grad_(False)
        log.info("Latent tool VAE loaded: %s (dtype=%s, temporal_decoder=%s)", self.source, self.dtype, self.temporal)

    def _recon_error(self, batch: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).to(self.dtype) / 127.5 - 1.0
        with torch.inference_mode():
            latent = self.vae.encode(x).latent_dist.mode()
            if self.temporal:
                recon = self.vae.decode(latent, num_frames=latent.shape[0]).sample
            else:
                recon = self.vae.decode(latent).sample
        err = (recon.float() - x.float()).pow(2).mean(dim=(1, 2, 3))
        if not torch.isfinite(err).all() and self.dtype != torch.float32:
            log.warning("Non-finite DIRE in %s (overflow) -> switching the VAE to float32 for the rest of the run",
                        self.dtype)
            self.dtype = torch.float32
            self.vae.to(torch.float32)
            return self._recon_error(batch)
        return err.cpu().numpy()

    def features(self, frames: List[np.ndarray], patches: List[Tuple[int, int]], patch: int) -> Dict[str, float]:
        picks = [frames[len(frames) // 3], frames[(2 * len(frames)) // 3]]
        global_err = float(self._recon_error(np.stack([_center_crop(f, 256) for f in picks])).mean())
        stacks = [f[y:y + patch, x:x + patch] for (y, x) in patches for f in picks]
        per = self._recon_error(np.stack(stacks)).reshape(len(patches), len(picks)).mean(axis=1)
        return {
            "dire_global": global_err,
            "dire_patch_mean": float(per.mean()),
            "dire_patch_min": float(per.min()),
            "dire_patch_std": float(per.std()),
            "dire_patch_global_ratio": float(per.mean() / (global_err + EPS)),
        }


def run_toolpool(frames: List[np.ndarray], patch: int, num_patches: int,
                 latent_tool: Optional[LatentTool], groups: Sequence[str] = TOOL_GROUPS) -> ToolResult:
    """Run the requested tool groups on one video; unrequested groups are NaN and masked out."""
    frames = ensure_min_size(frames, patch * 2)
    feats = np.full(len(ALL_FEATURES), np.nan, dtype=np.float32)
    mask = np.zeros(len(TOOL_GROUPS), dtype=bool)
    times: Dict[str, float] = {}
    t = time.perf_counter()
    patches = propose_patches(frames, patch, num_patches)
    times["proposal"] = time.perf_counter() - t

    for gi, group in enumerate(TOOL_GROUPS):
        if group not in groups:
            continue
        if group == "latent" and latent_tool is None:
            continue
        t = time.perf_counter()
        if group == "spatial":
            out = spatial_features(frames, patches, patch)
        elif group == "spectral":
            out = spectral_features(frames, patches, patch)
        else:
            out = latent_tool.features(frames, patches, patch)
            if latent_tool.device.type == "cuda":
                latent_tool.torch.cuda.synchronize()
        times[group] = time.perf_counter() - t
        vals = np.array([out[n] for n in FEATURE_NAMES[group]], dtype=np.float32)
        feats[GROUP_SLICES[group]] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        mask[gi] = True
    return ToolResult(feats, mask, times, patches)
