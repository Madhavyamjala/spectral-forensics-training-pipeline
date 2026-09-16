"""
CPU extraction worker for the feature cache (runs inside a process pool).

Kept free of torch / transformers imports so each spawned worker stays small (~100 MB instead of
~400 MB) and starts fast on Windows, where process pools use `spawn`.

Input : job dict (repo id / path, decoding and toolpool settings, delete flag).
Output: result dict with JPEG-encoded VLM frames, spatial + spectral features, latent-tool input crops
        (global 256px crops and 64px patches, processed on the GPU by the parent) and timings; or
        ok=False with the failing stage, error message and traceback.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from csf.data.video_io import decode_frames, download_video, encode_jpegs, to_vlm_frames
from csf.tools.toolpool import _center_crop, ensure_min_size, propose_patches, spatial_features, spectral_features


def cpu_extract(job: Dict[str, Any]) -> Dict[str, Any]:
    cv2.setNumThreads(1)
    out: Dict[str, Any] = {"key": job["key"], "ok": False}
    stage = "download"
    local = None
    try:
        local = download_video(job["repo_id"], job["repo_path"], Path(job["video_dir"]), job["revision"])
        stage = "decode"
        t = time.perf_counter()
        native = decode_frames(local, job["num_frames"], job["tool_max_side"])
        t_decode = time.perf_counter() - t
        vlm = to_vlm_frames(native, job["frame_size"])
        buf, off = encode_jpegs(vlm, job["jpeg_quality"])

        stage = "tools"
        patch = job["patch_size"]
        frames = ensure_min_size(native, patch * 2)
        t = time.perf_counter()
        patches = propose_patches(frames, patch, job["num_patches"])
        t_prop = time.perf_counter() - t
        t = time.perf_counter()
        sp = spatial_features(frames, patches, patch)
        t_sp = time.perf_counter() - t
        t = time.perf_counter()
        sc = spectral_features(frames, patches, patch)
        t_sc = time.perf_counter() - t

        picks = [frames[len(frames) // 3], frames[(2 * len(frames)) // 3]]
        latent_global = np.stack([_center_crop(f, 256) for f in picks])
        latent_patches = np.stack([f[y:y + patch, x:x + patch] for (y, x) in patches for f in picks])

        out.update(ok=True, frames_buf=buf, frames_off=off, spatial=sp, spectral=sc,
                   latent_global=latent_global, latent_patches=latent_patches, n_patches=len(patches),
                   times={"decode": t_decode, "proposal": t_prop, "spatial": t_sp, "spectral": t_sc},
                   native_hw=np.array(native[0].shape[:2], dtype=np.int32))
    except Exception as exc:
        out.update(stage=stage, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if local is not None and job["delete_after"]:
            try:
                Path(local).unlink(missing_ok=True)
            except OSError:
                pass
    return out
