"""
Distributed, resumable, disk-bounded feature cache.

For every manifest row the video is downloaded, decoded once and turned into everything the
later stages need, so no stage ever has to touch the raw video again:

    <cache_dir>/items/<class>/<video_id>.npz
        frames_buf, frames_off : JPEG-encoded VLM frames (T x frame_size^2)
        features               : float32 toolpool vector (csf.tools.ALL_FEATURES order)
        mask                   : bool per tool group (spatial, spectral, latent)
        times                  : seconds [decode, proposal, spatial, spectral, latent]
        native_hw              : original (height, width)

Work split: rank r takes rows r, r+W, r+2W ... . Inside a rank, a process pool does download +
decode + CPU tools (spatial / spectral) + latent-tool input crops, while the main process runs the
VAE (latent tool) on the GPU. The video file is deleted right after decoding (unless it is one of the
`keep_videos_for_latency` test videos kept for the end-to-end latency benchmark), so peak disk use
stays at roughly `prefetch` videos per rank. Items that already exist are skipped, so re-running
resumes. Failures are appended to <cache_dir>/failed_rank{R}.csv with the exact stage and error.

After all ranks finish, rank 0 writes
    <cache_dir>/index.parquet      manifest rows that were cached successfully (+ timings)
    <cache_dir>/feature_stats.json per-feature mean/std on the TRAIN split (for normalisation)
    <cache_dir>/tool_costs.json    median seconds per tool group (the dispatcher's cost model)

Input : run manifest DataFrame, Config, DistInfo.
Output: the files above; returns the index DataFrame.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from csf.config import Config
from csf.data.extract_worker import cpu_extract
from csf.distributed import DistInfo, barrier
from csf.logging_utils import Throughput, get_logger, set_context
from csf.tools.toolpool import ALL_FEATURES, FEATURE_NAMES, GROUP_SLICES, TOOL_GROUPS

log = get_logger("data.cache")
TIME_KEYS = ["decode", "proposal", "spatial", "spectral", "latent"]


def item_path(cache_dir: Path, cls: str, video_id: str) -> Path:
    return cache_dir / "items" / cls / f"{video_id}.npz"


def _latent_features(latent_tool, res: Dict[str, Any]) -> Dict[str, float]:
    g = latent_tool._recon_error(res["latent_global"]).mean()
    per = latent_tool._recon_error(res["latent_patches"]).reshape(res["n_patches"], -1).mean(axis=1)
    return {"dire_global": float(g), "dire_patch_mean": float(per.mean()), "dire_patch_min": float(per.min()),
            "dire_patch_std": float(per.std()), "dire_patch_global_ratio": float(per.mean() / (g + 1e-8))}


def _save_item(path: Path, res: Dict[str, Any], latent: Optional[Dict[str, float]], t_latent: float) -> None:
    feats = np.full(len(ALL_FEATURES), np.nan, dtype=np.float32)
    mask = np.zeros(len(TOOL_GROUPS), dtype=bool)
    for gi, (group, values) in enumerate([("spatial", res["spatial"]), ("spectral", res["spectral"]),
                                          ("latent", latent)]):
        if values is None:
            continue
        vals = np.array([values[n] for n in FEATURE_NAMES[group]], dtype=np.float32)
        feats[GROUP_SLICES[group]] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        mask[gi] = True
    times = np.array([res["times"]["decode"], res["times"]["proposal"], res["times"]["spatial"],
                      res["times"]["spectral"], t_latent], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, frames_buf=res["frames_buf"], frames_off=res["frames_off"], features=feats, mask=mask,
                 times=times, native_hw=res["native_hw"])
    os.replace(tmp, path)


def _append_failure(path: Path, row: Dict[str, Any]) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "video_id", "repo_path", "stage", "error"])
        if new:
            w.writeheader()
        w.writerow(row)


def build_feature_cache(df: pd.DataFrame, cfg: Config, dist_info: DistInfo) -> pd.DataFrame:
    import torch
    from tqdm import tqdm
    from csf.tools.toolpool import LatentTool

    cache_dir, d = cfg.cache_dir, cfg.data
    video_dir = Path(cfg.paths.video_dir)
    keep = set(df[df["split"] == "test"].sort_values("video_id").head(d.keep_videos_for_latency)["video_id"])

    mine = df.iloc[dist_info.rank::dist_info.world_size]
    todo = [r for r in mine.itertuples(index=False)
            if not item_path(cache_dir, r[mine.columns.get_loc("class")], r.video_id).exists()]
    log.info("Feature cache: %d rows assigned to this rank, %d already cached, %d to extract",
             len(mine), len(mine) - len(todo), len(todo))

    failed_path = cache_dir / f"failed_rank{dist_info.rank}.csv"
    if todo:
        dtype = torch.float16 if dist_info.device.type == "cuda" else torch.float32
        latent_tool = LatentTool(cfg.models.vae_id, cfg.models.vae_subfolder, cfg.models.vae_fallback_id,
                                 dist_info.device, dtype)
        workers = max(1, min(d.download_workers, (os.cpu_count() or 2) // dist_info.world_size))
        log.info("Extraction pool: %d worker process(es), prefetch=%d, delete_videos_after_cache=%s",
                 workers, d.prefetch, d.delete_videos_after_cache)
        cls_idx = mine.columns.get_loc("class")
        jobs = deque({
            "key": (r[cls_idx], r.video_id, r.repo_path), "repo_id": d.repo_id, "repo_path": r.repo_path,
            "revision": d.revision, "video_dir": str(video_dir), "num_frames": d.num_frames,
            "tool_max_side": d.tool_max_side, "frame_size": d.frame_size, "jpeg_quality": d.jpeg_quality,
            "patch_size": d.patch_size, "num_patches": d.num_patches,
            "delete_after": d.delete_videos_after_cache and r.video_id not in keep,
        } for r in todo)

        tp = Throughput(len(todo))
        pbar = tqdm(total=len(todo), desc=f"cache r{dist_info.rank}", unit="vid", dynamic_ncols=True,
                    disable=not dist_info.is_main)
        consecutive = {"download": 0, "other": 0}
        n_done = n_fail = 0
        last_log = time.time()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            inflight = deque()
            while jobs or inflight:
                while jobs and len(inflight) < max(d.prefetch, workers):
                    inflight.append(pool.submit(cpu_extract, jobs.popleft()))
                res = inflight.popleft().result()
                cls, vid, repo_path = res["key"]
                set_context(video=f"{cls}/{vid}", repo_path=repo_path)
                if not res["ok"]:
                    n_fail += 1
                    kind = "download" if res["stage"] == "download" else "other"
                    consecutive[kind] += 1
                    log.error("Extraction failed [%s] %s/%s at stage=%s: %s", repo_path, cls, vid, res["stage"],
                              res["error"])
                    log.debug("Worker traceback for %s:\n%s", repo_path, res.get("traceback"))
                    _append_failure(failed_path, {"class": cls, "video_id": vid, "repo_path": repo_path,
                                                  "stage": res["stage"], "error": res["error"][:500]})
                    limit = d.download_fail_fast if kind == "download" else d.extract_fail_fast
                    if consecutive[kind] >= limit:
                        raise RuntimeError(f"{consecutive[kind]} consecutive {kind} failures (limit {limit}); "
                                           f"last: {res['error']}. Likely systemic (network / auth / codec). "
                                           f"See {failed_path}. Re-run to resume.")
                else:
                    consecutive = {"download": 0, "other": 0}
                    try:
                        t = time.perf_counter()
                        latent = _latent_features(latent_tool, res)
                        if dist_info.device.type == "cuda":
                            torch.cuda.synchronize()
                        t_latent = time.perf_counter() - t
                    except Exception as exc:
                        log.error("Latent tool failed on %s/%s: %s", cls, vid, exc, exc_info=True)
                        latent, t_latent = None, 0.0
                    _save_item(item_path(cache_dir, cls, vid), res, latent, t_latent)
                    n_done += 1
                pbar.update(1)
                pbar.set_postfix(ok=n_done, fail=n_fail)
                if time.time() - last_log > 30:
                    log.info("cache progress %s | ok=%d fail=%d", tp.line(n_done + n_fail), n_done, n_fail)
                    last_log = time.time()
        pbar.close()
        del latent_tool
        if dist_info.device.type == "cuda":
            torch.cuda.empty_cache()
        log.info("Rank extraction finished: ok=%d failed=%d", n_done, n_fail)

    barrier()
    index = None
    if dist_info.is_main:
        index = finalize_index(df, cfg)
    barrier()
    if index is None:
        index = pd.read_parquet(cache_dir / "index.parquet")
    return index


def finalize_index(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cache_dir = cfg.cache_dir
    keep_rows, feats, times = [], [], []
    for i, (cls, vid) in enumerate(zip(df["class"], df["video_id"])):
        p = item_path(cache_dir, cls, vid)
        if not p.exists():
            continue
        with np.load(p) as z:
            feats.append(z["features"])
            times.append(z["times"])
        keep_rows.append(i)
    index = df.iloc[keep_rows].reset_index(drop=True)
    if index.empty:
        raise RuntimeError("Feature cache is empty -- every extraction failed. Check failed_rank*.csv.")
    t = np.stack(times)
    for i, k in enumerate(TIME_KEYS):
        index[f"t_{k}"] = t[:, i]
    index.to_parquet(cache_dir / "index.parquet", index=False)

    F = np.stack(feats)
    train = (index["split"] == "train").to_numpy()
    mean = np.nanmean(F[train], axis=0)
    std = np.nanstd(F[train], axis=0)
    stats = {"names": ALL_FEATURES, "mean": np.nan_to_num(mean).tolist(),
             "std": np.where(np.nan_to_num(std) < 1e-6, 1.0, np.nan_to_num(std)).tolist()}
    (cache_dir / "feature_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    costs = {k: float(np.median(t[:, i])) for i, k in enumerate(TIME_KEYS)}
    (cache_dir / "tool_costs.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")

    dropped = len(df) - len(index)
    log.info("Feature cache index: %d/%d rows cached (%d missing/failed)\n%s", len(index), len(df), dropped,
             index.groupby(["class", "split"]).size().unstack(fill_value=0).to_string())
    log.info("Median tool cost (s): %s", {k: round(v, 4) for k, v in costs.items()})
    failed = sorted(cache_dir.glob("failed_rank*.csv"))
    if failed:
        log.warning("Failure logs: %s", [str(f) for f in failed])
    return index
