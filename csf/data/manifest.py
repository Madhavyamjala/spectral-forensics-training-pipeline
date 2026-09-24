"""
Manifest loading and run-subset construction for Chrono-TriClass-100k.

The manifest (produced by dataset.py) has one row per video:
    class, generator_edit_method, video_id, split, duration_sec, width, height, fps, codec,
    bitrate, has_audio, sha256
with a leakage-aware (sha256-grouped) stratified train/valid/test split already assigned.

This module
  1. resolves the manifest: a local path if it exists, otherwise `manifest.csv` downloaded from
     the dataset repo on the Hub,
  2. maps every (class, video_id) to its repo file path from the repo's naming scheme, e.g.
     real-17 -> "Real/part-1/real-17.mp4" (ids that do not match fall back to a one-off repo listing
     cached in <cache_dir>/repo_index.json),
  3. fills `generator_edit_method` from the AI-Edited sub-folder name when the manifest says
     "unknown" (face_manipulation, inpainting, object_removal, ...),
  4. for the test run, draws a stratified subset of `max_rows` rows that keeps the manifest's
     split proportions and class balance (so train/valid/test all contain all three classes).

Input : DataConfig, cache_dir.
Output: pandas DataFrame with the manifest columns + `repo_path`, `label` (int), `method`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from csf import LABEL2ID
from csf.config import DataConfig
from csf.logging_utils import get_logger

log = get_logger("data.manifest")

CLASS_FOLDERS = {"real": "Real", "ai_generated": "AI_Generated", "ai_edited": "AI Edited"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
REQUIRED_COLUMNS = ["class", "video_id", "split"]


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def resolve_manifest(cfg: DataConfig, cache_dir: Path) -> Path:
    local = Path(cfg.manifest)
    if local.exists():
        log.info("Using local manifest: %s", local.resolve())
        return local
    log.info("Local manifest %s not found -> downloading '%s' from dataset repo %s",
             local, local.name, cfg.repo_id)
    path = hf_hub_download(cfg.repo_id, local.name, repo_type="dataset", revision=cfg.revision,
                           local_dir=str(cache_dir / "hub_manifest"))
    return Path(path)


def repo_path_for(cls: str, video_id: str) -> Optional[str]:
    """Deterministic repo layout (verified against the full repo listing: 0 mismatches over 97,804 files):
    Real/part-{n//10000+1}/real-n.mp4, AI_Generated/part-{n//10000+1}/aigen-n.mp4, AI Edited/<method>/aiedit-<method>-k.mp4"""
    if cls in ("real", "ai_generated"):
        m = re.match(r"^(real|aigen)-(\d+)$", video_id)
        return f"{CLASS_FOLDERS[cls]}/part-{int(m.group(2)) // 10000 + 1}/{video_id}.mp4" if m else None
    m = re.match(r"^aiedit-(.+)-\d+$", video_id)
    return f"{CLASS_FOLDERS[cls]}/{m.group(1)}/{video_id}.mp4" if m else None


def build_repo_index(cfg: DataConfig, cache_dir: Path) -> Dict[str, str]:
    """Map "<class>/<video_id>" -> repo path. Cached on disk because listing ~98k files is slow."""
    cache_file = cache_dir / "repo_index.json"
    if cache_file.exists():
        index = json.loads(cache_file.read_text(encoding="utf-8"))
        log.info("Loaded cached repo index (%d video files) from %s", len(index), cache_file)
        return index

    log.info("Listing files of dataset repo %s (one-off, cached afterwards)...", cfg.repo_id)
    files = HfApi().list_repo_files(cfg.repo_id, repo_type="dataset", revision=cfg.revision)
    folder_to_class = {_norm(folder): cls for cls, folder in CLASS_FOLDERS.items()}
    index: Dict[str, str] = {}
    for f in files:
        p = Path(f)
        if p.suffix.lower() not in VIDEO_EXTS or len(p.parts) < 2:
            continue
        cls = folder_to_class.get(_norm(p.parts[0]))
        if cls is not None:
            index[f"{cls}/{p.stem}"] = f
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(index), encoding="utf-8")
    log.info("Repo index built: %d video files", len(index))
    return index


def _method_from_path(cls: str, repo_path: str, manifest_method: str) -> str:
    if isinstance(manifest_method, str) and manifest_method not in ("", "unknown", "nan"):
        return manifest_method
    if cls == "ai_edited":
        parts = Path(repo_path).parts
        return parts[1] if len(parts) > 2 else "unknown"
    return "unknown"


def stratified_subset(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Proportional sample over (split, class); guarantees >=1 row per non-empty cell."""
    if max_rows is None or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    parts = []
    for _, g in df.groupby(["split", "class"], sort=True):
        n = max(1, int(round(len(g) * frac)))
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    out = pd.concat(parts)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    return out.sort_values(["split", "class", "video_id"]).reset_index(drop=True)


def load_run_manifest(cfg: DataConfig, cache_dir: Path, seed: int) -> pd.DataFrame:
    path = resolve_manifest(cfg, cache_dir)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing required columns {missing}; has {df.columns.tolist()}")
    df["video_id"] = df["video_id"].astype(str)
    selected = {c: LABEL2ID[c] for c in (cfg.classes or LABEL2ID)}
    if cfg.classes:
        log.warning("data.classes restricts this run to %s - the other class(es) are excluded from "
                    "train, valid and test.", sorted(selected))
    df = df[df["class"].isin(selected)].copy()
    df = df[df["split"].isin(["train", "valid", "test"])].copy()
    log.info("Manifest rows: %d | per class: %s | per split: %s", len(df),
             df["class"].value_counts().to_dict(), df["split"].value_counts().to_dict())

    df["repo_path"] = [repo_path_for(c, v) for c, v in zip(df["class"], df["video_id"])]
    if df["repo_path"].isna().any():
        log.info("%d video id(s) do not follow the standard naming -> resolving them via the repo listing",
                 int(df["repo_path"].isna().sum()))
        index = build_repo_index(cfg, cache_dir)
        df["repo_path"] = [p if isinstance(p, str) else index.get(f"{c}/{v}")
                           for c, v, p in zip(df["class"], df["video_id"], df["repo_path"])]
    n_missing = int(df["repo_path"].isna().sum())
    if n_missing:
        log.warning("%d manifest row(s) have no matching video file in %s -> dropped. Examples: %s",
                    n_missing, cfg.repo_id, df.loc[df["repo_path"].isna(), "video_id"].head(5).tolist())
        df = df[df["repo_path"].notna()].copy()

    method_col = df["generator_edit_method"] if "generator_edit_method" in df else pd.Series("unknown", index=df.index)
    df["method"] = [_method_from_path(c, p, m) for c, p, m in zip(df["class"], df["repo_path"], method_col)]
    df["label"] = df["class"].map(LABEL2ID).astype(int)

    df = stratified_subset(df, cfg.max_rows, seed)
    table = df.groupby(["class", "split"]).size().unstack(fill_value=0)
    log.info("Run subset: %d rows\n%s", len(df), table.to_string())
    for split in ("train", "valid", "test"):
        present = set(df.loc[df["split"] == split, "class"])
        if present != set(selected):
            raise ValueError(f"Split '{split}' is missing classes {set(selected) - present}; "
                             f"increase data.max_rows, or narrow data.classes.")
    return df.reset_index(drop=True)


def split_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (df[df["split"] == "train"].reset_index(drop=True),
            df[df["split"] == "valid"].reset_index(drop=True),
            df[df["split"] == "test"].reset_index(drop=True))
