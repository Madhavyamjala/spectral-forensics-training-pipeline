"""
External benchmark harness: run a trained CSF export, and the baselines, on public AI-video test sets.

What it does
    1. prepare   Builds one manifest per external dataset from files on disk (folder layout rules or a
                 CSV list), labels each video real / ai_generated / ai_edited with a category
                 (generator or edit type), flags generators that overlap the Chrono-TriClass training
                 sources, and re-encodes every video to one codec / fps / resolution so that no
                 detector can separate the classes by container format.
    2. csf       The exported CSF bundle (csf.inference.CSFDetector) in every mode: scanner, static,
                 and agentic for each dispatcher profile.
    3. qwen_zeroshot
                 The same Qwen2.5-VL-3B backbone without LoRA or head, asked "Real or Fake?". Scored by
                 the next-token probability of the two answer words, so it yields a probability and
                 an AUC rather than a single parsed string. Uses the frames CSF sees.
    4. metadata  The metadata-shortcut classifier from the ablation (the same gradient-boosting model
                 on the same seven container features, trained on the Chrono-TriClass train split),
                 applied to the original external files and to the normalised ones. If the first
                 score is high and the second is at chance, the shortcut is gone.
    5. external  Detectors that live in their own repositories and environments (D3, SPLIT, DeMamba,
                 NSG-VD, BusterX++, STALL, VideoMAE, X-CLIP, TALL, NPR, ...). The harness writes a
                 video list, optionally runs your command for each dataset, and reads back a
                 per-video score CSV (format below). Use --list-baselines to see what is known
                 about each one.
    6. report    Binary real-vs-fake metrics for every (dataset, detector), per-generator AUC against
                 all real videos (the usual GenVideo protocol), class-level metrics for CSF using the
                 same classification_report_dict as the ablation report, bootstrap confidence
                 intervals, and overlapping generators reported separately.

Datasets are given as NAME=ADAPTER:ARG (repeatable):
    --dataset genvideo=folders:/data/GenVideo-Val       GenVideo-Val (Real/ = MSR-VTT, Fake/<generator>/)
    --dataset fakeparts=folders:/data/FakePartsBench    FakeParts (T2V, IT2V, FaceSwap, Inpainting, ...)
    --dataset genvidbench=csv:/data/genvidbench_test.csv  any dataset from a CSV list (see below)
    --dataset vifbench=folders:/data/ViF-Bench
    --dataset ffpp=folders:/data/FaceForensics++        face sub-benchmark (not out-of-distribution)
    --dataset chrono=chrono:2000                         2,000 Chrono-TriClass test videos (in-distribution)
The folders adapter picks rules by the dataset NAME (genvideo, fakeparts, ffpp, celebdf, fakeavceleb;
anything else gets generic rules). Always check the prepare summary before running models: it prints
every category it found and every top-level folder it could not label. Add rules with
    --map fakeparts:Outpaint=ai_edited      (dataset:folder_name=label)

CSV list format (csv adapter):
    path,label[,category][,split]      path absolute or relative to the CSV; label real / fake /
                                       ai_generated / ai_edited / 0 / 1; rows with a split other than
                                       test are skipped unless --csv-all-splits

External detector output (one CSV per detector and dataset):
    video_id or path            identifies the video (both are in the list the harness writes)
    score  or  p_fake           higher = more likely fake; p_fake must be a probability
    pred        (optional)      the method's own decision, 0/1 or a label name
    p_real,p_generated,p_edited (optional) class probabilities for three-class methods
    latency_ms  (optional)
Run a detector per dataset with a command template:
    --external d3='cd /opt/D3 && /opt/D3/.venv/bin/python dump_scores.py --list {videos} --out {out}'
placeholders: {videos} (video_id,path), {labeled} (video_id,path,label,category - evaluation only,
for tools like D3 whose scripts want separate real and fake lists), {out}, {dataset}, {gpu}.
Or read scores you already have:
    --ingest npr=/preds/npr_{dataset}.csv

Examples
    # 1. build manifests and normalise (CPU; check the summary it prints)
    python benchmark_external.py --steps prepare --out runs/extbench \
        --dataset genvideo=folders:/data/GenVideo-Val --dataset fakeparts=folders:/data/FakePartsBench

    # 2. run CSF + zero-shot Qwen + metadata shortcut on 4 GPUs, then report
    python benchmark_external.py --steps csf,qwen_zeroshot,metadata,report --out runs/extbench \
        --model-dir runs/full_2class/export --config configs/full_2class.yaml --gpus 0,1,2,3 \
        --dataset genvideo=folders:/data/GenVideo-Val --dataset fakeparts=folders:/data/FakePartsBench

    # 3. add baselines later and re-report
    python benchmark_external.py --steps external,report --out runs/extbench \
        --ingest d3=/preds/d3_{dataset}.csv --dataset genvideo=folders:/data/GenVideo-Val ...

Every step is resumable: predictions are appended per video and already-scored videos are skipped.
Outputs: <out>/manifests/*.csv, <out>/preds/<detector>/<dataset>*.jsonl,
         <out>/report/{benchmark_report.md, benchmark_summary.csv, per_category.csv, benchmark_metrics.json}
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from csf import LABEL2ID, LABELS, PRETTY_LABELS  # noqa: E402

REAL, GENERATED, EDITED = LABEL2ID["real"], LABEL2ID["ai_generated"], LABEL2ID["ai_edited"]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".gif"}
STEPS = ["prepare", "csf", "qwen_zeroshot", "metadata", "external", "report"]
GPU_STEPS = {"csf", "qwen_zeroshot"}

# ----------------------------------------------------------------------------------------------------
# Known baselines. Only facts checked against each project's README are stated as such; everything
# else is marked as unverified. None of these write per-video scores in the format above out of the
# box as far as their READMEs show, so each needs a small dump script (or your existing predictions).
# ----------------------------------------------------------------------------------------------------
BASELINES: Dict[str, Dict[str, str]] = {
    "d3": {"repo": "https://github.com/Zig-HS/D3", "type": "training-free (second-order temporal features)",
           "notes": "README: frames are extracted first (utils/video2frame.py), then `eval.py --real-csv ... "
                    "--fake-csv ...` reports aggregate metrics for one real/fake pair. Dump its per-video "
                    "scores to the CSV format here; {labeled} gives the real/fake split it expects."},
    "split": {"repo": "arXiv 2607.02886", "type": "training-free, generated and partially edited video",
              "notes": "No code repository was linked in the material this harness was written from - check "
                       "the paper page before planning on it."},
    "demamba": {"repo": "https://github.com/chenhaoxing/DeMamba", "type": "supervised spatio-temporal",
                "notes": "Reference model on GenVideo. Checkpoint availability unverified; you may have to "
                         "train it (ideally on the Chrono-TriClass train split for the same-training comparison)."},
    "nsg_vd": {"repo": "https://github.com/ZSHsh98/NSG-VD", "type": "supervised", "notes": "Unverified CLI."},
    "busterx_pp": {"repo": "https://github.com/l8cv/BusterX", "type": "multimodal LLM detector with explanations",
                   "notes": "README (2026/06 revision): base models moved to Qwen3.5; evaluation scripts target "
                            "GenBuster-Bench (scripts/eval_genbuster_bench.sh). Running it on other videos needs an "
                            "adapter that turns its verdict into a score; prefer a token-probability score over a "
                            "parsed yes/no so AUC is defined."},
    "stall": {"repo": "https://github.com/OmerBenHayun/STALL", "type": "training-free spatio-temporal likelihood",
              "notes": "Unverified CLI."},
    "videomae": {"repo": "(generic backbone)", "type": "supervised baseline", "notes": "Train on Chrono-TriClass train split."},
    "xclip": {"repo": "(generic backbone)", "type": "supervised baseline", "notes": "Train on Chrono-TriClass train split."},
    "tall": {"repo": "(generic backbone)", "type": "supervised baseline", "notes": "Train on Chrono-TriClass train split."},
    "npr": {"repo": "(generic backbone)", "type": "supervised baseline", "notes": "Train on Chrono-TriClass train split."},
}

# ----------------------------------------------------------------------------------------------------
# Labelling rules for the folders adapter. Keys are folder names normalised to lowercase alphanumerics.
# A rule is (label, how): how="self" -> category is that folder's name; how="child" -> category is the
# next folder down (e.g. Fake/<generator>/...). The first matching path component, top-down, wins.
# ----------------------------------------------------------------------------------------------------
_GENERIC_REAL = {"real": ("real", "self"), "reals": ("real", "self"), "realvideos": ("real", "self"),
                 "original": ("real", "self"), "originals": ("real", "self"), "pristine": ("real", "self"),
                 "authentic": ("real", "self"), "msrvtt": ("real", "self"), "realmsrvtt": ("real", "self"),
                 "kinetics": ("real", "self"), "kinetics400": ("real", "self"), "youku": ("real", "self"),
                 "youkumplug": ("real", "self")}
_GENERIC_FAKE = {"fake": ("ai_generated", "child"), "fakes": ("ai_generated", "child"),
                 "generated": ("ai_generated", "child"), "aigc": ("ai_generated", "child"),
                 "synthetic": ("ai_generated", "child"), "aigenerated": ("ai_generated", "child")}
FOLDER_RULES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "generic": {**_GENERIC_REAL, **_GENERIC_FAKE},
    # GenVideo-Val.zip: GenVideo-Val/Real/<MSR-VTT videos>, GenVideo-Val/Fake/<generator>/<videos>
    # (layout as used by the D3 README: `mv GenVideo-Val/Real video/real_MSRVTT; mv GenVideo-Val/Fake/* video/`).
    "genvideo": {**_GENERIC_REAL, **_GENERIC_FAKE},
    "genvidbench": {**_GENERIC_REAL, **_GENERIC_FAKE},
    "vifbench": {**_GENERIC_REAL, **_GENERIC_FAKE},
    # FakePartsBench categories from its README: full-video T2V / IT2V; spatial FaceSwap / Inpainting /
    # Outpainting; temporal Interpolation / Extrapolation; Style. The on-disk folder names are not
    # documented there, so common spellings are listed - check the prepare summary.
    "fakeparts": {**_GENERIC_REAL,
                  "t2v": ("ai_generated", "self"), "it2v": ("ai_generated", "self"), "i2v": ("ai_generated", "self"),
                  "texttovideo": ("ai_generated", "self"), "imagetovideo": ("ai_generated", "self"),
                  "faceswap": ("ai_edited", "self"), "inpainting": ("ai_edited", "self"),
                  "inpaint": ("ai_edited", "self"), "outpainting": ("ai_edited", "self"),
                  "outpaint": ("ai_edited", "self"), "style": ("ai_edited", "self"),
                  "styletransfer": ("ai_edited", "self"), "interpolation": ("ai_edited", "self"),
                  "extrapolation": ("ai_edited", "self"), "fakeparts": ("ai_edited", "child"),
                  "fullfake": ("ai_generated", "child"), "full": ("ai_generated", "child")},
    # FaceForensics++: original_sequences/... real, manipulated_sequences/<method>/... edited
    "ffpp": {"originalsequences": ("real", "self"),
             "manipulatedsequences": ("ai_edited", "child"), **_GENERIC_REAL},
    # Celeb-DF (v2): Celeb-real, YouTube-real, Celeb-synthesis
    "celebdf": {"celebreal": ("real", "self"), "youtubereal": ("real", "self"),
                "celebsynthesis": ("ai_edited", "self"), **_GENERIC_REAL},
    # FakeAVCeleb: RealVideo-RealAudio real; FakeVideo-* edited (RealVideo-FakeAudio is audio-only -> real video)
    "fakeavceleb": {"realvideorealaudio": ("real", "self"), "realvideofakeaudio": ("real", "self"),
                    "fakevideorealaudio": ("ai_edited", "self"), "fakevideofakeaudio": ("ai_edited", "self")},
}
# Datasets that are face-manipulation sub-benchmarks already represented inside Chrono-TriClass's
# AI-Edited class: never report them as out-of-distribution.
IN_DISTRIBUTION_DATASETS = {"ffpp", "celebdf", "fakeavceleb", "chrono"}
DEFAULT_OVERLAP = ["pika", "crafter", "videocrafter", "videocrafter2", "videocraftv2"]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_label(value: Any) -> str:
    """Map the many spellings datasets use onto real / ai_generated / ai_edited."""
    v = _norm(str(value))
    if v in {"real", "0", "authentic", "pristine", "original", "genuine", "true"}:
        return "real"
    if v in {"fake", "1", "generated", "aigenerated", "aigc", "synthetic", "fullfake", "t2v", "i2v", "it2v"}:
        return "ai_generated"
    if v in {"edited", "aiedited", "2", "partial", "partialfake", "manipulated", "fakeparts", "edit"}:
        return "ai_edited"
    raise ValueError(f"cannot map label {value!r} to real / ai_generated / ai_edited")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


# ====================================================================================================
# prepare
# ====================================================================================================

def _stable_id(dataset: str, rel: str) -> str:
    return f"{dataset}-{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:16]}"


def _label_by_folders(rel_parts: Sequence[str], rules: Dict[str, Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    dirs = list(rel_parts[:-1])
    for i, part in enumerate(dirs):
        rule = rules.get(_norm(part))
        if rule is None:
            continue
        label, how = rule
        if how == "child":
            category = dirs[i + 1] if i + 1 < len(dirs) else part
        else:
            category = part
        return label, category
    return None


def adapter_folders(name: str, root: Path, extra_rules: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if not root.is_dir():
        raise FileNotFoundError(f"[{name}] folder {root} does not exist")
    rules = dict(FOLDER_RULES.get(_norm(name), FOLDER_RULES["generic"]))
    for folder, label in extra_rules.items():
        rules[_norm(folder)] = (canonical_label(label), "self")
    rows, unmatched = [], {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = p.relative_to(root)
        hit = _label_by_folders(rel.parts, rules)
        if hit is None:
            top = rel.parts[0] if len(rel.parts) > 1 else "(dataset root)"
            unmatched[top] = unmatched.get(top, 0) + 1
            continue
        label, category = hit
        rows.append({"video_id": _stable_id(name, rel.as_posix()), "src_path": str(p.resolve()),
                     "label": label, "category": category})
    return pd.DataFrame(rows), unmatched


def adapter_csv(name: str, csv_path: Path, all_splits: bool) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "path" not in cols or "label" not in cols:
        raise ValueError(f"[{name}] {csv_path} needs at least 'path' and 'label' columns, has {list(df.columns)}")
    if "split" in cols and not all_splits:
        before = len(df)
        df = df[df[cols["split"]].astype(str).str.lower().isin({"test", "testing", "eval", "evaluation"})]
        log(f"[{name}] kept {len(df)}/{before} rows with split=test (use --csv-all-splits to keep all)")
    rows, bad = [], {}
    for r in df.itertuples(index=False):
        rec = r._asdict()
        raw = str(rec[cols["path"]])
        p = Path(raw) if Path(raw).is_absolute() else (csv_path.parent / raw)
        if not p.is_file():
            bad["missing file"] = bad.get("missing file", 0) + 1
            continue
        try:
            label = canonical_label(rec[cols["label"]])
        except ValueError:
            bad[f"label {rec[cols['label']]!r}"] = bad.get(f"label {rec[cols['label']]!r}", 0) + 1
            continue
        category = str(rec[cols["category"]]) if "category" in cols else (
            str(rec[cols["generator"]]) if "generator" in cols else label)
        rows.append({"video_id": _stable_id(name, raw), "src_path": str(p.resolve()), "label": label,
                     "category": category})
    return pd.DataFrame(rows), bad


def adapter_chrono(name: str, arg: str, config: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Class-stratified Chrono-TriClass test videos, downloaded on demand through the pipeline's own
    ensure_latency_videos (which downloads whatever rows it is handed)."""
    if not config:
        raise SystemExit("--config is required for the chrono adapter")
    from csf.config import load_config
    from csf.eval.latency import ensure_latency_videos
    cfg = load_config(config)
    n = int(arg)
    index = _chrono_index(cfg)
    test = index[index["split"] == "test"]
    per = max(1, n // max(1, test["class"].nunique()))
    pick = (test.sort_values("video_id").groupby("class", group_keys=False)
            .apply(lambda g: g.sample(n=min(per, len(g)), random_state=cfg.seed)))
    rows = ensure_latency_videos(cfg, pick.reset_index(drop=True), len(pick))
    out = pd.DataFrame({"video_id": [f"{name}-{v}" for v in rows["video_id"]],
                        "src_path": [str(Path(p).resolve()) for p in rows["video_path"]],
                        "label": rows["class"].tolist(),
                        "category": [m if isinstance(m, str) and m else c
                                     for m, c in zip(rows.get("method", rows["class"]), rows["class"])]})
    return out, {}


def _chrono_index(cfg) -> pd.DataFrame:
    """The run's cached index restricted to the run manifest - the same frame main.py evaluates on."""
    index_file = cfg.cache_dir / "index.parquet"
    manifest = cfg.work_dir / "run_manifest.csv"
    for f in (index_file, manifest):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found - run the training pipeline for this config first")
    index = pd.read_parquet(index_file)
    ids = set(pd.read_csv(manifest, dtype={"video_id": str})["video_id"])
    return index[index["video_id"].astype(str).isin(ids)].reset_index(drop=True)


def _sample(df: pd.DataFrame, cap: Optional[int], seed: int) -> pd.DataFrame:
    if not cap:
        return df
    return (df.groupby(["label", "category"], group_keys=False)
            .apply(lambda g: g.sample(n=min(cap, len(g)), random_state=seed))
            .reset_index(drop=True))


def _probe(path: str) -> Optional[Dict[str, Any]]:
    from csf.generation.kinetics import probe_video
    try:
        return probe_video(Path(path))
    except Exception:
        return None


def _normalise_one(src: str, dst: Path, size: Tuple[int, int], fps: float, max_seconds: float, crf: int,
                   ffmpeg: str) -> Tuple[bool, str]:
    """Re-encode onto one fixed canvas. Every output has the same width, height, fps, codec, pixel
    format and no audio: scaling only the short side (and never upscaling) left resolution and
    aspect ratio as a class signal, which is exactly the container shortcut this step removes.
    Aspect ratio is kept by letterboxing, so no content is cropped away (an edit near a border
    would otherwise vanish). Duration is only capped - see the report notes."""
    if dst.exists() and dst.stat().st_size > 0:
        return True, "cached"
    info = _probe(src)
    if not info or not info.get("width") or not info.get("height"):
        return False, "unreadable"
    w, h = size
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=bicubic,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps:g}")
    cmd = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", src, "-t", f"{max_seconds:g}", "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an",
           "-map_metadata", "-1", "-movflags", "+faststart", str(tmp)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False, "ffmpeg timeout"
    if res.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False, (res.stderr or "ffmpeg failed").strip().splitlines()[-1][:200]
    tmp.replace(dst)
    return True, "ok"


def step_prepare(args, datasets: Dict[str, Tuple[str, str]], extra_maps: Dict[str, Dict[str, str]]) -> None:
    out = Path(args.out)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    overlap = {_norm(g) for g in args.overlap_generators.split(",") if g.strip()}
    ffmpeg = None
    if not args.no_normalize:
        from csf.generation.ffmpeg_tools import ffmpeg_exe
        ffmpeg = ffmpeg_exe()
        if ffmpeg is None:
            raise SystemExit("ffmpeg not found (needed to normalise videos). Install it, or pass --no-normalize "
                             "and accept that container differences stay visible to every detector.")
    for name, (adapter, arg) in datasets.items():
        log(f"[{name}] building manifest with the {adapter} adapter from {arg}")
        if adapter == "folders":
            df, skipped = adapter_folders(name, Path(arg), extra_maps.get(name, {}))
        elif adapter == "csv":
            df, skipped = adapter_csv(name, Path(arg), args.csv_all_splits)
        elif adapter == "chrono":
            df, skipped = adapter_chrono(name, arg, args.config)
        else:
            raise SystemExit(f"unknown adapter {adapter!r} for {name}; use folders, csv or chrono")
        if df.empty:
            raise SystemExit(f"[{name}] no labelled videos found. Unlabelled top-level folders: {skipped}. "
                             f"Add --map {name}:<folder>=<real|ai_generated|ai_edited> rules.")
        df = _sample(df.drop_duplicates("video_id"), args.max_per_category, args.seed)
        df["dataset"] = name
        df["overlap"] = [_norm(c) in overlap for c in df["category"]]
        df["in_distribution"] = name in IN_DISTRIBUTION_DATASETS

        if ffmpeg:
            norm_dir = out / "videos_norm" / name
            ok_col, why_col, paths = [], [], []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(_normalise_one, src, norm_dir / f"{vid}.mp4", _size(args.norm_size),
                                    args.norm_fps, args.norm_max_seconds, args.norm_crf, ffmpeg): i
                        for i, (vid, src) in enumerate(zip(df["video_id"], df["src_path"]))}
                results: Dict[int, Tuple[bool, str]] = {}
                done = 0
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()
                    done += 1
                    if done % 500 == 0:
                        log(f"[{name}] normalised {done}/{len(df)}")
            for i, vid in enumerate(df["video_id"]):
                ok, why = results[i]
                ok_col.append(ok)
                why_col.append(why)
                paths.append(str((norm_dir / f"{vid}.mp4").resolve()) if ok else "")
            df["path"], df["usable"], df["prep_note"] = paths, ok_col, why_col
        else:
            df["path"], df["usable"], df["prep_note"] = df["src_path"], True, "not normalised"

        bad = df[~df["usable"]]
        df.to_csv(out / "manifests" / f"{name}.csv", index=False)
        good = df[df["usable"]]
        log(f"[{name}] {len(good)} usable videos ({len(bad)} failed to normalise)")
        summary = good.groupby(["label", "category", "overlap"]).size().reset_index(name="n")
        print(summary.to_string(index=False))
        if skipped:
            print(f"  NOT labelled (skipped): {skipped}")
        if len(bad):
            print(f"  normalisation failures, first 5: {bad['prep_note'].head().tolist()}")
    meta = {"normalised": not args.no_normalize, "size": args.norm_size, "fps": args.norm_fps,
            "max_seconds": args.norm_max_seconds, "crf": args.norm_crf, "codec": "libx264 yuv420p, audio dropped",
            "overlap_generators": sorted(overlap), "max_per_category": args.max_per_category}
    (out / "manifests" / "_prepare.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_manifests(out: Path, names: Iterable[str]) -> Dict[str, pd.DataFrame]:
    res = {}
    for n in names:
        f = out / "manifests" / f"{n}.csv"
        if not f.exists():
            raise SystemExit(f"{f} missing - run --steps prepare for dataset {n} first")
        df = pd.read_csv(f, dtype={"video_id": str})
        res[n] = df[df["usable"].astype(bool)].reset_index(drop=True)
    return res


# ====================================================================================================
# prediction bookkeeping (resumable, sharded)
# ====================================================================================================

class PredWriter:
    def __init__(self, out: Path, detector: str, dataset: str, shard: Tuple[int, int]):
        self.path = out / "preds" / detector / f"{dataset}.shard{shard[0]}of{shard[1]}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue        # a line cut off by a crash; the video is simply redone
                if "error" not in rec:  # failed videos (e.g. a transient OOM) are retried on resume
                    self.done.add(rec["video_id"])
        self.fh = open(self.path, "a", encoding="utf-8")

    def write(self, rec: Dict[str, Any]) -> None:
        self.fh.write(json.dumps(rec, default=float) + "\n")
        self.fh.flush()
        self.done.add(rec["video_id"])     # within this run, never score a video twice

    def close(self) -> None:
        self.fh.close()


def write_kind(out: Path, detector: str, dataset: str, probability: bool,
               classes: Optional[Sequence[int]] = None) -> None:
    """What the report needs to know about a scorer: whether its score is a probability (so 0.5 is a
    meaningful threshold) and which label ids it can output (its class-level metrics average over
    those, not over all of LABELS)."""
    f = out / "preds" / detector / f"{dataset}.kind.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"probability": bool(probability),
                             "classes": [int(c) for c in classes] if classes is not None else None}),
                 encoding="utf-8")


def _shard(df: pd.DataFrame, shard: Tuple[int, int]) -> pd.DataFrame:
    k, n = shard
    return df.iloc[k::n] if n > 1 else df


def _probs_record(probs: Dict[str, float]) -> Dict[str, float]:
    p = {l: float(probs.get(PRETTY_LABELS[l], 0.0)) for l in LABELS}
    return {"p_real": p["real"], "p_generated": p["ai_generated"], "p_edited": p["ai_edited"],
            "score": 1.0 - p["real"], "pred": LABELS[int(np.argmax([p[l] for l in LABELS]))]}


# ====================================================================================================
# CSF
# ====================================================================================================

def step_csf(args, manifests: Dict[str, pd.DataFrame], shard: Tuple[int, int]) -> None:
    import torch
    from csf.inference import CSFDetector
    if not args.model_dir:
        raise SystemExit("--model-dir (the exported CSF bundle) is required for the csf step")
    modes = [m.strip() for m in args.csf_modes.split(",") if m.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    comps = set()
    if "scanner" in modes or "agentic" in modes:
        comps.add("qwen")
    if "static" in modes or "agentic" in modes:
        comps.update({"llama", "vae"})
    det = CSFDetector(args.model_dir, components=tuple(sorted(comps)), attn_implementation=args.attn)
    log(f"CSF bundle {args.model_dir} | classes {det.active_classes} | modes {modes} | profiles {profiles} "
        f"| device {det.device}")
    missing = [p for p in profiles if "agentic" in modes and p not in det.policies]
    if missing:
        raise SystemExit(f"bundle has no dispatcher for {missing}; available: {sorted(det.policies)}")
    names = (["csf_scanner"] if "scanner" in modes else []) + (["csf_static"] if "static" in modes else []) + \
            ([f"csf_agentic_{p}" for p in profiles] if "agentic" in modes else [])

    for ds, df in manifests.items():
        writers = {n: PredWriter(Path(args.out), n, ds, shard) for n in names}
        for n in names:
            write_kind(Path(args.out), n, ds, True, det.active_ids)
        todo = _shard(df, shard)
        todo = todo[[any(v not in w.done for w in writers.values()) for v in todo["video_id"]]]
        log(f"[csf/{ds}] {len(todo)} videos to score on this shard")
        t_last = time.time()
        for i, row in enumerate(todo.itertuples(index=False)):
            base = {"video_id": row.video_id}
            try:
                scanner_probs, scanner_ms = None, 0.0
                if "scanner" in modes or "agentic" in modes:
                    r = det.predict(row.path, "scanner")
                    scanner_probs = np.array([r["probs"][PRETTY_LABELS[l]] for l in LABELS], dtype=np.float32)
                    scanner_ms = float(r["latency_ms"].get("scanner", 0.0))
                    if "csf_scanner" in writers and row.video_id not in writers["csf_scanner"].done:
                        writers["csf_scanner"].write({**base, **_probs_record(r["probs"]),
                                                      "latency_ms": r["total_latency_ms"]})
                if "static" in modes and row.video_id not in writers["csf_static"].done:
                    r = det.predict(row.path, "static")
                    writers["csf_static"].write({**base, **_probs_record(r["probs"]),
                                                 "latency_ms": r["total_latency_ms"], "action": r["action"]})
                for p in (profiles if "agentic" in modes else []):
                    w = writers[f"csf_agentic_{p}"]
                    if row.video_id in w.done:
                        continue
                    # scanner probabilities are reused (as the latency benchmark does); its measured
                    # time is added back so the agentic latency is end-to-end
                    r = det.predict(row.path, "agentic", p, scanner_probs=scanner_probs)
                    w.write({**base, **_probs_record(r["probs"]),
                             "latency_ms": r["total_latency_ms"] + scanner_ms, "action": r["action"]})
            except Exception as exc:  # one bad video must not end a 20k-video run
                msg = f"{type(exc).__name__}: {str(exc)[:300]}"
                for n, w in writers.items():
                    if row.video_id not in w.done:
                        w.write({**base, "error": msg})
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            if time.time() - t_last > 60:
                log(f"[csf/{ds}] {i + 1}/{len(todo)}")
                t_last = time.time()
        for w in writers.values():
            w.close()
    del det
    torch.cuda.empty_cache()


# ====================================================================================================
# zero-shot Qwen2.5-VL (backbone only)
# ====================================================================================================

ZS_QUESTION = "Is this video real camera footage or AI-generated? Answer with exactly one word: Real or Fake."


def step_qwen_zeroshot(args, manifests: Dict[str, pd.DataFrame], shard: Tuple[int, int]) -> None:
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenVL
    except ImportError:                                   # very new transformers renamed the auto class
        from transformers import AutoModelForImageTextToText as QwenVL
    from csf.data.video_io import decode_frames, to_vlm_frames

    frame_cfg = {"num_frames": 8, "frame_size": 224, "tool_max_side": 1024}
    model_id = args.qwen_id
    if args.model_dir and (Path(args.model_dir) / "csf_config.json").exists():
        bundle = json.loads((Path(args.model_dir) / "csf_config.json").read_text(encoding="utf-8"))
        frame_cfg.update({k: bundle[k] for k in frame_cfg if k in bundle})   # same frames CSF sees
        model_id = model_id or bundle.get("qwen_id")
    model_id = model_id or "Qwen/Qwen2.5-VL-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = QwenVL.from_pretrained(model_id, torch_dtype=dtype, attn_implementation=args.attn).to(device).eval()
    tok = processor.tokenizer
    cand = {}
    for word in ("Real", "Fake"):
        ids = tok.encode(word, add_special_tokens=False)
        cand[word] = ids[0]
        if len(ids) != 1:
            log(f"note: '{word}' tokenises to {len(ids)} tokens; scoring its first sub-token")
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": ZS_QUESTION}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    log(f"zero-shot {model_id} | frames {frame_cfg} | device {device}")

    for ds, df in manifests.items():
        w = PredWriter(Path(args.out), "qwen_zeroshot", ds, shard)
        write_kind(Path(args.out), "qwen_zeroshot", ds, True, [REAL, GENERATED])
        todo = _shard(df, shard)
        todo = todo[~todo["video_id"].isin(w.done)]
        log(f"[qwen_zeroshot/{ds}] {len(todo)} videos to score on this shard")
        t_last = time.time()
        for i, row in enumerate(todo.itertuples(index=False)):
            try:
                native = decode_frames(Path(row.path), frame_cfg["num_frames"], frame_cfg["tool_max_side"])
                frames = to_vlm_frames(native, frame_cfg["frame_size"])
                enc = processor(text=[prompt], videos=[list(frames)], return_tensors="pt").to(device)
                if "pixel_values_videos" in enc:
                    enc["pixel_values_videos"] = enc["pixel_values_videos"].to(dtype)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = model(**enc).logits[0, -1].float()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                full = torch.softmax(logits, -1)
                pair = torch.softmax(logits[[cand["Fake"], cand["Real"]]], -1)
                p_fake = float(pair[0])
                w.write({"video_id": row.video_id, "score": p_fake, "p_real": 1 - p_fake, "p_generated": p_fake,
                         "p_edited": 0.0, "pred": "ai_generated" if p_fake >= 0.5 else "real", "latency_ms": ms,
                         # how much of the next-token mass is on the two answers: low means the model
                         # is not answering in the requested format and the score is less meaningful
                         "answer_mass": float(full[cand["Fake"]] + full[cand["Real"]])})
            except Exception as exc:
                w.write({"video_id": row.video_id, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            if time.time() - t_last > 60:
                log(f"[qwen_zeroshot/{ds}] {i + 1}/{len(todo)}")
                t_last = time.time()
        w.close()
    del model
    torch.cuda.empty_cache()


# ====================================================================================================
# metadata shortcut
# ====================================================================================================

META_COLS = ("width", "height", "fps", "duration_sec", "bitrate", "codec", "has_audio")


def _meta_matrix(df: pd.DataFrame) -> np.ndarray:
    # identical feature construction to csf/eval/ablation.py::_baselines.meta
    return np.column_stack([df["width"], df["height"], df["fps"], df["duration_sec"],
                            np.log1p(df["bitrate"].astype(float)), (df["codec"] == "h264").astype(int),
                            df["has_audio"].astype(int)]).astype(np.float32)


def step_metadata(args, manifests: Dict[str, pd.DataFrame]) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    if not args.config:
        raise SystemExit("--config is required for the metadata step (it trains on that run's train split)")
    from csf.config import load_config
    cfg = load_config(args.config)
    index = _chrono_index(cfg)
    missing = [c for c in META_COLS if c not in index.columns]
    if missing:
        raise SystemExit(f"the run index has no {missing} columns; cannot train the metadata shortcut")
    train = index[index["split"] == "train"]
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=cfg.seed).fit(_meta_matrix(train), train["label"])
    classes = [int(c) for c in clf.classes_]
    log(f"metadata shortcut trained on {len(train)} Chrono train videos, classes {[LABELS[c] for c in classes]}")

    for ds, df in manifests.items():
        for variant, col in (("raw", "src_path"), ("normalized", "path")):
            name = f"metadata_shortcut_{variant}"
            w = PredWriter(Path(args.out), name, ds, (0, 1))
            write_kind(Path(args.out), name, ds, True, classes)
            todo = df[~df["video_id"].isin(w.done)]
            if todo.empty:
                w.close()
                continue
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                probes = list(pool.map(_probe, todo[col].tolist()))
            ok = [p is not None for p in probes]
            feats = pd.DataFrame([p for p in probes if p is not None])
            if len(feats):
                proba = clf.predict_proba(_meta_matrix(feats))
                full = np.zeros((len(feats), len(LABELS)))
                full[:, classes] = proba
            j = 0
            for vid, good in zip(todo["video_id"], ok):
                if not good:
                    w.write({"video_id": vid, "error": "probe failed"})
                    continue
                p = full[j]
                j += 1
                w.write({"video_id": vid, "p_real": p[REAL], "p_generated": p[GENERATED], "p_edited": p[EDITED],
                         "score": 1.0 - p[REAL], "pred": LABELS[int(np.argmax(p))]})
            w.close()
            log(f"[{name}/{ds}] scored {sum(ok)} videos")


# ====================================================================================================
# external detectors
# ====================================================================================================

def _parse_kv(items: Sequence[str], what: str) -> Dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{what} expects NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _ingest_csv(path: Path, df: pd.DataFrame, detector: str, dataset: str, out: Path) -> int:
    pred = pd.read_csv(path)
    cols = {c.lower(): c for c in pred.columns}
    if "video_id" in cols:
        key = pred[cols["video_id"]].astype(str)
        known = set(df["video_id"])
    elif "path" in cols:
        by_path = {}
        for vid, p, s in zip(df["video_id"], df["path"], df["src_path"]):
            for q in (p, s):
                if isinstance(q, str) and q:
                    by_path[str(Path(q).resolve())] = vid
                    by_path.setdefault(Path(q).name, vid)
        key = pred[cols["path"]].astype(str).map(lambda q: by_path.get(str(Path(q).resolve()), by_path.get(Path(q).name)))
        known = set(df["video_id"])
    else:
        raise SystemExit(f"[{detector}/{dataset}] {path} needs a video_id or path column")
    if "score" in cols:
        score = pred[cols["score"]].astype(float)
    elif "p_fake" in cols:
        score = pred[cols["p_fake"]].astype(float)
    elif "p_real" in cols:
        score = 1.0 - pred[cols["p_real"]].astype(float)
    else:
        raise SystemExit(f"[{detector}/{dataset}] {path} needs a score, p_fake or p_real column")
    target = out / "preds" / detector / f"{dataset}.ingested.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(target, "w", encoding="utf-8") as fh:
        for i, vid in enumerate(key):
            if vid is None or (isinstance(vid, float) and math.isnan(vid)) or vid not in known:
                continue
            rec: Dict[str, Any] = {"video_id": vid, "score": float(score.iloc[i])}
            if "p_fake" in cols and "score" in cols:
                rec["p_fake"] = float(pred[cols["p_fake"]].iloc[i])
            for c in ("p_real", "p_generated", "p_edited", "latency_ms"):
                if c in cols:
                    rec[c] = float(pred[cols[c]].iloc[i])
            if "pred" in cols:
                v = pred[cols["pred"]].iloc[i]
                try:
                    rec["pred"] = canonical_label(v)
                except ValueError:
                    pass
            fh.write(json.dumps(rec) + "\n")
            n += 1
    # 0.5 is only a meaningful threshold on a probability: p_fake / p_real columns are, a bare score
    # only when the user says so with --prob-scores
    is_prob = "p_fake" in cols or "p_real" in cols or detector in _PROB_SCORE_DETECTORS
    write_kind(out, detector, dataset, is_prob)
    return n


_PROB_SCORE_DETECTORS: set = set()


def step_external(args, manifests: Dict[str, pd.DataFrame]) -> None:
    out = Path(args.out)
    commands = _parse_kv(args.external, "--external")
    ingests = _parse_kv(args.ingest, "--ingest")
    _PROB_SCORE_DETECTORS.update(d.strip() for d in args.prob_scores.split(",") if d.strip())
    if not commands and not ingests:
        log("external: nothing to do (no --external or --ingest given)")
        return
    for ds, df in manifests.items():
        lists = out / "external_inputs"
        lists.mkdir(parents=True, exist_ok=True)
        videos = lists / f"{ds}.videos.csv"
        labeled = lists / f"{ds}.labeled.csv"
        df[["video_id", "path"]].to_csv(videos, index=False)
        df[["video_id", "path", "label", "category"]].to_csv(labeled, index=False)
        for det, template in commands.items():
            target = out / "external_raw" / det / f"{ds}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or args.rerun_external:
                cmd = template.format(videos=shlex.quote(str(videos)), labeled=shlex.quote(str(labeled)),
                                      out=shlex.quote(str(target)), dataset=ds, gpu=args.gpus.split(",")[0] if args.gpus else "0")
                log(f"[{det}/{ds}] $ {cmd}")
                res = subprocess.run(cmd, shell=True)
                if res.returncode != 0 or not target.exists():
                    log(f"[{det}/{ds}] command failed (exit {res.returncode}); skipping")
                    continue
            n = _ingest_csv(target, df, det, ds, out)
            log(f"[{det}/{ds}] ingested {n}/{len(df)} scores")
        for det, pattern in ingests.items():
            matches = sorted(glob.glob(pattern.format(dataset=ds)))
            if not matches:
                log(f"[{det}/{ds}] no file matches {pattern.format(dataset=ds)}; skipping")
                continue
            n = _ingest_csv(Path(matches[0]), df, det, ds, out)
            log(f"[{det}/{ds}] ingested {n}/{len(df)} scores from {matches[0]}")


# ====================================================================================================
# report
# ====================================================================================================

def _load_preds(out: Path) -> Dict[Tuple[str, str], pd.DataFrame]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for f in sorted((out / "preds").glob("*/*.jsonl")):
        det, ds = f.parent.name, f.name.split(".")[0]
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                groups.setdefault((det, ds), []).append(json.loads(line))
            except json.JSONDecodeError:
                pass
    res = {}
    for k, rows in groups.items():
        d = pd.DataFrame(rows)
        # a video may have an error row from one attempt and a score from a later one: keep the score
        d["has_score"] = d["score"].notna() if "score" in d else False
        d = d.sort_values("has_score").drop_duplicates("video_id", keep="last")
        res[k] = d
    return res


def _kind(out: Path, det: str, ds: str) -> Dict[str, Any]:
    f = out / "preds" / det / f"{ds}.kind.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {"probability": False, "classes": None}


def binary_metrics(y: np.ndarray, score: np.ndarray, pred: Optional[np.ndarray], prob: bool) -> Dict[str, Any]:
    """y: 1 = fake. score: higher = more fake. pred: the method's own 0/1 decision if it has one."""
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    out: Dict[str, Any] = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    both = out["n_real"] > 0 and out["n_fake"] > 0
    if both:
        out["auc"] = float(roc_auc_score(y, score))
        out["ap"] = float(average_precision_score(y, score))
        fpr, tpr, _ = roc_curve(y, score)
        i = int(np.argmin(np.abs((1 - tpr) - fpr)))
        out["eer"] = float((fpr[i] + 1 - tpr[i]) / 2)
        out["tpr_at_fpr_1pct"] = float(np.interp(0.01, fpr, tpr))
        out["tpr_at_fpr_5pct"] = float(np.interp(0.05, fpr, tpr))
    if pred is None and prob:
        pred = (score >= 0.5).astype(int)
    if pred is not None:
        tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
        fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
        tpr_, tnr_ = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        out.update(accuracy=(tp + tn) / max(len(y), 1), fake_recall=tpr_ if out["n_fake"] else None,
                   real_specificity=tnr_ if out["n_real"] else None,
                   balanced_accuracy=(tpr_ + tnr_) / 2 if both else None,
                   precision=tp / max(tp + fp, 1), f1=2 * tp / max(2 * tp + fp + fn, 1))
    if prob:
        out["brier"] = float(np.mean((np.clip(score, 0, 1) - y) ** 2))
    return out


def bootstrap_ci(y: np.ndarray, score: np.ndarray, pred: Optional[np.ndarray], prob: bool, n_boot: int,
                 seed: int) -> Dict[str, Tuple[float, float]]:
    """Percentile 95% intervals, resampling real and fake videos separately so both stay present."""
    from sklearn.metrics import roc_auc_score
    if n_boot <= 0 or (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return {}
    rng = np.random.default_rng(seed)
    real, fake = np.where(y == 0)[0], np.where(y == 1)[0]
    dec = pred if pred is not None else ((score >= 0.5).astype(int) if prob else None)
    aucs, baccs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(real, len(real)), rng.choice(fake, len(fake))])
        aucs.append(roc_auc_score(y[idx], score[idx]))
        if dec is not None:
            d, t = dec[idx], y[idx]
            baccs.append((((d == 1) & (t == 1)).sum() / max((t == 1).sum(), 1) +
                          ((d == 0) & (t == 0)).sum() / max((t == 0).sum(), 1)) / 2)
    ci = {"auc_ci95": (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))}
    if baccs:
        ci["balanced_accuracy_ci95"] = (float(np.percentile(baccs, 2.5)), float(np.percentile(baccs, 97.5)))
    return ci


def _pred_vector(d: pd.DataFrame) -> Optional[np.ndarray]:
    """The method's own real/fake decision, or None when it did not give one for every video."""
    if "pred" not in d or d["pred"].isna().any():
        return None
    return np.array([0 if p == "real" else 1 for p in d["pred"]])


def step_report(args, manifests: Dict[str, pd.DataFrame]) -> None:
    from csf.eval.metrics import classification_report_dict
    out = Path(args.out)
    rdir = out / "report"
    rdir.mkdir(parents=True, exist_ok=True)
    preds = _load_preds(out)
    if not preds:
        raise SystemExit(f"no predictions under {out / 'preds'}; run a model step first")
    prep = out / "manifests" / "_prepare.json"
    prep = json.loads(prep.read_text(encoding="utf-8")) if prep.exists() else {}
    import warnings
    # overlap-only subsets and single-label categories legitimately have undefined AUCs; those come
    # back as "-" in the tables, so sklearn's per-call warnings are only noise here
    warnings.filterwarnings("ignore", module="sklearn")

    summary, per_cat, full = [], [], {}
    for (det, ds), d in sorted(preds.items()):
        if ds not in manifests:
            continue
        m = manifests[ds].merge(d, on="video_id", how="left")
        n_err = int(m["score"].isna().sum()) if "score" in m else len(m)
        m = m[m["score"].notna()] if "score" in m else m.iloc[0:0]
        if m.empty:
            continue
        kind = _kind(out, det, ds)
        prob = bool(kind.get("probability"))
        for scope, sub in (("primary", m[~m["overlap"].astype(bool)]), ("overlap_only", m[m["overlap"].astype(bool)])):
            if sub.empty:
                continue
            y = (sub["label"] != "real").astype(int).to_numpy()
            score = sub["score"].astype(float).to_numpy()
            pv = _pred_vector(sub)
            met = binary_metrics(y, score, pv, prob)
            met.update(bootstrap_ci(y, score, pv, prob, args.bootstrap if scope == "primary" else 0, args.seed))
            if "latency_ms" in sub and sub["latency_ms"].notna().any():
                lat = sub["latency_ms"].dropna().to_numpy()
                met.update(latency_ms_p50=float(np.percentile(lat, 50)), latency_ms_p95=float(np.percentile(lat, 95)))
            if "answer_mass" in sub:
                met["mean_answer_mass"] = float(sub["answer_mass"].mean())
            row = {"dataset": ds, "detector": det, "scope": scope, "errors": n_err if scope == "primary" else None,
                   "in_distribution": ds in IN_DISTRIBUTION_DATASETS, "score_is_probability": prob, **met}
            summary.append(row)
            full.setdefault(ds, {}).setdefault(det, {})[scope] = met

            # class-level metrics for detectors that emit class probabilities (CSF, metadata shortcut)
            if scope == "primary" and {"p_real", "p_generated", "p_edited"} <= set(sub.columns) \
                    and sub[["p_real", "p_generated", "p_edited"]].notna().all().all():
                # average over the classes this scorer can output (recorded by the step that ran it)
                ids = kind.get("classes") or list(range(len(LABELS)))
                keep = sub["label"].map(LABEL2ID).isin(ids)
                yy = sub.loc[keep, "label"].map(LABEL2ID).to_numpy()
                if len(np.unique(yy)) >= 2:
                    probs = sub.loc[keep, ["p_real", "p_generated", "p_edited"]].to_numpy(dtype=float)
                    rep = classification_report_dict(yy, probs, active_label_ids=ids)
                    full[ds][det]["class_level"] = {"n_used": int(keep.sum()),
                                                    "n_excluded_outside_classes": int((~keep).sum()), **rep}

        # per category: detection rate for each fake category, specificity for each real one, and
        # the per-generator AUC of that category's fakes against every real video (GenVideo protocol)
        reals = m[m["label"] == "real"]
        pv_all = _pred_vector(m)
        dec_all = pv_all if pv_all is not None else ((m["score"].to_numpy() >= 0.5).astype(int) if prob else None)
        for (lab, cat), g in m.groupby(["label", "category"]):
            rec = {"dataset": ds, "detector": det, "label": lab, "category": cat, "n": len(g),
                   "overlap": bool(g["overlap"].iloc[0])}
            if dec_all is not None:
                dec = dec_all[m.index.get_indexer(g.index)]
                rec["rate_called_fake" if lab != "real" else "rate_called_real"] = \
                    float(dec.mean()) if lab != "real" else float(1 - dec.mean())
            if lab != "real" and len(reals):
                from sklearn.metrics import roc_auc_score, average_precision_score
                yy = np.r_[np.zeros(len(reals)), np.ones(len(g))]
                ss = np.r_[reals["score"].to_numpy(), g["score"].to_numpy()]
                rec["auc_vs_all_real"] = float(roc_auc_score(yy, ss))
                rec["ap_vs_all_real"] = float(average_precision_score(yy, ss))
            per_cat.append(rec)

    sdf, cdf = pd.DataFrame(summary), pd.DataFrame(per_cat)
    sdf.to_csv(rdir / "benchmark_summary.csv", index=False)
    cdf.to_csv(rdir / "per_category.csv", index=False)
    (rdir / "benchmark_metrics.json").write_text(json.dumps({"prepare": prep, "results": full}, indent=2,
                                                            default=lambda o: list(o) if isinstance(o, tuple) else float(o)),
                                                 encoding="utf-8")
    _write_markdown(rdir, sdf, cdf, manifests, prep, full)
    log(f"report written to {rdir}")


def _f(v: Any, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return "-"
    return f"{100 * v:.1f}" if pct else f"{v:.3f}"


def _write_markdown(rdir: Path, sdf: pd.DataFrame, cdf: pd.DataFrame, manifests: Dict[str, pd.DataFrame],
                    prep: Dict[str, Any], full: Dict[str, Any]) -> None:
    L = ["# CSF external benchmark", ""]
    if prep:
        L += [f"Videos normalised: **{prep.get('normalised')}** - {prep.get('size')} letterboxed canvas, "
              f"{prep.get('fps')} fps, first {prep.get('max_seconds')} s, {prep.get('codec')} (CRF {prep.get('crf')}). "
              f"Overlapping generators (reported separately): {', '.join(prep.get('overlap_generators', [])) or 'none'}.", ""]
    L += ["## Datasets", "", "| Dataset | Real | AI-Generated | AI-Edited | Overlap videos | Out-of-distribution |",
          "|---|---|---|---|---|---|"]
    for ds, df in manifests.items():
        c = df["label"].value_counts()
        L.append(f"| {ds} | {c.get('real', 0)} | {c.get('ai_generated', 0)} | {c.get('ai_edited', 0)} | "
                 f"{int(df['overlap'].astype(bool).sum())} | {'no (in-distribution)' if ds in IN_DISTRIBUTION_DATASETS else 'yes'} |")
    if sdf.empty:
        (rdir / "benchmark_report.md").write_text("\n".join(L), encoding="utf-8")
        return
    prim = sdf[sdf["scope"] == "primary"]
    dsets = list(manifests)
    L += ["", "## Real vs fake: AUC (95% CI) / balanced accuracy %", "",
          "Fake = AI-Generated + AI-Edited. Balanced accuracy uses the method's own decision, or 0.5 on a "
          "probability; methods that output an uncalibrated score get AUC only.", "",
          "| Detector | " + " | ".join(dsets) + " |", "|---|" + "---|" * len(dsets)]
    for det in sorted(prim["detector"].unique()):
        cells = []
        for ds in dsets:
            r = prim[(prim["detector"] == det) & (prim["dataset"] == ds)]
            if r.empty:
                cells.append("-")
                continue
            r = r.iloc[0]
            ci = r.get("auc_ci95")
            ci_s = f" ({ci[0]:.3f}-{ci[1]:.3f})" if isinstance(ci, (list, tuple)) else ""
            cells.append(f"{_f(r.get('auc'))}{ci_s} / {_f(r.get('balanced_accuracy'), True)}")
        L.append(f"| {det} | " + " | ".join(cells) + " |")

    for ds in dsets:
        sub = sdf[sdf["dataset"] == ds]
        if sub.empty:
            continue
        L += ["", f"## {ds}", "", "| Detector | Scope | n | AUC | AP | EER | TPR@1%FPR | TPR@5%FPR | Acc % | Bal-Acc % | "
              "Fake recall % | Real spec. % | Brier | p50 ms | errors |", "|" + "---|" * 15]
        for _, r in sub.sort_values(["scope", "detector"]).iterrows():
            L.append(f"| {r['detector']} | {r['scope']} | {r['n']} | {_f(r.get('auc'))} | {_f(r.get('ap'))} | "
                     f"{_f(r.get('eer'))} | {_f(r.get('tpr_at_fpr_1pct'), True)} | {_f(r.get('tpr_at_fpr_5pct'), True)} | "
                     f"{_f(r.get('accuracy'), True)} | {_f(r.get('balanced_accuracy'), True)} | "
                     f"{_f(r.get('fake_recall'), True)} | {_f(r.get('real_specificity'), True)} | {_f(r.get('brier'))} | "
                     f"{_f(r.get('latency_ms_p50')) if pd.notna(r.get('latency_ms_p50')) else '-'} | "
                     f"{'' if pd.isna(r.get('errors')) else int(r['errors'])} |")
        cs = cdf[(cdf["dataset"] == ds) & (cdf["label"] != "real")] if not cdf.empty else cdf
        if not cs.empty and "auc_vs_all_real" in cs:
            piv = cs.pivot_table(index=["label", "category", "overlap"], columns="detector", values="auc_vs_all_real")
            L += ["", f"Per-category AUC against all real videos ({ds}); overlap=True rows are excluded from the "
                  "primary numbers above.", "", "| Label | Category | Overlap | " + " | ".join(piv.columns) + " |",
                  "|---|---|---|" + "---|" * len(piv.columns)]
            for (lab, cat, ov), vals in piv.iterrows():
                L.append(f"| {lab} | {cat} | {ov} | " + " | ".join(_f(v) for v in vals) + " |")
        for det, blocks in full.get(ds, {}).items():
            cl = blocks.get("class_level")
            if cl:
                L += ["", f"Class-level ({det}, classes {', '.join(cl['active_labels'])}, n={cl['n_used']}, "
                      f"{cl['n_excluded_outside_classes']} videos of other classes excluded): macro-F1 "
                      f"{_f(cl['macro_f1'], True)}%, accuracy {_f(cl['accuracy'], True)}%, confusion {cl['confusion_matrix']}"]
    L += ["", "## Notes", "",
          "- `overlap_only` rows use generators that also appear in the training sources; they are not "
          "out-of-distribution evidence.",
          "- In-distribution datasets (face sub-benchmarks, Chrono test) must not be reported as generalisation.",
          "- `metadata_shortcut_raw` vs `metadata_shortcut_normalized`: if the first is well above chance and the second "
          "is near 0.5 AUC, normalisation removed the container shortcut for every detector in this table. Duration is "
          "only capped (--norm-max-seconds), so where real and fake clip lengths differ it can survive; the normalized "
          "row measures what is left.",
          "- A two-class CSF bundle never predicts AI-Edited; on edit-only categories it can only be scored as "
          "real vs fake (rate_called_fake in per_category.csv).",
          "- qwen_zeroshot mean_answer_mass (in benchmark_summary.csv) below ~0.5 means the backbone mostly "
          "did not answer Real/Fake, so its score is weak evidence."]
    (rdir / "benchmark_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ====================================================================================================
# driver
# ====================================================================================================

def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--out", default="runs/external_benchmark")
    ap.add_argument("--dataset", action="append", default=[], help="NAME=ADAPTER:ARG (folders|csv|chrono)")
    ap.add_argument("--map", action="append", default=[], help="extra folder rule DATASET:FOLDER=LABEL")
    ap.add_argument("--steps", default="all", help=f"comma list of {STEPS} or 'all'")
    ap.add_argument("--model-dir", help="exported CSF bundle (runs/<run>/export)")
    ap.add_argument("--config", help="training config of that run (metadata shortcut, chrono adapter)")
    ap.add_argument("--csf-modes", default="scanner,static,agentic")
    ap.add_argument("--profiles", default="ultra_fast,balanced,max_security")
    ap.add_argument("--qwen-id", default=None, help="zero-shot backbone (default: the bundle's qwen_id)")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--gpus", default="", help="e.g. 0,1,2,3: one worker per GPU for the csf/qwen steps")
    ap.add_argument("--shard", default="0/1", help=argparse.SUPPRESS)
    ap.add_argument("--external", action="append", default=[], help="NAME='command with {videos} {out}'")
    ap.add_argument("--ingest", action="append", default=[], help="NAME=/path/scores_{dataset}.csv")
    ap.add_argument("--prob-scores", default="", help="external detectors whose 'score' column is a probability")
    ap.add_argument("--rerun-external", action="store_true")
    ap.add_argument("--overlap-generators", default=",".join(DEFAULT_OVERLAP))
    ap.add_argument("--max-per-category", type=int, default=None, help="cap videos per (label, category)")
    ap.add_argument("--csv-all-splits", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--norm-size", default="854x480", help="fixed WxH canvas (letterboxed)")
    ap.add_argument("--norm-fps", type=float, default=24.0)
    ap.add_argument("--norm-max-seconds", type=float, default=10.0)
    ap.add_argument("--norm-crf", type=int, default=23)
    ap.add_argument("--workers", type=int, default=8, help="parallel ffmpeg/ffprobe processes")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list-baselines", action="store_true")
    return ap.parse_args(argv)


def _size(spec: str) -> Tuple[int, int]:
    m = re.match(r"^(\d+)x(\d+)$", spec)
    if not m or int(m.group(1)) % 2 or int(m.group(2)) % 2:
        raise SystemExit(f"--norm-size expects even WxH such as 854x480, got {spec!r}")
    return int(m.group(1)), int(m.group(2))


def _parse_datasets(items: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    res = {}
    for it in items:
        m = re.match(r"^([A-Za-z0-9_\-]+)=(folders|csv|chrono):(.+)$", it)
        if not m:
            raise SystemExit(f"--dataset expects NAME=folders|csv|chrono:ARG, got {it!r}")
        res[m.group(1)] = (m.group(2), m.group(3))
    return res


def _parse_maps(items: Sequence[str]) -> Dict[str, Dict[str, str]]:
    res: Dict[str, Dict[str, str]] = {}
    for it in items:
        m = re.match(r"^([^:]+):([^=]+)=(.+)$", it)
        if not m:
            raise SystemExit(f"--map expects DATASET:FOLDER=LABEL, got {it!r}")
        canonical_label(m.group(3))
        res.setdefault(m.group(1), {})[m.group(2)] = m.group(3)
    return res


def _spawn_gpu_workers(args, steps: List[str], gpus: List[str]) -> None:
    """One child per GPU, each scoring a disjoint shard; the parent waits for all of them."""
    base = [a for a in sys.argv[1:]]
    cleaned, skip = [], False
    for a in base:                                  # drop --steps/--gpus/--shard from the child command
        if skip:
            skip = False
            continue
        if a in ("--steps", "--gpus", "--shard"):
            skip = True
            continue
        if a.startswith(("--steps=", "--gpus=", "--shard=")):
            continue
        cleaned.append(a)
    procs = []
    logdir = Path(args.out) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    for k, g in enumerate(gpus):
        cmd = [sys.executable, str(Path(__file__).resolve()), *cleaned, "--steps", ",".join(steps),
               "--shard", f"{k}/{len(gpus)}"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": g}
        fh = open(logdir / f"{'_'.join(steps)}.gpu{g}.log", "a", encoding="utf-8")
        log(f"GPU {g}: shard {k}/{len(gpus)} -> {fh.name}")
        procs.append((g, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
    failed = []
    for g, p, fh in procs:
        if p.wait() != 0:
            failed.append(g)
        fh.close()
    if failed:
        raise SystemExit(f"worker(s) on GPU {failed} failed - see {logdir}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_baselines:
        for k, v in BASELINES.items():
            print(f"{k:12s} {v['type']}\n{'':12s} {v['repo']}\n{'':12s} {v['notes']}\n")
        return 0
    steps = STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f"unknown step(s) {bad}; choose from {STEPS}")
    datasets = _parse_datasets(args.dataset)
    if not datasets:
        raise SystemExit("give at least one --dataset NAME=ADAPTER:ARG")
    k, n = (int(x) for x in args.shard.split("/"))
    shard = (k, n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _size(args.norm_size)
    if "csf" in steps and not (args.model_dir and (Path(args.model_dir) / "csf_config.json").exists()):
        raise SystemExit(f"--model-dir must be an exported CSF bundle (with csf_config.json), got {args.model_dir!r}")
    if "metadata" in steps and not args.config:
        raise SystemExit("--config (the training config of the run) is required for the metadata step")
    if "prepare" in steps and n == 1:
        step_prepare(args, datasets, _parse_maps(args.map))
    manifests = load_manifests(out, datasets)

    gpu_steps = [s for s in steps if s in GPU_STEPS]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if gpu_steps and len(gpus) > 1 and n == 1:
        _spawn_gpu_workers(args, gpu_steps, gpus)
    else:
        if len(gpus) == 1 and n == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
        if "csf" in gpu_steps:
            step_csf(args, manifests, shard)
        if "qwen_zeroshot" in gpu_steps:
            step_qwen_zeroshot(args, manifests, shard)
    if n > 1:                                   # a GPU worker: the parent does the rest
        return 0
    if "metadata" in steps:
        step_metadata(args, manifests)
    if "external" in steps:
        step_external(args, manifests)
    if "report" in steps:
        step_report(args, manifests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
