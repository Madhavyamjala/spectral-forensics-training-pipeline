"""
Live end-to-end latency benchmark from raw video files, using the exported bundle through the public
`CSFDetector` API (so it also validates that the published artefacts load and run).

For each kept test video it measures decode + tools + model forwards for every model type:
scanner (A), static arbiter (B) and the agentic CSF per profile (C). With `resident=False`
(12-16 GB GPUs) models are loaded stage-wise: Qwen first, then Llama + VAE, and the scanner latency
from stage one is added to the agentic totals.

Input : Config, export dir, list of video paths, torch device.
Output: dict {mode: latency stats (ms) + prediction agreement}, also saved to
        <work_dir>/metrics/latency_benchmark.json.
"""

from __future__ import annotations

import gc
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from csf.config import Config
from csf.eval.metrics import latency_stats
from csf.logging_utils import get_logger, set_context

log = get_logger("eval.latency")


TOOL_DEPENDENCY_GROUPS: Dict[str, Tuple[str, ...]] = {
    "no_tools": (),
    "spatial": ("spatial",),
    "spectral": ("spectral",),
    "latent": ("latent",),
    "spatial_spectral": ("spatial", "spectral"),
    "spatial_latent": ("spatial", "latent"),
    "spectral_latent": ("spectral", "latent"),
    "all_tools": ("spatial", "spectral", "latent"),
}


def _latency_rows(index, num_videos: int):
    """Select a deterministic held-out test set for raw-video benchmarks.

    Note:
        The selection is independent of the cached feature retention policy. This lets
        --latencynum request more raw videos after training without rebuilding features.

    TODO:
        Add an optional class-stratified benchmark sampler while preserving the deterministic
        default used by existing runs.
    """
    if num_videos < 1:
        raise ValueError(f"num_videos must be >= 1, got {num_videos}")
    test = index[index["split"] == "test"].sort_values("video_id").reset_index(drop=True)
    if len(test) < num_videos:
        raise ValueError(
            f"Requested {num_videos} latency videos, but only {len(test)} held-out test videos "
            "are available in the run manifest."
        )
    return test.head(num_videos).copy()


def ensure_latency_videos(cfg: Config, index, num_videos: int):
    """Ensure exactly num_videos held-out test videos exist locally, downloading missing files.

    Note:
        Downloads are placed directly under cfg.paths.video_dir using each manifest row's repo_path.
        Existing non-empty files are reused, so increasing a benchmark from 100 to 1000 only fetches
        the missing 900 videos.

    TODO:
        Persist per-video checksum verification for benchmark artifacts when the dataset exposes hashes.
    """
    from huggingface_hub import hf_hub_download

    rows = _latency_rows(index, num_videos)
    video_dir = Path(cfg.paths.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    rows["video_path"] = [str(video_dir / p) for p in rows["repo_path"]]

    pending = []
    for row in rows.itertuples(index=False):
        dest = Path(row.video_path)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        pending.append((row.repo_path, dest))

    if pending:
        workers = max(1, min(int(cfg.data.download_workers), 16))
        log.info("Latency benchmark: %d/%d raw videos missing -> downloading with %d worker(s)",
                 len(pending), len(rows), workers)

        def fetch(repo_path: str, dest: Path) -> str:
            downloaded = hf_hub_download(
                cfg.data.repo_id,
                repo_path,
                repo_type="dataset",
                revision=cfg.data.revision,
                local_dir=str(video_dir),
            )
            actual = Path(downloaded)
            if not actual.exists() or actual.stat().st_size <= 0:
                raise IOError(f"Hub download returned no usable file: {actual}")
            if actual.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    import shutil
                    shutil.copy2(actual, dest)
            return str(dest)

        errors = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, repo_path, dest): (repo_path, dest)
                        for repo_path, dest in pending}
            for fut in as_completed(futures):
                repo_path, dest = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    errors.append((repo_path, repr(exc)))
                    log.error("Latency video download failed: %s | %s", repo_path, exc)

        if errors:
            sample = "; ".join(f"{p}: {e}" for p, e in errors[:5])
            raise RuntimeError(
                f"Failed to download {len(errors)} latency benchmark video(s). "
                f"Example failures: {sample}"
            )

    missing = [str(p) for p in rows["video_path"] if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"Latency video preparation finished with {len(missing)} missing file(s).")
    return rows




def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def live_latency_benchmark(cfg: Config, export_dir: Path, videos: List[Path], labels: List[str],
                           resident: bool) -> Dict[str, Any]:
    from csf.inference import CSFDetector
    if not videos:
        log.warning("No kept raw videos found for the live latency benchmark - skipped.")
        return {}
    warm = min(cfg.eval.latency_warmup, len(videos))
    records: Dict[str, List[Dict[str, Any]]] = {}

    def run(det, mode, profile=None, scanner=None):
        key = mode if mode != "agentic" else f"agentic_{profile}"
        for w in range(warm):
            det.predict(str(videos[w]), mode, profile or "balanced",
                        scanner_probs=None if scanner is None else scanner[w]["probs_arr"])
        recs = []
        for i, v in enumerate(videos):
            set_context(latency_mode=key, video=str(v))
            r = det.predict(str(v), mode, profile or "balanced",
                            scanner_probs=None if scanner is None else scanner[i]["probs_arr"])
            if scanner is not None:
                r["latency_ms"]["scanner"] = scanner[i]["latency_ms"]["scanner"]
                r["total_latency_ms"] = round(sum(r["latency_ms"].values()), 2)
            r.pop("evidence_graph", None)
            recs.append(r)
        records[key] = recs
        log.info("live latency %-22s p50 %.1f ms over %d videos", key,
                 float(np.median([r["total_latency_ms"] for r in recs])), len(recs))

    if resident:
        det = CSFDetector(str(export_dir), components=("qwen", "llama", "vae"),
                          attn_implementation=cfg.models.attn_implementation)
        run(det, "scanner")
        run(det, "static")
        for p in cfg.eval.profiles:
            run(det, "agentic", p)
        del det
    else:
        det = CSFDetector(str(export_dir), components=("qwen",), attn_implementation=cfg.models.attn_implementation)
        run(det, "scanner")
        for r in records["scanner"]:
            r["probs_arr"] = np.array(list(r["probs"].values()), dtype=np.float32)
        scanner = records["scanner"]
        del det
        _free()
        det = CSFDetector(str(export_dir), components=("llama", "vae"), attn_implementation=cfg.models.attn_implementation)
        run(det, "static")
        for p in cfg.eval.profiles:
            run(det, "agentic", p, scanner=scanner)
        for r in scanner:
            r.pop("probs_arr", None)
        del det
    _free()

    summary: Dict[str, Any] = {"n_videos": len(videos), "resident_models": resident}
    for key, recs in records.items():
        totals = np.array([r["total_latency_ms"] for r in recs]) / 1000.0
        comp = {}
        for r in recs:
            for k, v in r["latency_ms"].items():
                comp.setdefault(k, []).append(v)
        summary[key] = {**latency_stats(totals),
                        "component_ms_mean": {k: float(np.mean(v)) for k, v in comp.items()},
                        "accuracy_on_benchmark_videos": float(np.mean([r["label"] == l for r, l in zip(recs, labels)])),
                        "actions": [r["action"] for r in recs]}
    path = cfg.work_dir / "metrics" / "latency_benchmark.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Live latency benchmark saved to %s", path)
    return summary



def tool_dependency_benchmark(cfg: Config, export_dir: Path, videos: List[Path], labels: List[str],
                              resident: bool) -> Dict[str, Any]:
    """Benchmark the causal-inference-style dependency of predictions on each tool subset.

    Note:
        This is a fixed-subset ablation, not a causal identification claim. Every condition uses
        the same videos, Qwen scanner output, Llama no-tool dispatcher-state pass, and vision-state
        cache. Only the requested forensic tool groups change. The static reference additionally
        runs the all-tool Llama pass without cached vision states.

    TODO:
        Add bootstrap confidence intervals and paired per-video significance tests.
    """
    from csf import LABELS, LABEL2ID
    from csf.eval.metrics import classification_report_dict
    from csf.graph import normalize_features
    from csf.models.dispatcher import action_mask_array

    if not videos:
        log.warning("No videos supplied for the tool dependency benchmark - skipped.")
        return {}

    if len(videos) != len(labels):
        raise ValueError(f"videos/labels length mismatch: {len(videos)} vs {len(labels)}")

    warm = min(cfg.eval.latency_warmup, len(videos))
    conditions: Dict[str, Tuple[str, ...]] = {
        "no_tools": (),
        "spatial": ("spatial",),
        "spectral": ("spectral",),
        "latent": ("latent",),
        "spatial_spectral": ("spatial", "spectral"),
        "spatial_latent": ("spatial", "latent"),
        "spectral_latent": ("spectral", "latent"),
        "all_tools": ("spatial", "spectral", "latent"),
    }

    det = CSFDetector(
        str(export_dir),
        components=("qwen", "llama", "vae"),
        attn_implementation=cfg.models.attn_implementation,
    )

    def evaluate_one(video_path: Path, groups: Sequence[str]) -> Tuple[np.ndarray, float, Dict[str, float]]:
        native, vlm, decode_s = det.load_video(str(video_path))

        scanner_probs, scanner_s = det.scan(vlm)
        empty_mask = np.zeros(3, dtype=bool)
        p0, _, state_s, cache = det.runner.pixel_pass(
            [{"frames": vlm, "z": np.zeros(len(det.stats["mean"]), np.float32),
              "mask": empty_mask, "label": 0, "key": "video"}],
            empty_mask,
        )

        if not groups:
            probs = (scanner_probs + p0[0]) / 2.0
            return probs, decode_s + scanner_s + state_s, {
                "decode": decode_s, "scanner": scanner_s, "dispatcher_state": state_s,
                "proposal": 0.0, "tools": 0.0, "arbiter": 0.0,
            }

        res, tool_s = det.tools(native, groups)
        mask = action_mask_array(
            "full_tri_domain" if set(groups) == {"spatial", "spectral", "latent"}
            else "spatial_spectral" if set(groups) == {"spatial", "spectral"}
            else "spatial_latent" if set(groups) == {"spatial", "latent"}
            else "spectral_latent" if set(groups) == {"spectral", "latent"}
            else groups[0]
        ) & res.mask

        z = normalize_features(res.features, det.stats)
        p, arbiter_s = det.runner.cached_pass(
            [{"frames": vlm, "z": z, "mask": mask, "label": 0, "key": "video"}],
            mask, cache,
        )
        total = decode_s + scanner_s + state_s + tool_s + arbiter_s
        return p[0], total, {
            "decode": decode_s, "scanner": scanner_s, "dispatcher_state": state_s,
            "proposal": float(res.times.get("proposal", 0.0)),
            "tools": tool_s - float(res.times.get("proposal", 0.0)),
            "arbiter": arbiter_s,
        }

    # Warmup each condition with the same first few videos.
    for groups in conditions.values():
        for i in range(warm):
            evaluate_one(videos[i], groups)

    rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in conditions}
    for name, groups in conditions.items():
        set_context(tool_dependency=name)
        for video_path, label in zip(videos, labels):
            probs, total_s, components = evaluate_one(video_path, groups)
            rows[name].append({
                "video": str(video_path),
                "label": label,
                "label_id": LABEL2ID[label],
                "probs": probs.tolist(),
                "predicted": PRETTY_LABELS[int(np.argmax(probs))],
                "total_latency_ms": total_s * 1000.0,
                "components_s": components,
            })

    # Static full-tool reference: same raw videos, all tools, but no cached vision-state reuse.
    static_rows: List[Dict[str, Any]] = []
    static_groups = ("spatial", "spectral", "latent")
    for i in range(warm):
        native, vlm, _ = det.load_video(str(videos[i]))
        res, _ = det.tools(native, static_groups)
        mask = np.ones(3, dtype=bool)
        z = normalize_features(res.features, det.stats)
        det.runner.pixel_pass(
            [{"frames": vlm, "z": z, "mask": mask, "label": 0, "key": "video"}], mask
        )
    for video_path, label in zip(videos, labels):
        native, vlm, decode_s = det.load_video(str(video_path))
        res, tool_s = det.tools(native, static_groups)
        z = normalize_features(res.features, det.stats)
        mask = np.ones(3, dtype=bool)
        p, _, arbiter_s, _ = det.runner.pixel_pass(
            [{"frames": vlm, "z": z, "mask": mask, "label": 0, "key": "video"}], mask
        )
        static_rows.append({
            "video": str(video_path), "label": label, "label_id": LABEL2ID[label],
            "probs": p[0].tolist(),
            "predicted": PRETTY_LABELS[int(np.argmax(p[0]))],
            "total_latency_ms": (decode_s + tool_s + arbiter_s) * 1000.0,
            "components_s": {
                "decode": decode_s,
                "proposal": float(res.times.get("proposal", 0.0)),
                "tools": tool_s - float(res.times.get("proposal", 0.0)),
                "arbiter": arbiter_s,
            },
        })

    rows["static_reference"] = static_rows
    del det
    _free()

    summary: Dict[str, Any] = {
        "n_videos": len(videos),
        "resident_models": bool(resident),
        "design": "paired fixed-tool-subset benchmark on the same held-out raw videos",
        "conditions": {},
    }

    for name, recs in rows.items():
        probs = np.stack([np.asarray(r["probs"], dtype=np.float64) for r in recs])
        y = np.asarray([r["label_id"] for r in recs], dtype=int)
        lat_s = np.asarray([r["total_latency_ms"] for r in recs], dtype=np.float64) / 1000.0
        comp: Dict[str, List[float]] = {}
        for r in recs:
            for k, v in r["components_s"].items():
                comp.setdefault(k, []).append(v)

        metrics = classification_report_dict(
            y, probs, latency_s=lat_s, active_label_ids=cfg.data.active_label_ids()
        )
        summary["conditions"][name] = {
            "tool_groups": list(conditions.get(name, static_groups if name == "static_reference" else ())),
            "tool_count": len(conditions.get(name, static_groups if name == "static_reference" else ())),
            **metrics,
            "component_ms_mean": {k: float(np.mean(v) * 1000.0) for k, v in comp.items()},
        }

    path = cfg.work_dir / "metrics" / "tool_dependency_benchmark.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Tool dependency benchmark saved to %s", path)
    return summary
