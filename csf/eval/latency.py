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
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from csf.config import Config
from csf.eval.metrics import latency_stats
from csf.logging_utils import get_logger, set_context

log = get_logger("eval.latency")


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
