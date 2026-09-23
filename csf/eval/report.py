"""
Unified Markdown results report for a CSF run.

Combines training summaries, held-out ablation metrics, live raw-video latency,
and the fixed forensic-tool subset benchmark into one readable artifact.

Input : Config and optional cached feature index.
Output: <work_dir>/metrics/full_results.md.

Note:
    The report is descriptive and does not assign an overall ranking.

TODO:
    Add confidence intervals and paired statistical tests when benchmark outputs support them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _load(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON object when the file exists and is valid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a scalar for Markdown."""
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "-"
    return f"{v:.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    """Format a fraction as a percentage."""
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(v):
        return "-"
    return f"{100.0 * v:.{digits}f}%"


def _ms(value: Any, digits: int = 1) -> str:
    """Format milliseconds."""
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(v):
        return "-"
    return f"{v:.{digits}f}"


def _dataset_counts(index) -> str:
    """Return cached train/valid/test counts."""
    if index is None:
        return "-"
    try:
        counts = index["split"].value_counts().to_dict()
        return ", ".join(
            f"{name}={int(counts.get(name, 0))}" for name in ("train", "valid", "test")
        )
    except (KeyError, TypeError):
        return "-"


def _training(lines: list[str], metrics_dir: Path) -> None:
    """Append model training summaries."""
    lines += [
        "## 1. Training",
        "",
        "| Model | Training time (s) | Optimizer steps | Epochs | Best validation F1 | Peak VRAM GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    found = False
    for kind in ("qwen", "llama"):
        data = _load(metrics_dir / f"train_{kind}.json")
        if not data:
            continue
        found = True
        time_s = data.get("training_time_s", data.get("train_time_s"))
        steps = data.get("optimizer_steps", data.get("steps"))
        epochs = data.get("epochs_seen", data.get("epochs"))
        best_f1 = data.get("best_val_f1", data.get("best_validation_f1"))
        vram = data.get("peak_vram_gib", data.get("peak_vram"))
        lines.append(
            f"| {kind.upper()} | {_fmt(time_s, 1)} | {steps if steps is not None else '-'} "
            f"| {_fmt(epochs, 2)} | {_fmt(best_f1, 4)} | {_fmt(vram, 2)} |"
        )
    if not found:
        lines.append("| - | Training summary files not found | - | - | - | - |")
    lines += [
        "",
        "Training loss logs: logs/train_qwen.jsonl and logs/train_llama.jsonl.",
        "",
    ]


def _ablation(lines: list[str], data: Optional[Dict[str, Any]]) -> None:
    """Append held-out test-set evaluation metrics."""
    lines += ["## 2. Held-out test-set evaluation", ""]
    if not data or "models" not in data:
        lines += ["Evaluation results are not present yet.", ""]
        return

    models = data["models"]
    lines += [
        "| Model | N | Accuracy | Balanced Acc. | Macro-F1 | Weighted-F1 | MCC | ROC-AUC | PR-AUC | Fake Recall | Miss Rate | False Alarm | ECE | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in models.items():
        fake = m.get("binary_fake_detection", {})
        lines.append(
            f"| {name} | {m.get('n', '-')} | {_pct(m.get('accuracy'))} "
            f"| {_pct(m.get('balanced_accuracy'))} | {_pct(m.get('macro_f1'))} "
            f"| {_pct(m.get('weighted_f1'))} | {_fmt(m.get('mcc'))} "
            f"| {_fmt(m.get('roc_auc_ovr_macro'))} | {_fmt(m.get('pr_auc_macro'))} "
            f"| {_pct(fake.get('recall'))} | {_pct(fake.get('miss_rate'))} "
            f"| {_pct(fake.get('false_alarm_rate'))} | {_fmt(m.get('ece'))} "
            f"| {_ms(m.get('latency_ms_p50'))} | {_ms(m.get('latency_ms_p95'))} |"
        )

    lines += [
        "",
        "### Per-class metrics",
        "",
        "| Model | Class | Precision | Recall | F1 | Support |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, m in models.items():
        for label, cm in m.get("per_class", {}).items():
            lines.append(
                f"| {name} | {label} | {_pct(cm.get('precision'))} | {_pct(cm.get('recall'))} "
                f"| {_pct(cm.get('f1'))} | {cm.get('support', '-')} |"
            )

    lines += [
        "",
        "### Calibration and binary-forensics metrics",
        "",
        "| Model | Log-loss | Brier | ECE | Fake ROC-AUC | Edited→Generated | Generated→Edited |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in models.items():
        fake = m.get("binary_fake_detection", {})
        lines.append(
            f"| {name} | {_fmt(m.get('log_loss'))} | {_fmt(m.get('brier'))} | {_fmt(m.get('ece'))} "
            f"| {_fmt(fake.get('roc_auc'))} | {_pct(fake.get('edited_as_generated_rate'))} "
            f"| {_pct(fake.get('generated_as_edited_rate'))} |"
        )

    profiles = {k: v for k, v in models.items() if k.startswith("C_csf_agentic_")}
    if profiles:
        lines += [
            "",
            "### Learned routing profiles",
            "",
            "| Profile | Accuracy | Macro-F1 | Miss Rate | p50 ms | Tool Cost ms | Early Exit |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in profiles.items():
            lines.append(
                f"| {name.replace('C_csf_agentic_', '')} | {_pct(m.get('accuracy'))} "
                f"| {_pct(m.get('macro_f1'))} | {_pct(m.get('miss_rate'))} "
                f"| {_ms(m.get('latency_ms_p50'))} | {_ms(m.get('mean_tool_cost_ms'))} "
                f"| {_pct(m.get('early_exit_rate'))} |"
            )


def _latency(lines: list[str], metrics_dir: Path) -> None:
    """Append live raw-video latency results."""
    lines += ["## 3. Live raw-video latency benchmark", ""]
    data = _load(metrics_dir / "latency_benchmark.json")
    if not data:
        lines += ["Live latency results are not present yet.", ""]
        return

    lines += [
        f"Benchmark videos: **{data.get('n_videos', '-')}**  ",
        f"Resident models: **{data.get('resident_models', '-')}**",
        "",
        "| Condition | Accuracy | Mean ms | p50 ms | p90 ms | p95 ms | p99 ms | Throughput videos/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in data.items():
        if name in {"n_videos", "resident_models"} or not isinstance(m, dict):
            continue
        lines.append(
            f"| {name} | {_pct(m.get('accuracy_on_benchmark_videos'))} "
            f"| {_ms(m.get('latency_ms_mean'))} | {_ms(m.get('latency_ms_p50'))} "
            f"| {_ms(m.get('latency_ms_p90'))} | {_ms(m.get('latency_ms_p95'))} "
            f"| {_ms(m.get('latency_ms_p99'))} | {_fmt(m.get('throughput_videos_per_s'), 2)} |"
        )
    lines.append("")


def _dependency(lines: list[str], metrics_dir: Path) -> None:
    """Append fixed forensic-tool subset results."""
    lines += ["## 4. Tool-dependency benchmark", ""]
    data = _load(metrics_dir / "tool_dependency_benchmark.json")
    if not data:
        lines += ["Tool-dependency results are not present yet.", ""]
        return

    lines += [
        f"Benchmark videos: **{data.get('n_videos', '-')}**",
        "",
        "| Condition | Tool count | Tools | Accuracy | Macro-F1 | Fake Recall | Miss Rate | False Alarm | p50 ms | p95 ms | Throughput videos/s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in data.get("conditions", {}).items():
        fake = m.get("binary_fake_detection", {})
        tools = ", ".join(m.get("tool_groups", [])) or "none"
        lines.append(
            f"| {name} | {m.get('tool_count', 0)} | {tools} | {_pct(m.get('accuracy'))} "
            f"| {_pct(m.get('macro_f1'))} | {_pct(fake.get('recall'))} "
            f"| {_pct(fake.get('miss_rate'))} | {_pct(fake.get('false_alarm_rate'))} "
            f"| {_ms(m.get('latency_ms_p50'))} | {_ms(m.get('latency_ms_p95'))} "
            f"| {_fmt(m.get('throughput_videos_per_s'), 2)} |"
        )
        comps = m.get("component_ms_mean", {})
        if comps:
            lines.append("Component means (ms): " + ", ".join(f"{k}={_ms(v)}" for k, v in comps.items()))
    lines += [
        "",
        "The all_tools row contains spatial + spectral + latent. static_reference is the separate "
        "full-tool Llama reference without cached vision-state reuse.",
        "",
    ]


def write_full_results_report(cfg, index=None) -> Path:
    """Write one consolidated Markdown report from completed run artifacts.

    Note:
        Missing sections are shown explicitly so the report remains useful for partial runs.

    TODO:
        Add direct plot embeds and a small run timeline when those artifacts are present.
    """
    metrics_dir = cfg.work_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Chrono-Spectral Forensics — Full Results ({cfg.run_name})",
        "",
        f"- Mode: {cfg.mode}",
        f"- Active classes: {', '.join(cfg.data.classes or ['real', 'ai_generated', 'ai_edited'])}",
        f"- Cached dataset splits: {_dataset_counts(index)}",
        f"- Latency sample target: {cfg.eval.latency_samples}",
        "",
        "This is a consolidated record of measured results. It does not assign an overall ranking.",
        "",
    ]

    _training(lines, metrics_dir)
    _ablation(lines, _load(metrics_dir / "ablation_metrics.json"))
    _latency(lines, metrics_dir)
    _dependency(lines, metrics_dir)

    lines += [
        "## 5. Artifact index",
        "",
        "| File | Contents |",
        "|---|---|",
        "| ablation_report.md | Detailed held-out ablation report |",
        "| ablation_summary.csv | Tabular ablation metrics |",
        "| ablation_metrics.json | Full machine-readable ablation metrics |",
        "| latency_benchmark.json | Live raw-video latency results |",
        "| tool_dependency_benchmark.json | Fixed tool-subset results |",
        "| dispatcher_curves.json | GRPO training/selection curves |",
        "| plots/ | Confusion, ROC, quality, Pareto, loss, and GRPO plots |",
        "",
        "## 6. Reproduction",
        "",
        "    python full_2class.py --stage evaluate --force evaluate",
        f"    python full_2class.py --stage latency --latencynum {cfg.eval.latency_samples} --tooldependency",
        "    python full_2class.py --stage report",
        "",
    ]

    path = metrics_dir / "full_results.md"
    path.write_text("\\n".join(lines), encoding="utf-8")
    return path
