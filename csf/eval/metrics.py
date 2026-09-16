"""
Evaluation metrics for tri-class video attribution.

`classification_report_dict(y, probs, ...)` returns
  quality     : accuracy, balanced accuracy, macro / weighted precision-recall-F1, MCC, Cohen's kappa,
                per-class precision / recall / F1 / support, confusion matrix (counts + row-normalised),
                one-vs-rest ROC-AUC and PR-AUC (macro + per class), log-loss, Brier score, ECE
  forensics   : binary fake-detection (non-Real vs Real) accuracy / precision / recall / F1 / ROC-AUC,
                miss rate (fake predicted Real - the proposal's beta_miss event), false-alarm rate
                (Real predicted fake), AI-Edited vs AI-Generated confusion
  robustness  : accuracy per AI-Edited method (face_manipulation, inpainting, ...)
  efficiency  : latency mean / std / p50 / p90 / p95 / p99 (ms), throughput (videos/s), mean tool cost,
                routing action distribution and early-exit rate when provided

Input : y (int[N]), probs (float[N,3]), optional latency seconds (float[N]), methods (str[N]),
        actions (int[N]) and action names.
Output: nested JSON-serialisable dict.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from csf import LABEL2ID, LABELS

REAL = LABEL2ID["real"]


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def _safe(fn, *args, **kwargs) -> Optional[float]:
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return None


def latency_stats(latency_s: np.ndarray) -> Dict[str, float]:
    ms = np.asarray(latency_s, dtype=np.float64) * 1000.0
    return {"latency_ms_mean": float(ms.mean()), "latency_ms_std": float(ms.std()),
            "latency_ms_p50": float(np.percentile(ms, 50)), "latency_ms_p90": float(np.percentile(ms, 90)),
            "latency_ms_p95": float(np.percentile(ms, 95)), "latency_ms_p99": float(np.percentile(ms, 99)),
            "throughput_videos_per_s": float(1000.0 / ms.mean()) if ms.mean() > 0 else None}


def classification_report_dict(y: np.ndarray, probs: np.ndarray, latency_s: Optional[np.ndarray] = None,
                               methods: Optional[Sequence[str]] = None, actions: Optional[np.ndarray] = None,
                               action_names: Optional[List[str]] = None, tool_cost_s: Optional[np.ndarray] = None,
                               ece_bins: int = 15) -> Dict:
    y = np.asarray(y).astype(int)
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0)
    probs = probs / probs.sum(1, keepdims=True)
    pred = probs.argmax(1)
    labels = list(range(len(LABELS)))

    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    onehot = np.eye(len(LABELS))[y]
    out: Dict = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "roc_auc_ovr_macro": _safe(roc_auc_score, onehot, probs, average="macro", multi_class="ovr"),
        "pr_auc_macro": _safe(average_precision_score, onehot, probs, average="macro"),
        "log_loss": float(log_loss(y, probs, labels=labels)),
        "brier": float(np.mean([brier_score_loss(onehot[:, k], probs[:, k]) for k in labels])),
        "ece": expected_calibration_error(y, probs, ece_bins),
        "per_class": {LABELS[k]: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                                  "support": int(s[k]),
                                  "roc_auc": _safe(roc_auc_score, onehot[:, k], probs[:, k]),
                                  "pr_auc": _safe(average_precision_score, onehot[:, k], probs[:, k])}
                      for k in labels},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": (cm / np.maximum(cm.sum(1, keepdims=True), 1)).round(4).tolist(),
    }

    fake_true = y != REAL
    fake_pred = pred != REAL
    fake_score = 1.0 - probs[:, REAL]
    tp_, fp_ = int((fake_true & fake_pred).sum()), int((~fake_true & fake_pred).sum())
    fn_, tn_ = int((fake_true & ~fake_pred).sum()), int((~fake_true & ~fake_pred).sum())
    edited, generated = LABEL2ID["ai_edited"], LABEL2ID["ai_generated"]
    out["binary_fake_detection"] = {
        "accuracy": (tp_ + tn_) / max(len(y), 1),
        "precision": tp_ / max(tp_ + fp_, 1), "recall": tp_ / max(tp_ + fn_, 1),
        "f1": 2 * tp_ / max(2 * tp_ + fp_ + fn_, 1),
        "roc_auc": _safe(roc_auc_score, fake_true.astype(int), fake_score),
        "miss_rate": fn_ / max(tp_ + fn_, 1), "false_alarm_rate": fp_ / max(fp_ + tn_, 1),
        "edited_as_generated_rate": float(((y == edited) & (pred == generated)).sum() / max((y == edited).sum(), 1)),
        "generated_as_edited_rate": float(((y == generated) & (pred == edited)).sum() / max((y == generated).sum(), 1)),
    }

    if methods is not None:
        methods = np.asarray(methods).astype(str)
        per_method = {}
        for m in sorted(set(methods[y == edited])):
            sel = (methods == m) & (y == edited)
            per_method[m] = {"n": int(sel.sum()), "accuracy": float((pred[sel] == y[sel]).mean()),
                             "detected_as_fake": float((pred[sel] != REAL).mean())}
        out["ai_edited_per_method"] = per_method

    if latency_s is not None:
        out.update(latency_stats(latency_s))
    if tool_cost_s is not None:
        out["mean_tool_cost_ms"] = float(np.mean(tool_cost_s) * 1000.0)
    if actions is not None and action_names:
        counts = np.bincount(np.asarray(actions), minlength=len(action_names))
        out["action_distribution"] = {a: float(c / max(len(actions), 1)) for a, c in zip(action_names, counts)}
        out["early_exit_rate"] = out["action_distribution"].get("early_exit", 0.0)
    return out
