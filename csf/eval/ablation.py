"""
Ablation study: dispatcher training per reward profile, evaluation of every model type, report.

Model types compared on the held-out TEST split:
  A  qwen_scanner                 Qwen2.5-VL-3B + LoRA, frames only (Phase 1 alone)
  B  llama_arbiter_static         Llama-3.2-11B-Vision + LoRA, every tool always run (Phases 3+4, no routing)
  C  csf_agentic_<profile>        full CSF: scanner -> GRPO dispatcher -> sparse tools -> shared-state arbiter,
                                  one dispatcher per proposal profile (ultra_fast / balanced / max_security)
Supporting rows:
  csf_fixed_<action>              fixed routing ladder (each tool subset without a learned dispatcher)
  baseline_metadata_shortcut      gradient boosting on container metadata only (resolution, fps, codec, ...)
                                  -> measures how much of the task is solvable by dataset-source shortcuts
  baseline_toolpool_gbdt          gradient boosting on the toolpool features only (no VLM)

Dispatchers are trained on the VALID split outcome table (85% train / 15% model selection), never on
data the VLMs were fitted on, so the scanner probabilities they see are not over-confident.

Latency per video = decode + model forwards + tool groups actually executed (all measured).

Input : Config, index, feature stats, scanner/outcome prediction dicts, tool cost medians.
Output: <work_dir>/metrics/{ablation_metrics.json, ablation_summary.csv, ablation_report.md, plots/*.png},
        <work_dir>/checkpoints/dispatcher/<profile>.pt
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from csf import LABELS, PRETTY_LABELS
from csf.config import Config
from csf.data.feature_cache import item_path
from csf.eval.metrics import classification_report_dict
from csf.logging_utils import get_logger, set_context
from csf.models.dispatcher import (ACTION_GROUPS, ACTIONS, PROFILES, DispatcherPolicy, GRPOTrainer, action_costs,
                                   dispatcher_scalars)
from csf.tools.toolpool import ALL_FEATURES

log = get_logger("eval.ablation")
TOOL_COL = {"proposal": 1, "spatial": 2, "spectral": 3, "latent": 4}

def _align_probs_to_labels(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map classifier probability columns into the canonical tri-class label order.

    Subset-mode runs can train a baseline on fewer than all canonical classes. Scikit-learn
    then returns one probability column per observed class; downstream CSF evaluation expects
    a stable [real, ai_generated, ai_edited] representation. Missing classes receive zero.

    Note:
        The pipeline remains tri-class internally. This helper only normalizes classifier output
        shape at the evaluation boundary.

    TODO:
        Distinguish "class absent from this evaluation" from a genuinely zero-probability class
        in the human-readable report when running subset experiments.
    """
    probs = np.asarray(probs, dtype=np.float64)
    classes = np.asarray(classes, dtype=int)
    aligned = np.zeros((len(probs), len(LABELS)), dtype=np.float64)
    for src_idx, label_id in enumerate(classes):
        if 0 <= int(label_id) < len(LABELS):
            aligned[:, int(label_id)] = probs[:, src_idx]
    return aligned



def _tensors(o: Dict[str, np.ndarray], device):
    state = torch.tensor(o["state"].astype(np.float32), device=device)
    scalars = torch.tensor(dispatcher_scalars(o["scanner_probs"], o["notool_probs"]), device=device)
    labels = torch.tensor(o["labels"], device=device, dtype=torch.long)
    preds = torch.tensor(o["action_probs"].argmax(-1), device=device, dtype=torch.long)
    return state, scalars, labels, preds


def train_dispatchers(cfg: Config, outcomes_valid: Dict[str, np.ndarray], tool_costs: Dict[str, float],
                      device: torch.device) -> Dict[str, Path]:
    dcfg = cfg.train.dispatcher
    arbiter_s = float(np.median(outcomes_valid["t_actions"][:, 1:]))
    costs = action_costs(tool_costs, arbiter_s)
    log.info("Dispatcher action costs (s): %s", dict(zip(ACTIONS, np.round(costs, 4).tolist())))
    n = len(outcomes_valid["labels"])
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    n_sel = max(1, int(0.15 * n))
    sel_idx, tr_idx = perm[:n_sel], perm[n_sel:]
    state, scalars, labels, preds = _tensors(outcomes_valid, device)
    out_dir = cfg.work_dir / "checkpoints" / "dispatcher"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves: Dict[str, List[Dict[str, float]]] = {}
    paths = {}

    oracle = (preds == labels.unsqueeze(1)).float()
    log.info("Outcome-table accuracy per action on dispatcher data: %s | oracle-routing upper bound %.4f",
             {a: round(float(oracle[:, i].mean()), 4) for i, a in enumerate(ACTIONS)}, float(oracle.max(1).values.mean()))

    for pname in cfg.eval.profiles:
        profile = PROFILES[pname]
        torch.manual_seed(cfg.seed)
        policy = DispatcherPolicy(state.shape[1], scalars.shape[1], dcfg.hidden_dim)
        trainer = GRPOTrainer(policy, profile, costs, dcfg, device)
        best, best_state, hist = -1e9, None, []
        set_context(profile=pname)
        for it in range(1, dcfg.iterations + 1):
            b = torch.tensor(rng.choice(tr_idx, size=min(dcfg.batch_videos, len(tr_idx)), replace=False), device=device)
            st = trainer.step(state[b], scalars[b], labels[b], preds[b])
            if it % 10 == 0 or it == dcfg.iterations:
                ev = trainer.greedy_eval(state[sel_idx], scalars[sel_idx], labels[sel_idx], preds[sel_idx])
                hist.append({"iter": it, **st, **{f"sel_{k}": v for k, v in ev.items()}})
                if ev["reward"] > best:
                    best, best_state = ev["reward"], {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                if it % 50 == 0 or it == dcfg.iterations:
                    log.info("[GRPO %s] iter %d/%d | loss %.4f kl %.4f ent %.3f train_reward %.3f | "
                             "select: reward %.3f acc %.3f cost %.3f early_exit %.2f",
                             pname, it, dcfg.iterations, st["loss"], st["kl"], st["entropy"], st["reward"],
                             ev["reward"], ev["acc"], ev["cost_norm"], ev["early_exit_rate"])
        path = out_dir / f"{pname}.pt"
        torch.save({"state_dict": best_state, "config": policy.config, "profile": profile.__dict__,
                    "action_costs_s": costs.tolist(), "actions": ACTIONS, "best_select_reward": best}, path)
        curves[pname] = hist
        paths[pname] = path
        log.info("[GRPO %s] saved best dispatcher (select reward %.4f) -> %s", pname, best, path)
    (cfg.work_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "metrics" / "dispatcher_curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    return paths


def _load_policy(path: Path, device) -> DispatcherPolicy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    policy = DispatcherPolicy(c["state_dim"], c["n_scalar"], c["hidden"], c["n_actions"])
    policy.load_state_dict(ck["state_dict"])
    return policy.to(device).eval()


def _tool_seconds(t_tools: np.ndarray, groups: List[str]) -> np.ndarray:
    if not groups:
        return np.zeros(len(t_tools))
    return t_tools[:, TOOL_COL["proposal"]] + sum(t_tools[:, TOOL_COL[g]] for g in groups)


def _load_raw_features(index: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    feats = []
    for c, v in zip(index["class"], index["video_id"]):
        with np.load(item_path(cache_dir, c, v)) as z:
            feats.append(z["features"])
    return np.stack(feats)


def _baselines(cfg: Config, index: pd.DataFrame, test_keys: np.ndarray) -> Dict[str, Dict[str, Any]]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    res = {}
    key = index["class"] + "/" + index["video_id"]
    train = index[index["split"] == "train"]
    test = index.set_index(key).loc[test_keys].reset_index(drop=True)

    def meta(df):
        return np.column_stack([df["width"], df["height"], df["fps"], df["duration_sec"],
                                np.log1p(df["bitrate"].astype(float)), (df["codec"] == "h264").astype(int),
                                df["has_audio"].astype(int)]).astype(np.float32)

    if all(c in index.columns for c in ("width", "height", "fps", "duration_sec", "bitrate", "codec", "has_audio")):
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=cfg.seed).fit(meta(train), train["label"])
        t0 = time.perf_counter()
        probs = _align_probs_to_labels(clf.predict_proba(meta(test)), clf.classes_)
        per = (time.perf_counter() - t0) / len(test)
        res["baseline_metadata_shortcut"] = {"probs": probs, "latency": np.full(len(test), per), "tool": np.zeros(len(test))}

    log.info("Loading toolpool features for GBDT baseline (%d train / %d test)...", len(train), len(test))
    Xtr, Xte = _load_raw_features(train, cfg.cache_dir), _load_raw_features(test, cfg.cache_dir)
    clf = HistGradientBoostingClassifier(max_iter=400, random_state=cfg.seed).fit(Xtr, train["label"])
    t0 = time.perf_counter()
    probs = _align_probs_to_labels(clf.predict_proba(Xte), clf.classes_)
    per = (time.perf_counter() - t0) / len(test)
    tools = test[["t_proposal", "t_spatial", "t_spectral", "t_latent"]].sum(1).to_numpy()
    res["baseline_toolpool_gbdt"] = {"probs": probs, "latency": test["t_decode"].to_numpy() + tools + per,
                                     "tool": tools}
    importances = dict(zip(ALL_FEATURES, np.round(np.abs(Xte).mean(0), 3).tolist()))
    log.debug("Toolpool feature mean |value| on test: %s", importances)
    return res


def evaluate_ablation(cfg: Config, index: pd.DataFrame, scanner_test: Dict[str, np.ndarray],
                      outcomes_test: Dict[str, np.ndarray], dispatcher_paths: Dict[str, Path],
                      device: torch.device) -> Dict[str, Dict[str, Any]]:
    o = outcomes_test
    y, methods, t_tools = o["labels"], o["methods"], o["t_tools"]
    decode = t_tools[:, 0]
    results: Dict[str, Dict[str, Any]] = {}
    raw: Dict[str, Dict[str, np.ndarray]] = {}

    active_ids = cfg.data.active_label_ids()

    def add(name, probs, latency, tool=None, actions=None):
        results[name] = classification_report_dict(y, probs, latency, methods, actions,
                                                    ACTIONS if actions is not None else None, tool, cfg.eval.ece_bins,
                                                    active_label_ids=active_ids)
        raw[name] = {"probs": probs, "latency": latency}

    add("A_qwen_scanner", o["scanner_probs"], decode + o["scanner_latency"])
    all_tools = _tool_seconds(t_tools, ["spatial", "spectral", "latent"])
    add("B_llama_arbiter_static", o["static_probs"], decode + all_tools + o["t_static"], all_tools)

    base = decode + o["scanner_latency"] + o["t_state"]
    state, scalars, _, _ = _tensors(o, device)
    for pname in cfg.eval.profiles:
        policy = _load_policy(dispatcher_paths[pname], device)
        from csf.models.dispatcher import allowed_actions
        with torch.no_grad():
            acts = policy(state, scalars, allowed_actions(PROFILES[pname])).argmax(-1).cpu().numpy()
        probs = o["action_probs"][np.arange(len(y)), acts]
        tool = np.array([_tool_seconds(t_tools[i:i + 1], ACTION_GROUPS[ACTIONS[a]])[0] for i, a in enumerate(acts)])
        lat = base + tool + o["t_actions"][np.arange(len(y)), acts]
        add(f"C_csf_agentic_{pname}", probs, lat, tool, acts)

    for ai, a in enumerate(ACTIONS):
        tool = _tool_seconds(t_tools, ACTION_GROUPS[a])
        add(f"csf_fixed_{a}", o["action_probs"][:, ai], base + tool + o["t_actions"][:, ai], tool,
            np.full(len(y), ai))

    for name, b in _baselines(cfg, index, o["keys"]).items():
        add(name, b["probs"], b["latency"], b["tool"])
    return results, raw


def _plots(results: Dict[str, Dict], raw: Dict[str, Dict], y: np.ndarray, cfg: Config) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    pdir = cfg.work_dir / "metrics" / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    made = []
    main = [k for k in results if k.startswith(("A_", "B_", "C_"))]
    active_ids = cfg.data.active_label_ids()

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in results.items():
        style = "o" if name in main else ("s" if name.startswith("baseline") else "x")
        ax.scatter(m["latency_ms_p50"], m["macro_f1"], marker=style, s=80 if name in main else 40)
        ax.annotate(name.replace("csf_", ""), (m["latency_ms_p50"], m["macro_f1"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("median end-to-end latency per video (ms, log)")
    ax.set_ylabel("macro-F1 (test)")
    ax.set_title("Pareto frontier: quality vs latency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "pareto_latency_vs_macro_f1.png", dpi=130)
    plt.close(fig)
    made.append("pareto_latency_vs_macro_f1.png")

    fig, axes = plt.subplots(1, len(main), figsize=(4.2 * len(main), 4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, main):
        ConfusionMatrixDisplay(np.array(results[name]["confusion_matrix_normalized"]),
                               display_labels=[PRETTY_LABELS[LABELS[l]] for l in active_ids]).plot(ax=ax, colorbar=False,
                                                                                         values_format=".2f")
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(pdir / "confusion_matrices.png", dpi=130)
    plt.close(fig)
    made.append("confusion_matrices.png")

    fig, axes = plt.subplots(1, len(active_ids), figsize=(4.5 * len(active_ids), 4.5))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, active_ids):
        for name in main:
            try:
                RocCurveDisplay.from_predictions((y == k).astype(int), raw[name]["probs"][:, k], name=name, ax=ax,
                                                 plot_chance_level=(name == main[0]))
            except ValueError:
                pass
        ax.set_title(f"ROC one-vs-rest: {PRETTY_LABELS[LABELS[k]]}")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(pdir / "roc_curves.png", dpi=130)
    plt.close(fig)
    made.append("roc_curves.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = list(results)
    x = np.arange(len(names))
    for off, metric in zip((-0.25, 0, 0.25), ("accuracy", "macro_f1", "balanced_accuracy")):
        ax.bar(x + off, [results[n][metric] for n in names], width=0.25, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Ablation - quality metrics (test)")
    fig.tight_layout()
    fig.savefig(pdir / "quality_bars.png", dpi=130)
    plt.close(fig)
    made.append("quality_bars.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for kind in ("qwen", "llama"):
        f = cfg.work_dir / "logs" / f"train_{kind}.jsonl"
        if f.exists():
            recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            if recs:
                ax.plot([r["step"] for r in recs], [r["loss"] for r in recs], label=f"{kind} train loss")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("cross-entropy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "training_loss.png", dpi=130)
    plt.close(fig)
    made.append("training_loss.png")

    curves_f = cfg.work_dir / "metrics" / "dispatcher_curves.json"
    if curves_f.exists():
        curves = json.loads(curves_f.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(8, 4))
        for pname, hist in curves.items():
            ax.plot([h["iter"] for h in hist], [h["sel_reward"] for h in hist], label=pname)
        ax.set_xlabel("GRPO iteration")
        ax.set_ylabel("greedy reward (selection split)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(pdir / "grpo_reward.png", dpi=130)
        plt.close(fig)
        made.append("grpo_reward.png")
    return made


def write_report(cfg: Config, results: Dict[str, Dict], raw: Dict[str, Dict], y: np.ndarray,
                 extra: Dict[str, Any]) -> Path:
    mdir = cfg.work_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "ablation_metrics.json").write_text(json.dumps({"models": results, **extra}, indent=2, default=float),
                                                encoding="utf-8")
    cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc", "roc_auc_ovr_macro", "pr_auc_macro",
            "log_loss", "ece", "latency_ms_p50", "latency_ms_p95", "throughput_videos_per_s", "mean_tool_cost_ms",
            "early_exit_rate"]
    rows = []
    for name, m in results.items():
        row = {"model": name, **{c: m.get(c) for c in cols}}
        row["fake_recall"] = m["binary_fake_detection"]["recall"]
        row["miss_rate"] = m["binary_fake_detection"]["miss_rate"]
        row["false_alarm_rate"] = m["binary_fake_detection"]["false_alarm_rate"]
        for lab in active_ids:
            row[f"f1_{LABELS[lab]}"] = m["per_class"][LABELS[lab]]["f1"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(mdir / "ablation_summary.csv", index=False)
    plots = _plots(results, raw, y, cfg)

    def fmt(v, pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return f"{100 * v:.2f}" if pct else (f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}")

    lines = [f"# Chrono-Spectral Forensics - ablation report ({cfg.run_name}, mode={cfg.mode})", "",
             f"Test videos: {len(y)} | class counts: " + ", ".join(f"{PRETTY_LABELS[LABELS[i]]}={int((y == i).sum())}"
                                                                     for i in active_ids), "",
             f"Active classes: " + ", ".join(PRETTY_LABELS[LABELS[i]] for i in active_ids), "",
             "## Main comparison (model types)", "",
             "| Model | Acc % | Bal-Acc % | Macro-F1 % | ROC-AUC | Fake recall % | Miss rate % | False alarm % | ECE | p50 ms | p95 ms | videos/s | Early exit |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        if not r["model"].startswith(("A_", "B_", "C_")):
            continue
        lines.append(f"| {r['model']} | {fmt(r['accuracy'], 1)} | {fmt(r['balanced_accuracy'], 1)} | {fmt(r['macro_f1'], 1)} | "
                     f"{fmt(r['roc_auc_ovr_macro'])} | {fmt(r['fake_recall'], 1)} | {fmt(r['miss_rate'], 1)} | "
                     f"{fmt(r['false_alarm_rate'], 1)} | {fmt(r['ece'])} | {fmt(r['latency_ms_p50'])} | "
                     f"{fmt(r['latency_ms_p95'])} | {fmt(r['throughput_videos_per_s'])} | {fmt(r['early_exit_rate'])} |")
    lines += ["", "## Routing ladder and baselines", "",
              "| Model | Acc % | Macro-F1 % | Miss rate % | p50 ms | Tool cost ms |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        if r["model"].startswith(("A_", "B_", "C_")):
            continue
        lines.append(f"| {r['model']} | {fmt(r['accuracy'], 1)} | {fmt(r['macro_f1'], 1)} | {fmt(r['miss_rate'], 1)} | "
                     f"{fmt(r['latency_ms_p50'])} | {fmt(r['mean_tool_cost_ms'])} |")
    for name, m in results.items():
        if m.get("action_distribution") and name.startswith("C_"):
            lines.append(f"\n**{name} routing:** " + ", ".join(f"{a}={100 * p:.1f}%" for a, p in m["action_distribution"].items()))
    per_method = results.get("C_csf_agentic_balanced", next(iter(results.values()))).get("ai_edited_per_method", {})
    if per_method:
        lines += ["", "## AI-Edited accuracy per edit method (balanced CSF)", "", "| Method | n | Acc % | Detected as fake % |",
                  "|---|---|---|---|"]
        for meth, v in per_method.items():
            lines.append(f"| {meth} | {v['n']} | {100 * v['accuracy']:.1f} | {100 * v['detected_as_fake']:.1f} |")
    if "baseline_metadata_shortcut" in results:
        lines += ["", "> **Shortcut check:** `baseline_metadata_shortcut` uses only container metadata "
                  f"(resolution, fps, codec, bitrate, duration, audio) and reaches macro-F1 "
                  f"{100 * results['baseline_metadata_shortcut']['macro_f1']:.1f}%. The closer the VLM models are to this "
                  "number, the more the task is being solved by dataset-source cues rather than forensic evidence."]
    lines += ["", "## Notes", "",
              "- Latency = decode + every model forward + every tool group actually executed, measured per video "
              f"(eval batch size {cfg.train.llama.eval_batch_size}).",
              "- R_attr (mask mIoU) is 0 in the GRPO reward: the dataset has no manipulation masks.",
              f"- Extra info: `{json.dumps({k: v for k, v in extra.items() if not isinstance(v, dict)})}`", "",
              "## Plots", ""] + [f"![{p}](plots/{p})" for p in plots]
    path = mdir / "ablation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Ablation report written: %s\n%s", path,
             df[["model", "accuracy", "macro_f1", "miss_rate", "latency_ms_p50"]].to_string(index=False))
    return path
