"""
The SAFER paper's analyses, computed from saved per-action outcome tables.

REIMPLEMENTATION NOTICE. This module was written from the paper's text (Sections 3.3-4.4, Appendices C-E),
not from the scripts that produced its numbers. Where the paper does not pin a definition, the choice made
here is stated in the docstring of the function that makes it (search for "Assumption"). Run it on the
paper's outcome tables and compare against the paper before citing any number it prints as reproduced.

Everything here needs only numpy, scikit-learn and scipy, and operates on two kinds of table:
    in-domain   <work_dir>/predictions/outcomes_{valid,test}.npz from the training pipeline (stage `outcomes`)
    external    <bench_out>/per_action/<dataset>*.jsonl from `benchmark_external.py --steps per_action`,
                joined with <bench_out>/manifests/<dataset>.csv for labels and categories
Both are turned into a `Table`: binary labels (1 = fake), P(fake) and latency per detector mode, and the
six dispatcher actions (no-tool exit + five tool subsets) in dispatcher order.

Modes (paper names -> keys):
    SAFER-Scanner            scanner          scanner alone
    Arbiter, no tools        notool           arbiter with an empty evidence graph (same-reasoner baseline)
    SAFER default            default          no-tool exit = mean(scanner, no-tool arbiter)
    SAFER-Static             static           arbiter once with all tools, pixel pass, no dispatcher state
    arbiter given tools T    arbiter:<a>      what a dispatcher tool action returns
    evidence mode, T         evidence:<a>     mean(scanner, arbiter given T)

Analyses (paper artefact -> function):
    Table 1, Figure 2        indomain_table, low_latency, choose_tau, oracle
    McNemar tests            mcnemar
    Table 9                  decision_changes
    Table 13 (0.5 column)    fixed_broken, least_confident_forcing
    Tables 10, 16, Fig. 3    tool_value (per-category and benchmark-level same-reasoner ΔAUC, TOST, correlations)
    Tables 4, 12, App. D     adaptation (recalibration, real-only recalibration, router refit, routing oracle)
    Table 3 (supervised row) SupervisedRouter

CLI:
    python -m safer.eval.paper indomain --outcomes-dir runs/safer/predictions --out runs/safer/paper
    python -m safer.eval.paper external --bench-out runs/extbench --out runs/extbench/paper \\
        [--fit-outcomes runs/safer/predictions/outcomes_valid.npz]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from safer import LABEL2ID

REAL = LABEL2ID["real"]
# dispatcher action order (safer.models.dispatcher.ACTIONS), repeated here so this module stays torch-free
ACTIONS = ["early_exit", "spatial", "spectral", "latent", "spatial_spectral", "full_tri_domain"]
TOOL_ACTIONS = ACTIONS[1:]
ACTION_GROUPS = {"early_exit": [], "spatial": ["spatial"], "spectral": ["spectral"], "latent": ["latent"],
                 "spatial_spectral": ["spatial", "spectral"], "full_tri_domain": ["spatial", "spectral", "latent"]}
ALL_TOOLS = "full_tri_domain"


# ================================================================================================
# the common table
# ================================================================================================

@dataclass
class Table:
    """One benchmark: y (1 = fake), P(fake) and latency (ms) per mode, categories for per-generator analyses."""
    ids: np.ndarray
    y: np.ndarray
    category: np.ndarray
    score: Dict[str, np.ndarray]
    latency: Dict[str, np.ndarray]
    tool_ms: Dict[str, np.ndarray] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=int)
        for k in list(self.score):
            self.score[k] = np.asarray(self.score[k], dtype=np.float64)
        for k in list(self.latency):
            self.latency[k] = np.asarray(self.latency[k], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def action_keys(self) -> List[str]:
        """The six dispatcher actions as mode keys: the no-tool exit, then the arbiter alone per tool subset."""
        return ["default"] + [f"arbiter:{a}" for a in TOOL_ACTIONS]

    def subset(self, idx: np.ndarray) -> "Table":
        idx = np.asarray(idx)
        return Table(self.ids[idx], self.y[idx], self.category[idx], {k: v[idx] for k, v in self.score.items()},
                     {k: v[idx] for k, v in self.latency.items()}, {k: v[idx] for k, v in self.tool_ms.items()},
                     self.name)


def _p_fake(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    return 1.0 - probs[..., REAL]


def table_from_outcomes(path: Path, name: str = "") -> Table:
    """In-domain outcome table (safer.pipeline.build_outcomes). Latencies are the pipeline's per-sample
    measurements (amortised over the eval batch), summed the way the paper's Table 8 sums them."""
    with np.load(path, allow_pickle=False) as z:
        o = {k: z[k] for k in z.files}
    y = (o["labels"] != REAL).astype(int)
    t_tools = o["t_tools"]                          # decode, proposal, spatial, spectral, latent (s)
    dec, prop = t_tools[:, 0], t_tools[:, 1]
    group = {"spatial": t_tools[:, 2], "spectral": t_tools[:, 3], "latent": t_tools[:, 4]}
    frontline = dec + o["scanner_latency"] + o["t_state"]
    score = {"scanner": _p_fake(o["scanner_probs"]), "notool": _p_fake(o["notool_probs"]),
             "default": _p_fake(o["action_probs"][:, 0]), "static": _p_fake(o["static_probs"])}
    lat = {"scanner": dec + o["scanner_latency"], "notool": dec + o["t_state"], "default": frontline,
           "static": dec + prop + sum(group.values()) + o["t_static"]}
    tool = {"scanner": np.zeros(len(y)), "notool": np.zeros(len(y)), "default": np.zeros(len(y)),
            "static": prop + sum(group.values())}
    for ai, a in enumerate(ACTIONS):
        if a == "early_exit":
            continue
        tools_s = prop + sum(group[g] for g in ACTION_GROUPS[a])
        score[f"arbiter:{a}"] = _p_fake(o["action_probs"][:, ai])
        score[f"evidence:{a}"] = (score["scanner"] + score[f"arbiter:{a}"]) / 2.0
        lat[f"arbiter:{a}"] = lat[f"evidence:{a}"] = frontline + tools_s + o["t_actions"][:, ai]
        tool[f"arbiter:{a}"] = tool[f"evidence:{a}"] = tools_s
    return Table(o["keys"].astype(str), y, o["methods"].astype(str),
                 score, {k: v * 1000.0 for k, v in lat.items()}, {k: v * 1000.0 for k, v in tool.items()},
                 name or Path(path).stem)


def table_from_per_action(bench_out: Path, dataset: str, include_overlap: bool = False) -> Table:
    """External per-action run (benchmark_external.py --steps per_action) joined with the prepared manifest.
    Generators that overlap the training sources are dropped unless include_overlap."""
    import pandas as pd
    man = pd.read_csv(Path(bench_out) / "manifests" / f"{dataset}.csv", dtype={"video_id": str})
    man = man[man["usable"].astype(bool)]
    if not include_overlap and "overlap" in man:
        man = man[~man["overlap"].astype(bool)]
    recs = {}
    for f in sorted((Path(bench_out) / "per_action").glob(f"{dataset}.*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                recs[r["video_id"]] = r
    man = man[man["video_id"].isin(recs)].reset_index(drop=True)
    if man.empty:
        raise ValueError(f"no per-action records for {dataset} under {bench_out}")
    R = [recs[v] for v in man["video_id"]]
    t = {k: np.array([r["t"][k] for r in R]) for k in R[0]["t"]}
    frontline = t["decode"] + t["scanner"] + t["state"]
    score = {"scanner": _p_fake([r["scanner"] for r in R]), "notool": _p_fake([r["notool"] for r in R]),
             "static": _p_fake([r["static"] for r in R])}
    score["default"] = (score["scanner"] + score["notool"]) / 2.0
    all_tools = t["proposal"] + t["spatial"] + t["spectral"] + t["latent"]
    lat = {"scanner": t["decode"] + t["scanner"], "notool": t["decode"] + t["state"], "default": frontline,
           "static": t["decode"] + all_tools + t["static"]}
    tool = {"scanner": np.zeros(len(R)), "notool": np.zeros(len(R)), "default": np.zeros(len(R)), "static": all_tools}
    for a in TOOL_ACTIONS:
        tools_s = t["proposal"] + sum(t[g] for g in ACTION_GROUPS[a])
        score[f"arbiter:{a}"] = _p_fake([r["actions"][a] for r in R])
        score[f"evidence:{a}"] = (score["scanner"] + score[f"arbiter:{a}"]) / 2.0
        lat[f"arbiter:{a}"] = lat[f"evidence:{a}"] = frontline + tools_s + t[f"arbiter_{a}"]
        tool[f"arbiter:{a}"] = tool[f"evidence:{a}"] = tools_s
    y = (man["label"] != "real").astype(int).to_numpy()
    return Table(man["video_id"].to_numpy(), y, man["category"].astype(str).to_numpy(), score,
                 {k: v * 1000.0 for k, v in lat.items()}, {k: v * 1000.0 for k, v in tool.items()}, dataset)


# ================================================================================================
# basic metrics
# ================================================================================================

def decisions(score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(score) >= threshold).astype(int)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    tpr = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    tnr = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return float((tpr + tnr) / 2.0)


def auc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, score))


def binary_report(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import f1_score
    y, pred = np.asarray(y), np.asarray(pred)
    fake, real = y == 1, y == 0
    return {"n": int(len(y)), "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
            "errors": int((pred != y).sum()),
            "miss_pct": float(100 * ((pred == 0) & fake).sum() / max(fake.sum(), 1)),
            "false_alarm_pct": float(100 * ((pred == 1) & real).sum() / max(real.sum(), 1)),
            "balanced_accuracy": balanced_accuracy(y, pred)}


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, Any]:
    """Exact (binomial) McNemar test on paired decisions: b = A right & B wrong, c = A wrong & B right."""
    from scipy.stats import binomtest
    ra, rb = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    b, c = int((ra & ~rb).sum()), int((~ra & rb).sum())
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": p}


# ================================================================================================
# routing: oracle, low-latency mode, supervised router
# ================================================================================================

def oracle(T: Table, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cheapest correct dispatcher action per video (Section 3.3). Among the six actions (no-tool exit and
    the arbiter alone for each tool subset), take the lowest-latency one whose decision is correct; when none
    is correct, take the no-tool exit. Returns (decisions, latency_ms, chosen action index)."""
    keys = T.action_keys
    correct = np.stack([decisions(T.score[k], threshold) == T.y for k in keys], 1)
    lat = np.stack([T.latency[k] for k in keys], 1)
    masked = np.where(correct, lat, np.inf)
    choice = np.where(correct.any(1), masked.argmin(1), 0)
    rows = np.arange(len(T))
    pred = np.stack([decisions(T.score[k], threshold) for k in keys], 1)[rows, choice]
    return pred, lat[rows, choice], choice


def _margin(p_fake: np.ndarray) -> np.ndarray:
    return np.abs(2.0 * np.asarray(p_fake) - 1.0)


def low_latency(T: Table, tau: float, escalate: str = "default") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low-latency mode. Assumption: the paper does not define the confidence it thresholds. Here the scanner
    decides alone when its margin |P(fake) - P(real)| is at least tau; otherwise the video escalates to
    `escalate` (default: the SAFER default) and pays that mode's latency, which already includes the scanner.
    Returns (decisions, latency_ms, escalated mask)."""
    esc = _margin(T.score["scanner"]) < tau
    score = np.where(esc, T.score[escalate], T.score["scanner"])
    lat = np.where(esc, T.latency[escalate], T.latency["scanner"])
    return decisions(score), lat, esc


def choose_tau(valid: Table, grid: Optional[Sequence[float]] = None, escalate: str = "default") -> float:
    """Assumption: the paper chooses tau on validation without stating the criterion. Here: fewest validation
    errors, ties broken by lower median latency, then by smaller tau."""
    grid = list(grid) if grid is not None else [round(x, 3) for x in np.linspace(0.0, 1.0, 101)]
    best = None
    for tau in grid:
        pred, lat, _ = low_latency(valid, tau, escalate)
        key = (int((pred != valid.y).sum()), float(np.median(lat)), tau)
        best = key if best is None or key < best else best
    return float(best[2])


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def frontline_features(T: Table) -> np.ndarray:
    """What a router may look at before paying for tools: the frontline's two scores and their entropies."""
    s, n = T.score["scanner"], T.score["notool"]
    return np.column_stack([s, n, _entropy(s), _entropy(n)])


class SupervisedRouter:
    """Supervised cost-aware router (Table 3's supervised row and the adaptation-mode router refit).

    Assumption: the paper does not give the model. Here one logistic regression per action predicts whether
    that action's decision (at 0.5) is correct from the frontline features; the router picks
    argmax_a P(correct_a) - lambda * C(a) / max C, with C(a) the median latency of action a on the fit table,
    the same cost normalisation as the GRPO reward (Equation 1)."""

    def __init__(self, lam: float = 0.2):
        self.lam = lam
        self.models: List[Any] = []
        self.cost: Optional[np.ndarray] = None

    def fit(self, T: Table) -> "SupervisedRouter":
        from sklearn.linear_model import LogisticRegression
        X = frontline_features(T)
        self.models = []
        for k in T.action_keys:
            c = (decisions(T.score[k]) == T.y).astype(int)
            self.models.append(float(c[0]) if c.min() == c.max() else
                               LogisticRegression(max_iter=1000).fit(X, c))
        med = np.array([np.median(T.latency[k]) for k in T.action_keys])
        self.cost = med / max(med.max(), 1e-9)
        return self

    def route(self, T: Table) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = frontline_features(T)
        pc = np.column_stack([np.full(len(T), m) if isinstance(m, float) else m.predict_proba(X)[:, 1]
                              for m in self.models])
        choice = (pc - self.lam * self.cost[None, :]).argmax(1)
        keys = T.action_keys
        rows = np.arange(len(T))
        pred = np.stack([decisions(T.score[k]) for k in keys], 1)[rows, choice]
        lat = np.stack([T.latency[k] for k in keys], 1)[rows, choice]
        return pred, lat, choice


# ================================================================================================
# Table 1 / Figure 2 / Table 9 / Table 13
# ================================================================================================

def indomain_table(T: Table, tau: Optional[float] = None) -> List[Dict[str, Any]]:
    """Rows of Table 1 (and the points of Figure 2). tau: low-latency threshold (see choose_tau)."""
    rows = []

    def add(name: str, pred: np.ndarray, lat: np.ndarray, tool: Optional[np.ndarray] = None, **extra):
        rows.append({"mode": name, **binary_report(T.y, pred), "p50_ms": float(np.median(lat)),
                     "tool_ms_mean": float(np.mean(tool)) if tool is not None else 0.0, **extra})

    add("SAFER-Scanner", decisions(T.score["scanner"]), T.latency["scanner"])
    add("Arbiter, no tools", decisions(T.score["notool"]), T.latency["notool"])
    add("SAFER-Static", decisions(T.score["static"]), T.latency["static"], T.tool_ms.get("static"))
    if tau is not None:
        pred, lat, esc = low_latency(T, tau)
        tool = np.where(esc, T.tool_ms.get("default", np.zeros(len(T))), 0.0)
        add(f"SAFER low-latency mode (tau={tau:g})", pred, lat, tool, escalated_pct=float(100 * esc.mean()))
    add("SAFER default (no-tool exit)", decisions(T.score["default"]), T.latency["default"])
    for a in TOOL_ACTIONS:
        add(f"Evidence mode, {a}", decisions(T.score[f"evidence:{a}"]), T.latency[f"evidence:{a}"],
            T.tool_ms.get(f"evidence:{a}"))
    for a in TOOL_ACTIONS:
        add(f"Arbiter with tools alone, {a}", decisions(T.score[f"arbiter:{a}"]), T.latency[f"arbiter:{a}"],
            T.tool_ms.get(f"arbiter:{a}"))
    pred, lat, choice = oracle(T)
    tool = None
    if all(k in T.tool_ms for k in T.action_keys):
        tool = np.stack([T.tool_ms[k] for k in T.action_keys], 1)[np.arange(len(T)), choice]
    add("Oracle (cheapest correct action)", pred, lat, tool)
    return rows


def decision_changes(T: Table, threshold: float = 0.5) -> Dict[str, int]:
    """Table 9: how often routing can change correctness. 'Tools fix exit': the no-tool exit is wrong and some
    tool action is right; 'tools break exit': the exit is right and some tool action is wrong."""
    keys = T.action_keys
    correct = np.stack([decisions(T.score[k], threshold) == T.y for k in keys], 1)
    exit_ok, tools_ok = correct[:, 0], correct[:, 1:]
    return {"videos": int(len(T)), "all_correct": int(correct.all(1).sum()), "all_wrong": int((~correct).all(1).sum()),
            "tools_fix_exit": int((~exit_ok & tools_ok.any(1)).sum()),
            "tools_break_exit": int((exit_ok & (~tools_ok).any(1)).sum())}


def fixed_broken(T: Table, mode: str = f"evidence:{ALL_TOOLS}", base: str = "default",
                 idx: Optional[np.ndarray] = None, threshold: float = 0.5) -> Dict[str, int]:
    """Decisions of `mode` relative to `base`: fixed = base wrong & mode right; broken = base right & mode wrong."""
    sel = np.arange(len(T)) if idx is None else np.asarray(idx)
    y = T.y[sel]
    rb = decisions(T.score[base][sel], threshold) == y
    rm = decisions(T.score[mode][sel], threshold) == y
    return {"fixed": int((~rb & rm).sum()), "broken": int((rb & ~rm).sum()), "n": int(len(sel))}


def least_confident_forcing(T: Table, mode: str = f"evidence:{ALL_TOOLS}", frac: float = 1 / 3) -> Dict[str, int]:
    """Section 4.3: force tools on the least-confident fraction of the default's predictions."""
    order = np.argsort(_margin(T.score["default"]))
    return fixed_broken(T, mode, "default", order[: int(round(frac * len(T)))])


# ================================================================================================
# Tables 10 / 16, Figure 3, Appendix E: the same-reasoner value of tools
# ================================================================================================

def _category_auc_pair(y_real_a, y_real_b, fake_a, fake_b) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    yy = np.r_[np.zeros(len(y_real_a)), np.ones(len(fake_a))]
    return (float(roc_auc_score(yy, np.r_[y_real_a, fake_a])), float(roc_auc_score(yy, np.r_[y_real_b, fake_b])))


def tool_value(tables: Sequence[Table], with_key: str = f"arbiter:{ALL_TOOLS}", without_key: str = "notool",
               n_boot: int = 1000, seed: int = 0, tost_margin: float = 0.02) -> Dict[str, Any]:
    """Same-reasoner tool value: AUC(arbiter with tools) - AUC(same arbiter, empty graph).

    Per fake category of each benchmark, against all real videos of that benchmark (Table 16). Per-category
    95% intervals resample the real and the category's fake videos (paired across the two scores). The mean
    over categories gets a 95% interval by resampling categories (Figure 3a). TOST: two one-sided t-tests of
    the per-category deltas against +-tost_margin; also reports the smallest margin that passes at 0.05.
    Correlations (Figure 3b): Spearman of the controlled delta against the no-tool AUC, and of the
    uncontrolled contrast (arbiter with tools - scanner) against the scanner AUC.
    Benchmark-level: AUC of every tool action minus the no-tool arbiter (Table 10)."""
    from scipy.stats import spearmanr, t as tdist
    rng = np.random.default_rng(seed)
    cats, benches = [], {}
    for T in tables:
        real = np.where(T.y == 0)[0]
        benches[T.name] = {a: (auc(T.y, T.score[f"arbiter:{a}"]) or float("nan")) - (auc(T.y, T.score["notool"]) or float("nan"))
                           for a in TOOL_ACTIONS}
        for cat in sorted(set(T.category[T.y == 1])):
            fake = np.where((T.y == 1) & (T.category == cat))[0]
            a_with, a_without = _category_auc_pair(T.score[with_key][real], T.score[without_key][real],
                                                   T.score[with_key][fake], T.score[without_key][fake])
            a_scan, _ = _category_auc_pair(T.score["scanner"][real], T.score["scanner"][real],
                                           T.score["scanner"][fake], T.score["scanner"][fake])
            boots = []
            for _ in range(n_boot):
                r, f = rng.choice(real, len(real)), rng.choice(fake, len(fake))
                w, wo = _category_auc_pair(T.score[with_key][r], T.score[without_key][r],
                                           T.score[with_key][f], T.score[without_key][f])
                boots.append(w - wo)
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
            cats.append({"benchmark": T.name, "category": cat, "n": int(len(fake)), "scanner_auc": a_scan,
                         "notool_auc": a_without, "tools_auc": a_with, "delta": a_with - a_without,
                         "ci95": [float(lo), float(hi)],
                         "verdict": "gain" if lo > 0 else ("loss" if hi < 0 else "no detectable change")})
    d = np.array([c["delta"] for c in cats])
    out: Dict[str, Any] = {"categories": cats, "benchmark_level_delta_auc": benches, "n_categories": int(len(d))}
    if len(d):
        means = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
        out["mean_delta"] = float(d.mean())
        out["mean_delta_ci95"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        out["counts"] = {v: sum(c["verdict"] == v for c in cats) for v in ("gain", "loss", "no detectable change")}
    if len(d) > 2:
        se = d.std(ddof=1) / math.sqrt(len(d))

        def tost_p(m: float) -> float:
            return float(max(tdist.sf((d.mean() + m) / se, len(d) - 1), tdist.cdf((d.mean() - m) / se, len(d) - 1)))

        out["tost"] = {"margin": tost_margin, "p_value": tost_p(tost_margin),
                       "smallest_margin_passing_0.05": float(next((m for m in np.arange(0.001, 0.2, 0.001)
                                                                  if tost_p(m) < 0.05), float("nan")))}
        rho_c = spearmanr([c["notool_auc"] for c in cats], d)
        rho_u = spearmanr([c["scanner_auc"] for c in cats], [c["tools_auc"] - c["scanner_auc"] for c in cats])
        out["spearman_controlled_vs_notool_auc"] = {"rho": float(rho_c.statistic), "p_value": float(rho_c.pvalue)}
        out["spearman_uncontrolled_vs_scanner_auc"] = {"rho": float(rho_u.statistic), "p_value": float(rho_u.pvalue)}
    return out


# ================================================================================================
# Tables 4 / 12, Appendix D: adaptation mode
# ================================================================================================

def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising balanced accuracy (Youden's J) on the given sample. Assumption: the paper says the
    adaptation mode 'computes a new threshold' on the labelled half without naming the criterion."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, score)
    i = int(np.argmax(tpr - fpr))
    return float(min(thr[i], 1.0))


def generator_disjoint_split(T: Table, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Fit/test halves: fake categories are split into two disjoint halves (so test generators stay unseen);
    real videos are split uniformly at random."""
    cats = np.array(sorted(set(T.category[T.y == 1])))
    perm = rng.permutation(cats)
    fit_cats = set(perm[: len(perm) // 2]) if len(perm) > 1 else set(perm)
    real = np.where(T.y == 0)[0]
    rp = rng.permutation(real)
    fit_real, test_real = rp[: len(rp) // 2], rp[len(rp) // 2:]
    fake_fit = np.where((T.y == 1) & np.isin(T.category, list(fit_cats)))[0]
    fake_test = np.where((T.y == 1) & ~np.isin(T.category, list(fit_cats)))[0]
    return np.r_[fit_real, fake_fit], np.r_[test_real, fake_test]


def adaptation(T: Table, n_splits: int = 10, seed: int = 0, lam: float = 0.2,
               zero_shot_router: Optional[SupervisedRouter] = None) -> Dict[str, Any]:
    """Balanced accuracy at 0.5 relative to the SAFER default (Table 4) and absolute values with AUC (Table 12).

    Zero-shot rows (whole benchmark): static, evidence mode (all tools), routing oracle and, if given, a router
    fitted in-domain. Adaptation rows (mean +- std over n_splits generator-disjoint splits, evaluated on the
    test half, relative to the default on that same half):
        recalibration   the default's threshold moved to the Youden point of the fit half
        real_only       threshold at 95% specificity on the fit half's real videos (no labelled fakes)
        router_refit    SupervisedRouter(lam) fitted on the fit half
    plus the share of the test-half routing oracle's gain that the refitted router recovers."""
    base_ba = balanced_accuracy(T.y, decisions(T.score["default"]))
    out: Dict[str, Any] = {"benchmark": T.name, "n": int(len(T)),
                           "default": {"balanced_accuracy": base_ba, "auc": auc(T.y, T.score["default"])}}
    for key, name in (("scanner", "scanner"), ("static", "static"), (f"evidence:{ALL_TOOLS}", "evidence_all_tools")):
        ba = balanced_accuracy(T.y, decisions(T.score[key]))
        out[name] = {"balanced_accuracy": ba, "delta": ba - base_ba, "auc": auc(T.y, T.score[key])}
    opred, _, _ = oracle(T)
    out["routing_oracle"] = {"balanced_accuracy": balanced_accuracy(T.y, opred),
                             "delta": balanced_accuracy(T.y, opred) - base_ba}
    if zero_shot_router is not None:
        rpred, _, choice = zero_shot_router.route(T)
        out["router_zero_shot"] = {"balanced_accuracy": balanced_accuracy(T.y, rpred),
                                   "delta": balanced_accuracy(T.y, rpred) - base_ba,
                                   "tool_share": float((choice > 0).mean())}
    rng = np.random.default_rng(seed)
    runs: Dict[str, List[float]] = {"recalibration": [], "real_only": [], "router_refit": [], "oracle_share": []}
    for _ in range(n_splits):
        fit_idx, test_idx = generator_disjoint_split(T, rng)
        F, E = T.subset(fit_idx), T.subset(test_idx)
        if F.y.min() == F.y.max() or E.y.min() == E.y.max():
            continue
        base = balanced_accuracy(E.y, decisions(E.score["default"]))
        thr = youden_threshold(F.y, F.score["default"])
        runs["recalibration"].append(balanced_accuracy(E.y, decisions(E.score["default"], thr)) - base)
        thr_real = float(np.quantile(F.score["default"][F.y == 0], 0.95))
        runs["real_only"].append(balanced_accuracy(E.y, decisions(E.score["default"], thr_real)) - base)
        rp, _, _ = SupervisedRouter(lam).fit(F).route(E)
        gain = balanced_accuracy(E.y, rp) - base
        runs["router_refit"].append(gain)
        og = balanced_accuracy(E.y, oracle(E)[0]) - base
        if og > 0:
            runs["oracle_share"].append(gain / og)
    for k, v in runs.items():
        out[k] = {"mean": float(np.mean(v)) if v else None, "std": float(np.std(v)) if v else None, "splits": len(v)}
    return out


# ================================================================================================
# CLI
# ================================================================================================

def _fmt(v: Any, pct: bool = False, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{100 * v:.1f}" if pct else f"{v:.{digits}f}"


def _write(out: Path, name: str, data: Any, md: List[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    (out / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


NOTICE = ("> Reimplementation of the paper's analyses from its text; see safer/eval/paper.py for every definition "
          "that is an assumption. Compare with the paper before citing.")


def cli_indomain(args) -> None:
    d = Path(args.outcomes_dir)
    test = table_from_outcomes(d / "outcomes_test.npz", "test")
    tau = args.tau
    if tau is None and (d / "outcomes_valid.npz").exists():
        tau = choose_tau(table_from_outcomes(d / "outcomes_valid.npz", "valid"))
    rows = indomain_table(test, tau)
    mc = {"default_vs_notool": mcnemar(test.y, decisions(test.score["default"]), decisions(test.score["notool"])),
          "static_vs_notool": mcnemar(test.y, decisions(test.score["static"]), decisions(test.score["notool"])),
          "arbiter_all_tools_vs_notool": mcnemar(test.y, decisions(test.score[f"arbiter:{ALL_TOOLS}"]),
                                                  decisions(test.score["notool"])),
          "evidence_all_tools_vs_default": mcnemar(test.y, decisions(test.score[f"evidence:{ALL_TOOLS}"]),
                                                   decisions(test.score["default"]))}
    changes = decision_changes(test)
    fb = fixed_broken(test)
    md = ["# In-domain analyses (test split)", "", NOTICE, "",
          f"Low-latency tau: {_fmt(tau)} ({'given' if args.tau is not None else 'chosen on validation'})", "",
          "| Mode | Macro-F1 | Errors | Miss % | FA % | Tool ms (mean) | p50 ms |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['mode']} | {r['macro_f1'] * 100:.2f} | {r['errors']} | {r['miss_pct']:.2f} | {r['false_alarm_pct']:.2f} | "
           f"{r['tool_ms_mean']:.1f} | {r['p50_ms']:.1f} |" for r in rows]
    md += ["", "McNemar (exact): " + "; ".join(f"{k}: {v['a_right_b_wrong']}/{v['a_wrong_b_right']}, p={v['p_value']:.3g}"
                                              for k, v in mc.items()),
           "", f"Decision changes (Table 9): {changes}", "",
           f"Evidence mode (all tools) vs default: fixed {fb['fixed']}, broken {fb['broken']}"]
    _write(Path(args.out), "indomain", {"tau": tau, "rows": rows, "mcnemar": mc, "decision_changes": changes,
                                        "evidence_vs_default": fb}, md)


def cli_external(args) -> None:
    bench = Path(args.bench_out)
    names = args.datasets or sorted({p.name.split(".")[0] for p in (bench / "per_action").glob("*.jsonl")})
    tables = [table_from_per_action(bench, n, args.include_overlap) for n in names]
    router = None
    if args.fit_outcomes:
        router = SupervisedRouter(args.lam).fit(table_from_outcomes(Path(args.fit_outcomes), "fit"))
    tv = tool_value(tables, n_boot=args.bootstrap, seed=args.seed)
    ad = [adaptation(T, args.splits, args.seed, args.lam, router) for T in tables]
    lcf = {T.name: least_confident_forcing(T) for T in tables}
    md = ["# External analyses", "", NOTICE, "",
          f"## Same-reasoner tool value (arbiter with all tools - arbiter without), {tv['n_categories']} categories", "",
          f"Mean ΔAUC {_fmt(tv.get('mean_delta'))} (95% CI {_fmt(tv.get('mean_delta_ci95', [None])[0])} to "
          f"{_fmt(tv.get('mean_delta_ci95', [None, None])[1])}); verdicts {tv.get('counts')}; TOST {tv.get('tost')}", "",
          "| Benchmark | Category | n | Scanner | No-tool | All tools | Δ | 95% CI |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {c['benchmark']} | {c['category']} | {c['n']} | {c['scanner_auc']:.3f} | {c['notool_auc']:.3f} | "
           f"{c['tools_auc']:.3f} | {c['delta']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |"
           for c in sorted(tv["categories"], key=lambda c: -c["delta"])]
    md += ["", "## Balanced accuracy relative to the SAFER default (points)", "",
           "| Benchmark | Default BA % | Static | Evidence (all) | Routing oracle | Router zero-shot | Router refit | "
           "Recalibration | Real-only recal. |", "|---|---|---|---|---|---|---|---|---|"]
    for a in ad:
        def ms(k: str) -> str:
            v = a.get(k) or {}
            return "-" if v.get("mean") is None else f"{100 * v['mean']:+.1f} ±{100 * v['std']:.1f}"

        def pt(k: str) -> str:
            return "-" if k not in a else f"{100 * a[k]['delta']:+.1f}"

        md.append(f"| {a['benchmark']} | {100 * a['default']['balanced_accuracy']:.1f} | {pt('static')} | "
                  f"{pt('evidence_all_tools')} | {pt('routing_oracle')} | {pt('router_zero_shot')} | "
                  f"{ms('router_refit')} | {ms('recalibration')} | {ms('real_only')} |")
    md += ["", f"Forcing all tools on the least-confident third (fixed / broken): {lcf}"]
    _write(Path(args.out), "external", {"tool_value": tv, "adaptation": ad, "least_confident_forcing": lcf}, md)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("indomain", help="Table 1 / Fig. 2 / Table 9 / McNemar from in-domain outcome tables")
    a.add_argument("--outcomes-dir", required=True, help="<work_dir>/predictions")
    a.add_argument("--tau", type=float, default=None, help="low-latency threshold (default: chosen on validation)")
    a.add_argument("--out", required=True)
    b = sub.add_parser("external", help="tool value, adaptation mode, routing on external per-action runs")
    b.add_argument("--bench-out", required=True, help="the --out of benchmark_external.py")
    b.add_argument("--datasets", nargs="*")
    b.add_argument("--fit-outcomes", help="in-domain outcomes_valid.npz for the zero-shot supervised router")
    b.add_argument("--include-overlap", action="store_true")
    b.add_argument("--splits", type=int, default=10)
    b.add_argument("--lam", type=float, default=0.2)
    b.add_argument("--bootstrap", type=int, default=1000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"indomain": cli_indomain, "external": cli_external}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
