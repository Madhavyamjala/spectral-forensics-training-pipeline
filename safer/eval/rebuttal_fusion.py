"""
Fusion upper bound and per-feature direction on the external benchmarks (CPU only).

Input : <dir>/<benchmark>.npz from benchmark_external --steps csf_outcomes,outcomes_merge with the 23 tool
        features saved (features_raw, N x 23). Overlapping generators are dropped, as in the paper.
Output: <out>/rebuttal_fusion.md and .json

    python -m safer.eval.rebuttal_fusion --dir ext/outcomes --out runs/rebuttal [--old runs/extbench_outcomes]

T1.1  fusion upper bound (rZyQ W1, t7Rq Q3). A classifier fit on the target's own labels, on the fit half of
      the paper's generator-disjoint split (10 splits, seed 0; test generators unseen), evaluated on the other
      half. Inputs: (a) the VLM scores (scanner, no-tool arbiter, all-tools arbiter, default; logits),
      (b) the 23 features, (c) both. Gain = AUC(c) - AUC(a), paired per split. Two learners: gradient-boosted
      trees (sklearn HistGradientBoosting, NaN-aware) and logistic regression (median-imputed, standardised).
      Reference rows: the default's AUC with no fitting, and its BA after threshold re-selection on the fit half.
T6.2  per-feature direction (4jrv Q3). AUC of each raw feature for fake vs. real on each benchmark, with a
      bootstrap 95% CI; a feature keeps its direction when every benchmark whose CI excludes 0.5 agrees.
repro the new tables against the original run's tables (same video_ids where they exist): AUC and BA of the
      default, scanner and all-tools evidence mode, and the max |diff| of p_fake on matched videos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from safer.eval.paper import ALL_TOOLS, auc, balanced_accuracy, decisions, youden_threshold
from safer.eval.rebuttal import _bootstrap_auc
from safer.eval.rebuttal_external import BENCHES, _ms, load, splits

# Same order as safer.tools.toolpool.FEATURE_NAMES (copied so this CPU script does not need cv2).
TOOL_GROUPS = ["spatial", "spectral", "latent"]
FEATURE_NAMES: Dict[str, List[str]] = {
    "spatial": ["sat_mean", "sat_std", "sat_entropy", "lum_jump_std", "lum_jump_max",
                "edge_lapvar_global", "edge_lapvar_patch_ratio", "flow_inconsistency",
                "patch_context_color_gap"],
    "spectral": ["fft_hf_global", "spectrum_slope", "fft_hf_patch_mean", "fft_hf_patch_std",
                 "dct3d_hf_global", "dct3d_hf_patch_mean", "phase_corr_global",
                 "phase_shift_spike", "noise_residual_cv"],
    "latent": ["dire_global", "dire_patch_mean", "dire_patch_min", "dire_patch_std",
               "dire_patch_global_ratio"],
}
ALL_FEATURES: List[str] = [f for g in TOOL_GROUPS for f in FEATURE_NAMES[g]]

GROUP_OF = {f: g for g in TOOL_GROUPS for f in FEATURE_NAMES[g]}
SCORE_KEYS = ["scanner", "notool", f"arbiter:{ALL_TOOLS}", "default"]


def load_with_features(d: Path, name: str):
    T, _ = load(d, name)
    with np.load(d / f"{name}.npz", allow_pickle=False) as z:
        if "features_raw" not in z.files:
            raise SystemExit(f"{name}: no features_raw in {d / (name + '.npz')} (needs the rerun with features)")
        keep = ~z["overlap"].astype(bool) if "overlap" in z.files else np.ones(len(z["labels"]), bool)
        X = z["features_raw"].astype(np.float64)[keep]
    assert len(X) == len(T), (name, len(X), len(T))
    return T, X


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def score_matrix(T) -> np.ndarray:
    return np.stack([_logit(T.score[k]) for k in SCORE_KEYS], 1)


def _learner(kind: str):
    if kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=20, l2_regularization=1.0, random_state=0)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))


def _fit_eval(kind: str, Xf, yf, Xe, ye) -> Dict[str, float]:
    m = _learner(kind)
    w = None
    if kind == "gbdt":                                     # balance classes the way the LR does
        w = np.where(yf == 1, 0.5 / max(yf.mean(), 1e-9), 0.5 / max(1 - yf.mean(), 1e-9))
        m.fit(Xf, yf, sample_weight=w)
    else:
        m.fit(Xf, yf)
    pf, pe = m.predict_proba(Xf)[:, 1], m.predict_proba(Xe)[:, 1]
    thr = youden_threshold(yf, pf)                         # operating point chosen on the fit half only
    return {"auc": auc(ye, pe), "ba": balanced_accuracy(ye, decisions(pe, thr))}


def fusion_bound(T, X: np.ndarray) -> Dict[str, Any]:
    S = score_matrix(T)
    sets = {"scores": S, "features": X, "scores+features": np.c_[S, X]}
    out: Dict[str, Any] = {"n": int(len(T)), "n_real": int((T.y == 0).sum()), "n_fake": int((T.y == 1).sum()),
                           "feature_nan_share": float(np.isnan(X).mean())}
    ref_auc, ref_ba = [], []
    res = {k: {s: {"auc": [], "ba": []} for s in sets} for k in ("gbdt", "logreg")}
    for f, e, F, E in splits(T):
        ref_auc.append(auc(E.y, E.score["default"]))
        ref_ba.append(balanced_accuracy(E.y, decisions(E.score["default"], youden_threshold(F.y, F.score["default"]))))
        for kind in res:
            for s, M in sets.items():
                r = _fit_eval(kind, M[f], F.y, M[e], E.y)
                res[kind][s]["auc"].append(r["auc"])
                res[kind][s]["ba"].append(r["ba"])
    out["default_no_fit"] = {"auc": _ms(ref_auc), "ba_recalibrated": _ms(ref_ba)}
    for kind, r in res.items():
        block: Dict[str, Any] = {s: {"auc": _ms(v["auc"]), "ba": _ms(v["ba"])} for s, v in r.items()}
        for metric in ("auc", "ba"):
            g = np.array(r["scores+features"][metric]) - np.array(r["scores"][metric])
            block[f"gain_{metric}"] = {**_ms(list(g)), "min": float(g.min()), "max": float(g.max()),
                                       "splits_positive": int((g > 0).sum())}
        g = np.array(r["scores+features"]["auc"]) - np.array(ref_auc)
        block["gain_auc_vs_default"] = _ms(list(g))
        out[kind] = block
    return out


def _rank_auc(y: np.ndarray, r: np.ndarray) -> float:
    """AUC from precomputed (tie-averaged) ranks of the full sample, rows given by a bootstrap index."""
    from scipy.stats import rankdata
    rr = rankdata(r)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((rr[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fast_boot_auc(y: np.ndarray, x: np.ndarray, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yi = y[i]
        if yi.min() != yi.max():
            vals.append(_rank_auc(yi, x[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def directions(tables: Dict[str, Any]) -> Dict[str, Any]:
    per: Dict[str, Dict[str, Any]] = {f: {} for f in ALL_FEATURES}
    for name, (T, X) in tables.items():
        for j, f in enumerate(ALL_FEATURES):
            ok = ~np.isnan(X[:, j])
            y, x = T.y[ok], X[ok, j]
            if ok.sum() < 20 or y.min() == y.max():
                per[f][name] = None
                continue
            a = auc(y, x)
            lo, hi = _fast_boot_auc(y, x, n=1000, seed=j)
            sig = +1 if lo > 0.5 else (-1 if hi < 0.5 else 0)
            per[f][name] = {"auc": a, "ci": [lo, hi], "sign": sig}
    summary = {"keep": [], "flip": [], "weak": []}
    for f, d in per.items():
        signs = [v["sign"] for v in d.values() if v and v["sign"] != 0]
        if len(signs) < 2:
            summary["weak"].append(f)
        elif all(s == signs[0] for s in signs):
            summary["keep"].append(f)
        else:
            summary["flip"].append(f)
    fam = {k: sorted({GROUP_OF[f] for f in v}) for k, v in summary.items()}
    return {"per_feature": per, "summary": summary, "families": fam,
            "counts": {k: len(v) for k, v in summary.items()}}


def reproduction(new_dir: Path, old_dir: Path, name: str) -> Optional[Dict[str, Any]]:
    if not (old_dir / f"{name}.npz").exists():
        return None
    Tn, _ = load(new_dir, name)
    To, _ = load(old_dir, name)
    out: Dict[str, Any] = {"n_new": int(len(Tn)), "n_old": int(len(To))}
    for k in ("default", "scanner", f"evidence:{ALL_TOOLS}"):
        out[k] = {"auc_new": auc(Tn.y, Tn.score[k]), "auc_old": auc(To.y, To.score[k]),
                  "ba_new": balanced_accuracy(Tn.y, decisions(Tn.score[k])),
                  "ba_old": balanced_accuracy(To.y, decisions(To.score[k]))}
    pos = {v: i for i, v in enumerate(To.ids)}
    m = [(i, pos[v]) for i, v in enumerate(Tn.ids) if v in pos]
    out["matched_ids"] = len(m)
    if m:
        a, b = np.array(m).T
        for k in ("default", "scanner"):
            d = np.abs(Tn.score[k][a] - To.score[k][b])
            out[k]["max_abs_diff_matched"] = float(d.max())
            out[k]["decision_agreement_matched"] = float((decisions(Tn.score[k][a]) == decisions(To.score[k][b])).mean())
    return out


def _f(v, p=3):
    return "-" if v is None else f"{v:.{p}f}"


def report(res: Dict[str, Any]) -> str:
    L = ["# Fusion upper bound and feature direction (external benchmarks, rerun with features)", ""]
    L += ["## T1.1 Fusion upper bound (fit half -> unseen-generator half, 10 splits)", "",
          "| Benchmark | n (real/fake) | Default AUC (no fit) | Learner | Scores AUC | Features AUC | Scores+features AUC | Gain AUC (splits > 0) | Gain AUC vs. default | Gain BA |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in res["fusion"].items():
        for kind in ("gbdt", "logreg"):
            k = r[kind]
            L.append(f"| {b} | {r['n']} ({r['n_real']}/{r['n_fake']}) | {_f(r['default_no_fit']['auc']['mean'])} | {kind} | "
                     f"{_f(k['scores']['auc']['mean'])} | {_f(k['features']['auc']['mean'])} | "
                     f"{_f(k['scores+features']['auc']['mean'])} | {k['gain_auc']['mean']:+.3f} ± {k['gain_auc']['std']:.3f} "
                     f"({k['gain_auc']['splits_positive']}/{k['gain_auc']['n']}) | {k['gain_auc_vs_default']['mean']:+.3f} | "
                     f"{100 * k['gain_ba']['mean']:+.1f} |")
    d = res["direction"]
    L += ["", "## T6.2 Per-feature direction (AUC fake vs. real, bootstrap 95% CI)", "",
          f"Keep direction: {d['counts']['keep']} ({', '.join(d['families']['keep'])}); "
          f"flip: {d['counts']['flip']} ({', '.join(d['families']['flip']) or '-'}); "
          f"not significant on 2+ benchmarks: {d['counts']['weak']}", "",
          "| Feature | Group | " + " | ".join(res["benches"]) + " | Verdict |", "|---|---|" + "---|" * len(res["benches"]) + "---|"]
    for f, per in d["per_feature"].items():
        cells = []
        for b in res["benches"]:
            v = per.get(b)
            cells.append("-" if not v else f"{v['auc']:.2f}{'*' if v['sign'] else ''}")
        verdict = next(k for k, v in d["summary"].items() if f in v)
        L.append(f"| {f} | {GROUP_OF[f]} | " + " | ".join(cells) + f" | {verdict} |")
    if res.get("repro"):
        L += ["", "## Reproduction vs. the original tables", "",
              "| Benchmark | n new/old | Mode | AUC new/old | BA new/old | Matched ids | max abs diff | Decision agreement |",
              "|---|---|---|---|---|---|---|---|"]
        for b, r in res["repro"].items():
            if not r:
                continue
            for k in ("default", "scanner", f"evidence:{ALL_TOOLS}"):
                v = r[k]
                L.append(f"| {b} | {r['n_new']}/{r['n_old']} | {k} | {_f(v['auc_new'])}/{_f(v['auc_old'])} | "
                         f"{_f(100 * v['ba_new'], 1)}/{_f(100 * v['ba_old'], 1)} | {r['matched_ids']} | "
                         f"{_f(v.get('max_abs_diff_matched'))} | {_f(v.get('decision_agreement_matched'))} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old", default=None, help="original outcome tables, for the reproduction check")
    ap.add_argument("--benches", default=",".join(BENCHES))
    a = ap.parse_args()
    d, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    benches = [b for b in a.benches.split(",") if (d / f"{b}.npz").exists()]
    tables = {b: load_with_features(d, b) for b in benches}
    res: Dict[str, Any] = {"benches": benches, "fusion": {}, "repro": {}}
    for b, (T, X) in tables.items():
        print(f"[fusion] {b} n={len(T)}", flush=True)
        res["fusion"][b] = fusion_bound(T, X)
    res["direction"] = directions(tables)
    (out / "rebuttal_fusion.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    if a.old:
        for b in benches:
            try:
                res["repro"][b] = reproduction(d, Path(a.old), b)
            except Exception as e:                       # the bound and directions are already saved
                print(f"[repro] {b} failed: {type(e).__name__}: {e}", flush=True)
                res["repro"][b] = None
    (out / "rebuttal_fusion.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    md = report(res)
    (out / "rebuttal_fusion.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
