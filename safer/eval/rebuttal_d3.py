"""Rebuttal (rZyQ W1, t7Rq Q3): D3 against SAFER on the same normalized external videos, and D3 as an
extra input to the fusion upper bound (CPU only).

Input : --dir    rerun outcome tables <benchmark>.npz (with features_raw; overlap dropped)
        --d3     per-benchmark score CSVs from safer.eval.d3_scores ({dataset} placeholder)
Output: <out>/rebuttal_d3.md and .json

  zero-shot  AUC (bootstrap 95% CI) and TPR at 5% FPR of D3 and of the SAFER default and scanner on the
             same videos; Spearman correlation of D3 with the default's score.
  fusion     the same 10 generator-disjoint splits and learners as rebuttal_fusion: VLM scores,
             scores + D3, scores + 23 features, scores + 23 features + D3. Gains are paired per split.

  python -m safer.eval.rebuttal_d3 --dir ext/outcomes --d3 'd3/{dataset}.csv' --out runs/rebuttal_d3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from safer.eval.paper import auc
from safer.eval.rebuttal import _bootstrap_auc
from safer.eval.rebuttal_external import BENCHES, _ms, splits
from safer.eval.rebuttal_fusion import _fit_eval, load_with_features, score_matrix


def _tpr_at(y: np.ndarray, s: np.ndarray, fpr: float = 0.05) -> float:
    from sklearn.metrics import roc_curve
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr, f, t))


def load_d3(pattern: str, name: str, T) -> np.ndarray:
    df = pd.read_csv(pattern.format(dataset=name), dtype={"video_id": str})
    m = dict(zip(df["video_id"], pd.to_numeric(df["score"], errors="coerce")))
    return np.array([m.get(k.split("/", 1)[1], np.nan) for k in T.ids], dtype=np.float64)


def analyse_one(T, X: np.ndarray, d3: np.ndarray) -> Dict[str, Any]:
    from scipy.stats import spearmanr
    ok = ~np.isnan(d3)
    y = T.y
    out: Dict[str, Any] = {"n": int(len(y)), "n_d3": int(ok.sum()), "n_real": int((y == 0).sum()),
                           "n_fake": int((y == 1).sum())}
    zs = {}
    for name, s in (("d3", d3), ("default", T.score["default"]), ("scanner", T.score["scanner"])):
        a, lo, hi = auc(y[ok], s[ok]), *_bootstrap_auc(y[ok], s[ok])
        zs[name] = {"auc": a, "ci": [lo, hi], "tpr_at_5fpr": _tpr_at(y[ok], s[ok])}
    zs["d3"]["auc_flipped_sign"] = 1 - zs["d3"]["auc"]
    out["zero_shot"] = zs
    out["spearman_d3_default"] = float(spearmanr(d3[ok], T.score["default"][ok]).statistic)

    S = score_matrix(T)
    d = d3[:, None]                                         # NaN for failed videos; GBDT handles it, LR imputes
    sets = {"scores": S, "scores+d3": np.c_[S, d], "scores+features": np.c_[S, X],
            "scores+features+d3": np.c_[S, X, d], "d3": d}
    res = {k: {s: {"auc": [], "ba": []} for s in sets} for k in ("gbdt", "logreg")}
    for f, e, F, E in splits(T):
        for kind in res:
            for s, M in sets.items():
                r = _fit_eval(kind, M[f], F.y, M[e], E.y)
                res[kind][s]["auc"].append(r["auc"])
                res[kind][s]["ba"].append(r["ba"])
    fus: Dict[str, Any] = {}
    for kind, r in res.items():
        blk: Dict[str, Any] = {s: {m: _ms(v[m]) for m in ("auc", "ba")} for s, v in r.items()}
        for a, b in (("scores+d3", "scores"), ("scores+features+d3", "scores+features")):
            g = np.array(r[a]["auc"]) - np.array(r[b]["auc"])
            blk[f"gain_auc[{a} - {b}]"] = {**_ms(list(g)), "splits_positive": int((g > 0).sum())}
        fus[kind] = blk
    out["fusion"] = fus
    return out


def report(res: Dict[str, Any]) -> str:
    f3 = lambda v: "-" if v is None else f"{v:.3f}"
    L = ["# D3 (training-free) vs. SAFER on the normalized external benchmarks", "",
         "## Zero-shot (no target labels)", "",
         "| Benchmark | n (D3 scored) | D3 AUC | D3 TPR@5%FPR | Default AUC | Default TPR@5%FPR | Scanner AUC | Spearman(D3, default) |",
         "|---|---|---|---|---|---|---|---|"]
    for b, r in res.items():
        z = r["zero_shot"]
        c = lambda k: f"{z[k]['auc']:.3f} ({z[k]['ci'][0]:.2f}-{z[k]['ci'][1]:.2f})"
        L.append(f"| {b} | {r['n_d3']}/{r['n']} | {c('d3')} | {100 * z['d3']['tpr_at_5fpr']:.1f} | {c('default')} | "
                 f"{100 * z['default']['tpr_at_5fpr']:.1f} | {c('scanner')} | {r['spearman_d3_default']:+.2f} |")
    L += ["", "D3 sign as in the official evaluation (lower second-order std = generated). "
          "`auc_flipped_sign` in the json gives 1 - AUC.", "",
          "## Fusion fit on target labels (10 generator-disjoint splits, mean AUC; BA at the fit-half threshold)", "",
          "| Benchmark | Learner | D3 alone | Scores | Scores + D3 | Gain | Scores + features | Scores + features + D3 | Gain |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b, r in res.items():
        for kind, k in r["fusion"].items():
            g1 = k["gain_auc[scores+d3 - scores]"]
            g2 = k["gain_auc[scores+features+d3 - scores+features]"]
            L.append(f"| {b} | {kind} | {f3(k['d3']['auc']['mean'])} | {f3(k['scores']['auc']['mean'])} | "
                     f"{f3(k['scores+d3']['auc']['mean'])} | {g1['mean']:+.3f} ± {g1['std']:.3f} ({g1['splits_positive']}/{g1['n']}) | "
                     f"{f3(k['scores+features']['auc']['mean'])} | {f3(k['scores+features+d3']['auc']['mean'])} | "
                     f"{g2['mean']:+.3f} ± {g2['std']:.3f} ({g2['splits_positive']}/{g2['n']}) |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--d3", required=True, help="CSV path pattern with {dataset}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for b in BENCHES:
        if not (Path(a.dir) / f"{b}.npz").exists():
            continue
        T, X = load_with_features(Path(a.dir), b)
        res[b] = analyse_one(T, X, load_d3(a.d3, b, T))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rebuttal_d3.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    md = report(res)
    (out / "rebuttal_d3.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
