"""
Rebuttal analyses on the external outcome tables (CPU only).

Input : <dir>/<benchmark>.npz in the build_outcomes format (keys, labels, categories, overlap, scanner_probs,
        notool_probs, state, static_probs, action_probs, t_actions, t_tools, ...). Overlapping generators are
        dropped, as in the paper.
Output: <out>/rebuttal_external.md and .json

    python -m safer.eval.rebuttal_external --dir runs/extbench_outcomes --out runs/rebuttal

Analyses (all fit on the fit half of the paper's generator-disjoint split, evaluated on the other half;
10 splits, seed 0, as safer.eval.paper.adaptation):
    sanity     Table 4 re-derived (threshold re-selection, real-only threshold, router refit)
    T3.1       label budget: threshold re-selection and router refit from k labelled fit-half videos
    T3.4       calibration under shift: ECE before/after temperature; BA at 0.5 vs after a threshold shift
    T4.1       recalibrate every action, then the oracle and a refit router on the recalibrated outputs
    gate       margin threshold (tau chosen in-domain on validation, matched all-tools action) vs always skip
    T5.2       can 'some tool action fixes the exit' be predicted from confidence / h0? (generator-disjoint)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from safer.eval.paper import (ALL_TOOLS, TOOL_ACTIONS, SupervisedRouter, Table, _margin, auc, balanced_accuracy,
                              decisions, generator_disjoint_split, mcnemar, oracle, table_from_outcomes,
                              youden_threshold)
from safer.eval.rebuttal import _bootstrap_auc, _logit, _sigmoid, ece, fit_temperature, tool_utility_targets

BENCHES = ["vifbench", "fakeparts", "genbuster", "genvidbench"]
BUDGETS = [10, 25, 50, 100, 250, 500]
GATE_TAU = 0.5          # chosen on the in-domain validation split (runs/rebuttal, matched outputs)


def load(d: Path, name: str) -> tuple[Table, np.ndarray]:
    with np.load(d / f"{name}.npz", allow_pickle=False) as z:
        keep = ~z["overlap"].astype(bool) if "overlap" in z.files else np.ones(len(z["labels"]), bool)
        state = z["state"].astype(np.float32)[keep]
    T = table_from_outcomes(d / f"{name}.npz", name)
    T.category = np.asarray(T.category)
    return T.subset(np.where(keep)[0]), state


def splits(T: Table, n: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        f, e = generator_disjoint_split(T, rng)
        F, E = T.subset(f), T.subset(e)
        if F.y.min() != F.y.max() and E.y.min() != E.y.max():
            yield f, e, F, E


def _ms(v: List[float]) -> Dict[str, Any]:
    return {"mean": float(np.mean(v)) if v else None, "std": float(np.std(v)) if v else None, "n": len(v)}


def budget_sample(F: Table, k: int, rng: np.random.Generator) -> np.ndarray:
    """k labelled fit-half videos, class-balanced where the half allows (at least one per class)."""
    real, fake = np.where(F.y == 0)[0], np.where(F.y == 1)[0]
    kr = int(min(len(real), max(1, k // 2)))
    kf = int(min(len(fake), k - kr))
    return np.r_[rng.choice(real, kr, replace=False), rng.choice(fake, kf, replace=False)]


def label_budget(T: Table, draws: int = 50, router_draws: int = 10) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rng = np.random.default_rng(1)
    for k in BUDGETS + ["full"]:
        rec, rt = [], []
        for _, _, F, E in splits(T):
            base = balanced_accuracy(E.y, decisions(E.score["default"]))
            if k == "full":
                idx_list = [np.arange(len(F))]
            else:
                idx_list = [budget_sample(F, k, rng) for _ in range(draws)]
            for j, idx in enumerate(idx_list):
                S = F.subset(idx)
                if S.y.min() == S.y.max():
                    continue
                thr = youden_threshold(S.y, S.score["default"])
                rec.append(balanced_accuracy(E.y, decisions(E.score["default"], thr)) - base)
                if j < router_draws:
                    rp, _, _ = SupervisedRouter(0.2).fit(S).route(E)
                    rt.append(balanced_accuracy(E.y, rp) - base)
        out[str(k)] = {"recalibration": _ms(rec), "router_refit": _ms(rt)}
    return out


def calibration_shift(T: Table) -> Dict[str, Any]:
    r = {"ece_raw": [], "ece_temp": [], "T": [], "ba_05": [], "ba_temp": [], "ba_thr": []}
    for _, _, F, E in splits(T):
        p_f, p_e = F.score["default"], E.score["default"]
        t = fit_temperature(F.y, p_f)
        pe_t = _sigmoid(_logit(p_e) / t)
        r["T"].append(t)
        r["ece_raw"].append(ece(E.y, p_e))
        r["ece_temp"].append(ece(E.y, pe_t))
        r["ba_05"].append(balanced_accuracy(E.y, decisions(p_e)))
        r["ba_temp"].append(balanced_accuracy(E.y, decisions(pe_t)))
        r["ba_thr"].append(balanced_accuracy(E.y, decisions(p_e, youden_threshold(F.y, p_f))))
    return {k: _ms(v) for k, v in r.items()}


def _shift_scores(T: Table, F: Table) -> Table:
    """Move each action's threshold to its Youden point on F by a monotone shift, so 0.5 decides as thr."""
    S = T.subset(np.arange(len(T)))
    for k in T.action_keys + [f"evidence:{a}" for a in TOOL_ACTIONS]:
        thr = youden_threshold(F.y, F.score[k])
        S.score[k] = np.clip(T.score[k] - thr + 0.5, 0.0, 1.0)
    return S


def recal_then_route(T: Table) -> Dict[str, Any]:
    r = {"recal_default": [], "recal_oracle": [], "recal_router": [], "recal_evidence_all": []}
    for _, _, F, E in splits(T):
        base = balanced_accuracy(E.y, decisions(E.score["default"]))
        Fs, Es = _shift_scores(F, F), _shift_scores(E, F)
        r["recal_default"].append(balanced_accuracy(E.y, decisions(Es.score["default"])) - base)
        r["recal_evidence_all"].append(balanced_accuracy(E.y, decisions(Es.score[f"evidence:{ALL_TOOLS}"])) - base)
        r["recal_oracle"].append(balanced_accuracy(E.y, oracle(Es)[0]) - base)
        rp, _, _ = SupervisedRouter(0.2).fit(Fs).route(Es)
        r["recal_router"].append(balanced_accuracy(E.y, rp) - base)
    return {k: _ms(v) for k, v in r.items()}


def gate(T: Table) -> Dict[str, Any]:
    skip = decisions(T.score["default"])
    esc = _margin(T.score["default"]) < GATE_TAU
    g = np.where(esc, decisions(T.score[f"evidence:{ALL_TOOLS}"]), skip)
    m = mcnemar(T.y, skip, g)
    return {"tau": GATE_TAU, "tool_rate": float(esc.mean()), "errors_skip": int((skip != T.y).sum()),
            "errors_gate": int((g != T.y).sum()), "fixed": m["a_wrong_b_right"], "broken": m["a_right_b_wrong"],
            "p": m["p_value"], "ba_skip": balanced_accuracy(T.y, skip), "ba_gate": balanced_accuracy(T.y, g)}


def predictability(T: Table, state: np.ndarray) -> Dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    s, n, d = T.score["scanner"], T.score["notool"], T.score["default"]
    conf = np.column_stack([s, n, d, _margin(s), _margin(n), _margin(d), np.abs(s - n)])
    out: Dict[str, Any] = {}
    for kind in ("arbiter", "evidence"):
        tgt = tool_utility_targets(T, kind)
        for target in ("tool_fixes_exit", "tool_changes_decision"):
            yv = tgt[target]
            res: Dict[str, Any] = {"positives": int(yv.sum()), "n": int(len(yv))}
            for name, C in (("confidence", 1.0), ("h0 PCA-64 + confidence", 0.05)):
                score = np.full(len(yv), np.nan)
                for f, e, F, _ in splits(T, n=1):          # one generator-disjoint split, both directions
                    for tr, te in ((f, e), (e, f)):
                        if yv[tr].min() == yv[tr].max():
                            continue
                        Xtr, Xte = conf[tr], conf[te]
                        if name.startswith("h0"):
                            pca = PCA(64, random_state=0).fit(state[tr])
                            Xtr = np.hstack([pca.transform(state[tr]), Xtr])
                            Xte = np.hstack([pca.transform(state[te]), Xte])
                        clf = make_pipeline(StandardScaler(),
                                            LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
                        score[te] = clf.fit(Xtr, yv[tr]).predict_proba(Xte)[:, 1]
                ok = ~np.isnan(score)
                if ok.sum() == 0 or yv[ok].min() == yv[ok].max():
                    res[name] = {"auc": None, "ci95": [None, None], "scored": int(ok.sum())}
                    continue
                res[name] = {"auc": auc(yv[ok], score[ok]), "ci95": _bootstrap_auc(yv[ok], score[ok], n=1000),
                             "scored": int(ok.sum())}
            out[f"{target} [{kind}]"] = res
        ch = tgt["tool_changes_decision"].astype(bool)
        d0 = decisions(T.score["default"])
        dec = np.stack([decisions(T.score[f"{kind}:{a}"]) for a in TOOL_ACTIONS], 1)
        fix = ((d0 != T.y)[:, None] & (dec == T.y[:, None])).any(1)
        out[f"changes [{kind}]"] = {"changed": int(ch.sum()), "fix": int((fix & ch).sum()),
                                    "break": int((ch & ~fix & (d0 == T.y)).sum())}
    return out


def report(res: Dict[str, Any]) -> str:
    L = ["# Rebuttal analyses on the external outcome tables", ""]
    pct = lambda m: "-" if m["mean"] is None else f"{100 * m['mean']:+.1f} ± {100 * m['std']:.1f}"
    L += ["## Label budget (balanced-accuracy gain over the default at 0.5, test half; mean ± sd over splits × draws)", "",
          "| Benchmark | Arm | " + " | ".join(f"k={k}" for k in BUDGETS) + " | full half |",
          "|---|---|" + "---|" * (len(BUDGETS) + 1)]
    for b, r in res.items():
        for arm in ("recalibration", "router_refit"):
            L.append(f"| {b} | {arm} | " + " | ".join(pct(r["budget"][str(k)][arm]) for k in BUDGETS + ["full"]) + " |")
    L += ["", "## Calibration under shift (default score; test half)", "",
          "| Benchmark | T | ECE raw | ECE temp | BA@0.5 | BA temp | BA threshold |", "|---|---|---|---|---|---|---|"]
    for b, r in res.items():
        c = r["calibration"]
        f = lambda k, s=100: f"{s * c[k]['mean']:.{1 if s == 100 else 3}f}"
        L.append(f"| {b} | {c['T']['mean']:.2f} | {c['ece_raw']['mean']:.3f} | {c['ece_temp']['mean']:.3f} | "
                 f"{f('ba_05')} | {f('ba_temp')} | {f('ba_thr')} |")
    L += ["", "## Recalibrate every action, then route (BA gain over default at 0.5)", "",
          "| Benchmark | recal default | recal all-tools evidence | recal + router refit | recal + oracle |", "|---|---|---|---|---|"]
    for b, r in res.items():
        x = r["recal_then_route"]
        L.append(f"| {b} | {pct(x['recal_default'])} | {pct(x['recal_evidence_all'])} | {pct(x['recal_router'])} | {pct(x['recal_oracle'])} |")
    L += ["", f"## Margin threshold (tau={GATE_TAU}, chosen in-domain) vs always skip", "",
          "| Benchmark | tool rate | errors skip → gate | fixed / broken | p | BA skip → gate |", "|---|---|---|---|---|---|"]
    for b, r in res.items():
        g = r["gate"]
        L.append(f"| {b} | {100 * g['tool_rate']:.1f}% | {g['errors_skip']} → {g['errors_gate']} | {g['fixed']} / {g['broken']} | "
                 f"{g['p']:.3g} | {100 * g['ba_skip']:.1f} → {100 * g['ba_gate']:.1f} |")
    L += ["", "## Is tool value predictable under shift? (generator-disjoint, both directions)", "",
          "| Benchmark | Target | Positives | Confidence AUC [CI] | h0+confidence AUC [CI] |", "|---|---|---|---|---|"]
    for b, r in res.items():
        for t, v in r["predictability"].items():
            if t.startswith("changes"):
                continue
            a = lambda m: "-" if m["auc"] is None or m["ci95"][0] is None else f"{m['auc']:.3f} [{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}]"
            L.append(f"| {b} | {t} | {v['positives']}/{v['n']} | {a(v['confidence'])} | {a(v['h0 PCA-64 + confidence'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/extbench_outcomes")
    ap.add_argument("--out", default="runs/rebuttal")
    ap.add_argument("--benches", nargs="+", default=BENCHES)
    a = ap.parse_args()
    res: Dict[str, Any] = {}
    for b in a.benches:
        T, state = load(Path(a.dir), b)
        print(f"[{b}] n={len(T)} fake={int(T.y.sum())}", flush=True)
        res[b] = {"n": len(T), "budget": label_budget(T), "calibration": calibration_shift(T),
                  "recal_then_route": recal_then_route(T), "gate": gate(T), "predictability": predictability(T, state)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "rebuttal_external.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    (out / "rebuttal_external.md").write_text(report(res), encoding="utf-8")
    print(f"wrote {out / 'rebuttal_external.md'}")


if __name__ == "__main__":
    main()
