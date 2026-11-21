"""
In-domain analyses for the ICLR 2027 rebuttal (Submission 46748), computed from the saved outcome tables.

Every analysis fits on the validation outcome table and reports on the test table, so nothing here touches
test labels before evaluation. Test IDs refer to the rebuttal plan:

    T5.3  dispatcher lambda sweep (several seeds), native and matched action outputs, supervised router and
          confidence gate on the same axis, exact McNemar of each against always-skip (the no-tool exit)
    T5.2  can tool utility be predicted from the frontline state? (logistic probes on confidence features and
          on the pooled arbiter state h0)
    T3.4  calibration diagnostics: ECE, Brier, NLL, balanced accuracy, before and after temperature scaling,
          Platt scaling and a threshold shift fit on validation
    T4.2  threshold-free operating points: TPR at 1% and 5% FPR, partial AUC up to 10% FPR

Only T5.3 needs torch (it retrains the GRPO dispatcher, a small MLP, on CPU). Everything else is numpy,
scipy and scikit-learn.

CLI:
    python -m safer.eval.rebuttal all --outcomes-dir runs/full_2class/predictions --out runs/rebuttal
    python -m safer.eval.rebuttal sweep --outcomes-dir ... --out ... --seeds 0 1 2 --iterations 1500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from safer.eval.paper import (ACTIONS, ALL_TOOLS, TOOL_ACTIONS, SupervisedRouter, Table, _margin, auc,
                              balanced_accuracy, decisions, mcnemar, oracle, table_from_outcomes)

SWEEP_LAMBDAS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
CAL_MODES = ["scanner", "notool", "default", f"arbiter:{ALL_TOOLS}", f"evidence:{ALL_TOOLS}"]


def _load(d: Path) -> Tuple[Table, Table, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    raw = {}
    for split in ("valid", "test"):
        with np.load(d / f"outcomes_{split}.npz", allow_pickle=False) as z:
            raw[split] = {k: z[k] for k in z.files}
    return (table_from_outcomes(d / "outcomes_valid.npz", "valid"), table_from_outcomes(d / "outcomes_test.npz", "test"),
            raw["valid"], raw["test"])


def _bootstrap_auc(y: np.ndarray, s: np.ndarray, n: int = 2000, seed: int = 0) -> List[Optional[float]]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a = auc(y[i], s[i])
        if a is not None:
            vals.append(a)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


# ================================================================================================
# T4.2 threshold-free operating points
# ================================================================================================

def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float) -> float:
    """Largest TPR whose FPR does not exceed the target (no interpolation)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def partial_auc(y: np.ndarray, s: np.ndarray, max_fpr: float = 0.1) -> Dict[str, float]:
    """Raw area under the ROC up to max_fpr (out of max_fpr) and sklearn's McClish-standardised value."""
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    x = np.append(fpr[:stop], max_fpr)
    yv = np.append(tpr[:stop], np.interp(max_fpr, fpr, tpr))
    raw = float(np.trapezoid(yv, x) / max_fpr)
    return {"raw": raw, "mcclish": float(roc_auc_score(y, s, max_fpr=max_fpr))}


def operating_points(test: Table) -> List[Dict[str, Any]]:
    modes = ["scanner", "notool", "default", "static"] + [f"evidence:{a}" for a in TOOL_ACTIONS] \
        + [f"arbiter:{a}" for a in TOOL_ACTIONS]
    rows = []
    for m in modes:
        s, y = test.score[m], test.y
        rows.append({"mode": m, "auc": auc(y, s), "tpr_at_1fpr": tpr_at_fpr(y, s, 0.01),
                     "tpr_at_5fpr": tpr_at_fpr(y, s, 0.05), "pauc10": partial_auc(y, s, 0.1)})
    return rows


# ================================================================================================
# T3.4 calibration diagnostics
# ================================================================================================

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the binary P(fake), equal-width bins on [0, 1]."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _cal_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {"ece": ece(y, p), "brier": float(np.mean((p - y) ** 2)),
            "nll": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "balanced_accuracy": balanced_accuracy(y, (p >= thr).astype(int)),
            "errors": int(((p >= thr).astype(int) != y).sum())}


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    z = _logit(p)

    def nll(log_t):
        q = np.clip(_sigmoid(z / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(np.exp(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x))


def fit_platt(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(_logit(p)[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def calibration(valid: Table, test: Table) -> List[Dict[str, Any]]:
    from safer.eval.paper import youden_threshold
    rows = []
    for m in CAL_MODES:
        pv, pt = valid.score[m], test.score[m]
        t = fit_temperature(valid.y, pv)
        a, b = fit_platt(valid.y, pv)
        thr = youden_threshold(valid.y, pv)
        rows.append({"mode": m, "temperature": t, "platt": [a, b], "threshold": thr,
                     "raw": _cal_metrics(test.y, pt),
                     "temperature_scaled": _cal_metrics(test.y, _sigmoid(_logit(pt) / t)),
                     "platt_scaled": _cal_metrics(test.y, _sigmoid(a * _logit(pt) + b)),
                     "threshold_shift": _cal_metrics(test.y, pt, thr)})
    return rows


# ================================================================================================
# T5.2 predictability of tool utility
# ================================================================================================

def tool_utility_targets(T: Table, kind: str) -> Dict[str, np.ndarray]:
    """Per video: does some tool action fix the no-tool exit? (`kind` = 'arbiter' for what a dispatcher action
    returns, 'evidence' for the scanner-arbiter average.) Also: does some tool action change the decision?"""
    d0 = decisions(T.score["default"])
    wrong = d0 != T.y
    dec = np.stack([decisions(T.score[f"{kind}:{a}"]) for a in TOOL_ACTIONS], 1)
    fixes = wrong & (dec == T.y[:, None]).any(1)
    changes = (dec != d0[:, None]).any(1)
    return {"tool_fixes_exit": fixes.astype(int), "tool_changes_decision": changes.astype(int)}


def _probe(Xv, yv, Xt, yt, C: float) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if yv.min() == yv.max() or yt.min() == yt.max():
        return {"auc": None, "ci95": [None, None]}
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    clf.fit(Xv, yv)
    s = clf.predict_proba(Xt)[:, 1]
    a = auc(yt, s)
    return {"auc": a, "ci95": _bootstrap_auc(yt, s)}


def predictability(valid: Table, test: Table, raw_v: Dict[str, np.ndarray], raw_t: Dict[str, np.ndarray]) -> Dict[str, Any]:
    from sklearn.decomposition import PCA

    def conf(T: Table) -> np.ndarray:
        s, n, d = T.score["scanner"], T.score["notool"], T.score["default"]
        return np.column_stack([s, n, d, _margin(s), _margin(n), _margin(d), np.abs(s - n)])

    hv, ht = raw_v["state"].astype(np.float32), raw_t["state"].astype(np.float32)
    pca = PCA(n_components=64, random_state=0).fit(hv)
    feats = {"confidence (7 scalars)": (conf(valid), conf(test), 1.0),
             "h0 PCA-64 + confidence": (np.hstack([pca.transform(hv), conf(valid)]),
                                        np.hstack([pca.transform(ht), conf(test)]), 0.05)}
    out: Dict[str, Any] = {}
    for kind in ("arbiter", "evidence"):
        tv, tt = tool_utility_targets(valid, kind), tool_utility_targets(test, kind)
        for target in tv:
            key = f"{target} [{kind}]"
            out[key] = {"positives_valid": int(tv[target].sum()), "positives_test": int(tt[target].sum()),
                        "probes": {name: _probe(Xv, tv[target], Xt, tt[target], C)
                                   for name, (Xv, Xt, C) in feats.items()}}
        # the question a router must answer: among videos where tools change the decision, fix or break?
        # Too few cases for a valid/test split, so pool both splits and use stratified 5-fold CV.
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = np.vstack([conf(valid), conf(test)])
        ch = np.concatenate([tv["tool_changes_decision"], tt["tool_changes_decision"]]).astype(bool)
        fx = np.concatenate([tv["tool_fixes_exit"], tt["tool_fixes_exit"]])[ch]
        Xc = X[ch]
        res: Dict[str, Any] = {"changed": int(ch.sum()), "fixes": int(fx.sum()), "breaks": int((1 - fx).sum())}
        k = int(min(5, fx.sum(), (1 - fx).sum()))
        if k >= 2:
            s = np.zeros(len(fx))
            for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(Xc, fx):
                clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
                s[te] = clf.fit(Xc[tr], fx[tr]).predict_proba(Xc[te])[:, 1]
            res.update({"cv_folds": k, "auc": auc(fx, s), "ci95": _bootstrap_auc(fx, s)})
        out[f"fix_vs_break_given_change [{kind}]"] = res
    return out


# ================================================================================================
# T5.3 lambda sweep: GRPO (torch), supervised router, confidence gate, all against always-skip
# ================================================================================================

def _matched_preds(raw: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-action 3-class predictions when every tool action returns mean(scanner, arbiter given tools),
    the same output form as the no-tool exit (action 0 is already that average)."""
    ap = raw["action_probs"].astype(np.float64)
    out = ap.copy()
    out[:, 1:] = (raw["scanner_probs"][:, None, :] + ap[:, 1:]) / 2.0
    return out.argmax(-1)


def _action_latency(T: Table, matched: bool) -> np.ndarray:
    keys = ["default"] + [f"{'evidence' if matched else 'arbiter'}:{a}" for a in TOOL_ACTIONS]
    return np.stack([T.latency[k] for k in keys], 1)


def _action_decisions(T: Table, matched: bool) -> np.ndarray:
    keys = ["default"] + [f"{'evidence' if matched else 'arbiter'}:{a}" for a in TOOL_ACTIONS]
    return np.stack([decisions(T.score[k]) for k in keys], 1)


def _summarise(name: str, T: Table, choice: np.ndarray, dec: np.ndarray, lat: np.ndarray,
               skip_pred: np.ndarray, **extra) -> Dict[str, Any]:
    rows = np.arange(len(T))
    pred, l = dec[rows, choice], lat[rows, choice]
    mc = mcnemar(T.y, pred, skip_pred)
    return {"router": name, **extra, "errors": int((pred != T.y).sum()),
            "balanced_accuracy": balanced_accuracy(T.y, pred), "tool_rate": float((choice != 0).mean()),
            "action_share": {a: float((choice == i).mean()) for i, a in enumerate(ACTIONS)},
            "p50_ms": float(np.median(l)), "mean_ms": float(np.mean(l)),
            "vs_always_skip": {"fixed": mc["a_right_b_wrong"], "broken": mc["a_wrong_b_right"], "p_value": mc["p_value"]}}


def train_grpo(raw_v: Dict[str, np.ndarray], lam: float, seed: int, matched: bool, iterations: int,
               beta_miss: float = 0.5) -> Any:
    """Same recipe as safer.eval.ablation.train_dispatchers (85/15 split of the validation table, model
    selection on greedy reward), with a custom lambda and seed."""
    import torch
    from types import SimpleNamespace
    from safer.models.dispatcher import (AblationProfile, DispatcherPolicy, GRPOTrainer, action_costs,
                                         dispatcher_scalars)
    tool_costs = json.loads(Path(raw_v["_tool_costs_path"]).read_text())
    costs = action_costs(tool_costs, float(np.median(raw_v["t_actions"][:, 1:])))
    cfg = SimpleNamespace(iterations=iterations, batch_videos=256, group_size=16, inner_epochs=2, lr=3e-4,
                          clip_eps=0.2, kl_beta=0.02, entropy_coef=0.01, hidden_dim=256, alpha_attr=0.0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    state = torch.tensor(raw_v["state"].astype(np.float32))
    scalars = torch.tensor(dispatcher_scalars(raw_v["scanner_probs"], raw_v["notool_probs"]))
    labels = torch.tensor(raw_v["labels"], dtype=torch.long)
    preds = torch.tensor(_matched_preds(raw_v) if matched else raw_v["action_probs"].argmax(-1), dtype=torch.long)
    n = len(labels)
    perm = rng.permutation(n)
    sel, tr = perm[: int(0.15 * n)], perm[int(0.15 * n):]
    profile = AblationProfile(f"lam{lam}", lambda_cost=lam, beta_miss=beta_miss)
    policy = DispatcherPolicy(state.shape[1], scalars.shape[1], cfg.hidden_dim)
    trainer = GRPOTrainer(policy, profile, costs, cfg, torch.device("cpu"))
    best, best_state = -1e9, None
    for it in range(1, iterations + 1):
        b = torch.tensor(rng.choice(tr, size=min(cfg.batch_videos, len(tr)), replace=False))
        trainer.step(state[b], scalars[b], labels[b], preds[b])
        if it % 10 == 0 or it == iterations:
            ev = trainer.greedy_eval(state[sel], scalars[sel], labels[sel], preds[sel])
            if ev["reward"] > best:
                best, best_state = ev["reward"], {k: v.detach().clone() for k, v in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    return policy.eval(), trainer.allowed


def grpo_choice(policy, allowed, raw_t: Dict[str, np.ndarray]) -> np.ndarray:
    import torch
    from safer.models.dispatcher import dispatcher_scalars
    with torch.no_grad():
        st = torch.tensor(raw_t["state"].astype(np.float32))
        sc = torch.tensor(dispatcher_scalars(raw_t["scanner_probs"], raw_t["notool_probs"]))
        return policy(st, sc, allowed).argmax(-1).numpy()


def gate_curve(test: Table, matched: bool, taus: Sequence[float]) -> List[Dict[str, Any]]:
    """Confidence gate: keep the no-tool exit when its margin is at least tau, else take all tools."""
    dec, lat = _action_decisions(test, matched), _action_latency(test, matched)
    skip = dec[:, 0]
    full = ACTIONS.index(ALL_TOOLS)
    return [_summarise("confidence gate", test, np.where(_margin(test.score["default"]) < t, full, 0), dec, lat, skip,
                       tau=float(t)) for t in taus]


def sweep(valid: Table, test: Table, raw_v, raw_t, lambdas: Sequence[float], seeds: Sequence[int],
          iterations: int, log=print) -> Dict[str, Any]:
    out: Dict[str, Any] = {"lambdas": list(lambdas), "seeds": list(seeds), "iterations": iterations}
    for matched in (False, True):
        tag = "matched" if matched else "native"
        dec, lat = _action_decisions(test, matched), _action_latency(test, matched)
        skip = dec[:, 0]
        rows: List[Dict[str, Any]] = [_summarise("always skip", test, np.zeros(len(test), int), dec, lat, skip)]
        o_pred, _, o_choice = oracle(test) if not matched else (None, None, None)
        if matched:
            correct = dec == test.y[:, None]
            masked = np.where(correct, lat, np.inf)
            o_choice = np.where(correct.any(1), masked.argmin(1), 0)
        rows.append(_summarise("oracle", test, o_choice, dec, lat, skip))
        for lam in lambdas:
            r = SupervisedRouter(lam)
            # the supervised router scores the same action set the dispatcher sees
            vt = _swap_actions(valid, matched)
            r.fit(vt)
            _, _, ch = r.route(_swap_actions(test, matched))
            rows.append(_summarise("supervised router", test, ch, dec, lat, skip, lam=lam))
        rows += gate_curve(test, matched, [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01])
        for lam in lambdas:
            for seed in seeds:
                t0 = time.time()
                policy, allowed = train_grpo(raw_v, lam, seed, matched, iterations)
                ch = grpo_choice(policy, allowed, raw_t)
                rows.append(_summarise("GRPO dispatcher", test, ch, dec, lat, skip, lam=lam, seed=seed))
                log(f"[{tag}] GRPO lam={lam} seed={seed}: errors {rows[-1]['errors']} tool_rate "
                    f"{rows[-1]['tool_rate']:.3f} p50 {rows[-1]['p50_ms']:.0f} ms ({time.time() - t0:.0f}s)")
        out[tag] = rows
    return out


def _swap_actions(T: Table, matched: bool) -> Table:
    """A view of T whose arbiter:<a> scores are the evidence-mode scores, so SupervisedRouter (which reads
    T.action_keys) routes over the matched action set."""
    if not matched:
        return T
    score = dict(T.score)
    lat = dict(T.latency)
    for a in TOOL_ACTIONS:
        score[f"arbiter:{a}"] = T.score[f"evidence:{a}"]
        lat[f"arbiter:{a}"] = T.latency[f"evidence:{a}"]
    return Table(T.ids, T.y, T.category, score, lat, dict(T.tool_ms), T.name)


# ================================================================================================
# reports
# ================================================================================================

def _f(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def report_md(res: Dict[str, Any]) -> str:
    L = ["# Rebuttal in-domain analyses (Chrono-66k, fit on valid, reported on test)", ""]
    if "operating_points" in res:
        L += ["## T4.2 Threshold-free operating points", "",
              "| Mode | AUC | TPR@1%FPR | TPR@5%FPR | pAUC≤10% (raw) | pAUC≤10% (McClish) |", "|---|---|---|---|---|---|"]
        L += [f"| {r['mode']} | {_f(r['auc'], 5)} | {_f(r['tpr_at_1fpr'], 4)} | {_f(r['tpr_at_5fpr'], 4)} | "
              f"{_f(r['pauc10']['raw'], 4)} | {_f(r['pauc10']['mcclish'], 4)} |" for r in res["operating_points"]]
        L.append("")
    if "calibration" in res:
        L += ["## T3.4 Calibration (test split; maps fit on valid)", "",
              "| Mode | Variant | ECE | Brier | NLL | BA % | Errors |", "|---|---|---|---|---|---|---|"]
        for r in res["calibration"]:
            for v in ("raw", "temperature_scaled", "platt_scaled", "threshold_shift"):
                m = r[v]
                L.append(f"| {r['mode']} | {v} | {m['ece']:.4f} | {m['brier']:.4f} | {m['nll']:.4f} | "
                         f"{100 * m['balanced_accuracy']:.2f} | {m['errors']} |")
        L += ["", "Fitted maps: " + "; ".join(f"{r['mode']}: T={r['temperature']:.3f}, Platt a={r['platt'][0]:.3f} "
                                              f"b={r['platt'][1]:.3f}, thr={r['threshold']:.3f}" for r in res["calibration"]), ""]
    if "predictability" in res:
        L += ["## T5.2 Is tool utility predictable from the frontline state?", "",
              "| Target | Pos. valid | Pos. test | Probe | AUC | 95% CI |", "|---|---|---|---|---|---|"]
        for k, v in res["predictability"].items():
            if "probes" not in v:
                L.append(f"| {k} | {v['fixes']} fix / {v['breaks']} break | (pooled, {v.get('cv_folds', '-')}-fold CV) | "
                         f"confidence (7 scalars) | {_f(v.get('auc'))} | [{_f((v.get('ci95') or [None])[0])}, "
                         f"{_f((v.get('ci95') or [None, None])[1])}] |")
                continue
            for name, p in v["probes"].items():
                L.append(f"| {k} | {v['positives_valid']} | {v['positives_test']} | {name} | {_f(p['auc'])} | "
                         f"[{_f(p['ci95'][0])}, {_f(p['ci95'][1])}] |")
        L.append("")
    if "sweep" in res:
        s = res["sweep"]
        L += [f"## T5.3 Lambda sweep (seeds {s['seeds']}, {s['iterations']} GRPO iterations)", ""]
        for tag in ("native", "matched"):
            if tag not in s:
                continue
            L += [f"### Action outputs: {tag}", "",
                  "| Router | λ / τ | Seed | Errors | Tool rate | p50 ms | Fixed / broken vs skip | McNemar p |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in s[tag]:
                knob = r.get("lam", r.get("tau", ""))
                vs = r["vs_always_skip"]
                L.append(f"| {r['router']} | {knob} | {r.get('seed', '')} | {r['errors']} | {r['tool_rate']:.3f} | "
                         f"{r['p50_ms']:.0f} | {vs['fixed']} / {vs['broken']} | {vs['p_value']:.3g} |")
            L.append("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuttal in-domain analyses")
    ap.add_argument("cmd", choices=["all", "fast", "sweep"])
    ap.add_argument("--outcomes-dir", default="runs/full_2class/predictions")
    ap.add_argument("--tool-costs", default="runs/full_2class/export/tool_costs.json")
    ap.add_argument("--out", default="runs/rebuttal")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lambdas", type=float, nargs="+", default=SWEEP_LAMBDAS)
    ap.add_argument("--iterations", type=int, default=1500)
    a = ap.parse_args(argv)
    valid, test, raw_v, raw_t = _load(Path(a.outcomes_dir))
    raw_v["_tool_costs_path"] = a.tool_costs
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res: Dict[str, Any] = {}
    if a.cmd in ("all", "fast"):
        res["operating_points"] = operating_points(test)
        res["calibration"] = calibration(valid, test)
        res["predictability"] = predictability(valid, test, raw_v, raw_t)
    if a.cmd in ("all", "sweep"):
        res["sweep"] = sweep(valid, test, raw_v, raw_t, a.lambdas, a.seeds, a.iterations)
    name = {"all": "rebuttal_indomain", "fast": "rebuttal_fast", "sweep": "rebuttal_sweep"}[a.cmd]
    (out / f"{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / f"{name}.md").write_text(report_md(res), encoding="utf-8")
    print(f"wrote {out / name}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
