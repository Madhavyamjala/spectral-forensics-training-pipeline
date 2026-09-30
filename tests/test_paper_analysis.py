"""
Tests for csf.eval.paper (the SAFER paper's analyses) on small synthetic outcome tables whose answers
can be worked out by hand. Torch-free: numpy, scikit-learn, scipy, pandas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csf.eval import paper  # noqa: E402
from csf.eval.paper import (ACTIONS, TOOL_ACTIONS, SupervisedRouter, Table, adaptation, choose_tau,  # noqa: E402
                            decision_changes, fixed_broken, generator_disjoint_split, low_latency, mcnemar,
                            oracle, table_from_outcomes, table_from_per_action, tool_value)


def make_table(y, scores, lat=None, category=None, name="t"):
    n = len(y)
    lat = lat or {}
    keys = ["scanner", "notool", "default", "static"] + [f"arbiter:{a}" for a in TOOL_ACTIONS] + \
           [f"evidence:{a}" for a in TOOL_ACTIONS]
    score = {k: np.asarray(scores.get(k, scores.get("default")), dtype=float) for k in keys}
    base = {"scanner": 100.0, "notool": 150.0, "default": 240.0, "static": 350.0}
    latency = {k: np.full(n, lat.get(k, base.get(k, 240.0 + 20 * (TOOL_ACTIONS.index(k.split(":")[1]) + 1)
                                            if ":" in k else 240.0))) for k in keys}
    return Table(np.array([f"v{i}" for i in range(n)]), np.asarray(y),
                 np.asarray(category if category is not None else ["c"] * n), score, latency, {}, name)


# ------------------------------------------------------------------------------------------ routing

def test_oracle_takes_the_cheapest_correct_action_and_falls_back_to_exit():
    y = np.array([1, 1, 0])
    s = {"default": [0.2, 0.2, 0.2],                     # exit: wrong, wrong, right
         f"arbiter:spectral": [0.9, 0.1, 0.1],           # right on video 0
         f"arbiter:latent": [0.9, 0.1, 0.1]}             # also right on 0 but slower
    T = make_table(y, s)
    pred, lat, choice = oracle(T)
    keys = T.action_keys
    assert keys[choice[0]] == "arbiter:spectral"          # cheapest correct
    assert keys[choice[1]] == "default"                   # nothing correct -> exit
    assert keys[choice[2]] == "default"                   # exit correct and cheapest
    assert list(pred) == [1, 0, 0]


def test_decision_changes_counts_fix_and_break():
    y = np.array([1, 1, 0, 0])
    s = {"default": [0.2, 0.9, 0.1, 0.1],
         "arbiter:spatial": [0.9, 0.2, 0.1, 0.8]}         # fixes 0, breaks 1 and 3
    c = decision_changes(make_table(y, s))
    assert c == {"videos": 4, "all_correct": 1, "all_wrong": 0, "tools_fix_exit": 1, "tools_break_exit": 2}


def test_low_latency_escalates_only_uncertain_scanner_calls():
    y = np.array([1, 0, 1])
    s = {"scanner": [0.95, 0.52, 0.45], "default": [0.9, 0.1, 0.8]}
    T = make_table(y, s)
    pred, lat, esc = low_latency(T, tau=0.5)             # margins 0.9, 0.04, 0.1
    assert list(esc) == [False, True, True]
    assert list(pred) == [1, 0, 1]
    assert lat[0] == 100.0 and lat[1] == 240.0


def test_choose_tau_prefers_fewer_errors_then_lower_latency():
    y = np.array([1, 0])
    s = {"scanner": [0.9, 0.6], "default": [0.9, 0.1]}   # scanner wrong on video 1 (margin 0.2)
    tau = choose_tau(make_table(y, s), grid=[0.0, 0.1, 0.3, 0.9])
    assert tau == 0.3                                     # smallest tau that escalates video 1 only


def test_mcnemar_counts_discordant_pairs():
    y = np.array([1, 1, 1, 0, 0])
    r = mcnemar(y, np.array([1, 1, 0, 0, 0]), np.array([0, 1, 1, 1, 0]))
    assert r["a_right_b_wrong"] == 2 and r["a_wrong_b_right"] == 1
    assert 0 < r["p_value"] <= 1


def test_fixed_broken_relative_to_default():
    y = np.array([1, 0, 1])
    s = {"default": [0.1, 0.1, 0.9], f"evidence:full_tri_domain": [0.9, 0.9, 0.9]}
    assert fixed_broken(make_table(y, s)) == {"fixed": 1, "broken": 1, "n": 3}


def test_supervised_router_learns_to_skip_tools_that_never_help():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.where(y == 1, 0.9, 0.1)
    s = {"scanner": good, "notool": good, "default": good,
         **{f"arbiter:{a}": np.where(rng.random(400) < 0.3, 1 - good, good) for a in TOOL_ACTIONS}}
    T = make_table(y, s)
    _, _, choice = SupervisedRouter(0.2).fit(T).route(T)
    assert (choice == 0).mean() > 0.95                    # exits without tools


# ------------------------------------------------------------------------------------------ tool value

def two_category_benchmark(delta_by_cat):
    rng = np.random.default_rng(1)
    n_real, n_fake = 200, 150
    y, cat, notool, tools = [], [], [], []
    for _ in range(n_real):
        y.append(0), cat.append("real"), notool.append(rng.normal(0.4, 0.15)), tools.append(None)
    for c, d in delta_by_cat.items():
        for _ in range(n_fake):
            v = rng.normal(0.6, 0.15)
            y.append(1), cat.append(c), notool.append(v), tools.append(v + d)
    notool = np.clip(notool, 0, 1)
    tools = np.array([n if t is None else t for n, t in zip(notool, tools)])
    return make_table(np.array(y), {"scanner": notool, "notool": notool, "default": notool,
                                    **{f"arbiter:{a}": np.clip(tools, 0, 1) for a in TOOL_ACTIONS}},
                      category=np.array(cat), name="bench")


def test_tool_value_is_zero_when_tools_change_nothing():
    tv = tool_value([two_category_benchmark({"genA": 0.0, "genB": 0.0})], n_boot=50)
    assert tv["n_categories"] == 2
    assert abs(tv["mean_delta"]) < 1e-9
    assert all(c["verdict"] == "no detectable change" for c in tv["categories"])


def test_tool_value_detects_a_real_gain_per_category():
    tv = tool_value([two_category_benchmark({"genA": 0.3, "genB": 0.0})], n_boot=200)
    by = {c["category"]: c for c in tv["categories"]}
    assert by["genA"]["delta"] > 0.1 and by["genA"]["verdict"] == "gain"
    assert by["genB"]["verdict"] == "no detectable change"


# ------------------------------------------------------------------------------------------ adaptation

def shifted_benchmark():
    """Ranks well (AUC high) but every fake scores below 0.5: the paper's miscalibration case."""
    rng = np.random.default_rng(2)
    cats = [f"gen{i}" for i in range(6)]
    y = np.r_[np.zeros(300), np.ones(600)].astype(int)
    cat = np.r_[["real"] * 300, np.repeat(cats, 100)]
    default = np.r_[rng.uniform(0.0, 0.2, 300), rng.uniform(0.2, 0.45, 600)]
    return make_table(y, {"default": default, "scanner": default, "notool": default}, category=cat, name="shift")


def test_generator_disjoint_split_keeps_test_generators_unseen():
    T = shifted_benchmark()
    fit, test = generator_disjoint_split(T, np.random.default_rng(0))
    fit_c = set(T.category[fit][T.y[fit] == 1])
    test_c = set(T.category[test][T.y[test] == 1])
    assert fit_c and test_c and not (fit_c & test_c)
    assert not (set(fit) & set(test))


def test_recalibration_recovers_a_miscalibrated_threshold():
    a = adaptation(shifted_benchmark(), n_splits=5)
    assert a["default"]["balanced_accuracy"] == pytest.approx(0.5)
    assert a["recalibration"]["mean"] > 0.4               # near-perfect after moving the threshold
    assert a["routing_oracle"]["delta"] == pytest.approx(0.0)   # tools identical -> nothing to route to


# ------------------------------------------------------------------------------------------ loaders + CLI

def write_outcomes(path: Path, n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)

    def probs(noise):
        p_fake = np.clip(np.where(labels == 1, 0.8, 0.2) + rng.normal(0, noise, n), 0.01, 0.99)
        return np.column_stack([1 - p_fake, p_fake, np.zeros(n)])

    scanner, notool = probs(0.2), probs(0.25)
    actions = np.stack([(scanner + notool) / 2] + [probs(0.25) for _ in TOOL_ACTIONS], 1)
    np.savez(path, keys=np.array([f"k{i}" for i in range(n)]), labels=labels, methods=np.array(["unknown"] * n),
             scanner_probs=scanner, scanner_latency=np.full(n, 0.064), notool_probs=notool,
             state=np.zeros((n, 4), np.float16), t_state=np.full(n, 0.1318), static_probs=probs(0.25),
             t_static=np.full(n, 0.2), action_probs=actions,
             t_actions=np.column_stack([np.zeros(n)] + [np.full(n, 0.05)] * len(TOOL_ACTIONS)),
             t_tools=np.column_stack([np.full(n, 0.0488), np.full(n, 0.0015), np.full(n, 0.059),
                                      np.full(n, 0.015), np.full(n, 0.041)]),
             vision_cache_used=np.array(True))


def test_outcome_table_latencies_follow_the_paper_accounting(tmp_path):
    write_outcomes(tmp_path / "outcomes_test.npz")
    T = table_from_outcomes(tmp_path / "outcomes_test.npz")
    # Table 8 components: decode 48.8 + scanner 64.0 + dispatcher state 131.8 = 244.6 ms no-tool exit
    assert T.latency["default"][0] == pytest.approx(244.6)
    # all tools: + proposal 1.5 + 59 + 15 + 41 tools + 50 arbiter
    assert T.latency["arbiter:full_tri_domain"][0] == pytest.approx(244.6 + 1.5 + 115 + 50)
    assert T.score["default"] == pytest.approx((T.score["scanner"] + T.score["notool"]) / 2)
    assert T.score["evidence:spectral"] == pytest.approx((T.score["scanner"] + T.score["arbiter:spectral"]) / 2)


def test_indomain_cli_end_to_end(tmp_path, capsys):
    write_outcomes(tmp_path / "outcomes_test.npz", seed=0)
    write_outcomes(tmp_path / "outcomes_valid.npz", seed=1)
    paper.main(["indomain", "--outcomes-dir", str(tmp_path), "--out", str(tmp_path / "out")])
    res = json.loads((tmp_path / "out" / "indomain.json").read_text())
    modes = [r["mode"] for r in res["rows"]]
    assert modes[0] == "SAFER-Scanner" and modes[-1] == "Oracle (cheapest correct action)"
    assert any(m.startswith("SAFER low-latency mode") for m in modes)
    oracle_row = res["rows"][-1]
    assert oracle_row["errors"] <= min(r["errors"] for r in res["rows"])


def test_external_cli_end_to_end(tmp_path):
    import pandas as pd
    bench = tmp_path / "bench"
    (bench / "manifests").mkdir(parents=True)
    (bench / "per_action").mkdir()
    rng = np.random.default_rng(3)
    rows, recs = [], []
    for i in range(240):
        label = "real" if i < 80 else "ai_generated"
        cat = "real" if label == "real" else f"gen{i % 4}"
        rows.append({"video_id": f"x{i}", "label": label, "category": cat, "usable": True,
                     "overlap": cat == "gen3", "path": f"/v/{i}.mp4"})
        pf = float(np.clip((0.35 if label != "real" else 0.15) + rng.normal(0, 0.1), 0.01, 0.99))

        def pr(p):
            return [1 - p, p, 0.0]

        recs.append({"video_id": f"x{i}", "scanner": pr(pf), "notool": pr(pf), "static": pr(pf),
                     "actions": {a: pr(float(np.clip(pf + rng.normal(0, 0.05), 0.01, 0.99))) for a in TOOL_ACTIONS},
                     "tool_mask": [True, True, True],
                     "t": {"decode": 0.05, "scanner": 0.06, "state": 0.13, "static": 0.2, "proposal": 0.002,
                           "spatial": 0.06, "spectral": 0.015, "latent": 0.04,
                           **{f"arbiter_{a}": 0.05 for a in TOOL_ACTIONS}}})
    pd.DataFrame(rows).to_csv(bench / "manifests" / "toy.csv", index=False)
    (bench / "per_action" / "toy.shard0of1.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    T = table_from_per_action(bench, "toy")
    assert "gen3" not in set(T.category)                 # overlapping generator dropped
    write_outcomes(tmp_path / "valid.npz")
    paper.main(["external", "--bench-out", str(bench), "--out", str(tmp_path / "out"), "--splits", "3",
                "--bootstrap", "50", "--fit-outcomes", str(tmp_path / "valid.npz")])
    res = json.loads((tmp_path / "out" / "external.json").read_text())
    assert res["tool_value"]["n_categories"] == 3
    ad = res["adaptation"][0]
    assert ad["recalibration"]["splits"] == 3 and "router_zero_shot" in ad
    assert ad["recalibration"]["mean"] > 0                # fakes sit below 0.5 but rank above reals
