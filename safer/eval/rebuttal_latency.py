"""Rebuttal: end-to-end latency of every mode in ONE harness, in-domain and under shift (CPU only).

The per-action outcome run (benchmark_external_outcomes --steps csf_outcomes) times every component per
video at batch size 1 with resident models, and it includes 1,000 in-domain Chrono-66k test videos
(chrono.npz) next to the four external benchmarks. This script sums the components into the modes of
Table 1 / Table 6 so in-domain and external latency come from the same harness:

  scanner            decode + scanner
  no-tool exit       decode + scanner + no-tool arbiter pass (dispatcher state)
  static             decode + scanner + proposal + all tools + static arbiter pass
  static (arbiter)   as static, without the scanner (the definition used in the external tables)
  evidence, spectral no-tool exit + proposal + spectral tools + cached arbiter pass
  evidence, all      no-tool exit + proposal + all tools + cached arbiter pass

  python -m safer.eval.rebuttal_latency --dir runs/extbench_outcomes --out runs/rebuttal_latency
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SETS = ["chrono", "vifbench", "fakeparts", "genbuster", "genvidbench"]


def modes(z) -> dict:
    dec, prop, spa, spe, lat = (z["t_tools"] * 1000).T
    sc, st, sta, A = z["scanner_latency"] * 1000, z["t_state"] * 1000, z["t_static"] * 1000, z["t_actions"] * 1000
    return {"scanner": dec + sc, "no-tool exit": dec + sc + st,
            "static": dec + sc + prop + spa + spe + lat + sta, "static (arbiter only)": dec + prop + spa + spe + lat + sta,
            "evidence, spectral": dec + sc + st + prop + spe + A[:, 2],
            "evidence, all tools": dec + sc + st + prop + spa + spe + lat + A[:, 5]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for s in SETS:
        p = Path(a.dir) / f"{s}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            m = modes(z)
            res[s] = {"n": int(len(z["labels"])),
                      "p50_ms": {k: float(np.median(v)) for k, v in m.items()},
                      "p95_ms": {k: float(np.percentile(v, 95)) for k, v in m.items()},
                      "components_p50_ms": {k: float(np.median(v) * 1000) for k, v in
                                            (("decode", z["t_tools"][:, 0]), ("scanner", z["scanner_latency"]),
                                             ("dispatcher state", z["t_state"]), ("proposal", z["t_tools"][:, 1]),
                                             ("spatial", z["t_tools"][:, 2]), ("spectral", z["t_tools"][:, 3]),
                                             ("latent", z["t_tools"][:, 4]), ("static pass", z["t_static"]),
                                             ("cached pass, all", z["t_actions"][:, 5]))}}
            p = res[s]["p50_ms"]
            res[s]["ratio_all_over_exit"] = p["evidence, all tools"] / p["no-tool exit"]
            res[s]["ratio_exit_over_all"] = p["no-tool exit"] / p["evidence, all tools"]
    names = list(next(iter(res.values()))["p50_ms"])
    L = ["# Latency of every mode in one harness (per-action run: batch 1, resident models, one worker per GPU)", "",
         "Median end-to-end ms per video. chrono = 1,000 in-domain test videos (500 real / 500 generated).", "",
         "| Set | n | " + " | ".join(names) + " | all / exit | exit / all |", "|---|---|" + "---|" * (len(names) + 2)]
    for s, r in res.items():
        L.append(f"| {s} | {r['n']} | " + " | ".join(f"{r['p50_ms'][k]:.1f}" for k in names) +
                 f" | {r['ratio_all_over_exit']:.2f} | {r['ratio_exit_over_all']:.2f} |")
    L += ["", "## Component medians (ms)", "", "| Set | " + " | ".join(res["chrono"]["components_p50_ms"]) + " |",
          "|---|" + "---|" * len(res["chrono"]["components_p50_ms"])]
    for s, r in res.items():
        L.append(f"| {s} | " + " | ".join(f"{v:.1f}" for v in r["components_p50_ms"].values()) + " |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rebuttal_latency.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "rebuttal_latency.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
