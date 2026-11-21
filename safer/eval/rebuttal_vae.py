"""Rebuttal (4jrv W2/Q3): do the tool features respond to a generation step with content held fixed?

Reads the outcome tables of the variants made by safer.eval.vae_roundtrip (datasets vae_identity,
vae_h264_crf*, vae_h265_crf*, vae_wan, vae_cog), pairs every variant with the identity copy of the
same clip, and reports per feature:

  es       mean paired shift / SD of the feature over the identity clips (effect size, bootstrap CI)
  gen      on how many external benchmarks the shift points from real toward generated
           (sign of median(generated) - median(real) in the rerun tables)

A feature counts as VAE-responsive when |es| >= 0.5 for a VAE, the CI excludes 0, and the VAE shift
is at least twice the largest re-encoding shift. It also reports how often the detector's decisions
on the real clips flip to "generated" after each variant.

  python -m safer.eval.rebuttal_vae --dir OUT/outcomes --manifests OUT/manifests --ext EXT/outcomes --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from safer.eval.paper import ALL_TOOLS, table_from_outcomes
from safer.eval.rebuttal_external import BENCHES
from safer.eval.rebuttal_fusion import ALL_FEATURES, GROUP_OF, load_with_features

SCORES = ["scanner", "notool", f"arbiter:{ALL_TOOLS}", "default"]


def load_variant(d: Path, man_dir: Path, name: str):
    T = table_from_outcomes(d / f"{name}.npz", name)
    with np.load(d / f"{name}.npz", allow_pickle=False) as z:
        X = z["features_raw"].astype(np.float64)
    man = pd.read_csv(man_dir / f"{name}.csv", dtype={"video_id": str})
    clip = dict(zip(man["video_id"], man["src_path"].map(lambda p: Path(p).stem)))
    ids = [clip[k.split("/", 1)[1]] for k in T.ids]
    return pd.Index(ids), T, X


def ext_gap_signs(ext: Path) -> Dict[str, np.ndarray]:
    out = {}
    for b in BENCHES:
        if not (ext / f"{b}.npz").exists():
            continue
        T, X = load_with_features(ext, b)
        out[b] = np.sign(np.nanmedian(X[T.y == 1], 0) - np.nanmedian(X[T.y == 0], 0))
    return out


def _boot_ci(d: np.ndarray, sd: float, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() / sd for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyse(d: Path, man_dir: Path, ext: Path) -> Dict[str, Any]:
    names = sorted(p.stem for p in d.glob("vae_*.npz") if not p.stem.endswith("_nostate"))
    if "vae_identity" not in names:
        raise SystemExit(f"vae_identity.npz missing in {d}")
    data = {n: load_variant(d, man_dir, n) for n in names}
    ids0, T0, X0 = data["vae_identity"]
    sd = np.nanstd(X0, 0)
    signs = ext_gap_signs(ext)
    res: Dict[str, Any] = {"n_identity": len(ids0), "variants": {}, "benches": list(signs)}
    for n in names:
        if n == "vae_identity":
            continue
        ids, T, X = data[n]
        common = ids0.intersection(ids)
        i0, i1 = ids0.get_indexer(common), ids.get_indexer(common)
        D = X[i1] - X0[i0]
        feats = {}
        for j, f in enumerate(ALL_FEATURES):
            dj = D[:, j][~np.isnan(D[:, j])]
            if len(dj) < 10 or not sd[j] > 0:
                feats[f] = None
                continue
            es = float(dj.mean() / sd[j])
            lo, hi = _boot_ci(dj, sd[j], seed=j)
            gen = [b for b, s in signs.items() if s[j] != 0 and np.sign(es) == s[j]]
            feats[f] = {"es": es, "ci": [lo, hi], "toward_generated_on": gen, "n_benches": len(signs)}
        det = {}
        for k in SCORES:
            p0, p1 = T0.score[k][i0], T.score[k][i1]
            det[k] = {"fake_rate_identity": float((p0 > 0.5).mean()), "fake_rate_variant": float((p1 > 0.5).mean()),
                      "real_to_fake": int(((p0 <= 0.5) & (p1 > 0.5)).sum()),
                      "fake_to_real": int(((p0 > 0.5) & (p1 <= 0.5)).sum()),
                      "mean_dp": float((p1 - p0).mean())}
        res["variants"][n] = {"n": len(common), "features": feats, "detector": det}

    vaes = [n for n in res["variants"] if n in ("vae_wan", "vae_cog")]
    encs = [n for n in res["variants"] if n not in vaes]
    responsive: Dict[str, List[str]] = {}
    for v in vaes:
        hits = []
        for f in ALL_FEATURES:
            r = res["variants"][v]["features"][f]
            if r is None:
                continue
            enc_max = max([abs(res["variants"][e]["features"][f]["es"]) for e in encs
                           if res["variants"][e]["features"][f]] or [0.0])
            if abs(r["es"]) >= 0.5 and (r["ci"][0] > 0 or r["ci"][1] < 0) and abs(r["es"]) >= 2 * enc_max:
                hits.append(f)
        responsive[v] = hits
    res["responsive"] = responsive
    return res


def report(res: Dict[str, Any]) -> str:
    vs = list(res["variants"])
    L = [f"# VAE round-trip vs. re-encoding on {res['n_identity']} held-out real clips", "",
         "Effect size = mean paired shift / SD over identity clips. `*` = 95% CI excludes 0. "
         f"`g/k` = shift points toward generated on g of k external benchmarks ({', '.join(res['benches'])}).", "",
         "| feature | group | " + " | ".join(v.removeprefix("vae_") for v in vs) + " |",
         "|---|---|" + "---|" * len(vs)]
    for f in ALL_FEATURES:
        cells = []
        for v in vs:
            r = res["variants"][v]["features"][f]
            if r is None:
                cells.append("-")
                continue
            star = "*" if (r["ci"][0] > 0 or r["ci"][1] < 0) else ""
            cells.append(f"{r['es']:+.2f}{star} ({len(r['toward_generated_on'])}/{r['n_benches']})")
        L.append(f"| {f} | {GROUP_OF[f]} | " + " | ".join(cells) + " |")
    L += ["", "## VAE-responsive features (|es| >= 0.5, CI excludes 0, >= 2x the largest re-encoding shift)", ""]
    for v, hits in res["responsive"].items():
        L.append(f"- {v}: {len(hits)} of {len(ALL_FEATURES)}: {', '.join(hits) or 'none'}")
    L += ["", "## Detector decisions on the real clips (p > 0.5 = generated)", "",
          "| variant | score | fake rate identity | fake rate variant | real->fake | fake->real | mean dp |",
          "|---|---|---|---|---|---|---|"]
    for v in vs:
        for k, r in res["variants"][v]["detector"].items():
            L.append(f"| {v.removeprefix('vae_')} | {k} | {100 * r['fake_rate_identity']:.1f}% | "
                     f"{100 * r['fake_rate_variant']:.1f}% | {r['real_to_fake']} | {r['fake_to_real']} | "
                     f"{r['mean_dp']:+.3f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="merged outcome tables of the vae_* datasets")
    ap.add_argument("--manifests", required=True, help="manifests dir of the same run (video_id -> src_path)")
    ap.add_argument("--ext", required=True, help="external rerun outcome tables, for the generated direction")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = analyse(Path(a.dir), Path(a.manifests), Path(a.ext))
    (out / "rebuttal_vae.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    md = report(res)
    (out / "rebuttal_vae.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
