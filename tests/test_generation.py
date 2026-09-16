"""
Regression tests for the AI-Edited regeneration planner.

Deliberately dependency-free (standard library only) so they run anywhere - no torch, no pandas,
no GPU, no network:

    python tests/test_generation.py

They lock down the properties that are expensive to discover are broken after three days of
generation: the counts match the specification document, the plan is reproducible, and no source
clip leaks across the train/valid/test boundary.
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csf.generation import spec as S
from csf.generation.adapters import ADAPTERS, coverage, env_specs
from csf.generation.jobs import Job, build_jobs, object_operations, summarise

FAILURES: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}: {detail}")
        FAILURES.append(name)


def synthetic_pool(per_label: int = 400, qualifying: bool = True) -> list:
    rows = []
    for li, label in enumerate(S.all_labels()):
        for k in range(per_label):
            rows.append({
                "clip_id": f"clip{li:03d}_{k:04d}", "label": label,
                "path": f"/pool/{li}/{k}.mp4", "frames_scanned": "12",
                "face_ratio": "0.95" if qualifying else "0.0",
                "mean_face_size": "0.25" if qualifying else "0.0",
                "mouth_ratio": "0.9" if qualifying else "0.0",
                "mean_abs_yaw": "10", "motion_score": "0.05", "edge_density": "0.08",
                "temporal_stability": "0.8", "error": ""})
    return rows


def test_spec() -> None:
    print("spec")
    S.validate()
    check("document base total is 28,333", sum(S.BASE_TARGETS.values()) == 28333)
    check("run total is 33,333", sum(S.FAMILY_TARGETS.values()) == 33333,
          str(sum(S.FAMILY_TARGETS.values())))
    for fam in S.FACE_FAMILIES:
        check(f"{fam} topped up by 1,250",
              S.FAMILY_TARGETS[fam] - S.BASE_TARGETS[fam] == 1250)
    for fam in S.FAMILIES:
        if fam not in S.FACE_FAMILIES:
            check(f"{fam} unchanged", S.FAMILY_TARGETS[fam] == S.BASE_TARGETS[fam])

    # the document's own cross-split tables, reproduced at the base targets
    expected = {
        ("video_to_video", 0): [116, 116, 112, 106],
        ("video_to_video", 6): [64, 64, 63, 59],
        ("video_inpainting", 0): [180, 150, 130, 140],
        ("video_inpainting", 1): [150, 125, 108, 117],
        ("background_manipulation", 1): [175, 175, 175, 175],
        ("object_insertion_removal", 0): [180, 180, 170, 170],
    }
    for (fam_key, row), want in expected.items():
        family = S.FAMILIES[fam_key]
        _, _, grid = S.family_matrix(family, S.BASE_TARGETS[fam_key])
        check(f"{fam_key} row {row} matches the document", grid[row] == want,
              f"got {grid[row]}, want {want}")

    check("apportion sums exactly", sum(S.apportion(1000, [3, 3, 2, 2])) == 1000)
    grid = S.cross_split([7, 11, 5], [9, 8, 6])
    check("cross_split keeps row margins", [sum(r) for r in grid] == [7, 11, 5])
    check("cross_split keeps column margins",
          [sum(grid[i][j] for i in range(3)) for j in range(3)] == [9, 8, 6])


def test_jobs() -> None:
    print("jobs")
    features = synthetic_pool()
    jobs = build_jobs(features, seed=42)
    summary = summarise(jobs)
    check("33,333 jobs planned", summary["total"] == 33333, str(summary["total"]))
    check("per-family counts match targets",
          Counter(j.family for j in jobs) == Counter(S.FAMILY_TARGETS))
    check("video ids are unique", len({j.video_id for j in jobs}) == len(jobs))
    check("job ids are unique", len({j.job_id for j in jobs}) == len(jobs))

    again = build_jobs(features, seed=42)
    check("plan is reproducible",
          [(j.job_id, j.source_clip_id, j.split) for j in jobs]
          == [(j.job_id, j.source_clip_id, j.split) for j in again])

    splits = defaultdict(set)
    for j in jobs:
        splits[j.source_clip_id].add(j.split)
    straddling = [k for k, v in splits.items() if len(v) > 1]
    check("no source clip straddles splits", not straddling,
          f"{len(straddling)} clips in multiple splits")
    check("every job has a split", all(j.split for j in jobs))

    ratios = Counter(j.split for j in jobs)
    train_frac = ratios["train"] / len(jobs)
    check("train share is near 80%", 0.74 <= train_frac <= 0.86, f"{train_frac:.3f}")

    # the object family's removal/insertion split is fixed by the document
    ops = Counter((j.model, j.operation) for j in jobs if j.family == "object_insertion_removal")
    check("ProPainter 700 removal / 200 insertion",
          ops[("propainter_object", "object_removal")] == 700
          and ops[("propainter_object", "object_insertion")] == 200)
    check("Object-WIPER is removal-only",
          ops[("object_wiper", "object_removal")] == 900
          and ops[("object_wiper", "object_insertion")] == 0)
    check("AnyV2V is insertion-only",
          ops[("anyv2v_object", "object_insertion")] == 850
          and ops[("anyv2v_object", "object_removal")] == 0)
    check("VideoComposer 150 removal / 700 insertion",
          ops[("videocomposer", "object_removal")] == 150
          and ops[("videocomposer", "object_insertion")] == 700)

    # family-specific bindings
    reen = [j for j in jobs if j.family == "facial_reenactment"]
    check("reenactment drives from a different clip",
          all(j.driving_clip_id and j.driving_clip_id != j.source_clip_id for j in reen))
    lip = [j for j in jobs if j.family == "lip_sync"]
    check("lip-sync borrows audio from another clip",
          all(j.audio_clip_id and j.audio_clip_id != j.source_clip_id for j in lip))
    v2v = [j for j in jobs if j.family == "video_to_video"]
    check("video-to-video jobs carry a prompt", all(j.prompt for j in v2v))
    inp = [j for j in jobs if j.family == "video_inpainting"]
    check("inpainting jobs carry mask size + motion",
          all(j.mask_size and j.mask_motion for j in inp))


def test_degraded_pool() -> None:
    print("degraded source pools")
    scarce = synthetic_pool(per_label=3)
    jobs = build_jobs(scarce, seed=42)
    check("plans in full even when clips must be reused", len(jobs) == 33333)
    splits = defaultdict(set)
    for j in jobs:
        splits[j.source_clip_id].add(j.split)
    check("reuse still does not leak across splits",
          not [k for k, v in splits.items() if len(v) > 1])

    faceless = synthetic_pool(per_label=20, qualifying=False)
    jobs = build_jobs(faceless, seed=42)
    check("falls back rather than crashing with no qualifying faces", len(jobs) == 33333)


def test_naming() -> None:
    print("manifest compatibility")
    # the regex from csf/data/manifest.py:repo_path_for must resolve every generated id
    pattern = re.compile(r"^aiedit-(.+)-\d+$")
    ok = True
    for family in S.FAMILIES:
        for idx in (0, 7, 1234, 33332):
            vid = f"aiedit-{family}-{idx}"
            m = pattern.match(vid)
            job = Job(job_id="x", video_id=vid, family=family, model="m", source_group="g",
                      source_clip_id="c", source_path="p", source_label="l")
            if not m or m.group(1) != family or \
                    job.repo_path != f"AI Edited/{family}/{vid}.mp4":
                ok = False
    check("every video id resolves to its repo path unchanged", ok)


def test_adapters() -> None:
    print("adapters")
    models = set(S.all_models())
    check("every spec model has an adapter", models <= set(ADAPTERS),
          str(sorted(models - set(ADAPTERS))))
    check("no orphan adapters", set(ADAPTERS) <= models,
          str(sorted(set(ADAPTERS) - models)))
    specs = env_specs()
    check("every adapter's environment is defined",
          all(a.env_name in specs for a in ADAPTERS.values()))
    missing_worker = [a.key for a in ADAPTERS.values() if not a.worker_path.exists()]
    check("every adapter's worker script exists", not missing_worker, str(missing_worker))
    unwired = [a.key for a in ADAPTERS.values()
               if not a.implemented and a.worker != "worker_unimplemented.py"]
    check("unwired models use the refusing worker", not unwired, str(unwired))
    cov = coverage()
    check("coverage accounts for every video",
          cov["videos_wired"] + cov["videos_pending"] == 33333)
    print(f"       tier-1 coverage: {cov['videos_wired']:,}/{cov['total_videos']:,} "
          f"({cov['fraction_wired']:.1%})")


def main() -> int:
    for fn in (test_spec, test_jobs, test_degraded_pool, test_naming, test_adapters):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all regeneration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
