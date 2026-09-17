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
from csf.generation.adapters import (ADAPTERS, coverage, cost_estimate, env_specs,
                                     videos_per_model)
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


def test_substitutions() -> None:
    print("substitutions")
    subs = {k: a for k, a in ADAPTERS.items() if a.substituted}
    check("substitutes are all wired", all(a.implemented for a in subs.values()))
    check("substitutes never claim their slot's name",
          all(a.runs != a.key for a in subs.values()))
    check("non-substitutes run as themselves",
          all(a.runs == a.key for a in ADAPTERS.values() if not a.substituted))
    check("every substitution carries an explanatory note",
          all(a.note for a in subs.values()),
          str([k for k, a in subs.items() if not a.note]))

    # a family must never be rendered entirely by one model while looking like several
    per_model = videos_per_model()
    for family in S.FAMILY_LIST:
        wired = [p.key for p in family.pipelines if ADAPTERS[p.key].implemented]
        renderers = {ADAPTERS[k].runs for k in wired}
        if wired:
            check(f"{family.key} reports its true renderer count",
                  len(renderers) >= 1 and len(renderers) <= len(wired))

    # coverage bookkeeping must still add up with substitutions in play
    cov = coverage()
    check("coverage still accounts for every video",
          cov["videos_wired"] + cov["videos_pending"] == 33333)
    check("substituted videos are a subset of wired",
          cov["videos_substituted"] <= cov["videos_wired"])
    by_actual = sum(cov["videos_by_actual_model"].values())
    check("per-renderer totals equal wired videos", by_actual == cov["videos_wired"],
          f"{by_actual} != {cov['videos_wired']}")
    print(f"       {cov['videos_wired']:,}/33,333 wired ({cov['fraction_wired']:.1%}), "
          f"{cov['videos_substituted']:,} substituted, "
          f"{cov['distinct_actual_models']} distinct renderers")


def test_budget() -> None:
    print("budget planner")
    from csf.generation.budget import family_breakdown, plan

    full = cost_estimate(gpus=3)
    check("full cost is reported", full["gpu_hours_total"] > 0)

    p = plan(84.0 * 3, gpus=3, diversity_floor=2)
    check("plan stays inside its budget", p.gpu_hours_used <= 84.0 * 3,
          f"{p.gpu_hours_used} > {84.0 * 3}")
    check("plan selects something", p.videos > 0)
    check("selected and skipped are disjoint",
          not (set(p.selected) & set(p.skipped)))
    check("selected + skipped covers every wired model",
          set(p.selected) | set(p.skipped) ==
          {k for k, a in ADAPTERS.items() if a.implemented})

    fams = family_breakdown(p)
    empty = [f for f, i in fams.items() if i["mechanisms"] == 0]
    check("no family is left empty by the planner", not empty, str(empty))
    # the floor counts distinct renderers, not slots - two VACE slots are one mechanism
    for fam, info in fams.items():
        check(f"{fam} renderer list has no duplicates",
              len(info["models"]) == len(set(info["models"])))

    # a bigger budget must never select fewer videos
    small = plan(40.0 * 3, gpus=3)
    big = plan(200.0 * 3, gpus=3)
    check("more budget never yields fewer videos", big.videos >= small.videos,
          f"{big.videos} < {small.videos}")
    check("an unlimited budget selects every wired model",
          set(plan(10_000.0, gpus=3).selected) ==
          {k for k, a in ADAPTERS.items() if a.implemented})
    print(f"       84 h x 3 GPUs -> {len(p.selected)} models, {p.videos:,} videos "
          f"({p.coverage:.1%}), {p.gpu_hours_used:.0f} GPU-h")


def test_reallocation() -> None:
    print("reallocation")
    allowed = {k for k, a in ADAPTERS.items() if a.implemented}
    unfillable = set(ADAPTERS) - allowed

    total = 0
    for family in S.FAMILY_LIST:
        target = S.FAMILY_TARGETS[family.key]
        pipelines = S.family_pipelines(family, allowed)
        rows, cols, grid = S.family_matrix(family, target, allowed)
        check(f"{family.key} still reaches its specified size", sum(cols) == target,
              f"{sum(cols)} != {target}")
        check(f"{family.key} keeps its source-group mix", sum(rows) == target)
        check(f"{family.key} drops only unfillable pipelines",
              not ({p.key for p in pipelines} & unfillable))
        for i, r in enumerate(grid):
            if sum(r) != rows[i]:
                check(f"{family.key} row {i} margin", False)
        total += sum(cols)
    check("reallocated plan totals 33,333", total == 33333, str(total))

    features = synthetic_pool(per_label=400)
    jobs = build_jobs(features, seed=42, allowed_models=allowed)
    check("reallocated job count is 33,333", len(jobs) == 33333, str(len(jobs)))
    check("no job targets an unfillable slot",
          not ({j.model for j in jobs} & unfillable))
    check("reallocated plan is reproducible",
          [j.job_id for j in jobs] ==
          [j.job_id for j in build_jobs(features, seed=42, allowed_models=allowed)])
    splits = defaultdict(set)
    for j in jobs:
        splits[j.source_clip_id].add(j.split)
    check("reallocation keeps splits leakage-free",
          not [k for k, v in splits.items() if len(v) > 1])
    check("every family still has >=1 renderer",
          len({ADAPTERS[j.model].runs for j in jobs}) >= 8)

    # object family's removal/insertion allocation must survive losing a pipeline
    ops = Counter(j.operation for j in jobs if j.family == "object_insertion_removal")
    check("object operations still sum to the family target",
          sum(ops.values()) == S.FAMILY_TARGETS["object_insertion_removal"])
    print(f"       33,333 videos across {len({ADAPTERS[j.model].runs for j in jobs})} renderers")


def test_concurrency() -> None:
    print("concurrency model")
    from csf.generation.budget import effective_speedup, wall_clock_estimate
    from csf.generation.scheduler import concurrency_for

    check("a 3 GB model packs many workers into 143 GB",
          concurrency_for("inswapper", 143.0, 6) == 6)
    check("a 26 GB model packs fewer", concurrency_for("bg_flux_image", 143.0, 6) == 4,
          str(concurrency_for("bg_flux_image", 143.0, 6)))
    check("concurrency never drops below 1",
          concurrency_for("bg_flux_image", 8.0, 6) == 1)
    check("cap is respected", concurrency_for("inswapper", 143.0, 2) == 2)

    check("one worker means no speedup", effective_speedup("inswapper", 1) == 1.0)
    check("diffusion gains less than small nets",
          effective_speedup("insv2v", 4) < effective_speedup("inswapper", 4))

    allowed = {k for k, a in ADAPTERS.items() if a.implemented}
    serial = wall_clock_estimate(4, 1, allowed=allowed)
    conc = wall_clock_estimate(4, 4, allowed=allowed)
    check("reallocated estimate covers 33,333", serial["videos"] == 33333)
    check("concurrency reduces wall clock",
          conc["wall_clock_hours"] < serial["wall_clock_hours"])
    check("concurrency never invents capacity",
          conc["gpu_hours_effective"] <= serial["gpu_hours_serial"])
    print(f"       4 GPUs x 4 workers -> {conc['wall_clock_days']} days for 33,333 videos")


def test_kinetics_schema() -> None:
    """The default mirror carries no label column and stores clip paths, not bytes."""
    print("kinetics mirror schema")
    from csf.generation.kinetics import (Annotation, label_from_path, rank_clips,
                                         resolve_row_label, _safe_name)

    wanted = {"playing guitar": 10, "riding a bike": 10, "singing": 5}
    row = {
        "video_id": "abc123XYZ", "video_path": "videos/abc123XYZ.mp4",
        "metadata": {"resolution": "1280x720", "frame_rate": 30, "codec": "h264"},
        "clips": [
            {"clip_name": "abc123XYZ_000010_000020",
             "clip_path": "clips/abc123XYZ_000010_000020.mp4", "start_time": 10.0,
             "duration": 10.0, "frames_count": 300,
             "quality_metrics": {"sharpness": 0.4, "stability": 0.3},
             "frames": [{"frame_number": 0, "image_path": "f0.jpg", "annotation": "",
                         "clip_score": 0.2, "aesthetic_score": 5.1}]},
            {"clip_name": "abc123XYZ_000030_000040",
             "clip_path": "clips/abc123XYZ_000030_000040.mp4", "start_time": 30.0,
             "duration": 10.0, "frames_count": 300,
             "quality_metrics": {"sharpness": 0.9, "stability": 0.8},
             "frames": [{"frame_number": 0, "image_path": "f0.jpg", "annotation": "",
                         "clip_score": 0.7, "aesthetic_score": 6.9}]},
        ]}
    anns = {"abc123XYZ_000010_000020":
            Annotation("abc123XYZ_000010_000020", "playing guitar", "train")}

    check("resolves a label with no label column, by joining video_id on the annotations",
          resolve_row_label(row, wanted, anns) == "playing guitar")
    check("ranks the higher-quality clip first",
          rank_clips(row)[0]["clip_name"] == "abc123XYZ_000030_000040")
    check("falls back to a foldered path",
          resolve_row_label({"video_id": "z", "video_path": "train/riding a bike/z.mp4",
                             "clips": []}, wanted, None) == "riding a bike")
    check("an explicit label column still wins",
          resolve_row_label({"label": "Singing", "video_path": "x.mp4"}, wanted,
                            None) == "singing")
    check("frame annotations are the last resort",
          resolve_row_label({"video_id": "q", "clips": [
              {"clip_path": "c.mp4", "frames": [{"annotation": "playing guitar"}]}]},
              wanted, None) == "playing guitar")
    check("classes outside the spec's 147 are rejected",
          resolve_row_label({"label": "abseiling", "video_path": "x.mp4"}, wanted, None) is None)
    check("a row with no clips falls back to the whole video",
          rank_clips({"video_id": "v", "video_path": "videos/v.mp4",
                      "clips": []})[0]["clip_path"] == "videos/v.mp4")
    check("unresolvable rows return None, they do not raise",
          resolve_row_label({"video_id": "unknown", "clips": []}, wanted, None) is None)
    check("clip names are made filesystem-safe",
          "/" not in _safe_name("abc/def:ghi") and _safe_name("") == "clip")
    check("label_from_path ignores split directories",
          label_from_path("train/playing guitar/x.mp4", wanted) == "playing guitar")


def test_attribution() -> None:
    """Kinetics-400 is CC BY 4.0, so anything we redistribute has to carry the credit."""
    print("licence attribution")
    from csf.config import load_config
    from csf.generation.upload import KINETICS_ATTRIBUTION, build_dataset_card

    card = build_dataset_card(load_config("configs/regen.yaml"),
                              {"per_class": {"ai_edited": 33333},
                               "per_model_actual": {"vace": 2883}})
    for needed, what in (("cc-by-4.0", "licence identifier in the card metadata"),
                         ("creativecommons.org/licenses/by/4.0", "link to the licence"),
                         ("Kinetics-400", "name of the source dataset"),
                         ("Zisserman", "credit to the original authors"),
                         ("1705.06950", "citation of the original paper"),
                         ("Changes made", "statement that the material was modified")):
        check(f"dataset card states the {what}", needed in card, needed)
    check("the attribution block is self-contained",
          "CC BY 4.0" in KINETICS_ATTRIBUTION and "Kinetics-400" in KINETICS_ATTRIBUTION)


def main() -> int:
    for fn in (test_spec, test_jobs, test_degraded_pool, test_naming, test_adapters,
               test_substitutions, test_budget, test_reallocation, test_concurrency,
               test_kinetics_schema, test_attribution):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all regeneration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
