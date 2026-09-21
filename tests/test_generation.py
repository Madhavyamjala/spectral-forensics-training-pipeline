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

import inspect
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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

    # every source label must be a real Kinetics-400 class. A label the spec invents downloads
    # nothing and silently starves the source groups that reference it, so it is a hard failure
    # here rather than a "below quota" warning eight hours into a run.
    classes = S.known_classes()
    check("the checked-in Kinetics-400 class list holds 400 names", len(classes) == 400,
          str(len(classes)))
    unknown = [lbl for lbl in S.all_labels() if lbl not in classes]
    check("every spec label is a real Kinetics-400 class", not unknown, str(unknown))
    for invented, real in (("playing golf", "golf driving"), ("boxing", "punching person (boxing)"),
                           ("sitting up", "situp"), ("riding a segway", "using segway"),
                           ("painting", "brush painting"), ("hiking", "marching"),
                           ("repairing puncture", "checking tires")):
        check(f"'{invented}' is not used; upstream spells it '{real}'",
              invented not in classes and real in classes)

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
    check("classes outside the spec's 149 are rejected",
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


def test_metadata() -> None:
    """Per-video metadata must land as real columns, not a truncated JSON blob."""
    print("per-video metadata")
    import json as _json

    from csf.generation.metadata import (CONTAINER_COLUMNS, IDENTITY_COLUMNS, coverage,
                                         metadata_columns, metadata_row, spec_metadata_columns)

    cols = metadata_columns()
    check("columns are unique", len(cols) == len(set(cols)))
    check("every family's spec fields are columns",
          all(f in cols for fam in S.FAMILY_LIST for f in fam.metadata_fields),
          str([f for fam in S.FAMILY_LIST for f in fam.metadata_fields if f not in cols]))
    check("identity and container groups are present",
          all(c in cols for c in IDENTITY_COLUMNS + CONTAINER_COLUMNS))
    check("the spec contributes a substantial share", len(spec_metadata_columns()) >= 40,
          str(len(spec_metadata_columns())))

    probed = {"duration_sec": 4.0, "width": 640, "height": 360, "fps": 25.0,
              "codec": "h264", "bitrate": 500000, "has_audio": True}
    job = Job(job_id="j", video_id="aiedit-lip_sync-3", family="lip_sync",
              model="videoretalking", source_group="g", source_clip_id="c",
              source_path="/x.mp4", source_label="singing", split="valid",
              audio_clip_id="donor", seed=7,
              metadata=_json.dumps({"expression_intensity": 0.9, "language": "unknown",
                                    "speech_duration": 1.0}))

    class _P:
        def exists(self):
            return True

        def stat(self):
            return type("S", (), {"st_size": 4242})()

    rendered = {"lip_sync_model": "latentsync", "audio_source": "donor",
                "speech_duration": 6.4, "substitutes_for": "videoretalking"}
    row = metadata_row(job, _P(), probed, "deadbeef", rendered, render_seconds=11.5)

    check("the renderer's value beats the planned one",
          str(row["speech_duration"]) == "6.4", str(row["speech_duration"]))
    check("planned values survive where nothing was measured",
          str(row["expression_intensity"]) == "0.9", str(row["expression_intensity"]))
    check("container fields come from the probed file", row["bitrate"] == 500000)
    check("substitution is recorded",
          row["model"] == "latentsync" and row["spec_model"] == "videoretalking"
          and row["substituted"] is True)
    check("render time is recorded", row["render_seconds"] == 11.5)
    check("file size and digest are recorded",
          row["file_bytes"] == 4242 and row["sha256"] == "deadbeef")
    check("unknown renderer keys are dropped, not crammed in",
          "substitutes_for" not in row)
    check("every column is a scalar a CSV cell can hold",
          all(isinstance(v, (str, int, float, bool)) for v in row.values()),
          str([k for k, v in row.items() if not isinstance(v, (str, int, float, bool))]))

    cov = coverage([row])
    check("coverage reports per-family fill rates", "lip_sync" in
          cov["per_family_field_fill_rate"])
    check("coverage is honest about unfilled fields",
          cov["per_family_field_fill_rate"]["lip_sync"]["occlusion_level"] == 0.0)
    print(f"       {len(cols)} columns, {len(spec_metadata_columns())} from the specification")


def test_metadata_merge() -> None:
    """Regenerating replaces the old AI-Edited metadata without touching the other classes."""
    print("metadata merge")
    import csv as _csv
    import tempfile

    from csf.generation.metadata import merge_metadata, read_metadata, write_metadata

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metadata.csv"
        old_cols = ["video_id", "class", "duration_sec", "legacy_note"]
        old_rows = (
            [{"video_id": f"real-{i}", "class": "real", "duration_sec": 2.0,
              "legacy_note": "keep"} for i in range(4)]
            + [{"video_id": f"aigen-{i}", "class": "ai_generated", "duration_sec": 3.0,
                "legacy_note": "keep"} for i in range(3)]
            + [{"video_id": f"aiedit-old-{i}", "class": "ai_edited", "duration_sec": 4.0,
                "legacy_note": "stale"} for i in range(5)])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=old_cols)
            w.writeheader()
            w.writerows(old_rows)

        new_rows = [{"video_id": f"aiedit-face_swap-{i}", "class": "ai_edited",
                     "family": "face_swap", "model": "inswapper", "yaw": -3.1}
                    for i in range(6)]

        res = merge_metadata(new_rows, path)
        check("old ai_edited rows are dropped", res["dropped_ai_edited"] == 5,
              str(res["dropped_ai_edited"]))
        check("real and ai_generated rows are kept", res["kept_other_classes"] == 7,
              str(res["kept_other_classes"]))

        write_metadata(new_rows, path)
        final = read_metadata(path)
        counts = Counter(r["class"] for r in final)
        check("the other two classes survive the rewrite",
              counts["real"] == 4 and counts["ai_generated"] == 3, str(dict(counts)))
        check("only the regenerated ai_edited rows remain", counts["ai_edited"] == 6,
              str(counts["ai_edited"]))
        check("no stale ai_edited id survives",
              not any(r["video_id"].startswith("aiedit-old-") for r in final))
        check("a column only the old file had is preserved",
              any(r.get("legacy_note") == "keep" for r in final))
        check("the new schema's columns are present", "yaw" in final[0])

        write_metadata(new_rows, path)
        check("re-running is idempotent", len(read_metadata(path)) == len(final),
              f"{len(final)} -> {len(read_metadata(path))}")

        kept = merge_metadata(new_rows, path, keep_old_edited=True)
        check("keep_old_edited retains both sets",
              sum(1 for r in kept["rows"] if str(r.get("class")) == "ai_edited") == 6)

        missing = merge_metadata(new_rows, Path(tmp) / "absent.csv")
        check("an absent file is not an error",
              missing["existing"] == 0 and len(missing["rows"]) == 6)


def test_stage_scoping() -> None:
    """Generation-only runs must not be gated on the training stack."""
    print("stage scoping")
    import inspect

    import main as driver

    check("generation and training stages are disjoint",
          not (set(driver.GENERATION_STAGES) & set(driver.TRAINING_STAGES)))
    check("STAGES is exactly their union",
          driver.STAGES == driver.GENERATION_STAGES + driver.TRAINING_STAGES)
    check("preflight takes the selected stages",
          "selected" in inspect.signature(driver.preflight).parameters)
    src = inspect.getsource(driver.preflight)
    check("the training stack is only demanded when training",
          "if training:" in src and src.index("if training:") < src.index("import peft"))
    check("bitsandbytes is gated on training too",
          src.index("if training:") < src.index("import bitsandbytes"))
    check("the generation stages require ffmpeg", "ffprobe" in src)
    check("generation.gpus is validated", "_check_generation_gpus" in src)


def test_probe_fallback() -> None:
    """A missing ffprobe must degrade the metadata, not empty the source pool."""
    print("container probe")
    import shutil as _shutil
    import subprocess as _sp
    import tempfile

    import csf.generation.kinetics as K

    original_run = _sp.run
    original_cv = K._probe_with_opencv
    try:
        # ffprobe absent, OpenCV able to read: must still return metadata
        _sp.run = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("ffprobe"))
        K._probe_with_opencv = lambda p: {"duration_sec": 10.0, "width": 340, "height": 256,
                                          "fps": 25.0, "codec": "unknown", "bitrate": 0,
                                          "has_audio": False, "probe": "opencv"}
        K._FFPROBE_WARNED = False
        meta = K.probe_video(Path("clip.mp4"))
        check("falls back to OpenCV when ffprobe is missing",
              meta is not None and meta.get("probe") == "opencv")
        check("the fallback still yields usable geometry",
              meta["width"] == 340 and meta["fps"] == 25.0)

        # neither backend can read it: None, so the caller can drop the clip
        K._probe_with_opencv = lambda p: None
        check("an undecodable file still returns None", K.probe_video(Path("bad.mp4")) is None)
    finally:
        _sp.run = original_run
        K._probe_with_opencv = original_cv

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # nothing downloaded
        pool = K.SourcePool(root / "a")
        try:
            pool.write_pool(["playing guitar"])
            check("an empty directory raises", False)
        except RuntimeError as exc:
            check("an empty directory says nothing was downloaded",
                  "No clips were downloaded" in str(exc), str(exc)[:80])

        # files present, none decodable - the case that used to say "run the kinetics stage"
        pool = K.SourcePool(root / "b")
        d = pool.clips_dir / "playing guitar"
        d.mkdir(parents=True)
        for i in range(5):
            (d / f"c{i}.mp4").write_bytes(b"not a video")
        try:
            pool.write_pool(["playing guitar"])
            check("undecodable clips raise", False)
        except RuntimeError as exc:
            msg = str(exc)
            check("the message reports how many clips are present", "5 clip(s) are present" in msg,
                  msg[:90])
            check("it does not tell you to re-run the stage you are in",
                  "Run the 'kinetics' stage first" not in msg)
            check("it names ffmpeg as the usual cause", "ffmpeg" in msg)
            check("it offers probe_clips=false as an escape", "probe_clips=false" in msg)

        # probing disabled: the pool builds regardless
        pool = K.SourcePool(root / "c")
        d = pool.clips_dir / "playing guitar"
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"c{i}.mp4").write_bytes(b"x" * 10)
        out = pool.write_pool(["playing guitar"], probe=False)
        import csv as _csv
        with open(out, newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        check("probe_clips=false still writes a usable pool", len(rows) == 3, str(len(rows)))


def test_ffmpeg_resolution() -> None:
    """ffmpeg must be usable without root, conda, or anything on PATH."""
    print("ffmpeg resolution")
    import subprocess as _sp
    import sys as _sys
    import types as _types

    import csf.generation.ffmpeg_tools as F

    orig_usable, orig_run = F._usable, _sp.run
    orig_module = _sys.modules.get("imageio_ffmpeg")
    try:
        fake = _types.ModuleType("imageio_ffmpeg")
        fake.get_ffmpeg_exe = lambda: "/opt/fake/ffmpeg"
        _sys.modules["imageio_ffmpeg"] = fake
        F._usable = lambda p: "/opt/fake/ffmpeg" if p == "/opt/fake/ffmpeg" else None
        F.ffmpeg_exe.cache_clear()
        F.ffprobe_exe.cache_clear()
        check("falls back to the imageio-ffmpeg binary",
              F.ffmpeg_exe() == "/opt/fake/ffmpeg", str(F.ffmpeg_exe()))
        check("have_ffmpeg is true with only the bundled binary", F.have_ffmpeg())
        check("ffprobe stays None, since imageio-ffmpeg ships none",
              F.ffprobe_exe() is None)

        F._usable = lambda p: p or None
        os.environ["CSF_FFMPEG"] = "/custom/ffmpeg"
        F.ffmpeg_exe.cache_clear()
        check("an explicit CSF_FFMPEG wins", F.ffmpeg_exe() == "/custom/ffmpeg")

        sample = ("  Duration: 00:00:10.05, start: 0.000000, bitrate: 502 kb/s\n"
                  "  Stream #0:0(und): Video: h264 (High), yuv420p, 340x256 "
                  "[SAR 1:1 DAR 85:64], 497 kb/s, 25 fps, 25 tbr\n"
                  "  Stream #0:1(und): Audio: aac (LC), 44100 Hz, stereo, fltp, 128 kb/s")
        _sp.run = lambda *a, **k: _types.SimpleNamespace(stderr=sample, stdout="", returncode=1)
        meta = F.probe_with_ffmpeg(Path("x.mp4"))
        check("ffmpeg -i yields geometry without ffprobe",
              meta and meta["width"] == 340 and meta["height"] == 256, str(meta))
        check("it parses duration, fps and codec",
              meta["duration_sec"] == 10.05 and meta["fps"] == 25.0
              and meta["codec"] == "h264", str(meta))
        check("it detects the audio track", meta["has_audio"] is True)
        check("the backend is labelled", meta.get("probe") == "ffmpeg")
    finally:
        os.environ.pop("CSF_FFMPEG", None)
        F._usable, _sp.run = orig_usable, orig_run
        if orig_module is None:
            _sys.modules.pop("imageio_ffmpeg", None)
        else:
            _sys.modules["imageio_ffmpeg"] = orig_module
        F.ffmpeg_exe.cache_clear()
        F.ffprobe_exe.cache_clear()

    # the workers carry their own copy, since they cannot import csf
    common = (Path(__file__).resolve().parent.parent / "csf" / "generation" / "adapters"
              / "workers" / "_common.py").read_text(encoding="utf-8")
    check("workers resolve ffmpeg rather than hardcoding it",
          'subprocess.run(["ffmpeg"' not in common and '["ffprobe"' not in common)
    check("workers fall back to imageio-ffmpeg too", "imageio_ffmpeg" in common)
    check("workers honour CSF_FFMPEG", "CSF_FFMPEG" in common)


#: What each renderer dereferences from its job payload. Kept next to the tests rather than in
#: the registry because it is an assertion about the workers, not configuration: a worker that
#: needs an input the planner never binds fails on every single job, and only at render time.
WORKER_REQUIREMENTS = {
    "inswapper": ["driving_path"],            # identity donor
    "reface": ["driving_path"],
    "dreamid_v": ["driving_path"],
    "fomm": ["driving_path"],                 # driving motion
    "tpsmm": ["driving_path"],
    "wav2lip": ["audio_path"],                # speech donor
    "latentsync": ["audio_path"],
    "musetalk": ["audio_path"],
    "sadtalker": ["audio_path"],
    "bg_real_composite": ["driving_path"],    # the background plate
    "bg_flux_image": ["prompt"],
    "bg_svd_video": ["prompt"],
    "tokenflow": ["prompt"],
    "styleganex": ["variant"],
    "liveportrait_expr": ["variant"],
    "propainter_inpaint": ["mask_size", "mask_motion"],
    "e2fgvi_hq": ["mask_size", "mask_motion"],
    "sttn": ["mask_size", "mask_motion"],
    "fuseformer": ["mask_size", "mask_motion"],
}
#: Slots whose renderer needs a *second* clip, which must differ from the target.
DISTINCT_SECOND_CLIP = {
    "driving_path": ["inswapper", "reface", "dreamid_v", "fomm", "tpsmm", "bg_real_composite"],
    "audio_path": ["wav2lip", "latentsync", "musetalk", "sadtalker"],
}


def test_worker_inputs() -> None:
    """Every wired renderer must actually receive what it dereferences."""
    print("worker inputs")
    allowed = {k for k, a in ADAPTERS.items() if a.implemented}
    jobs = build_jobs(synthetic_pool(per_label=400), seed=42, allowed_models=allowed)

    by_renderer = defaultdict(list)
    for job in jobs:
        by_renderer[ADAPTERS[job.model].runs].append(job)

    for renderer, fields in sorted(WORKER_REQUIREMENTS.items()):
        rows = by_renderer.get(renderer)
        if not rows:
            continue
        for field in fields:
            missing = [j.video_id for j in rows if not getattr(j, field, "")]
            check(f"{renderer} always receives {field}", not missing,
                  f"{len(missing)}/{len(rows)} jobs missing it, e.g. {missing[:2]}")

    for field, renderers in DISTINCT_SECOND_CLIP.items():
        key = "driving_clip_id" if field == "driving_path" else "audio_clip_id"
        for renderer in renderers:
            rows = by_renderer.get(renderer)
            if not rows:
                continue
            same = [j.video_id for j in rows if getattr(j, key) == j.source_clip_id]
            check(f"{renderer}'s second clip differs from the target", not same,
                  f"{len(same)} jobs reuse the source, e.g. {same[:2]}")

    # the generative background pipelines must not be handed a plate they would ignore
    for renderer in ("bg_flux_image", "bg_svd_video", "bg_propainter_recon"):
        rows = by_renderer.get(renderer, [])
        check(f"{renderer} is not given a redundant background plate",
              not [j for j in rows if j.driving_path])

    covered = set(WORKER_REQUIREMENTS) | {"propainter_object", "bg_propainter_recon",
                                          "diffueraser", "vace", "liveportrait"}
    unchecked = {ADAPTERS[k].runs for k in allowed} - covered
    check("every wired renderer is accounted for", not unchecked, str(sorted(unchecked)))
    print(f"       {len(by_renderer)} renderers, {len(jobs):,} jobs checked")


def test_progress() -> None:
    """Long stages must show they are alive, and keep working without a terminal."""
    print("progress reporting")
    import inspect

    from csf.generation import progress as P

    check("duration formatting is human", P._fmt_duration(75) == "1m15s"
          and P._fmt_duration(3725) == "1h02m", P._fmt_duration(3725))

    with P.bar(10, "unit test", "item", log_every=1000) as handle:
        for _ in range(10):
            handle.update(1)
        handle.set_postfix_str("x")
        check("the bar counts", getattr(handle, "n", 10) == 10)

    proc = P.run_streaming(["bash", "-c", "echo first; echo second"], desc="stub")
    check("run_streaming returns the exit status", proc.returncode == 0)
    check("run_streaming keeps a tail for error reporting", "second" in proc.stdout)
    proc = P.run_streaming(["bash", "-c", "echo boom >&2; exit 3"], desc="stub")
    check("a failing command reports its status and output",
          proc.returncode == 3 and "boom" in proc.stdout)

    with P.Heartbeat("stub", interval=0.05) as beat:
        beat.set_status("working")
        time.sleep(0.15)
    check("the heartbeat tracks elapsed time", beat.elapsed > 0)

    # the stages a long run spends its time in must all report progress
    from csf.generation import envs, filters, kinetics, manifest_build, prefetch, scheduler
    for module, name in ((envs, "envs"), (filters, "filters"), (kinetics, "kinetics"),
                         (manifest_build, "manifest_build"), (prefetch, "prefetch"),
                         (scheduler, "scheduler")):
        src = inspect.getsource(module)
        check(f"{name} reports progress",
              "progress.bar" in src or "progress_ui.bar" in src or "progress.track" in src
              or "Heartbeat" in src or "run_streaming" in src)

    check("the environment build streams rather than buffering",
          "run_streaming" in inspect.getsource(envs._run))
    check("clip scoring checkpoints so it can resume",
          "checkpoint()" in inspect.getsource(filters.score_pool))
    check("progress can be silenced for headless runs",
          "CSF_NO_PROGRESS" in inspect.getsource(P))


def test_env_paths() -> None:
    """Env paths must be absolute: workers are launched with cwd set to the env root."""
    print("environment paths")
    import inspect
    import shutil as _shutil
    import tempfile

    import csf.generation.envs as E
    from csf.generation.adapters import base as adapter_base

    with tempfile.TemporaryDirectory() as tmp:
        rel = os.path.relpath(tmp, os.getcwd())
        original = E._run
        try:
            def fake_run(cmd, cwd=None, env=None, timeout=3600, what=""):
                if "virtual environment" in what or "venv" in what:
                    venv_dir = Path(cmd[-1])
                    py = E._venv_python(venv_dir)
                    py.parent.mkdir(parents=True, exist_ok=True)
                    # stand in for a real interpreter: the build probes it with
                    # `-c "import sys; print(sys.prefix)"` to confirm it runs as its own venv
                    py.write_text(f'#!/bin/sh\necho "{venv_dir.resolve()}"\n')
                    py.chmod(0o755)
            E._run = fake_run
            ready = E.build_env(E.EnvSpec(name="probe", torch="", requirements=()), Path(rel))
        finally:
            E._run = original

        check("the env root is absolute", Path(ready.root).is_absolute(), str(ready.root))
        check("the interpreter path is absolute", Path(ready.python).is_absolute(),
              str(ready.python))
        # the actual failure: the worker is spawned with cwd=env.root
        check("the interpreter resolves from inside the env root",
              (Path(ready.root) / ready.python).exists() or Path(ready.python).exists())

    # the same trap as the interpreter: a worker's cwd is its env root, so a relative video
    # root would put every mp4 under cache/.../envs/<env>/ and the driver would record "ok"
    # for a file it cannot find
    from csf.generation.scheduler import GenerationScheduler
    sched = GenerationScheduler(video_root=Path("cache/rel/videos"), envs_root=Path("cache/envs"),
                                log_dir=Path("logs"), gpus=[0])
    check("the video root is absolute", sched.video_root.is_absolute(), str(sched.video_root))
    check("output paths are absolute",
          sched.output_path(Job(job_id="j", video_id="v", family="face_swap", model="inswapper",
                                source_group="g", source_clip_id="c", source_path="c.mp4",
                                source_label="singing")).is_absolute())

    check("the worker is launched with cwd set to the env root",
          "cwd=str(self.env.root)" in inspect.getsource(adapter_base.WorkerProcess.start))
    check("a missing interpreter is refused before spawning",
          "is missing at" in inspect.getsource(adapter_base.WorkerPool._get_locked))
    check("an env is not marked ready without its interpreter",
          "interpreter is missing at" in inspect.getsource(E.build_env))


def test_no_job_left_behind() -> None:
    """However a worker group ends, every job it held must reach the ledger."""
    print("job accounting")
    import inspect

    from csf.generation import scheduler as S_

    src = inspect.getsource(S_.GenerationScheduler._run_group)
    check("slot failures are caught rather than killing the thread",
          "except Exception as exc:" in src and "_slot_loop" in src)
    check("an unexpected slot crash drains the queue", "worker slot crashed" in src)
    check("the group sweeps anything left unattempted", "never attempted" in src)
    check("OSError from a missing interpreter is handled",
          "AdapterError, EnvBuildError, OSError" in src)
    check("workers are capped at the number of jobs", "min(len(jobs)," in src)


def test_retry_policy() -> None:
    """A failed job is picked back up on the next run, up to a cap."""
    print("retry policy")
    import tempfile

    from csf.generation.scheduler import Ledger, Outcome, group_jobs

    def job(n: int, model: str = "inswapper") -> Job:
        return Job(job_id=f"j{n}", video_id=f"v{n}", family="face_swap", model=model,
                   source_group="g", source_clip_id=f"c{n}", source_path=f"/tmp/c{n}.mp4",
                   source_label="singing")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.jsonl"
        ledger = Ledger(path)
        jobs = [job(0), job(1), job(2)]
        ledger.record(Outcome(job_id="j0", ok=True, output_path="/tmp/v0.mp4", error="",
                              seconds=1.0, metadata={}))
        ledger.record(Outcome(job_id="j1", ok=False, output_path="", error="env build failed",
                              seconds=1.0, metadata={}))

        groups = group_jobs(jobs, ledger, retry_failed=True, max_attempts=3)
        pending = {j.job_id for v in groups.values() for j in v}
        check("a succeeded job is never re-run", "j0" not in pending)
        check("a failed job is retried by default", "j1" in pending)
        check("an unattempted job is still pending", "j2" in pending)

        check("retry_failed=False keeps the old one-shot behaviour",
              "j1" not in {j.job_id for v in group_jobs(jobs, ledger, retry_failed=False).values()
                           for j in v})

        # burn the remaining attempts
        for _ in range(2):
            ledger.record(Outcome(job_id="j1", ok=False, output_path="", error="env build failed",
                                  seconds=1.0, metadata={}))
        check("attempts accumulate across records", ledger.attempt_count("j1") == 3)
        check("a job that used up its attempts stops being retried",
              "j1" not in {j.job_id for v in group_jobs(jobs, ledger, max_attempts=3).values()
                           for j in v})
        check("raising max_attempts brings it back",
              "j1" in {j.job_id for v in group_jobs(jobs, ledger, max_attempts=4).values()
                       for j in v})

        # a reopened ledger must see the same history, not a blank slate
        reopened = Ledger(path)
        check("attempt counts survive a restart", reopened.attempt_count("j1") == 3)
        check("success survives a restart", reopened.succeeded("j0"))
        reasons = reopened.failure_reasons()
        check("failures are tallied by message",
              reasons and reasons[0] == ("env build failed", 1), str(reasons))

        # an env that will not build is not the job's fault: record it, but do not spend the
        # job's attempts on it, or an offline node permanently strands work that would run
        env_path = Path(tmp) / "envledger.jsonl"
        env_ledger = Ledger(env_path)
        for _ in range(4):
            env_ledger.record(Outcome(job_id="j9", ok=False, env_error=True,
                                      error="worker unavailable: EnvBuildError"))
        check("an env failure costs no attempts", env_ledger.attempt_count("j9") == 0)
        check("but the job is still on record", env_ledger.seen("j9"))
        check("an env failure does not block a retry",
              "j9" in {j.job_id for v in group_jobs([job(9)], env_ledger).values() for j in v})
        env_ledger.record(Outcome(job_id="j9", ok=False, error="the model crashed"))
        check("a real failure still counts", env_ledger.attempt_count("j9") == 1)

        # ledgers written before env_error existed still carry the distinction in their text
        legacy = Path(tmp) / "legacy.jsonl"
        legacy.write_text("".join(
            '{"job_id": "j8", "ok": false, "error": "worker unavailable: no pip"}\n'
            for _ in range(3)), encoding="utf-8")
        check("a legacy 'worker unavailable' row is read as an env failure",
              Ledger(legacy).attempt_count("j8") == 0)
        check("and the job is retried rather than left exhausted",
              "j8" in {j.job_id for v in group_jobs([job(8)], Ledger(legacy)).values() for j in v})

        # a success whose video has vanished must be regenerated, not skipped forever
        gone = Path(tmp) / "gone.mp4"
        present = Path(tmp) / "present.mp4"
        present.write_bytes(b"x" * 16)
        out_ledger = Ledger(Path(tmp) / "out.jsonl")
        out_ledger.record(Outcome(job_id="j0", ok=True, output_path=str(gone)))
        out_ledger.record(Outcome(job_id="j1", ok=True, output_path=str(present)))
        paths = {"j0": gone, "j1": present}
        pending = {j.job_id for v in group_jobs([job(0), job(1)], out_ledger,
                                                output_for=lambda j: paths[j.job_id]).values()
                   for j in v}
        check("a recorded success with no file is re-generated", "j0" in pending)
        check("a recorded success whose file is there is left alone", "j1" not in pending)

        # a job that failed and later succeeded must not be reported as a failure
        reopened.record(Outcome(job_id="j1", ok=True, output_path="/tmp/v1.mp4", error="",
                                seconds=1.0, metadata={}))
        check("a later success clears the recorded error", not reopened.failure_reasons())


def test_stage_staleness() -> None:
    """A completed stage must re-run when the inputs it depended on have changed."""
    print("stage staleness")
    from csf.generation import run as R

    class _K:
        demand_margin = 1.2
        rescore = False

    class _G:
        total_videos = 33333
        kinetics = _K()

    # a cache with the stage's outputs present, so these checks isolate the fingerprint rule
    import tempfile as _tf
    _cache = _tf.mkdtemp()
    _kin = Path(_cache) / "kinetics"
    _kin.mkdir()
    for _name in ("source_pool.csv", "clip_features.csv"):
        (_kin / _name).write_text("clip_id\nabc\n")

    class _P:
        cache_dir = _cache

    class _C:
        generation = _G()
        paths = _P()

    cfg = _C()
    fp = R.kinetics_fingerprint(cfg)
    check("the fingerprint pins the label set", fp["n_labels"] == len(S.all_labels()))
    check("nothing is stale when the fingerprint matches",
          R.stale_reason("kinetics", cfg, {"result": {"fingerprint": dict(fp)}}) is None)

    stale = dict(fp, labels_sha="deadbeefdeadbeef")
    reason = R.stale_reason("kinetics", cfg, {"result": {"fingerprint": stale}})
    check("a changed label set forces the stage to re-run", bool(reason) and "labels_sha" in reason)

    _K.rescore = True
    check("rescore=true overrides the completed marker",
          R.stale_reason("kinetics", cfg, {"result": {"fingerprint": dict(fp)}}) is not None)
    _K.rescore = False

    check("a state.json written before fingerprints existed is trusted",
          R.stale_reason("kinetics", cfg, {"result": {"clips": 200}}) is None)

    # a completed marker whose output is gone must not be honoured: state.json outlives the
    # cache it describes, and skipping on the marker alone defers the failure to a later stage
    with _tf.TemporaryDirectory() as tmp:
        class _P2:
            cache_dir = tmp
        cfg_real = type("C", (), {"generation": _G(), "paths": _P2()})()
        fp = R.kinetics_fingerprint(cfg_real)
        done = {"result": {"fingerprint": dict(fp)}}
        reason = R.stale_reason("kinetics", cfg_real, done)
        check("a marker with no output on disk is refused", bool(reason) and "missing" in reason,
              str(reason))
        check("the reason names the files it looked for",
              "clip_features.csv" in (reason or ""))

        kin = Path(tmp) / "kinetics"
        kin.mkdir()
        (kin / "source_pool.csv").write_text("clip_id\n")
        (kin / "clip_features.csv").write_text("")          # present but empty
        check("an empty artifact counts as missing",
              "clip_features.csv" in (R.stale_reason("kinetics", cfg_real, done) or ""))
        (kin / "clip_features.csv").write_text("clip_id\nabc\n")
        check("a complete cache with a matching fingerprint is honoured",
              R.stale_reason("kinetics", cfg_real, done) is None)
        check("missing_artifacts reports nothing when both files are there",
              R.missing_artifacts("kinetics", cfg_real) == [])
    check("other stages are unaffected",
          R.stale_reason("generate", cfg, {"result": {"fingerprint": stale}}) is None)

    src = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    check("main.py consults the staleness check before skipping",
          "_stale_reason(name, cfg, state.info(name))" in src)


def test_env_interpreter() -> None:
    """The venv interpreter path must stay inside the venv, symlink and all."""
    print("env interpreter")
    import subprocess
    import tempfile

    from csf.generation.envs import EnvSpec, _venv_is_sane, _venv_python, diagnose

    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "env" / "venv"
        (venv / "bin").mkdir(parents=True)
        # a venv's bin/python is a symlink to the base interpreter; resolving it walks out of
        # the environment and silently runs the system Python instead (no pip, no packages).
        link = venv / "bin" / "python"
        link.symlink_to(sys.executable)
        py = _venv_python(venv)
        check("the interpreter path stays inside the venv", str(py).startswith(str(venv)),
              str(py))
        check("the path is absolute", py.is_absolute())
        check("the symlink is not followed", py.resolve() != py or not link.is_symlink())
        check("a venv with no pyvenv.cfg is reported as not its own venv",
              not _venv_is_sane(py, venv))

        spec = EnvSpec(name="env", torch="", requirements=())
        report = diagnose([spec], Path(tmp))["env"]
        check("the doctor reports a broken env as broken", report["verdict"] == "broken",
              str(report))
        check("the doctor names the interpreter it checked", report["python"] == str(py))

        missing = diagnose([EnvSpec(name="nope", torch="")], Path(tmp))["nope"]
        check("an env that was never built reads as missing", missing["verdict"] == "missing")

    # a worker must not inherit the driver's interpreter state: the driver runs from its own
    # venv (often inside conda), and its PYTHONPATH would shadow the env's pinned packages
    saved = {k: os.environ.get(k) for k in ("PYTHONPATH", "PYTHONHOME")}
    try:
        os.environ["PYTHONPATH"] = "/driver/site-packages"
        os.environ["PYTHONHOME"] = "/conda/base"
        from csf.generation.envs import ReadyEnv
        ready = ReadyEnv(EnvSpec(name="x", torch=""), Path("/envs/x"),
                         Path("/envs/x/venv/bin/python"), {"r": Path("/envs/x/repos/r")})
        env = ready.environ()
        check("the driver's PYTHONPATH does not reach the worker",
              env["PYTHONPATH"] == str(Path("/envs/x/repos/r")), env.get("PYTHONPATH", ""))
        check("PYTHONHOME is dropped", "PYTHONHOME" not in env)
        check("user site-packages are switched off", env.get("PYTHONNOUSERSITE") == "1")
        check("the env's bin dir leads PATH",
              env["PATH"].split(os.pathsep)[0] == "/envs/x/venv/bin")
        check("VIRTUAL_ENV points at the venv", env.get("VIRTUAL_ENV") == "/envs/x/venv")
        no_repos = ReadyEnv(EnvSpec(name="x", torch=""), Path("/envs/x"),
                            Path("/envs/x/venv/bin/python"), {}).environ()
        check("an env with no repos sets no PYTHONPATH at all", "PYTHONPATH" not in no_repos)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # a requirement must not be able to swap the pinned torch out from under the env
    import csf.generation.envs as E_
    with tempfile.TemporaryDirectory() as tmp:
        spec = EnvSpec(name="pinned", torch="torch==2.5.1 torchvision==0.20.1")
        check("torch pins are read off the spec",
              E_.torch_pins(spec) == ["torch==2.5.1", "torchvision==0.20.1"])
        written = E_._write_constraints(spec, Path(tmp))
        check("a constraints file pins every torch package",
              written.read_text().split() == ["torch==2.5.1", "torchvision==0.20.1"])
        check("an env with no torch needs no constraints",
              E_._write_constraints(EnvSpec(name="x", torch=""), Path(tmp)) is None)

        # a stub interpreter reporting the wrong torch must fail the build
        stub = Path(tmp) / "python"
        stub.write_text('#!/bin/sh\necho "2.14.0+cu130"\n')
        stub.chmod(0o755)
        try:
            E_._verify_torch(stub, spec)
            check("a drifted torch fails the build", False, "no error raised")
        except E_.EnvBuildError as exc:
            check("a drifted torch fails the build, naming both versions",
                  "2.5.1" in str(exc) and "2.14.0" in str(exc), str(exc)[:120])
        stub.write_text('#!/bin/sh\necho "2.5.1+cu121"\n')
        stub.chmod(0o755)
        E_._verify_torch(stub, spec)          # the pinned build passes, local CUDA tag and all

    build_src_ = inspect.getsource(E_.build_env)
    check("the constraints file is handed to pip for the requirements install",
          'build_env_vars["PIP_CONSTRAINT"]' in build_src_)
    check("the torch pin is verified before the env is marked ready",
          "_verify_torch(py, spec)" in build_src_)
    check("sam2_diffusers pins a torch that satisfies SAM2's floor",
          env_specs()["sam2_diffusers"].torch.startswith("torch==2.5.1"),
          env_specs()["sam2_diffusers"].torch)

    # burning: only registered environments, nothing else living under the envs root
    from csf.generation.envs import burn_envs
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("insightface", "sam2_diffusers", "not_an_env"):
            (root / name / "venv").mkdir(parents=True)
            (root / name / "venv" / "blob").write_bytes(b"x" * 1024)
        specs = list(env_specs().values())
        freed = burn_envs(specs, root, keep=("sam2_diffusers",))
        check("burn removes a registered env", "insightface" in freed
              and not (root / "insightface").exists())
        check("burn honours keep", (root / "sam2_diffusers").exists())
        check("burn leaves unregistered directories alone", (root / "not_an_env").exists())
        check("burn on an empty root is a no-op", burn_envs(specs, root / "nope") == {})

    # 'all' must mean "every env a runnable model needs", and a list must be accepted
    import csf.generation.envs as _E
    src_cli = inspect.getsource(_E._main)
    check("'all' skips envs no implemented adapter uses",
          "a.implemented" in src_cli and "no implemented adapter uses" in src_cli)
    check("names can be given as a comma-separated list", 'name.split(",")' in src_cli)
    live = {a.env_name for a in ADAPTERS.values() if a.implemented}
    check("vid2vid is excluded from 'all'", "vid2vid" not in live and "vid2vid" in env_specs())
    check("every other env survives the filter", len(live) == len(env_specs()) - 1,
          f"{len(live)} vs {len(env_specs())}")

    # a Git LFS pointer is not a checkpoint
    from csf.generation.envs import WeightFile, _is_lfs_pointer, _staged_file
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        (staged / "real.pth").write_bytes(b"\x80\x02}q\x00." * 100)
        (staged / "pointer.pth").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 164535938\n")
        check("a real checkpoint is not mistaken for a pointer",
              not _is_lfs_pointer(staged / "real.pth"))
        check("an LFS pointer is recognised", _is_lfs_pointer(staged / "pointer.pth"))
        found = _staged_file(WeightFile(dest="w/real.pth", staged_name="real.pth"), staged, "e")
        check("a staged checkpoint is found by name", found == staged / "real.pth")
        try:
            _staged_file(WeightFile(dest="w/p.pth", staged_name="pointer.pth"), staged, "e")
            check("an LFS pointer is refused", False, "no error raised")
        except Exception as exc:
            check("an LFS pointer is refused with the git lfs pull remedy",
                  "git lfs pull" in str(exc), str(exc)[:100])
        try:
            _staged_file(WeightFile(dest="w/x.pth", staged_name="absent.pth",
                                    where="https://example.invalid"), staged, "e")
            check("a missing staged file is refused", False, "no error raised")
        except Exception as exc:
            check("a missing staged file names its source and what is present",
                  "example.invalid" in str(exc) and "real.pth" in str(exc), str(exc)[:120])

    # the insightface env must end up with the GPU runtime and headless OpenCV, not the CPU
    # runtime and full OpenCV that insightface 2.0 depends on
    ins = env_specs()["insightface"]
    hook = " ".join(" ".join(c) for c in ins.post_install)
    check("insightface installs without a compiler", "insightface==2.0" in ins.requirements)
    check("the CPU runtime is replaced with onnxruntime-gpu",
          "uninstall" in hook and "onnxruntime-gpu" in hook)
    check("full OpenCV is replaced with the headless build",
          "opencv-python'" in hook and "opencv-python-headless" in hook)
    check("the swap is verified by the import check",
          {"onnxruntime", "cv2"} <= set(ins.checks()))

    # Hub downloads must not go through a CLI: `huggingface-cli` was removed in favour of `hf`
    specs_all = env_specs()
    hooks = {n: " ".join(" ".join(c) for c in sp.post_install) for n, sp in specs_all.items()}
    check("no post-install hook shells out to huggingface-cli",
          not any("huggingface-cli" in h for h in hooks.values()),
          str([n for n, h in hooks.items() if "huggingface-cli" in h]))
    hub_hooks = [n for n, h in hooks.items() if "snapshot_download" in h]
    check("the Hub pulls use snapshot_download", len(hub_hooks) == 4, str(sorted(hub_hooks)))
    check("each snapshot lands under the env root",
          all("CSF_ENV_ROOT" in hooks[n] for n in hub_hooks))

    # every env that touches the Hub must have the library installed in it
    missing = [n for n, sp in specs_all.items() if sp.needs_hub()
               and not any("huggingface" in r.lower() for r in sp.pip_requirements())]
    check("every Hub-using env installs huggingface_hub", not missing, str(missing))
    check("an env that never touches the Hub does not gain the dependency",
          not any("huggingface" in r.lower()
                  for r in specs_all["propainter"].pip_requirements()))
    check("the derived requirement is part of the digest",
          EnvSpec(name="x", torch="", weights=(WeightFile(dest="w", hf_repo="r", hf_file="f"),)
                  ).digest()
          != EnvSpec(name="x", torch="").digest())

    # nothing may require a C toolchain or Python.h: the cluster has neither
    all_reqs = {n: sp.pip_requirements() for n, sp in env_specs().items()}
    check("no env pins the insightface source distribution",
          not any("insightface==0.7" in r for reqs in all_reqs.values() for r in reqs),
          str([n for n, reqs in all_reqs.items() if any("0.7" in r for r in reqs)]))
    check("dlib comes from the prebuilt wheel",
          "dlib-bin" in all_reqs["stylegan"] and "dlib" not in all_reqs["stylegan"])
    check("the wheel still provides the dlib import name",
          "dlib" in env_specs()["stylegan"].checks())
    for name in ("insightface", "dreamid", "reface"):
        spec = env_specs()[name]
        hook = " ".join(" ".join(c) for c in spec.post_install)
        check(f"{name} replaces the CPU runtime insightface pulls in",
              "onnxruntime-gpu" in hook and "uninstall" in hook)
        check(f"{name} verifies the swap took effect",
              {"onnxruntime", "cv2"} <= set(spec.checks()), str(spec.checks()))

    # a weight source that does not exist is worse than one that has to be staged
    fomm_weights = env_specs()["fomm"].weights
    check("the FOMM checkpoint is staged, not fetched from a guessed repo",
          all(w.staged_name and not w.hf_repo for w in fomm_weights))
    check("it says where to get it", all(w.where for w in fomm_weights))

    # a framework pin with no ceiling is not a pin: transformers 5 requires torch>=2.5 and,
    # below that, disables PyTorch and loads tokenizers only - an env that imports fine and
    # cannot load a single model
    unbounded = [(n, r) for n, sp in specs_all.items() for r in sp.pip_requirements()
                 if r.split(">")[0].split("=")[0].split("<")[0] in ("transformers", "diffusers")
                 and "<" not in r]
    check("every transformers/diffusers pin has a major-version ceiling", not unbounded,
          str(unbounded))
    check("the ceiling keeps transformers on the 4.x line",
          all("transformers>=4" in r and ",<5" in r
              for sp in specs_all.values() for r in sp.pip_requirements()
              if r.startswith("transformers")))
    from csf.generation.envs import FRAMEWORK_PROBE, _verify_framework

    # a stub interpreter whose framework probe reports PyTorch disabled: both the build and
    # the doctor must refuse it, because every other signal in such an env looks healthy
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        venv = root / "probe" / "venv"
        (venv / "bin").mkdir(parents=True)
        stub = venv / "bin" / "python"
        stub.write_text(f"""#!/bin/sh
case "$2" in
  *is_torch_available*) echo "transformers 5.17.0 | torch 2.4.1 | torch enabled: False"; exit 3 ;;
  *sys.prefix*)         echo "{venv.resolve()}" ;;
  *torch.__version__*)  echo "2.4.1+cu121" ;;
  *)                    exit 0 ;;
esac
""")
        stub.chmod(0o755)
        broken = EnvSpec(name="probe", torch="torch==2.4.1",
                         requirements=("transformers>=4.44,<5",),
                         verify_imports=("torch", "transformers"))
        try:
            _verify_framework(stub, broken)
            check("the build refuses an env with PyTorch disabled", False, "no error raised")
        except E_.EnvBuildError as exc:
            check("the build refuses an env with PyTorch disabled",
                  "PyTorch disabled" in str(exc), str(exc)[:80])
        report = E_.diagnose([broken], root)["probe"]
        check("the doctor calls it broken, not ok", report["verdict"] == "broken",
              str(report.get("verdict")))
        check("the doctor reports torch_enabled", report.get("torch_enabled") is False)
        check("an env with no transformers is not framework-checked",
              "torch_enabled" not in E_.diagnose(
                  [EnvSpec(name="probe", torch="torch==2.4.1", verify_imports=("torch",))],
                  root)["probe"])
    check("the build asserts transformers can use torch",
          "is_torch_available" in FRAMEWORK_PROBE)
    build_all = inspect.getsource(E_.build_env)
    check("the framework check runs before an env is marked ready",
          "_verify_framework(py, spec)" in build_all)

    # the swap must survive the state that broke it: opencv-python uninstalled out from under
    # opencv-python-headless, whose metadata then suppresses the reinstall
    from csf.generation.adapters import RUNTIME_SWAP
    import ast as _ast
    _ast.parse(RUNTIME_SWAP)
    for pkg in ("onnxruntime", "onnxruntime-gpu", "opencv-python", "opencv-python-headless",
                "opencv-contrib-python"):
        check(f"the swap removes {pkg} before installing", f"'{pkg}'" in RUNTIME_SWAP)
    check("the swap imports what it installed, so a silent no-op cannot pass",
          "importlib.import_module(name)" in RUNTIME_SWAP)
    from csf.generation.adapters import ENVS as ALL_ENVS
    check("no env pins opencv alongside insightface, which pulls its own",
          not any("opencv" in r
                  for n in ("dreamid", "reface", "insightface", "faceshifter")
                  for r in ALL_ENVS[n].pip_requirements()))

    # numpy must be constrained for every later install in the env, not just torch
    with tempfile.TemporaryDirectory() as tmp:
        written = E_._write_constraints(specs_all["dreamid"], Path(tmp))
        check("the constraints file pins numpy as well as torch",
              "numpy<2" in written.read_text())
        plain = E_._write_constraints(specs_all["propainter"], Path(tmp))
        check("an env without a numpy pin still gets its torch pins",
              "torch==" in plain.read_text())

    sad = " ".join(" ".join(c) for c in env_specs()["sadtalker"].post_install)
    check("the basicsr patch does not import basicsr to find it",
          "find_spec" not in sad and "sysconfig" in sad)
    check("the basicsr patch verifies the import afterwards",
          "import_module('basicsr.data.degradations')" in sad)
    check("basicsr's removed torchvision import is patched",
          "functional_tensor" in sad and "basicsr.data.degradations" in sad)
    check("the patch runs before SadTalker's own downloader",
          sad.index("functional_tensor") < sad.index("download_models.sh"))

    body = inspect.getsource(_venv_python).split('"""')[-1]      # skip the docstring's prose
    check("the interpreter path is never resolved through its symlink",
          ".resolve()" not in body and "os.path.abspath" in body)

    # the derived import check has to cover what the workers actually import
    specs = env_specs()
    check("every env checks for cv2 when it installs opencv",
          all("cv2" in sp.checks() for sp in specs.values()
              if any("opencv" in r for r in sp.requirements)))
    check("every env with torch checks for torch",
          all("torch" in sp.checks() for sp in specs.values() if sp.torch))
    check("sam2_diffusers checks the modules its worker imports",
          {"cv2", "torch", "diffusers"} <= set(specs["sam2_diffusers"].checks()))
    check("an explicit verify_imports overrides the derived set",
          EnvSpec(name="x", torch="t", verify_imports=("only_this",)).checks() == ("only_this",))
    check("verify_imports is part of the env digest",
          EnvSpec(name="x", verify_imports=("a",)).digest()
          != EnvSpec(name="x", verify_imports=("b",)).digest())

    # the build must confirm the env works rather than trusting its own marker
    build_src = inspect.getsource(__import__("csf.generation.envs", fromlist=["build_env"]).build_env)
    check("a ready marker is re-verified before the env is handed out",
          "_venv_is_sane(py, venv_dir)" in build_src and "_verify_imports" in build_src)
    check("pip is bootstrapped when the venv has none", "_ensure_pip(py," in build_src)
    check("a venv that is not its own venv is recreated", "recreating it" in build_src)


def test_variant_capability() -> None:
    """No job may carry a manipulation_type its renderer cannot actually perform."""
    print("variant capability")
    from csf.generation.jobs import variant_pools

    allowed = {k for k, a in ADAPTERS.items() if a.implemented}
    jobs = build_jobs(synthetic_pool(400), S.FAMILY_TARGETS, seed=42, allowed_models=allowed)
    bad = [j for j in jobs if j.variant and ADAPTERS[j.model].variants
           and j.variant not in ADAPTERS[j.model].variants]
    check("every job's variant is one its renderer supports", not bad,
          str([(j.model, j.variant) for j in bad[:3]]))

    expr = [j for j in jobs if j.family == "expression_attribute_editing"]
    by_model = defaultdict(set)
    for job in expr:
        by_model[job.model].add(job.variant)
    check("StyleGANEX only renders its two released directions",
          by_model["styleganex"] == {"age", "hair_color"}, str(sorted(by_model["styleganex"])))
    check("LivePortrait never renders age or hair colour",
          not ({"age", "hair_color"} & by_model["ganimation"]), str(sorted(by_model["ganimation"])))
    check("a variant no renderer supports is dropped, not faked",
          "facial_attributes" not in {j.variant for j in expr})

    # the worker tables must agree with what the registry advertises
    lp = (Path(__file__).resolve().parents[1] / "csf/generation/adapters/workers"
          / "worker_liveportrait.py").read_text(encoding="utf-8")
    check("LivePortrait has no no-op retargeting entry",
          '"hair_color": (0.0, 0.0)' not in lp)
    check("LivePortrait refuses a variant it cannot perform", "cannot produce the" in lp)
    sg = (Path(__file__).resolve().parents[1] / "csf/generation/adapters/workers"
          / "worker_styleganex.py").read_text(encoding="utf-8")
    check("StyleGANEX maps each variant to a released checkpoint",
          "styleganex_edit_age.pt" in sg and "styleganex_edit_hair.pt" in sg)
    check("StyleGANEX calls the repo's real video-editing entry point",
          "video_editing.py" in sg and "--task" not in sg)

    # the per-model split must follow variant demand where renderers differ, so the realised
    # manipulation_type mix matches the document instead of being dictated by the model split
    fam_expr = S.FAMILIES["expression_attribute_editing"]
    caps = {k: a.variants for k, a in ADAPTERS.items() if a.variants}
    doc_cols = S.family_matrix(fam_expr, 4750, allowed)[1]
    cap_cols = S.family_matrix(fam_expr, 4750, allowed, capabilities=caps)[1]
    check("capability weighting moves the model split", doc_cols != cap_cols,
          f"{doc_cols} vs {cap_cols}")
    check("the family still totals the same", sum(cap_cols) == sum(doc_cols) == 4750)
    check("the model that supports fewer variants gets the smaller share",
          cap_cols[[p.key for p in S.family_pipelines(fam_expr, allowed)].index("styleganex")]
          < cap_cols[[p.key for p in S.family_pipelines(fam_expr, allowed)].index("ganimation")])

    weights = dict(fam_expr.variants)
    producible = [v for v in weights if v != "facial_attributes"]
    want = dict(zip(producible, S.apportion(len(expr), [float(weights[v]) for v in producible])))
    got = Counter(j.variant for j in expr)
    worst = max(abs(got.get(v, 0) - want[v]) for v in want)
    check("every producible variant lands on its document share (within rounding)", worst <= 2,
          f"worst deviation {worst}: " + str({v: (got.get(v, 0), want[v]) for v in want}))
    check("the variant with no renderer stays at zero", got.get("facial_attributes", 0) == 0)

    # only families whose renderers differ may be re-weighted; everything else is untouched
    for key in ("face_swap", "lip_sync", "video_inpainting", "object_insertion_removal",
                "background_manipulation", "video_to_video", "facial_reenactment"):
        fam = S.FAMILIES[key]
        target = S.FAMILY_TARGETS[key]
        check(f"{key}'s split is unchanged by capability weighting",
              S.family_matrix(fam, target, allowed)[1]
              == S.family_matrix(fam, target, allowed, capabilities=caps)[1])
    check("a family with uniform capability keeps the document's weights",
          S.capability_weights(S.FAMILIES["lip_sync"],
                               S.family_pipelines(S.FAMILIES["lip_sync"], allowed), caps) is None)
    # build_plan() is the document's own view and must stay that way: it passes no
    # capabilities, so the printed plan still reproduces the proposal's per-model split
    plan_cols = Counter()
    for cell in S.build_plan(allowed=allowed):
        if cell.family == "expression_attribute_editing":
            plan_cols[cell.model] += cell.videos
    check("build_plan keeps the document's per-model split",
          [plan_cols[p.key] for p in S.family_pipelines(fam_expr, allowed)] == doc_cols,
          f"{dict(plan_cols)} vs {doc_cols}")

    # family totals must survive the re-routing exactly
    fam = S.FAMILIES["expression_attribute_editing"]
    capacity = {"styleganex": 2250, "ganimation": 2500}
    pools = variant_pools(fam, capacity, seed=1)
    check("every pipeline's capacity is filled exactly",
          all(len(pools[m]) == n for m, n in capacity.items()),
          str({m: len(v) for m, v in pools.items()}))
    check("a pipeline with no declared capability may use any variant",
          len(set(variant_pools(fam, {"inswapper": 100}, seed=1)["inswapper"])) > 1)


def test_disk_probe() -> None:
    """Free space must be measured where the data lands, not on the root filesystem."""
    print("disk probe")
    import importlib.util
    import shutil as _sh
    import tempfile

    spec = importlib.util.spec_from_file_location("csf_main", ROOT / "main.py")
    main_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_mod)

    src = (ROOT / "main.py").read_text(encoding="utf-8")
    import ast as _a
    calls = [n for n in _a.walk(_a.parse(src))
             if isinstance(n, _a.Attribute) and n.attr == "anchor"]
    check("no code path measures Path(...).anchor any more", not calls,
          f"{len(calls)} anchor access(es) remain")
    check("preflight logs which directory it measured", '"disk_probe"' in src)

    with tempfile.TemporaryDirectory() as tmp:
        deep = Path(tmp) / "cache" / "regen" / "kinetics" / "clips"
        gib, probed = main_mod.free_space_gib(deep)
        check("a path that does not exist yet resolves to an existing parent", probed.exists())
        check("it walks up no further than it must", str(probed) == str(Path(tmp).resolve()),
              f"{probed} vs {Path(tmp).resolve()}")
        check("it reports that filesystem's free space",
              abs(gib - _sh.disk_usage(tmp).free / 2 ** 30) < 1.0)

        existing = Path(tmp) / "here"
        existing.mkdir()
        _, probed_existing = main_mod.free_space_gib(existing)
        check("an existing directory is measured directly",
              probed_existing == existing.resolve())

    # the bug: on POSIX the anchor of any absolute path is "/", so the old call measured the
    # root filesystem no matter where the cache actually lived
    check("the anchor of a project path is the root filesystem",
          Path("./cache/regen").resolve().anchor == "/")
    _, probed = main_mod.free_space_gib("./cache/regen")
    check("the new probe does not collapse to the root", str(probed) != "/")


def test_worker_dependencies() -> None:
    """Every import a worker's upstream repo makes at module scope must be installed.

    Each check here is one model that rendered nothing in a two-day run: the traceback was
    always a bare ModuleNotFoundError or a signature change, hours after the env was declared
    ready. An env's requirements are the only place that knowledge can live, because the repos
    themselves either ship no requirements file or ship one that fights the pinned torch.
    """
    print("worker dependencies")
    envs = env_specs()

    # (env, distribution, who imports it)
    required = [
        ("videoinpaint", "matplotlib", "E2FGVI / STTN / FuseFormer test.py"),
        ("stylegan", "matplotlib", "StyleGANEX models/psp.py"),
        ("propainter", "requests", "ProPainter utils/download_util.py"),
        ("liveportrait", "onnx", "LivePortrait's vendored insightface arcface_onnx.py"),
        ("tpsmm", "face-alignment", "TPSMM demo.py find_best_frame"),
    ]
    for env_name, dist, who in required:
        reqs = [r.split("=")[0].split("<")[0].split(">")[0].strip()
                for r in envs[env_name].pip_requirements()]
        check(f"env '{env_name}' installs {dist} for {who}", dist in reqs, f"has {reqs}")

    # onnxruntime is a different distribution from onnx and does not provide that import
    lp = [r for r in envs["liveportrait"].pip_requirements()]
    check("onnxruntime is not mistaken for onnx",
          any(r.startswith("onnx") and not r.startswith("onnxruntime") for r in lp))

    vace = envs["vace"]
    wan_install = [c for c in vace.post_install if "pip" in c and any("Wan2.1" in a for a in c)]
    check("the vace env installs the Wan2.1 package its entry point imports", wan_install)
    check("it installs it without dependencies (flash_attn would build from source)",
          wan_install and "--no-deps" in wan_install[0])
    check("a failed wan install fails the build, not 2,883 jobs", "wan" in vace.verify_imports)

    w2l = "".join("".join(c) for c in envs["wav2lip"].post_install)
    check("wav2lip rewrites the positional librosa.filters.mel call",
          "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft," in w2l)
    check("it keeps librosa itself unpinned below 0.10",
          not any("librosa<0.10" in r for r in envs["wav2lip"].pip_requirements()))

    sad = [c for c in envs["sadtalker"].post_install if "np.float" in "".join(c)]
    check("sadtalker rewrites the numpy aliases removed in 1.24", sad)
    if sad:
        pattern = re.search(r"re\.compile\(r'([^']+)'\)", "".join(sad[0]))
        check("the patch's pattern is recoverable", pattern is not None)
        if pattern:
            rx = re.compile(pattern.group(1))
            check("it rewrites the bare alias",
                  rx.sub(lambda m: m.group(1), "x.astype(np.float, copy=False)")
                  == "x.astype(float, copy=False)")
            check("it leaves the sized dtypes alone",
                  rx.sub(lambda m: m.group(1), "np.float32 np.int64 np.bool_")
                  == "np.float32 np.int64 np.bool_")

    dreamid = (ROOT / "csf/generation/adapters/workers/worker_dreamid.py").read_text()
    for flag in ("--ref_image", "--ref_video", "--save_file", "--dreamidv_ckpt",
                 "generate_dreamidv.py"):
        check(f"the DreamID-V worker uses {flag}", flag in dreamid)
    for gone in ("--target_video", "--source_image", "--output_dir", '"inference.py"'):
        check(f"it no longer uses {gone}, which upstream never had", gone not in dreamid)
    check("the Wan2.1 backbone DreamID-V borrows its VAE and T5 from is staged",
          any("Wan2.1-T2V-1.3B" in repo for repo in envs["dreamid"].hub_repos))
    check("frame_num stays on upstream's 4n+1 grid",
          re.search(r"N_FRAMES = (\d+)", dreamid) and
          (int(re.search(r"N_FRAMES = (\d+)", dreamid).group(1)) - 1) % 4 == 0)


def test_child_process_errors() -> None:
    """A failed upstream CLI must report the head of its traceback, not only the tail."""
    print("child process errors")
    common = ROOT / "csf/generation/adapters/workers/_common.py"
    src = common.read_text()
    namespace: dict = {}
    body = src[src.index("def clip_output"):src.index("def run_cmd")]
    exec(compile(body, str(common), "exec"), namespace)
    clip_output = namespace["clip_output"]

    short = "traceback\nline\nerror"
    check("output that fits is passed through untouched", clip_output(short) == short)

    head, tail = "H" * 4000, "T" * 4000
    clipped = clip_output(head + "middle" * 500 + tail)
    check("the first frames survive", clipped.startswith("H" * 1500))
    check("the last frames survive", clipped.endswith("T" * 1500))
    check("the elision is stated, not silent", "characters elided" in clipped)
    check("nothing is smuggled through the middle", "middle" not in clipped)

    check("run_cmd no longer keeps the tail alone", "[-1500:]" not in src,
          "the import chain that names the broken dependency lives in the head")
    check("run_cmd reports through clip_output", "clip_output(proc.stderr" in src)


def test_smoke_selection() -> None:
    """The per-model smoke command must refuse what it cannot honestly test."""
    print("smoke selection")
    from csf.generation import smoke

    wired = smoke.selectable()
    check("every selectable model is wired up", all(ADAPTERS[k].implemented for k in wired))
    check("--all offers every wired model",
          set(smoke._resolve([], "", True)) == set(wired))
    check("a comma-separated list is split",
          smoke._resolve(["inswapper,wav2lip"], "", False) == ["inswapper", "wav2lip"])
    check("a family selects its wired models",
          set(smoke._resolve([], "video_inpainting", False))
          == {k for k in wired if ADAPTERS[k].family == "video_inpainting"})

    for bad, why in ((["no_such_model"], "unknown model"),
                     ([""], "empty selection")):
        try:
            resolved = smoke._resolve(bad, "", False)
        except SystemExit:
            resolved = None
        check(f"{why} is rejected or empty", not resolved)

    scaffolds = [k for k, a in ADAPTERS.items() if not a.implemented]
    if scaffolds:
        try:
            smoke._resolve([scaffolds[0]], "", False)
            refused = False
        except SystemExit:
            refused = True
        check("a tier-2 scaffold cannot be smoke-tested", refused)

    src = (ROOT / "csf/generation/smoke.py").read_text()
    check("smoke renders into its own directory, never the dataset tree",
          "AI Edited" not in src)
    check("one resident worker, so a full sweep needs one model's VRAM",
          "max_resident=1" in src)
    check("a worker that reports success but writes nothing is a failure",
          "empty file" in src)


def main() -> int:
    for fn in (test_spec, test_jobs, test_degraded_pool, test_naming, test_adapters,
               test_substitutions, test_budget, test_reallocation, test_concurrency,
               test_kinetics_schema, test_attribution, test_metadata,
               test_metadata_merge, test_stage_scoping, test_probe_fallback,
               test_ffmpeg_resolution, test_worker_inputs, test_progress,
               test_env_paths, test_no_job_left_behind, test_retry_policy,
               test_stage_staleness, test_env_interpreter, test_variant_capability,
               test_disk_probe, test_worker_dependencies, test_child_process_errors,
               test_smoke_selection):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all regeneration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
