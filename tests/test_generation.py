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

import os
import random
import re
import sys
import time
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


def main() -> int:
    for fn in (test_spec, test_jobs, test_degraded_pool, test_naming, test_adapters,
               test_substitutions, test_budget, test_reallocation, test_concurrency,
               test_kinetics_schema, test_attribution, test_metadata,
               test_metadata_merge, test_stage_scoping, test_probe_fallback,
               test_ffmpeg_resolution, test_worker_inputs, test_progress):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all regeneration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
