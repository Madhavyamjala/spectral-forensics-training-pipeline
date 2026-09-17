"""
The four regeneration stages, as called by `main.py`.

    kinetics        acquire Kinetics-400 source clips and score them for qualification
    generate        render the 33,333 AI-Edited videos across the configured GPUs
    regen_manifest  rebuild manifest.csv around what was actually produced
    push_dataset    (opt-in) replace the AI-Edited class in the dataset repo

All four are rank-0 only: generation does its own multi-process, multi-GPU scheduling through
per-model workers, so it must not be run under torchrun's DDP ranks. `main.py` enforces that.

Each stage is independently resumable and safe to re-run - the expensive state lives in the
source pool, the clip-feature cache and the generation ledger, none of which is rebuilt if it is
already there.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from csf.generation import spec as S
from csf.generation.filters import load_features, score_pool
from csf.generation.jobs import build_jobs, read_jobs, summarise, write_jobs
from csf.generation.kinetics import SourcePool, acquire
from csf.logging_utils import get_logger

log = get_logger("generation.run")


def video_root_for(cfg) -> Path:
    """Where generated videos land.

    Defaults to the run's own `paths.video_dir`, because `csf.data.video_io.download_video`
    checks `<video_dir>/<repo_path>` before reaching for the Hub - so writing there makes the
    feature-extraction stage pick the local files up with no download and no code change.
    """
    root = cfg.generation.video_root
    return Path(root) if root else Path(cfg.paths.video_dir)


def _check_video_root(cfg) -> Path:
    root = video_root_for(cfg)
    expected = Path(cfg.paths.video_dir)
    if root.resolve() != expected.resolve():
        log.warning("generation.video_root (%s) differs from paths.video_dir (%s). The feature "
                    "stage looks for videos under paths.video_dir, so it would try to download "
                    "the regenerated clips from the Hub instead of reading them locally. Set "
                    "them to the same path unless you intend to push first.", root, expected)
    return root


# --------------------------------------------------------------------------------------
# stage: kinetics
# --------------------------------------------------------------------------------------


def stage_kinetics(cfg) -> Dict[str, object]:
    gen = cfg.generation
    cache = Path(cfg.paths.cache_dir)
    targets = S.family_targets(gen.total_videos)

    demand = S.label_demand(targets)
    margin = max(1.0, float(gen.kinetics.demand_margin))
    wanted = {label: int(math.ceil(n * margin)) for label, n in demand.items()}
    log.info("Kinetics demand: %d clips over %d labels (x%.2f margin -> %d requested)",
             sum(demand.values()), len(demand), margin, sum(wanted.values()))

    pool_csv = acquire(gen, cache, wanted)
    features_csv = cache / "kinetics" / "clip_features.csv"
    score_pool(pool_csv, features_csv, workers=gen.kinetics.score_workers,
               num_frames=gen.kinetics.score_frames, need_faces=True,
               force=gen.kinetics.rescore)

    features = load_features(features_csv)
    from csf.generation.filters import _row_qualifies
    counts = {name: sum(1 for r in features if _row_qualifies(r, name))
              for name in ("any", "face", "mouth", "object")}
    report = {"clips": len(features), "qualifying": counts,
              "labels": len({r["label"] for r in features}),
              "requested": sum(wanted.values())}
    log.info("Source pool ready: %s", json.dumps(report))
    for name, need in (("face", "face_swap + reenactment + expression"), ("mouth", "lip_sync")):
        if counts[name] < 500:
            log.warning("Only %d clip(s) pass the '%s' filter, which the %s families need. "
                        "Install insightface or mediapipe so faces are detected, or widen "
                        "generation.kinetics.demand_margin.", counts[name], name, need)
    return report


# --------------------------------------------------------------------------------------
# stage: generate
# --------------------------------------------------------------------------------------


def _filter_models(jobs, only: Sequence[str], skip: Sequence[str]):
    if only:
        jobs = [j for j in jobs if j.model in set(only)]
        log.info("generation.only_models -> %d job(s) for %s", len(jobs), sorted(set(only)))
    if skip:
        jobs = [j for j in jobs if j.model not in set(skip)]
        log.info("generation.skip_models -> %d job(s) remain", len(jobs))
    return jobs


def _apply_budget(cfg, jobs):
    """Drop the model groups that do not fit `generation.budget_wall_clock_hours`.

    Deciding here rather than letting `deadline_hours` truncate the run matters: a deadline stops
    whichever group is in flight when it fires, so the dataset ends up shaped by scheduling
    order. The planner instead keeps at least two distinct renderers per family and spends what
    is left on volume, and says exactly what it dropped.
    """
    gen = cfg.generation
    if not gen.budget_wall_clock_hours:
        return jobs
    from csf.generation.budget import family_breakdown, plan

    budget = float(gen.budget_wall_clock_hours) * max(1, len(gen.gpus))
    p = plan(budget, gpus=len(gen.gpus), targets=S.family_targets(gen.total_videos),
             diversity_floor=gen.budget_diversity_floor, pinned=gen.budget_pin_models)
    log.info("Budget planner: %s", json.dumps(p.to_dict()))
    for fam, info in family_breakdown(p, S.family_targets(gen.total_videos)).items():
        log.info("  %-32s %5d/%-5d videos | %d renderer(s): %s", fam, info["kept"],
                 info["planned"], info["mechanisms"], ", ".join(info["models"]) or "-")
    if p.skipped:
        log.warning("Budget of %.0f h wall clock on %d GPU(s) does not fit every wired model; "
                    "skipping %s. Raise generation.budget_wall_clock_hours (or set it to null) "
                    "to run them.", gen.budget_wall_clock_hours, len(gen.gpus), p.skipped)
    keep = set(p.selected)
    return [j for j in jobs if j.model in keep]


def plan_jobs(cfg, rebuild: bool = False):
    """Load the job plan, building it from the scored source pool the first time."""
    gen = cfg.generation
    jobs_csv = Path(gen.jobs_csv)
    if jobs_csv.exists() and not rebuild:
        jobs = read_jobs(jobs_csv)
        log.info("Loaded %d job(s) from %s", len(jobs), jobs_csv)
        return jobs

    features_csv = Path(cfg.paths.cache_dir) / "kinetics" / "clip_features.csv"
    if not features_csv.exists():
        raise RuntimeError(f"{features_csv} is missing - run the 'kinetics' stage first.")
    features = load_features(features_csv)
    targets = S.family_targets(gen.total_videos)
    jobs = build_jobs(features, targets, seed=cfg.seed)
    write_jobs(jobs, jobs_csv)
    log.info("Job plan: %s", json.dumps(summarise(jobs)))
    return jobs


def stage_generate(cfg) -> Dict[str, object]:
    from csf.generation.scheduler import GenerationScheduler, Ledger

    gen = cfg.generation
    jobs = _filter_models(plan_jobs(cfg), gen.only_models, gen.skip_models)
    jobs = _apply_budget(cfg, jobs)
    if not jobs:
        raise RuntimeError("No jobs to run after applying only_models / skip_models / budget")

    # REFace's checkpoint carries a non-commercial-research restriction that the videos it
    # produces inherit; its worker refuses to start unless this is acknowledged explicitly.
    if gen.accept_noncommercial:
        os.environ["CSF_ACCEPT_NONCOMMERCIAL"] = "1"
        log.warning("generation.accept_noncommercial is set: adapters with non-commercial "
                    "research-only weights (REFace) are enabled, and the videos they produce "
                    "inherit that restriction.")

    root = _check_video_root(cfg)
    log_dir = Path(cfg.paths.work_dir) / "logs" / "generation"
    scheduler = GenerationScheduler(
        video_root=root, envs_root=Path(gen.envs_root), log_dir=log_dir, gpus=gen.gpus,
        job_timeout=gen.job_timeout_s, fail_fast=gen.fail_fast, min_free_gb=gen.min_free_gb,
        offline=gen.offline, deadline_hours=gen.deadline_hours)
    ledger = Ledger(Path(gen.ledger))

    summary = scheduler.run(jobs, ledger, retry_failed=gen.retry_failed)
    scheduler.write_failures(Path(cfg.paths.work_dir) / "metrics" / "generation_failures.csv")

    ok = sum(1 for v in ledger.done.values() if v)
    summary["total_ok"] = ok
    summary["total_planned"] = len(jobs)
    out = Path(cfg.paths.work_dir) / "metrics" / "generation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Generation summary -> %s: %s", out, json.dumps(summary))
    if ok == 0:
        raise RuntimeError("Generation produced no videos at all. Check "
                           "runs/<run>/logs/generation/ and metrics/generation_failures.csv - "
                           "most likely no adapter environment could be built.")
    return summary


# --------------------------------------------------------------------------------------
# stage: regen_manifest
# --------------------------------------------------------------------------------------


def stage_regen_manifest(cfg) -> Dict[str, object]:
    from csf.generation.manifest_build import build_manifest, write_report

    gen = cfg.generation
    jobs = plan_jobs(cfg)
    out = Path(gen.manifest_out)
    report = build_manifest(old_manifest=Path(cfg.data.manifest), jobs=jobs,
                            ledger_path=Path(gen.ledger), video_root=video_root_for(cfg),
                            out_path=out, keep_old_edited=gen.keep_old_edited)
    write_report(report, Path(cfg.paths.work_dir) / "metrics" / "regeneration_report.json")
    log.info("Manifest rebuilt at %s. Point data.manifest at it for training:\n"
             "    --set data.manifest=%s", out, out)
    return report


# --------------------------------------------------------------------------------------
# stage: push_dataset
# --------------------------------------------------------------------------------------


def stage_push_dataset(cfg) -> Optional[str]:
    from csf.generation.upload import push_interactive

    if not cfg.generation.push.enabled:
        log.info("generation.push.enabled is false -> not touching the dataset repo")
        return None
    return push_interactive(cfg, Path(cfg.generation.manifest_out), video_root_for(cfg))
