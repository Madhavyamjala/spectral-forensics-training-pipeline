"""
AI-Edited dataset regeneration.

Builds the AI-Edited class of Chrono-TriClass-100k from scratch, following the
"AI Edited Data Source and Pipeline" specification: 33,333 videos across eight manipulation
families and 32 models, all sourced from Kinetics-400.

Module map:
    spec.py            the specification as data + exact integer allocation of every cell
    kinetics.py        Kinetics-400 acquisition (disk-bounded shard streaming) and the source pool
    filters.py         per-clip qualification (face / mouth / object / any)
    jobs.py            plan -> 33,333 deterministic, resumable work items with leakage-aware splits
    envs.py            one virtual environment per model, built on demand
    adapters/          model registry + persistent-worker protocol + the worker scripts
    scheduler.py       multi-GPU execution, batched by model, resumable via a ledger
    manifest_build.py  the regenerated videos -> a new dataset manifest
    upload.py          publishing the new class back to the Hub

Entry point: the `kinetics`, `generate` and `regen_manifest` stages in `main.py`.
"""

from __future__ import annotations

__all__ = ["spec", "kinetics", "filters", "jobs", "envs", "scheduler", "manifest_build", "upload"]
