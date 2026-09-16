"""
Rebuilding the dataset manifest around the regenerated AI-Edited videos.

The old manifest carries 31,128 ai_edited rows, 89% of which have `generator_edit_method` =
"unknown" - which is exactly why per-method forensic analysis was impossible on it. The new rows
replace them wholesale and every one of them knows which of the 32 models produced it.

What this module does:
  1. keeps every `real` and `ai_generated` row from the old manifest untouched,
  2. drops all old `ai_edited` rows,
  3. adds one row per *successfully generated* video - failed jobs are simply absent, so the
     manifest never points at a file that does not exist,
  4. probes each new file for real container metadata (duration, resolution, fps, codec, bitrate,
     audio, sha256) rather than copying the source clip's, because those container fingerprints
     are precisely what the `baseline_metadata_shortcut` model in the ablation is there to expose,
  5. writes the extra provenance columns (family, model, source clip, variant, ...) that the
     per-method breakdown and any later attribution study need.

Column layout stays a superset of the original, so `csf.data.manifest.load_run_manifest` reads it
with no change: `video_id` is `aiedit-<family>-<n>`, which its existing regex maps to
`AI Edited/<family>/<video_id>.mp4`, and `generator_edit_method` holds the model key.

Input : old manifest.csv, jobs.csv, the generation ledger, the video root.
Output: the new manifest.csv plus regeneration_report.json.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from csf.generation.jobs import Job
from csf.generation.kinetics import probe_video, sha256_file
from csf.logging_utils import get_logger

log = get_logger("generation.manifest")

BASE_COLUMNS = ["class", "generator_edit_method", "video_id", "split", "duration_sec", "width",
                "height", "fps", "codec", "bitrate", "has_audio", "sha256"]
EXTRA_COLUMNS = ["family", "model", "source_clip_id", "source_label", "source_group", "variant",
                 "operation", "mask_size", "mask_motion", "driving_clip_id", "audio_clip_id",
                 "provenance"]
ALL_COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS


def read_ledger(path: Path) -> Dict[str, dict]:
    """job_id -> last recorded outcome (the ledger is append-only, so later wins)."""
    out: Dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["job_id"]] = rec
    return out


def _row_for(job: Job, path: Path, meta: dict) -> Optional[Dict[str, object]]:
    probed = probe_video(path)
    if probed is None:
        log.debug("Generated file is unreadable, dropping from the manifest: %s", path)
        return None
    row: Dict[str, object] = {
        "class": "ai_edited",
        "generator_edit_method": job.model,
        "video_id": job.video_id,
        "split": job.split,
        "duration_sec": probed["duration_sec"],
        "width": probed["width"],
        "height": probed["height"],
        "fps": probed["fps"],
        "codec": probed["codec"],
        "bitrate": probed["bitrate"],
        "has_audio": probed["has_audio"],
        "sha256": sha256_file(path),
        "family": job.family,
        "model": job.model,
        "source_clip_id": job.source_clip_id,
        "source_label": job.source_label,
        "source_group": job.source_group,
        "variant": job.variant,
        "operation": job.operation,
        "mask_size": job.mask_size,
        "mask_motion": job.mask_motion,
        "driving_clip_id": job.driving_clip_id,
        "audio_clip_id": job.audio_clip_id,
        "provenance": json.dumps({**(meta or {}), "regenerated": True}, sort_keys=True)[:2000],
    }
    return row


def build_manifest(old_manifest: Path, jobs: Sequence[Job], ledger_path: Path, video_root: Path,
                   out_path: Path, keep_old_edited: bool = False,
                   workers: int = 8) -> Dict[str, object]:
    """Write the new manifest; returns a report dict."""
    old_manifest, out_path, video_root = Path(old_manifest), Path(out_path), Path(video_root)
    ledger = read_ledger(ledger_path)

    kept: List[Dict[str, object]] = []
    old_counts: Counter = Counter()
    if old_manifest.exists():
        with open(old_manifest, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                cls = row.get("class", "")
                old_counts[cls] += 1
                if cls == "ai_edited" and not keep_old_edited:
                    continue
                if cls == "ai_edited" and keep_old_edited:
                    row.setdefault("provenance", json.dumps({"regenerated": False}))
                kept.append(row)
        log.info("Old manifest: %s | keeping %d non-ai_edited row(s)", dict(old_counts), len(kept))
    else:
        log.warning("Old manifest %s not found - the new manifest will contain only the "
                    "regenerated ai_edited rows", old_manifest)

    by_id = {j.job_id: j for j in jobs}
    new_rows: List[Dict[str, object]] = []
    missing_file = 0
    failed = 0
    for job_id, rec in ledger.items():
        job = by_id.get(job_id)
        if job is None:
            continue
        if not rec.get("ok"):
            failed += 1
            continue
        path = Path(rec.get("output_path") or (video_root / job.repo_path))
        if not path.exists() or path.stat().st_size == 0:
            missing_file += 1
            continue
        row = _row_for(job, path, rec.get("metadata") or {})
        if row is not None:
            new_rows.append(row)

    log.info("Regenerated rows: %d usable | %d failed | %d recorded-ok but file missing",
             len(new_rows), failed, missing_file)
    if not new_rows:
        raise RuntimeError("No regenerated videos are usable - refusing to write a manifest with "
                           "an empty ai_edited class. Check the generation ledger and failures.csv.")

    rows = kept + new_rows
    fieldnames = list(ALL_COLUMNS)
    for row in kept:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    report = _report(rows, new_rows, old_counts, failed, missing_file)
    log.info("New manifest written: %s\n%s", out_path, json.dumps(report, indent=2)[:1500])
    return report


def _report(rows: Sequence[dict], new_rows: Sequence[dict], old_counts: Counter, failed: int,
            missing_file: int) -> Dict[str, object]:
    per_class = Counter(r.get("class", "") for r in rows)
    per_split = Counter(r.get("split", "") for r in rows)
    per_family = Counter(r.get("family", "") for r in new_rows)
    per_model = Counter(r.get("model", "") for r in new_rows)
    edited_split = Counter(r.get("split", "") for r in new_rows)
    return {
        "total_rows": len(rows),
        "per_class": dict(per_class),
        "per_split": dict(per_split),
        "ai_edited_regenerated": len(new_rows),
        "ai_edited_old_dropped": old_counts.get("ai_edited", 0),
        "jobs_failed": failed,
        "recorded_ok_but_missing": missing_file,
        "per_family": dict(sorted(per_family.items(), key=lambda kv: -kv[1])),
        "per_model": dict(sorted(per_model.items(), key=lambda kv: -kv[1])),
        "ai_edited_per_split": dict(edited_split),
        "class_balance_delta": {
            cls: per_class.get(cls, 0) - max(per_class.values()) for cls in per_class},
    }


def write_report(report: Dict[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
