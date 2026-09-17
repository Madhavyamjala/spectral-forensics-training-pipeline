"""
Per-video metadata for the regenerated AI-Edited class.

The specification asks for per-video metadata to be tracked for every family - "for later
artifact-attribution analysis, separating *learned the swap* from *learned the model / pose /
compression fingerprint*". Face swap wants source and target identity, face quality, visibility,
occlusion and pose; lip-sync wants the audio source, speaker, language and mouth visibility;
inpainting wants mask size and motion class, and so on.

`manifest.csv` is the training-facing file and deliberately stays narrow: `csf.data.manifest`
reads it on every run, and the ablation only needs the class, the split and the method. Burying
forty analysis fields in it - or in one truncated JSON column, which is what `provenance` was -
makes them unqueryable.

So the generator writes a second file, `metadata.csv`: one row per produced video, every
specification field as a real column, nothing truncated. The two share a key (`video_id`) and are
built from the same single pass over the files, because probing and hashing 33,333 videos is the
expensive part and doing it twice would be wasteful.

Values come from two places, with the renderer winning:

  * **planned** - what the job asked for (the requested edit magnitude, the mask class drawn from
    the document's distribution, which clip was bound as the audio donor),
  * **measured** - what the worker reported after rendering (the pose it actually found, the
    fraction of frames with a detected face, the real mask area).

Where both exist the measured value is authoritative; a planned value that the renderer silently
could not honour would otherwise be recorded as fact.

Input : the job plan, the generation ledger, and the produced files.
Output: `metadata.csv` next to the manifest, plus `metadata_schema.json` describing the columns.

    python -m csf.generation.metadata --config configs/regen.yaml
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from csf.generation import spec as S
from csf.generation.jobs import Job
from csf.logging_utils import get_logger

log = get_logger("generation.metadata")

#: Identity and provenance of the produced file.
IDENTITY_COLUMNS = [
    "video_id", "class", "family", "model", "spec_model", "substituted", "split",
    "repo_path", "file_bytes", "sha256",
]
#: Container fingerprint. These are what `baseline_metadata_shortcut` in the ablation exploits,
#: so they are recorded from the produced file itself, never copied from the source clip.
CONTAINER_COLUMNS = [
    "duration_sec", "width", "height", "fps", "codec", "bitrate", "has_audio",
]
#: What the job bound before rendering.
JOB_COLUMNS = [
    "source_clip_id", "source_label", "source_group", "driving_clip_id", "audio_clip_id",
    "variant", "operation", "mask_size", "mask_motion", "prompt", "seed", "render_seconds",
]


def spec_metadata_columns() -> List[str]:
    """Union of every family's `metadata_fields`, in specification order.

    Derived from the spec rather than hand-listed, so a field added to a family there
    automatically becomes a column here.
    """
    seen: Dict[str, None] = {}
    for family in S.FAMILY_LIST:
        for field in family.metadata_fields:
            seen.setdefault(field, None)
    return list(seen)


def metadata_columns() -> List[str]:
    cols = IDENTITY_COLUMNS + CONTAINER_COLUMNS + JOB_COLUMNS
    return cols + [c for c in spec_metadata_columns() if c not in cols]


def _coerce(value) -> object:
    """Flatten a value into something a CSV cell can hold without losing it."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return json.dumps(value, sort_keys=True)


def metadata_row(job: Job, path: Path, probed: Dict[str, object], sha256: str,
                 rendered: Optional[Dict[str, object]] = None,
                 render_seconds: Optional[float] = None) -> Dict[str, object]:
    """Build one metadata row from the plan, the probed file and the worker's report."""
    from csf.generation.adapters import ADAPTERS

    adapter = ADAPTERS.get(job.model)
    actual = adapter.runs if adapter is not None else job.model

    planned: Dict[str, object] = {}
    if job.metadata:
        try:
            planned = json.loads(job.metadata) or {}
        except (TypeError, json.JSONDecodeError):
            planned = {}

    row: Dict[str, object] = {c: "" for c in metadata_columns()}
    row.update({
        "video_id": job.video_id,
        "class": "ai_edited",
        "family": job.family,
        "model": actual,
        "spec_model": job.model,
        "substituted": bool(adapter is not None and adapter.substituted),
        "split": job.split,
        "repo_path": job.repo_path,
        "file_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256,
        "source_clip_id": job.source_clip_id,
        "source_label": job.source_label,
        "source_group": job.source_group,
        "driving_clip_id": job.driving_clip_id,
        "audio_clip_id": job.audio_clip_id,
        "variant": job.variant,
        "operation": job.operation,
        "mask_size": job.mask_size,
        "mask_motion": job.mask_motion,
        "prompt": job.prompt,
        "seed": job.seed,
        "render_seconds": round(render_seconds, 2) if render_seconds is not None else "",
    })
    for key in CONTAINER_COLUMNS:
        row[key] = _coerce(probed.get(key))

    known = set(row)
    # planned first, then measured - the renderer's own numbers override the request
    for source in (planned, rendered or {}):
        for key, value in source.items():
            if key in known and value not in (None, ""):
                row[key] = _coerce(value)
    return row


def read_metadata(path: Path) -> List[Dict[str, object]]:
    """Existing metadata rows, or [] when the file is absent or unreadable."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except (OSError, csv.Error) as exc:
        log.warning("Could not read existing metadata at %s (%s) -> starting fresh", path, exc)
        return []


def merge_metadata(new_rows: Sequence[Dict[str, object]], existing_path: Path,
                   keep_old_edited: bool = False) -> Dict[str, object]:
    """Combine freshly generated rows with whatever `metadata.csv` already holds.

    The regenerated AI-Edited videos replace the old ones, so the old `ai_edited` rows have to go
    - leaving them would describe files the new manifest no longer references, and any
    per-method analysis would double-count the class.

    Rows for `real` and `ai_generated` are kept untouched: those classes are not regenerated, and
    a wholesale overwrite would silently discard their metadata. That is the trap here - writing
    only the new rows looks correct until you notice two thirds of the dataset lost its metadata.

    Rows with no class column are treated as stale AI-Edited entries, since earlier versions of
    this file only ever described the generated class.
    """
    existing = read_metadata(existing_path)
    kept: List[Dict[str, object]] = []
    dropped = 0
    for row in existing:
        cls = str(row.get("class") or "").strip().lower()
        is_edited = cls in ("ai_edited", "ai-edited", "") or not cls
        if is_edited and not keep_old_edited:
            dropped += 1
            continue
        kept.append(row)

    # a regenerated video_id always supersedes an identical one that was kept
    new_ids = {str(r.get("video_id")) for r in new_rows}
    before = len(kept)
    kept = [r for r in kept if str(r.get("video_id")) not in new_ids]
    superseded = before - len(kept)

    if existing:
        log.info("Existing %s: %d row(s) | dropped %d old ai_edited | kept %d other-class | "
                 "%d superseded by a regenerated id", existing_path, len(existing), dropped,
                 len(kept), superseded)
    return {"rows": kept + list(new_rows), "existing": len(existing), "dropped_ai_edited": dropped,
            "kept_other_classes": len(kept), "superseded": superseded, "new": len(new_rows)}


def write_metadata(rows: Sequence[Dict[str, object]], path: Path,
                   merge_existing: bool = True, keep_old_edited: bool = False) -> Path:
    """Write `metadata.csv` plus a schema file, preserving other classes' rows.

    Set `merge_existing=False` only when the file is known to describe nothing worth keeping.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    merged = (merge_metadata(rows, path, keep_old_edited) if merge_existing
              else {"rows": list(rows), "existing": 0, "dropped_ai_edited": 0,
                    "kept_other_classes": 0, "superseded": 0, "new": len(rows)})
    rows = merged["rows"]

    # union the schema so a column an older file carried is never silently dropped
    columns = metadata_columns()
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    tmp.replace(path)                      # atomic: never leave a half-written metadata file

    schema = {
        "rows": len(rows),
        "columns": len(columns),
        "merge": {k: v for k, v in merged.items() if k != "rows"},
        "groups": {
            "identity": IDENTITY_COLUMNS,
            "container": CONTAINER_COLUMNS,
            "job": JOB_COLUMNS,
            "specification": spec_metadata_columns(),
        },
        "per_family_fields": {f.key: list(f.metadata_fields) for f in S.FAMILY_LIST},
        "notes": ("Container fields are probed from the produced file, not copied from the "
                  "source clip. Where a field was both planned and measured, the measured "
                  "value is recorded."),
    }
    schema_path = path.with_name("metadata_schema.json")
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    log.info("Wrote %d metadata row(s) x %d column(s) -> %s (schema: %s)",
             len(rows), len(columns), path, schema_path)
    return path


def coverage(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """How completely each specification field was actually filled, per family.

    A field the document asks for but that no renderer reports will show up as 0% here, which is
    the honest signal that the attribution analysis cannot use it.
    """
    per_family: Dict[str, Dict[str, float]] = {}
    for family in S.FAMILY_LIST:
        subset = [r for r in rows if r.get("family") == family.key]
        if not subset:
            continue
        per_family[family.key] = {
            field: round(sum(1 for r in subset if r.get(field) not in ("", None)) / len(subset), 3)
            for field in family.metadata_fields
        }
    return {"rows": len(rows), "per_family_field_fill_rate": per_family}


def _main() -> int:
    """Rebuild metadata.csv from the job plan and the ledger, without re-running a stage."""
    import argparse

    ap = argparse.ArgumentParser(description="Rebuild metadata.csv for the generated videos")
    ap.add_argument("--config", default="configs/regen.yaml")
    ap.add_argument("--out", default="", help="defaults to metadata.csv beside the manifest")
    ap.add_argument("--columns", action="store_true", help="print the column list and exit")
    args = ap.parse_args()

    if args.columns:
        cols = metadata_columns()
        print(f"{len(cols)} columns:")
        for group, names in (("identity", IDENTITY_COLUMNS), ("container", CONTAINER_COLUMNS),
                             ("job", JOB_COLUMNS), ("specification", spec_metadata_columns())):
            print(f"\n  {group} ({len(names)}):")
            for n in names:
                print(f"    {n}")
        return 0

    from csf.config import load_config
    from csf.generation.kinetics import probe_video, sha256_file
    from csf.generation.manifest_build import read_ledger
    from csf.generation.run import plan_jobs, video_root_for

    cfg = load_config(args.config)
    jobs = {j.job_id: j for j in plan_jobs(cfg)}
    ledger = read_ledger(Path(cfg.generation.ledger))
    video_root = video_root_for(cfg)
    out = Path(args.out) if args.out else Path(cfg.generation.manifest_out).with_name(
        "metadata.csv")

    rows, missing = [], 0
    for job_id, rec in ledger.items():
        job = jobs.get(job_id)
        if job is None or not rec.get("ok"):
            continue
        path = Path(rec.get("output_path") or (video_root / job.repo_path))
        if not path.exists() or path.stat().st_size == 0:
            missing += 1
            continue
        probed = probe_video(path)
        if probed is None:
            missing += 1
            continue
        rows.append(metadata_row(job, path, probed, sha256_file(path),
                                 rec.get("metadata") or {}, rec.get("seconds")))
    if not rows:
        print("No generated videos found - run the generate stage first.")
        return 1
    write_metadata(rows, out)
    if missing:
        log.warning("%d ledger entry(ies) had no readable file and were skipped", missing)
    print(json.dumps(coverage(rows), indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
