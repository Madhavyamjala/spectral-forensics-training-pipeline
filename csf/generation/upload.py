"""
Publishing the regenerated AI-Edited videos back to the dataset repo.

The user's decision for this run is "remove old and push new", so this does three things, in an
order chosen so the repo is never left inconsistent for long:

    1. upload the new `AI Edited/<family>/*.mp4` files in batched commits,
    2. upload the new manifest.csv,
    3. delete the old `AI Edited/` paths that the new manifest no longer references.

Deletion runs last and only for paths the new manifest does not use, so an interrupted push
leaves a repo with extra files rather than missing ones.

This is a destructive, outward-facing operation against a public artefact, so it never runs from
the normal stage sequence without an explicit opt-in: `generation.push.enabled` must be true, and
an interactive run asks for confirmation and a WRITE token first.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from csf.logging_utils import get_logger

log = get_logger("generation.upload")

BATCH = 400


#: Kinetics-400 is CC BY 4.0, and every AI-Edited video here is a derivative of a Kinetics clip.
#: The licence requires that redistribution credits the original authors, links the licence and
#: states that changes were made - so the dataset card carries all three, and the push refuses to
#: run without it.
KINETICS_ATTRIBUTION = """\
## Source data and attribution

The AI-Edited class of this dataset is derived from **Kinetics-400**, via the Hugging Face mirror
[`liuhuanjim013/kinetics400`](https://huggingface.co/datasets/liuhuanjim013/kinetics400).

- **Original dataset**: Kinetics-400
- **Original authors**: Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier,
  Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa
  Suleyman, Andrew Zisserman
- **Original paper**: *The Kinetics Human Action Video Dataset*,
  [arXiv:1705.06950](https://arxiv.org/abs/1705.06950)
- **Original licence**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

### Changes made

Every AI-Edited video is a **modified** Kinetics-400 clip. Source clips were re-encoded and then
altered by one of the manipulation models listed above - face swapping, reenactment, lip-sync,
expression editing, object insertion/removal, inpainting, background replacement or whole-frame
transformation. The `spec_model`, `model` and `source_clip_id` columns record, for every row,
which model produced it and which Kinetics clip it came from.

This dataset is released under **CC BY 4.0**, the same licence as the source.
"""


def build_dataset_card(cfg, report: Optional[Dict[str, object]] = None) -> str:
    """The dataset card pushed as README.md, including the CC BY 4.0 attribution."""
    report = report or {}
    per_model = report.get("per_model_actual") or {}
    rows = "\n".join(f"| `{k}` | {v:,} |" for k, v in sorted(per_model.items(),
                                                              key=lambda kv: -kv[1]))
    counts = report.get("per_class") or {}
    class_rows = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(counts.items()))
    return f"""---
license: cc-by-4.0
task_categories:
- video-classification
tags:
- deepfake-detection
- video-forensics
- kinetics400
---

# {cfg.data.repo_id.split('/')[-1]}

Three-class video forensics dataset: **Real / AI-Generated / AI-Edited**.

| class | videos |
|---|---:|
{class_rows or "| (see manifest.csv) | |"}

## AI-Edited class

Regenerated from Kinetics-400 following *AI Edited Data Source and Pipeline*: eight manipulation
families rendered by the models below. Each row records the model that actually produced it
(`model`), the specification slot it fills (`spec_model`), the source clip (`source_clip_id`) and
the edit parameters.

| model | videos |
|---|---:|
{rows or "| (see manifest.csv) | |"}

{KINETICS_ATTRIBUTION}
"""


def _api(token: Optional[str]):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def manifest_paths(manifest: Path) -> Set[str]:
    """`AI Edited/...` repo paths referenced by the new manifest."""
    out: Set[str] = set()
    with open(Path(manifest), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("class") != "ai_edited":
                continue
            family = row.get("family") or ""
            vid = row.get("video_id") or ""
            if family and vid:
                out.add(f"AI Edited/{family}/{vid}.mp4")
    return out


def existing_edited_paths(repo_id: str, token: Optional[str], revision: Optional[str] = None
                          ) -> List[str]:
    files = _api(token).list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return [f for f in files if f.replace("_", " ").lower().startswith("ai edited/")]


def push(repo_id: str, manifest: Path, video_root: Path, token: Optional[str] = None,
         delete_old: bool = True, private: bool = True, dry_run: bool = False,
         card: Optional[str] = None) -> Dict[str, object]:
    """Upload the regenerated videos + manifest, then prune the superseded ones."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    api = _api(token)
    wanted = manifest_paths(manifest)
    video_root = Path(video_root)
    present = [(p, video_root / p) for p in sorted(wanted) if (video_root / p).exists()]
    absent = len(wanted) - len(present)
    if absent:
        log.warning("%d manifest row(s) have no local file and will not be uploaded", absent)

    try:
        old = existing_edited_paths(repo_id, token)
    except Exception as exc:                                   # noqa: BLE001 - repo may be new
        log.warning("Could not list %s (%s); assuming nothing to delete", repo_id, str(exc)[:200])
        old = []
    stale = sorted(set(old) - wanted)

    plan = {"repo_id": repo_id, "upload_videos": len(present), "missing_locally": absent,
            "delete_old": len(stale) if delete_old else 0, "dry_run": dry_run,
            "metadata_csv": manifest.with_name("metadata.csv").exists()}
    log.info("Push plan: %s", plan)
    if dry_run:
        return plan

    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    for i in range(0, len(present), BATCH):
        chunk = present[i:i + BATCH]
        ops = [CommitOperationAdd(path_in_repo=rp, path_or_fileobj=str(lp)) for rp, lp in chunk]
        api.create_commit(repo_id, repo_type="dataset", operations=ops,
                          commit_message=f"Add regenerated AI-Edited videos "
                                         f"({i + 1}-{i + len(chunk)} of {len(present)})")
        log.info("Uploaded %d/%d videos", i + len(chunk), len(present))

    api.upload_file(path_or_fileobj=str(manifest), path_in_repo="manifest.csv",
                    repo_id=repo_id, repo_type="dataset",
                    commit_message="Update manifest for regenerated AI-Edited class")

    # metadata.csv travels with the manifest - it is what the per-video attribution analysis
    # reads, and a stale copy on the Hub would describe the deleted videos
    for extra in (manifest.with_name("metadata.csv"),
                  manifest.with_name("metadata_schema.json")):
        if extra.exists():
            api.upload_file(path_or_fileobj=str(extra), path_in_repo=extra.name,
                            repo_id=repo_id, repo_type="dataset",
                            commit_message=f"Update {extra.name} for the regenerated class")
            log.info("Uploaded %s", extra.name)
        elif extra.name == "metadata.csv":
            log.warning("%s not found next to the manifest - the Hub copy (if any) will be "
                        "stale. Run the regen_manifest stage to produce it.", extra)

    if card is not None:
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="dataset",
                        commit_message="Dataset card with CC BY 4.0 Kinetics-400 attribution")
        api.upload_file(path_or_fileobj=KINETICS_ATTRIBUTION.encode("utf-8"),
                        path_in_repo="ATTRIBUTION.md", repo_id=repo_id, repo_type="dataset",
                        commit_message="Kinetics-400 attribution (CC BY 4.0)")
        log.info("Uploaded the dataset card and ATTRIBUTION.md (CC BY 4.0 requires credit, a "
                 "licence link and a statement of changes)")

    if delete_old and stale:
        for i in range(0, len(stale), BATCH):
            chunk = stale[i:i + BATCH]
            api.create_commit(repo_id, repo_type="dataset",
                              operations=[CommitOperationDelete(path_in_repo=p) for p in chunk],
                              commit_message=f"Remove superseded AI-Edited videos "
                                             f"({i + 1}-{i + len(chunk)} of {len(stale)})")
            log.info("Deleted %d/%d superseded files", i + len(chunk), len(stale))

    log.info("Push complete: https://huggingface.co/datasets/%s", repo_id)
    return {**plan, "url": f"https://huggingface.co/datasets/{repo_id}"}


def push_interactive(cfg, manifest: Path, video_root: Path) -> Optional[str]:
    """Confirm, take a WRITE token, then push. Returns the repo url, or None if declined."""
    gen = cfg.generation
    repo_id = gen.push.repo_id or cfg.data.repo_id
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    interactive = bool(sys.stdin and sys.stdin.isatty())
    if interactive:
        print(f"\nAbout to replace the AI-Edited class of dataset '{repo_id}'.")
        print(f"  upload : regenerated videos listed in {manifest}")
        print(f"  delete : every existing 'AI Edited/...' file the new manifest does not use")
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            log.info("Push declined by the operator")
            return None
        if not token:
            import getpass
            token = getpass.getpass("Hugging Face WRITE token (hidden): ").strip()
    if not token:
        raise RuntimeError("No Hugging Face token available. Set HF_TOKEN or run interactively.")

    report = {}
    report_path = Path(cfg.paths.work_dir) / "metrics" / "regeneration_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    result = push(repo_id, manifest, video_root, token=token, delete_old=gen.push.delete_old,
                  private=gen.push.private, dry_run=gen.push.dry_run,
                  card=build_dataset_card(cfg, report))
    return result.get("url")
