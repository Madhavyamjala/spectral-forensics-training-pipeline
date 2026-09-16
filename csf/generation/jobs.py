"""
Turning the spec into concrete, resumable work items.

`build_jobs` expands the (family x source group x model) plan from `spec.py` into one row per
output video - 33,333 of them - binding each to a real Kinetics-400 source clip that passes the
family's filter, plus everything the adapter needs to run:

  * a variant (manipulation type / inpainting target / transformation family), drawn to match
    the document's secondary breakdown for that family,
  * a driving clip for reenactment (different source clip, so the motion is not the target's own),
  * an audio donor for lip-sync,
  * mask size and motion class for the inpainting family,
  * the removal/insertion operation for the object family, following the document's exact
    per-model split (ProPainter 700/200, Object-WIPER 900/0, AnyV2V 0/850, VideoComposer 150/700).

Two properties matter and are enforced:

  * **Deterministic.** Everything is drawn from `random.Random(seed + stable hash of the bucket)`,
    so re-running produces byte-identical job ids and source bindings. That is what makes the
    generation stage resumable after a crash.
  * **Leakage-aware splits.** Videos are grouped by their source clip before train/valid/test is
    assigned, so no Kinetics clip contributes to more than one split. Ratios follow the existing
    manifest (80 / 10 / 10).

Video ids follow the repo's established naming, `aiedit-<family>-<n>`, so `csf.data.manifest`
resolves them to `AI Edited/<family>/<video_id>.mp4` without any change to that module. The
manipulation model goes in the `generator_edit_method` column, which is what the per-method
accuracy breakdown in the ablation report keys on.

Input : the spec plan + scored clip features.
Output: `jobs.csv` - one row per video to generate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from csf.generation import spec as S
from csf.generation.filters import qualifying
from csf.logging_utils import get_logger

log = get_logger("generation.jobs")

SPLIT_RATIOS = (("train", 0.80), ("valid", 0.10), ("test", 0.10))


@dataclass
class Job:
    job_id: str
    video_id: str
    family: str
    model: str
    source_group: str
    source_clip_id: str
    source_path: str
    source_label: str
    variant: str = ""
    operation: str = ""
    mask_size: str = ""
    mask_motion: str = ""
    driving_clip_id: str = ""
    driving_path: str = ""
    audio_clip_id: str = ""
    audio_path: str = ""
    prompt: str = ""
    split: str = ""
    seed: int = 0
    metadata: str = "{}"          # JSON blob of the family's per-video metadata fields

    @property
    def repo_path(self) -> str:
        return f"AI Edited/{self.family}/{self.video_id}.mp4"


JOB_FIELDS: Tuple[str, ...] = tuple(Job.__dataclass_fields__)


def _rng(seed: int, *parts: object) -> random.Random:
    """Deterministic RNG keyed by a stable digest of `parts` (hash() is salted per process)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return random.Random((seed * 1_000_003) ^ digest)


# --------------------------------------------------------------------------------------
# prompts for the generative families
# --------------------------------------------------------------------------------------

BACKGROUND_PROMPTS: Dict[str, Sequence[str]] = {
    "bg_flux_image": (
        "a sunlit city street, photorealistic, shallow depth of field",
        "a quiet forest clearing at golden hour, photorealistic",
        "a modern office interior, soft daylight, photorealistic",
        "a sandy beach with breaking waves, photorealistic",
        "a snow-covered mountain slope under clear sky, photorealistic",
        "a busy indoor market hall, warm lighting, photorealistic",
        "an empty parking garage, fluorescent lighting, photorealistic",
    ),
    "bg_svd_video": (
        "drifting clouds over a wide open field",
        "gentle ocean waves rolling toward the shore",
        "traffic moving along a night-time city avenue",
        "rain falling on a quiet suburban street",
        "leaves moving in a windy park",
    ),
}

V2V_PROMPTS: Dict[str, Sequence[str]] = {
    "photorealistic_to_artistic": (
        "in the style of an oil painting, visible brush strokes",
        "as a watercolour painting, soft bleeding edges",
        "as a charcoal sketch, high contrast",
        "in the style of Van Gogh, swirling impasto",
    ),
    "photorealistic_to_stylized_cg": (
        "as a 3D animated film still, stylized shading",
        "in the style of cel-shaded anime",
        "as a claymation scene, visible fingerprints",
        "as a low-poly video game render",
    ),
    "appearance_material_transformation": (
        "everything made of polished marble",
        "everything made of brushed metal",
        "everything made of translucent glass",
        "everything covered in autumn foliage",
    ),
    "semantic_environmental_transformation": (
        "the same scene in heavy snowfall",
        "the same scene at night under street lights",
        "the same scene during a desert sandstorm",
        "the same scene underwater",
    ),
}

OBJECT_PROMPTS: Sequence[str] = (
    "a red rubber ball", "a ceramic coffee mug", "a potted houseplant", "a cardboard box",
    "a bicycle helmet", "a stack of books", "a wooden stool", "a small dog",
)

EXPRESSION_INTENSITY = (0.3, 0.5, 0.7, 0.9)


# --------------------------------------------------------------------------------------
# operation / variant allocation
# --------------------------------------------------------------------------------------


def object_operations(target: int) -> Dict[str, List[str]]:
    """Per-model removal/insertion lists for the object family, scaled to `target`.

    The document fixes the split per model; scaling keeps those ratios and still lands on the
    column totals produced by `spec.family_matrix`.
    """
    _, cols, _ = S.family_matrix(S.OBJECT_EDIT, target)
    out: Dict[str, List[str]] = {}
    for pipe, col_total in zip(S.OBJECT_EDIT.pipelines, cols):
        base = S.OBJECT_OPERATION_SPLIT[pipe.key]
        names = ["object_removal", "object_insertion"]
        weights = [base[n] for n in names]
        if sum(weights) == 0:
            counts = [col_total, 0]
        else:
            counts = S.apportion(col_total, weights)
        ops: List[str] = []
        for name, n in zip(names, counts):
            ops.extend([name] * n)
        out[pipe.key] = ops
    return out


def _weighted_pool(target: int, pairs: Sequence[Tuple[str, float]]) -> List[str]:
    names = [n for n, _ in pairs]
    counts = S.apportion(target, [w for _, w in pairs])
    pool: List[str] = []
    for name, n in zip(names, counts):
        pool.extend([name] * n)
    return pool


# --------------------------------------------------------------------------------------
# source binding
# --------------------------------------------------------------------------------------


class ClipAllocator:
    """Hands out source clips per (filter, label), reusing clips only when the pool is short.

    Kinetics has far fewer clips in some classes than the spec asks of them, so a clip may have
    to back more than one output video. Reuse is spread as evenly as possible and, critically,
    a reused clip keeps the same split (see `assign_splits`), so reuse never leaks across splits.
    """

    def __init__(self, features: Sequence[Dict[str, object]], seed: int):
        self.seed = seed
        self._by_filter: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
        for name in ("any", "face", "mouth", "object"):
            by_label: Dict[str, List[Dict[str, object]]] = defaultdict(list)
            for row in qualifying(features, name):
                by_label[str(row["label"])].append(row)
            for label in by_label:
                by_label[label].sort(key=lambda r: str(r["clip_id"]))
            self._by_filter[name] = by_label
        self._cursor: Dict[Tuple[str, str], int] = defaultdict(int)
        self._pools: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        self.shortfalls: Dict[str, int] = defaultdict(int)

    def available(self, filter_name: str, labels: Sequence[str]) -> int:
        by_label = self._by_filter.get(filter_name, {})
        return sum(len(by_label.get(l, ())) for l in labels)

    def take(self, filter_name: str, labels: Sequence[str], n: int,
             bucket: str) -> List[Dict[str, object]]:
        """`n` clips drawn round-robin across `labels`, cycling when the pool is exhausted."""
        by_label = self._by_filter.get(filter_name, {})
        pool: List[Dict[str, object]] = []
        for label in labels:
            pool.extend(by_label.get(label, ()))
        if not pool:
            # fall back to the widest pool so the bucket can still be produced
            fallback = "any" if filter_name != "any" else None
            if fallback:
                by_any = self._by_filter.get(fallback, {})
                for label in labels:
                    pool.extend(by_any.get(label, ()))
            if not pool:
                pool = [r for rows in self._by_filter.get("any", {}).values() for r in rows]
            if pool:
                log.debug("Bucket %s: no clip passed '%s' -> fell back to a wider pool",
                          bucket, filter_name)
        if not pool:
            raise RuntimeError(
                f"No usable source clips at all for bucket {bucket} (filter={filter_name}, "
                f"labels={list(labels)[:4]}...). The Kinetics pool is empty - run the "
                f"'kinetics' stage and check <cache>/kinetics/clip_features.csv.")
        # Shuffle once per (filter, label-set), not per bucket: the cursor below walks that one
        # ordering, so consecutive buckets over the same labels take *different* clips instead of
        # re-drawing from a freshly shuffled list. That is what keeps source reuse to a minimum
        # when a label is in demand from several models.
        key = (filter_name, "|".join(sorted(labels)))
        cached = self._pools.get(key)
        if cached is None:
            cached = sorted(pool, key=lambda r: str(r["clip_id"]))
            _rng(self.seed, "alloc", key[0], key[1]).shuffle(cached)
            self._pools[key] = cached
        pool = cached
        start = self._cursor[key]
        if n > len(pool):
            self.shortfalls[bucket] += n - len(pool)
        out = [pool[(start + i) % len(pool)] for i in range(n)]
        self._cursor[key] = (start + n) % len(pool)
        return out

    def any_clip(self, rng: random.Random, exclude: str, filter_name: str = "face") -> Optional[Dict[str, object]]:
        """A clip for a secondary role (driving video, audio donor), different from `exclude`."""
        by_label = self._by_filter.get(filter_name) or self._by_filter.get("any") or {}
        labels = sorted(by_label)
        if not labels:
            return None
        for _ in range(8):
            rows = by_label[rng.choice(labels)]
            if rows:
                pick = rows[rng.randrange(len(rows))]
                if str(pick["clip_id"]) != exclude:
                    return pick
        return None


# --------------------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------------------


def assign_splits(jobs: Sequence[Job], seed: int) -> None:
    """Assign train/valid/test per *source clip group*, in place.

    Grouping by source clip is what keeps the split leakage-aware: every output video derived
    from one Kinetics clip lands in the same split, however many families reused that clip.
    """
    groups: Dict[str, List[Job]] = defaultdict(list)
    for job in jobs:
        groups[job.source_clip_id].append(job)
    keys = sorted(groups)
    _rng(seed, "splits").shuffle(keys)

    total = len(jobs)
    targets = {name: int(round(total * ratio)) for name, ratio in SPLIT_RATIOS}
    targets["train"] += total - sum(targets.values())
    filled = {name: 0 for name, _ in SPLIT_RATIOS}

    order = ["test", "valid", "train"]          # fill the small splits first, remainder to train
    idx = 0
    for key in keys:
        group = groups[key]
        placed = False
        for _ in range(len(order)):
            name = order[idx % len(order)]
            idx += 1
            if filled[name] + len(group) <= targets[name]:
                for job in group:
                    job.split = name
                filled[name] += len(group)
                placed = True
                break
        if not placed:
            for job in group:
                job.split = "train"
            filled["train"] += len(group)
    log.info("Split assignment over %d source-clip groups: %s", len(keys), filled)


# --------------------------------------------------------------------------------------
# job construction
# --------------------------------------------------------------------------------------


def build_jobs(features: Sequence[Dict[str, object]], targets: Optional[Dict[str, int]] = None,
               seed: int = 42) -> List[Job]:
    targets = targets or S.FAMILY_TARGETS
    allocator = ClipAllocator(features, seed)
    jobs: List[Job] = []
    counter: Dict[str, int] = defaultdict(int)

    for family in S.FAMILY_LIST:
        target = targets[family.key]
        rows, cols, grid = S.family_matrix(family, target)

        # secondary breakdowns, drawn per family then consumed per cell
        variant_pool: List[str] = []
        if family.variants:
            variant_pool = _weighted_pool(target, [(n, float(w)) for n, w in family.variants])
            _rng(seed, family.key, "variants").shuffle(variant_pool)
        mask_sizes = _weighted_pool(target, S.INPAINT_MASK_SIZES) if family.key == "video_inpainting" else []
        mask_motions = _weighted_pool(target, S.INPAINT_MASK_MOTION) if family.key == "video_inpainting" else []
        if mask_sizes:
            _rng(seed, family.key, "mask_size").shuffle(mask_sizes)
            _rng(seed, family.key, "mask_motion").shuffle(mask_motions)
        operations = object_operations(target) if family.key == "object_insertion_removal" else {}
        for key in operations:
            _rng(seed, family.key, "ops", key).shuffle(operations[key])
        op_cursor: Dict[str, int] = defaultdict(int)

        v_cursor = 0
        for i, group in enumerate(family.source_groups):
            for j, pipe in enumerate(family.pipelines):
                n = grid[i][j]
                if n == 0:
                    continue
                bucket = f"{family.key}/{group.key}/{pipe.key}"
                clips = allocator.take(family.source_filter, group.labels, n, bucket)
                for k, clip in enumerate(clips):
                    rng = _rng(seed, bucket, k)
                    idx = counter[family.key]
                    counter[family.key] += 1
                    job = Job(
                        job_id=f"{family.key}:{pipe.key}:{idx}",
                        video_id=f"aiedit-{family.key}-{idx}",
                        family=family.key,
                        model=pipe.key,
                        source_group=group.key,
                        source_clip_id=str(clip["clip_id"]),
                        source_path=str(clip["path"]),
                        source_label=str(clip["label"]),
                        seed=rng.randrange(1 << 30),
                    )
                    if variant_pool:
                        job.variant = variant_pool[v_cursor % len(variant_pool)]
                        v_cursor += 1
                    if mask_sizes:
                        job.mask_size = mask_sizes[(v_cursor - 1) % len(mask_sizes)]
                        job.mask_motion = mask_motions[(v_cursor - 1) % len(mask_motions)]
                    if operations:
                        ops = operations.get(pipe.key) or ["object_removal"]
                        job.operation = ops[op_cursor[pipe.key] % len(ops)]
                        op_cursor[pipe.key] += 1
                        job.variant = job.operation
                    _bind_extras(job, family, pipe, allocator, rng)
                    jobs.append(job)

    assign_splits(jobs, seed)

    if allocator.shortfalls:
        worst = sorted(allocator.shortfalls.items(), key=lambda kv: -kv[1])[:8]
        log.warning("%d bucket(s) had fewer qualifying clips than videos requested, so some source "
                    "clips back more than one output video. Worst: %s",
                    len(allocator.shortfalls), worst)
    log.info("Built %d jobs over %d families / %d models",
             len(jobs), len({j.family for j in jobs}), len({j.model for j in jobs}))
    return jobs


def _bind_extras(job: Job, family: S.Family, pipe: S.Pipeline, allocator: ClipAllocator,
                 rng: random.Random) -> None:
    """Attach the family-specific inputs and the per-video metadata blob."""
    meta: Dict[str, object] = {"family": family.key, "model": pipe.key,
                               "model_name": pipe.name, "source_group": job.source_group,
                               "source_label": job.source_label}

    if family.key == "facial_reenactment":
        driving = allocator.any_clip(rng, job.source_clip_id, "face")
        if driving is not None:
            job.driving_clip_id = str(driving["clip_id"])
            job.driving_path = str(driving["path"])
        meta.update(target_video_id=job.source_clip_id, driving_video_id=job.driving_clip_id,
                    reenactment_model=pipe.key,
                    expression_intensity=rng.choice(EXPRESSION_INTENSITY))

    elif family.key == "lip_sync":
        donor = allocator.any_clip(rng, job.source_clip_id, "mouth")
        if donor is not None:
            job.audio_clip_id = str(donor["clip_id"])
            job.audio_path = str(donor["path"])
        meta.update(lip_sync_model=pipe.key, audio_source=job.audio_clip_id,
                    speaker_id=job.audio_clip_id, language="unknown",
                    expression_intensity=rng.choice(EXPRESSION_INTENSITY))

    elif family.key == "face_swap":
        source_id = allocator.any_clip(rng, job.source_clip_id, "face")
        if source_id is not None:
            job.driving_clip_id = str(source_id["clip_id"])     # identity donor
            job.driving_path = str(source_id["path"])
        meta.update(source_identity_id=job.driving_clip_id, target_identity_id=job.source_clip_id,
                    face_swap_model=pipe.key)

    elif family.key == "expression_attribute_editing":
        meta.update(edit_model=pipe.key, manipulation_type=job.variant,
                    edit_magnitude=rng.choice(EXPRESSION_INTENSITY))

    elif family.key == "object_insertion_removal":
        meta.update(edit_model=pipe.key, operation=job.operation)
        if job.operation == "object_insertion":
            job.prompt = rng.choice(OBJECT_PROMPTS)
            meta["object_class"] = job.prompt

    elif family.key == "video_inpainting":
        meta.update(inpaint_model=pipe.key, inpaint_target=job.variant,
                    mask_size_class=job.mask_size, mask_motion_pattern=job.mask_motion)

    elif family.key == "background_manipulation":
        prompts = BACKGROUND_PROMPTS.get(pipe.key)
        if prompts:
            job.prompt = rng.choice(list(prompts))
        meta.update(segmentation_model="sam2", background_source=pipe.key, composite_mode=pipe.key)

    elif family.key == "video_to_video":
        prompts = V2V_PROMPTS.get(job.variant) or V2V_PROMPTS["photorealistic_to_artistic"]
        job.prompt = rng.choice(list(prompts))
        meta.update(transform_model=pipe.key, transformation_family=job.variant, prompt=job.prompt,
                    strength=round(rng.uniform(0.45, 0.85), 2))

    job.metadata = json.dumps(meta, sort_keys=True)


# --------------------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------------------


def write_jobs(jobs: Sequence[Job], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(JOB_FIELDS))
        writer.writeheader()
        for job in jobs:
            writer.writerow(asdict(job))
    log.info("Wrote %d jobs -> %s", len(jobs), path)
    return path


def read_jobs(path: Path) -> List[Job]:
    with open(Path(path), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        r = {k: v for k, v in r.items() if k in Job.__dataclass_fields__}
        r["seed"] = int(r.get("seed") or 0)
        out.append(Job(**r))
    return out


def summarise(jobs: Sequence[Job]) -> Dict[str, object]:
    per_family: Dict[str, int] = defaultdict(int)
    per_model: Dict[str, int] = defaultdict(int)
    per_split: Dict[str, int] = defaultdict(int)
    for job in jobs:
        per_family[job.family] += 1
        per_model[job.model] += 1
        per_split[job.split] += 1
    return {"total": len(jobs), "per_family": dict(per_family), "per_model": dict(per_model),
            "per_split": dict(per_split),
            "distinct_source_clips": len({j.source_clip_id for j in jobs})}
