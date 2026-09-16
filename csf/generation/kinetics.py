"""
Kinetics-400 acquisition and source-clip pooling.

The regeneration spec draws every source clip from Kinetics-400, but only references 147 of
its 400 classes. Downloading the full ~450 GB release to keep ~37% of it is wasteful and slow,
so this module streams the CVDF mirror shard by shard:

    download shard -> extract only the clips whose label is needed -> delete the shard

which keeps peak disk use at roughly (kept clips + one shard) instead of the whole dataset.
It stops as soon as every label quota is met, so a run that needs 60k clips does not pull all
241 training shards.

Sources, in preference order:
  1. `local_root`   - an already-extracted Kinetics-400 tree on the cluster (nothing downloaded)
  2. `mirror_base`  - CVDF-style HTTP mirror of shards + annotation CSVs (the default)
  3. `hf_repo`      - a Hugging Face dataset mirror, fetched with the Hub client

Everything is resumable: extracted clips and a JSON ledger live under `<cache>/kinetics/`, and
re-running skips shards that were already consumed.

Input : GenerationConfig, the set of labels the spec needs, a per-label quota.
Output: `<cache>/kinetics/clips/<label>/<clip_id>.mp4` plus `source_pool.csv`
        (clip_id, label, path, duration, width, height, fps, has_audio, sha256).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from csf.logging_utils import get_logger

log = get_logger("generation.kinetics")

#: CVDF public mirror. Shards are gzipped tars of 10-second clips.
DEFAULT_MIRROR = "https://s3.amazonaws.com/kinetics/400"
TRAIN_SHARDS = 242
VAL_SHARDS = 20
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def normalize_label(label: str) -> str:
    """Kinetics labels appear with spaces, underscores and varying case across mirrors."""
    return " ".join(str(label).strip().strip('"').replace("_", " ").lower().split())


def _http_get(url: str, dest: Path, retries: int = 5, timeout: int = 120) -> Path:
    """Download `url` to `dest` with exponential backoff. Resumes nothing - shards are one-shot."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "csf-regen/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            code = getattr(exc, "code", None)
            if code == 404:
                tmp.unlink(missing_ok=True)
                raise FileNotFoundError(f"{url} does not exist (404)") from exc
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"download failed for {url} after {attempt} attempts: {exc}") from exc
            wait = min(5 * 2 ** (attempt - 1), 120)
            log.warning("Fetch error on %s (attempt %d/%d): %s -> retry in %.0fs",
                        url, attempt, retries, str(exc)[:160], wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def probe_video(path: Path) -> Optional[Dict[str, object]]:
    """Container metadata via ffprobe; returns None when the file is unreadable."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60, check=True).stdout
        info = json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError, OSError):
        return None
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        return None
    audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    fps = 0.0
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    try:
        duration = float(info.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    return {"duration_sec": round(duration, 3), "width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0), "fps": round(fps, 3),
            "codec": video.get("codec_name", "unknown"),
            "bitrate": int(info.get("format", {}).get("bit_rate") or 0), "has_audio": audio}


# --------------------------------------------------------------------------------------
# annotations
# --------------------------------------------------------------------------------------


@dataclass
class Annotation:
    clip_id: str
    label: str
    split: str


def load_annotations(cache: Path, mirror_base: str, splits: Sequence[str] = ("train", "val")) -> Dict[str, Annotation]:
    """youtube-id-keyed annotation table, downloaded once and cached.

    Kinetics annotation CSVs have columns: label, youtube_id, time_start, time_end, split.
    The clip file name in the shards is `<youtube_id>_<start:06d>_<end:06d>`.
    """
    ann_dir = cache / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    table: Dict[str, Annotation] = {}
    for split in splits:
        local = ann_dir / f"{split}.csv"
        if not local.exists():
            url = f"{mirror_base}/annotations/{split}.csv"
            log.info("Downloading Kinetics-400 %s annotations from %s", split, url)
            _http_get(url, local)
        with open(local, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                yid = (row.get("youtube_id") or row.get("video_id") or "").strip()
                if not yid:
                    continue
                try:
                    start, end = int(float(row["time_start"])), int(float(row["time_end"]))
                    clip_id = f"{yid}_{start:06d}_{end:06d}"
                except (KeyError, ValueError):
                    clip_id = yid
                table[clip_id] = Annotation(clip_id, normalize_label(row.get("label", "")), split)
    log.info("Kinetics annotations: %d clips across %d labels", len(table),
             len({a.label for a in table.values()}))
    return table


def _label_of(member_name: str, annotations: Dict[str, Annotation]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve (clip_id, label) for a tar member, from its parent folder or the annotation table."""
    path = Path(member_name)
    clip_id = path.stem
    ann = annotations.get(clip_id)
    if ann is not None and ann.label:
        return clip_id, ann.label
    # some mirrors store train/<label>/<clip>.mp4
    if len(path.parts) >= 2:
        folder = normalize_label(path.parts[-2])
        if folder and folder not in ("train", "val", "test", "videos"):
            return clip_id, folder
    return clip_id, None


# --------------------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------------------


class SourcePool:
    """Builds and records the pool of Kinetics clips available as manipulation sources."""

    def __init__(self, cache_dir: Path):
        self.root = Path(cache_dir) / "kinetics"
        self.clips_dir = self.root / "clips"
        self.pool_csv = self.root / "source_pool.csv"
        self.ledger_path = self.root / "ledger.json"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.ledger: Dict[str, object] = {"shards_done": [], "counts": {}}
        if self.ledger_path.exists():
            try:
                self.ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("Ledger %s is corrupt -> starting a fresh one", self.ledger_path)
        self.ledger.setdefault("shards_done", [])
        self.ledger.setdefault("counts", {})

    # ---------------- ledger ----------------

    def _save_ledger(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(self.ledger, indent=2), encoding="utf-8")

    def counts(self) -> Dict[str, int]:
        return dict(self.ledger.get("counts", {}))

    def _count_on_disk(self, label: str) -> int:
        d = self.clips_dir / label.replace("/", "_")
        return sum(1 for p in d.glob("*.mp4")) if d.exists() else 0

    def refresh_counts(self, labels: Iterable[str]) -> Dict[str, int]:
        counts = {label: self._count_on_disk(label) for label in labels}
        self.ledger["counts"] = counts
        self._save_ledger()
        return counts

    # ---------------- source 1: already on disk ----------------

    def ingest_local(self, local_root: Path, wanted: Dict[str, int]) -> Dict[str, int]:
        """Link clips from an existing Kinetics tree (label-foldered) into the pool."""
        local_root = Path(local_root)
        if not local_root.exists():
            raise FileNotFoundError(f"generation.kinetics.local_root does not exist: {local_root}")
        log.info("Ingesting Kinetics clips from local tree %s", local_root)
        counts = {label: self._count_on_disk(label) for label in wanted}
        for sub in sorted(p for p in local_root.rglob("*") if p.is_dir()):
            label = normalize_label(sub.name)
            need = wanted.get(label, 0) - counts.get(label, 0)
            if need <= 0:
                continue
            dest_dir = self.clips_dir / label.replace("/", "_")
            dest_dir.mkdir(parents=True, exist_ok=True)
            for clip in sorted(sub.iterdir()):
                if need <= 0:
                    break
                if clip.suffix.lower() not in VIDEO_EXTS:
                    continue
                dest = dest_dir / f"{clip.stem}.mp4"
                if dest.exists():
                    continue
                try:
                    os.link(clip, dest)          # hard link: no copy, no extra disk
                except OSError:
                    shutil.copy2(clip, dest)
                counts[label] = counts.get(label, 0) + 1
                need -= 1
        self.ledger["counts"] = counts
        self._save_ledger()
        log.info("Local ingest complete: %d clips over %d labels", sum(counts.values()), len(counts))
        return counts

    # ---------------- source 2: streaming shard download ----------------

    def stream_mirror(self, mirror_base: str, wanted: Dict[str, int], annotations: Dict[str, Annotation],
                      max_shards: Optional[int] = None, splits: Sequence[str] = ("train", "val"),
                      workdir: Optional[Path] = None) -> Dict[str, int]:
        """Download shards one at a time, keep only needed clips, delete the shard.

        Stops early once every label quota in `wanted` is satisfied.
        """
        counts = {label: self._count_on_disk(label) for label in wanted}
        done: Set[str] = set(self.ledger.get("shards_done", []))
        tmp_root = Path(workdir) if workdir else self.root / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)

        shards: List[Tuple[str, str]] = []
        for split in splits:
            n = TRAIN_SHARDS if split == "train" else VAL_SHARDS
            shards.extend((split, f"{mirror_base}/{split}/part_{i}.tar.gz") for i in range(n))

        pulled = 0
        for split, url in shards:
            if all(counts.get(k, 0) >= v for k, v in wanted.items()):
                log.info("All label quotas satisfied -> stopping shard download")
                break
            if max_shards is not None and pulled >= max_shards:
                log.info("Reached generation.kinetics.max_shards=%d -> stopping", max_shards)
                break
            if url in done:
                continue
            shard = tmp_root / Path(url).name
            try:
                _http_get(url, shard)
            except FileNotFoundError:
                log.info("Shard %s absent on the mirror -> skipping", url)
                done.add(url)
                continue
            except RuntimeError as exc:
                log.warning("Giving up on shard %s: %s", url, exc)
                continue
            pulled += 1
            kept = self._consume_shard(shard, wanted, counts, annotations)
            shard.unlink(missing_ok=True)
            done.add(url)
            self.ledger["shards_done"] = sorted(done)
            self.ledger["counts"] = counts
            self._save_ledger()
            remaining = sum(max(0, v - counts.get(k, 0)) for k, v in wanted.items())
            log.info("Shard %s: kept %d clips | pool %d | still needed %d",
                     Path(url).name, kept, sum(counts.values()), remaining)
        return counts

    def _consume_shard(self, shard: Path, wanted: Dict[str, int], counts: Dict[str, int],
                       annotations: Dict[str, Annotation]) -> int:
        kept = 0
        try:
            with tarfile.open(shard, "r:*") as tar:
                for member in tar:
                    if not member.isfile() or Path(member.name).suffix.lower() not in VIDEO_EXTS:
                        continue
                    clip_id, label = _label_of(member.name, annotations)
                    if not label or wanted.get(label, 0) <= counts.get(label, 0):
                        continue
                    dest_dir = self.clips_dir / label.replace("/", "_")
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / f"{clip_id}.mp4"
                    if dest.exists():
                        continue
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False) as tmp:
                        shutil.copyfileobj(src, tmp, length=1 << 20)
                        tmp_path = Path(tmp.name)
                    if tmp_path.stat().st_size == 0:
                        tmp_path.unlink(missing_ok=True)
                        continue
                    tmp_path.replace(dest)
                    counts[label] = counts.get(label, 0) + 1
                    kept += 1
        except (tarfile.TarError, OSError) as exc:
            log.warning("Shard %s could not be read (%s) -> skipped", shard.name, exc)
        return kept

    # ---------------- source 3: Hugging Face mirror ----------------

    def ingest_hf(self, repo_id: str, wanted: Dict[str, int], revision: Optional[str] = None) -> Dict[str, int]:
        """Pull clips from a Hugging Face dataset mirror that stores them under <label>/ folders."""
        from huggingface_hub import HfApi, hf_hub_download

        counts = {label: self._count_on_disk(label) for label in wanted}
        files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
        by_label: Dict[str, List[str]] = {}
        for f in files:
            p = Path(f)
            if p.suffix.lower() not in VIDEO_EXTS or len(p.parts) < 2:
                continue
            by_label.setdefault(normalize_label(p.parts[-2]), []).append(f)
        for label, need in wanted.items():
            have = counts.get(label, 0)
            dest_dir = self.clips_dir / label.replace("/", "_")
            dest_dir.mkdir(parents=True, exist_ok=True)
            for repo_path in sorted(by_label.get(label, []))[:max(0, need - have)]:
                dest = dest_dir / f"{Path(repo_path).stem}.mp4"
                if dest.exists():
                    continue
                try:
                    got = hf_hub_download(repo_id, repo_path, repo_type="dataset", revision=revision)
                    shutil.copy2(got, dest)
                    counts[label] = counts.get(label, 0) + 1
                except Exception as exc:                      # noqa: BLE001 - mirror layouts vary
                    log.warning("HF mirror fetch failed for %s: %s", repo_path, str(exc)[:160])
        self.ledger["counts"] = counts
        self._save_ledger()
        return counts

    # ---------------- pool table ----------------

    def write_pool(self, labels: Iterable[str], probe: bool = True) -> Path:
        """Probe every clip on disk and write `source_pool.csv` (the sampler's input)."""
        rows = []
        for label in sorted(set(labels)):
            d = self.clips_dir / label.replace("/", "_")
            if not d.exists():
                continue
            for clip in sorted(d.glob("*.mp4")):
                if clip.stat().st_size == 0:
                    continue
                row = {"clip_id": clip.stem, "label": label, "path": str(clip.resolve())}
                if probe:
                    meta = probe_video(clip)
                    if meta is None:
                        log.debug("Unreadable clip dropped from pool: %s", clip)
                        continue
                    row.update(meta)
                    row["sha256"] = sha256_file(clip)
                rows.append(row)
        if not rows:
            raise RuntimeError(f"Source pool is empty under {self.clips_dir}. Run the 'kinetics' stage first.")
        fields = list(rows[0].keys())
        self.pool_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pool_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        log.info("Source pool written: %d clips over %d labels -> %s",
                 len(rows), len({r["label"] for r in rows}), self.pool_csv)
        return self.pool_csv


# --------------------------------------------------------------------------------------
# entry point used by the `kinetics` stage
# --------------------------------------------------------------------------------------


def acquire(gen_cfg, cache_dir: Path, wanted: Dict[str, int]) -> Path:
    """Fill the source pool to `wanted` clips per label using the configured sources."""
    pool = SourcePool(cache_dir)
    have = pool.refresh_counts(wanted)
    missing = {k: v for k, v in wanted.items() if have.get(k, 0) < v}
    log.info("Source pool: %d/%d clips present; %d label(s) still short",
             sum(have.values()), sum(wanted.values()), len(missing))

    if missing and gen_cfg.kinetics.local_root:
        have = pool.ingest_local(Path(gen_cfg.kinetics.local_root), wanted)
        missing = {k: v for k, v in wanted.items() if have.get(k, 0) < v}

    if missing and gen_cfg.kinetics.hf_repo:
        have = pool.ingest_hf(gen_cfg.kinetics.hf_repo, wanted, gen_cfg.kinetics.hf_revision)
        missing = {k: v for k, v in wanted.items() if have.get(k, 0) < v}

    if missing and gen_cfg.kinetics.mirror_base:
        annotations = load_annotations(pool.root, gen_cfg.kinetics.mirror_base,
                                       tuple(gen_cfg.kinetics.splits))
        have = pool.stream_mirror(gen_cfg.kinetics.mirror_base, wanted, annotations,
                                  max_shards=gen_cfg.kinetics.max_shards,
                                  splits=tuple(gen_cfg.kinetics.splits))
        missing = {k: v for k, v in wanted.items() if have.get(k, 0) < v}

    if missing:
        short = sorted(missing.items(), key=lambda kv: have.get(kv[0], 0) - kv[1])[:10]
        log.warning("%d label(s) below quota after all sources; worst: %s. The sampler will "
                    "redistribute within each source group.", len(missing),
                    [(k, have.get(k, 0), v) for k, v in short])
    return pool.write_pool(wanted.keys(), probe=gen_cfg.kinetics.probe_clips)
