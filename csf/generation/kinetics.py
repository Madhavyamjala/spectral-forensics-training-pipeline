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
  1. `local_root` - an already-extracted Kinetics-400 tree on the cluster (nothing downloaded)
  2. `hf_repo`    - a Hugging Face mirror (the default, `liuhuanjim013/kinetics400`). Community
                    mirrors differ in layout, so `probe_hf_layout` inspects the repo and picks
                    the right reader: one file per clip, tar/zip shards, or a `datasets`-loadable
                    table. The default mirror is the last of those: 241,181 rows of
                    video_id / video_path / metadata / clips[] / frames[], with no label column
                    and clip *paths* rather than inline bytes. `resolve_row_label` recovers the
                    class (falling back to a join on the official annotation CSVs, since the ids
                    are YouTube ids) and `rank_clips` picks the best clip per video using the
                    mirror's own quality metrics.
  3. `mirror_base` - the CVDF S3 shards, kept as a fallback

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

from csf.generation import progress
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


_FFPROBE_WARNED = False


def _probe_with_opencv(path: Path) -> Optional[Dict[str, object]]:
    """Fallback container probe using OpenCV, which is already a hard dependency.

    It cannot report codec, bitrate or the presence of an audio track, so those come back as
    unknown/zero/False. That is enough to build the source pool - the sampler only needs
    resolution, duration and frame rate - and it means a missing ffprobe binary degrades the
    metadata instead of emptying the pool.
    """
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            return None
        if frames <= 0:                      # some containers do not report a frame count
            ok, _ = cap.read()
            if not ok:
                return None
    finally:
        cap.release()
    duration = round(frames / fps, 3) if fps > 0 and frames > 0 else 0.0
    return {"duration_sec": duration, "width": width, "height": height,
            "fps": round(fps, 3), "codec": "unknown", "bitrate": 0, "has_audio": False,
            "probe": "opencv"}


def probe_video(path: Path) -> Optional[Dict[str, object]]:
    """Container metadata, via ffprobe when available and OpenCV otherwise.

    Returns None only when neither backend can read the file, i.e. the video really is unusable.
    An earlier version returned None whenever ffprobe was missing, which silently discarded every
    clip and made a fully-populated pool look empty.
    """
    global _FFPROBE_WARNED

    from csf.generation.ffmpeg_tools import INSTALL_HINT, ffprobe_exe, probe_with_ffmpeg

    exe = ffprobe_exe()
    info = None
    if exe is None:
        if not _FFPROBE_WARNED:
            _FFPROBE_WARNED = True
            log.warning("ffprobe was not found - container metadata will come from ffmpeg or "
                        "OpenCV instead, so codec/bitrate/audio may be recorded as unknown. The "
                        "generation workers still need ffmpeg to encode.\n%s", INSTALL_HINT)
    else:
        try:
            out = subprocess.run([exe, "-v", "error", "-print_format", "json", "-show_format",
                                  "-show_streams", str(path)],
                                 capture_output=True, timeout=60, check=True).stdout
            info = json.loads(out)
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
            info = None

    if info is not None:
        video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if video is not None:
            audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
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
                    "bitrate": int(info.get("format", {}).get("bit_rate") or 0),
                    "has_audio": audio, "probe": "ffprobe"}

    # ffmpeg carries the same stream details on stderr; imageio-ffmpeg ships it without ffprobe
    via_ffmpeg = probe_with_ffmpeg(path)
    if via_ffmpeg is not None:
        return via_ffmpeg
    return _probe_with_opencv(path)


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


#: Row keys that sometimes carry the action class directly.
LABEL_KEYS = ("label", "labels", "class", "category", "action", "action_label", "class_name")
#: Row keys that carry the video/clip payload or its path.
VIDEO_KEYS = ("clip_path", "video_path", "video", "mp4", "clip", "bytes", "path", "file")


def _safe_name(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))
    return keep[:120] or "clip"


def _first_str(value) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (list, tuple)):
        for v in value:
            got = _first_str(v)
            if got:
                return got
    return None


def label_from_path(path: str, wanted: Optional[Dict[str, int]] = None) -> Optional[str]:
    """Pull a Kinetics class out of a path like `train/playing guitar/abc_000001_000011.mp4`."""
    parts = [normalize_label(p) for p in Path(str(path)).parts[:-1]]
    for part in reversed(parts):
        if not part or part in ("train", "val", "test", "videos", "clips", "data", "."):
            continue
        if wanted is None or part in wanted:
            return part
    return None


def resolve_row_label(row: Dict[str, object], wanted: Optional[Dict[str, int]] = None,
                      annotations: Optional[Dict[str, Annotation]] = None) -> Optional[str]:
    """Work out a row's Kinetics class, whatever the mirror happens to record.

    The `liuhuanjim013/kinetics400` schema (video_id / video_path / metadata / clips / frames)
    carries no explicit label column, so the class has to be recovered. Tried in order of how
    trustworthy each source is:

      1. an explicit label field, if the mirror has one,
      2. the same inside `metadata`,
      3. the directory component of `video_path` / `clips[].clip_path`,
      4. the official Kinetics annotation table, joined on `video_id` (this mirror's ids are
         YouTube ids, which is what the annotation CSVs key on),
      5. a frame-level `annotation` string, which on some mirrors holds the action name.

    Returns a normalized label, or None when nothing matches.
    """
    def accept(cand: Optional[str]) -> Optional[str]:
        if not cand:
            return None
        norm = normalize_label(cand)
        if not norm:
            return None
        return norm if (wanted is None or norm in wanted) else None

    for key in LABEL_KEYS:
        got = accept(_first_str(row.get(key)))
        if got:
            return got

    meta = row.get("metadata")
    if isinstance(meta, dict):
        for key in LABEL_KEYS:
            got = accept(_first_str(meta.get(key)))
            if got:
                return got

    clips = row.get("clips") if isinstance(row.get("clips"), (list, tuple)) else []
    for path_value in [row.get("video_path")] + [c.get("clip_path") for c in clips
                                                 if isinstance(c, dict)]:
        path = _first_str(path_value)
        if path:
            got = label_from_path(path, wanted)
            if got:
                return got

    if annotations:
        vid = _first_str(row.get("video_id")) or _first_str(row.get("id"))
        if vid:
            ann = annotations.get(vid)
            if ann is None:                       # ids may carry the _start_end suffix
                ann = next((a for cid, a in annotations.items() if cid.startswith(vid)), None)
            if ann is not None:
                got = accept(ann.label)
                if got:
                    return got

    for clip in clips:
        if not isinstance(clip, dict):
            continue
        for frame in (clip.get("frames") or [])[:3]:
            if isinstance(frame, dict):
                got = accept(_first_str(frame.get("annotation"))
                             or _first_str(frame.get("annotation_extra")))
                if got:
                    return got
    return None


def _clip_quality(clip: Dict[str, object]) -> float:
    """A single score for ranking a row's clips. This mirror ships per-clip quality metrics and
    per-frame aesthetic scores; preferring the best clip of each video is free quality."""
    score = 0.0
    metrics = clip.get("quality_metrics")
    if isinstance(metrics, dict):
        for v in metrics.values():
            if isinstance(v, (int, float)):
                score += float(v)
    frames = clip.get("frames") or []
    aesthetic = [f.get("aesthetic_score") for f in frames
                 if isinstance(f, dict) and isinstance(f.get("aesthetic_score"), (int, float))]
    if aesthetic:
        score += float(sum(aesthetic)) / len(aesthetic)
    return score


def rank_clips(row: Dict[str, object]) -> List[Dict[str, object]]:
    """The row's clips, best first. Falls back to the whole video when there are no clips."""
    clips = [c for c in (row.get("clips") or []) if isinstance(c, dict)]
    if not clips:
        path = _first_str(row.get("video_path")) or row.get("video")
        return [{"clip_name": _first_str(row.get("video_id")) or "video", "clip_path": path}]
    return sorted(clips, key=_clip_quality, reverse=True)


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
        remaining_shards = [u for u in shards if u[1] not in done]
        shard_bar = progress.bar(len(remaining_shards), "shards", "shard", log_every=1)
        bar_handle = shard_bar.__enter__()
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
            bar_handle.update(1)
            bar_handle.set_postfix_str(f"pool {sum(counts.values())}, need {remaining}")
            log.info("Shard %s: kept %d clips | pool %d | still needed %d",
                     Path(url).name, kept, sum(counts.values()), remaining)
        shard_bar.__exit__(None, None, None)
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

    # ---------------- source 3: Hugging Face mirror (primary) ----------------

    def probe_hf_layout(self, repo_id: str, revision: Optional[str] = None
                        ) -> Tuple[str, List[str]]:
        """Work out how a Hugging Face Kinetics mirror stores its clips.

        Community mirrors are not consistent - some store `<split>/<label>/<clip>.mp4`, some ship
        WebDataset tar shards, some publish parquet with the video inline. Rather than hard-code
        one layout and fail on the others, list the repo once and classify what is actually
        there. Returns (layout, files).
        """
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
        videos = [f for f in files if Path(f).suffix.lower() in VIDEO_EXTS]
        archives = [f for f in files if f.endswith((".tar", ".tar.gz", ".tgz", ".zip"))]
        parquet = [f for f in files if f.endswith(".parquet")]

        if videos:
            # label-foldered only if the parent directory looks like a Kinetics class
            labelled = [f for f in videos if len(Path(f).parts) >= 2]
            layout = "label_folders" if labelled else "flat_videos"
            log.info("HF mirror %s: %d video file(s), layout=%s", repo_id, len(videos), layout)
            return layout, videos
        if archives:
            log.info("HF mirror %s: %d archive shard(s)", repo_id, len(archives))
            return "archives", archives
        if parquet:
            log.info("HF mirror %s: %d parquet file(s) -> Hugging Face Dataset", repo_id,
                     len(parquet))
            return "dataset", parquet
        raise RuntimeError(
            f"Could not find videos, archives or parquet in dataset {repo_id}. Files seen: "
            f"{files[:20]}. Point generation.kinetics.hf_repo at a different mirror, or set "
            f"generation.kinetics.local_root to an already-extracted tree.")

    def ingest_hf(self, repo_id: str, wanted: Dict[str, int], revision: Optional[str] = None,
                  annotations: Optional[Dict[str, Annotation]] = None,
                  workdir: Optional[Path] = None) -> Dict[str, int]:
        """Fill the pool from a Hugging Face mirror, whichever layout it uses."""
        layout, files = self.probe_hf_layout(repo_id, revision)
        if layout in ("label_folders", "flat_videos"):
            return self._hf_individual(repo_id, files, wanted, revision, layout, annotations)
        if layout == "archives":
            return self._hf_archives(repo_id, files, wanted, revision, annotations, workdir)
        return self._hf_dataset(repo_id, wanted, revision, annotations)

    # -- layout: one file per clip ------------------------------------------------------

    def _hf_individual(self, repo_id: str, files: Sequence[str], wanted: Dict[str, int],
                       revision: Optional[str], layout: str,
                       annotations: Optional[Dict[str, Annotation]]) -> Dict[str, int]:
        from huggingface_hub import hf_hub_download

        by_label: Dict[str, List[str]] = {}
        for f in files:
            path = Path(f)
            label = None
            if layout == "label_folders":
                for part in reversed(path.parts[:-1]):
                    cand = normalize_label(part)
                    if cand in wanted:
                        label = cand
                        break
            if label is None and annotations:
                ann = annotations.get(path.stem)
                label = ann.label if ann else None
            if label in wanted:
                by_label.setdefault(label, []).append(f)

        matched = sum(len(v) for v in by_label.values())
        absent = sorted(set(wanted) - set(by_label))
        log.info("HF mirror: %d file(s) match the %d label(s) the spec needs", matched, len(wanted))
        if absent:
            log.warning("%d label(s) the spec needs have no file in this mirror: %s%s",
                        len(absent), absent[:8], " ..." if len(absent) > 8 else "")
        if matched == 0:
            raise RuntimeError(
                f"No file in {repo_id} matched any of the {len(wanted)} Kinetics classes the "
                f"spec needs. The mirror may use different class spellings - check a few paths "
                f"and adjust csf/generation/spec.py, or use a different mirror.")

        counts = {label: self._count_on_disk(label) for label in wanted}
        outstanding = sum(max(0, wanted[l] - counts.get(l, 0)) for l in by_label)
        log.info("Downloading %d clip(s) from %s", outstanding, repo_id)
        with progress.bar(outstanding, "fetching clips", "clip", log_every=100) as pbar:
            counts = self._download_individual(repo_id, by_label, wanted, counts, revision, pbar)
        return counts

    def _download_individual(self, repo_id, by_label, wanted, counts, revision, pbar):
        from huggingface_hub import hf_hub_download

        for label, repo_paths in sorted(by_label.items()):
            need = wanted[label] - counts.get(label, 0)
            if need <= 0:
                continue
            dest_dir = self.clips_dir / label.replace("/", "_")
            dest_dir.mkdir(parents=True, exist_ok=True)
            taken = 0
            pbar.set_postfix_str(label[:28])
            for repo_path in sorted(repo_paths):
                if taken >= need:
                    break
                dest = dest_dir / f"{Path(repo_path).stem}.mp4"
                if dest.exists():
                    continue
                try:
                    got = hf_hub_download(repo_id, repo_path, repo_type="dataset",
                                          revision=revision)
                    shutil.copy2(got, dest)
                    counts[label] = counts.get(label, 0) + 1
                    taken += 1
                    pbar.update(1)
                except Exception as exc:                      # noqa: BLE001 - keep going
                    log.warning("HF fetch failed for %s: %s", repo_path, str(exc)[:160])
            self.ledger["counts"] = counts
            self._save_ledger()
            log.debug("  %-34s %4d/%-4d clips", label, counts.get(label, 0), wanted[label])
        return counts

    # -- layout: tar / zip shards -------------------------------------------------------

    def _hf_archives(self, repo_id: str, shards: Sequence[str], wanted: Dict[str, int],
                     revision: Optional[str], annotations: Optional[Dict[str, Annotation]],
                     workdir: Optional[Path]) -> Dict[str, int]:
        """Stream shards: download one, keep the clips we need, delete it."""
        from huggingface_hub import hf_hub_download

        counts = {label: self._count_on_disk(label) for label in wanted}
        done: Set[str] = set(self.ledger.get("shards_done", []))
        tmp_root = Path(workdir) if workdir else self.root / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        annotations = annotations or {}

        for shard in sorted(shards):
            if all(counts.get(k, 0) >= v for k, v in wanted.items()):
                log.info("All label quotas satisfied -> stopping shard download")
                break
            if shard in done:
                continue
            try:
                local = Path(hf_hub_download(repo_id, shard, repo_type="dataset",
                                             revision=revision))
            except Exception as exc:                          # noqa: BLE001
                log.warning("Could not fetch shard %s: %s", shard, str(exc)[:160])
                continue

            kept = (self._consume_zip(local, wanted, counts, annotations)
                    if shard.endswith(".zip")
                    else self._consume_shard(local, wanted, counts, annotations))
            # hf_hub_download caches into the hub cache; drop it so disk stays bounded
            try:
                local.unlink(missing_ok=True)
            except OSError:
                pass
            done.add(shard)
            self.ledger["shards_done"] = sorted(done)
            self.ledger["counts"] = counts
            self._save_ledger()
            remaining = sum(max(0, v - counts.get(k, 0)) for k, v in wanted.items())
            log.info("Shard %s: kept %d | pool %d | still needed %d",
                     Path(shard).name, kept, sum(counts.values()), remaining)
        return counts

    def _consume_zip(self, archive: Path, wanted: Dict[str, int], counts: Dict[str, int],
                     annotations: Dict[str, Annotation]) -> int:
        import zipfile

        kept = 0
        try:
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.is_dir() or Path(info.filename).suffix.lower() not in VIDEO_EXTS:
                        continue
                    clip_id, label = _label_of(info.filename, annotations)
                    if not label or wanted.get(label, 0) <= counts.get(label, 0):
                        continue
                    dest_dir = self.clips_dir / label.replace("/", "_")
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / f"{clip_id}.mp4"
                    if dest.exists():
                        continue
                    with zf.open(info) as src, tempfile.NamedTemporaryFile(
                            dir=dest_dir, delete=False) as tmp:
                        shutil.copyfileobj(src, tmp, length=1 << 20)
                        tmp_path = Path(tmp.name)
                    if tmp_path.stat().st_size == 0:
                        tmp_path.unlink(missing_ok=True)
                        continue
                    tmp_path.replace(dest)
                    counts[label] = counts.get(label, 0) + 1
                    kept += 1
        except (zipfile.BadZipFile, OSError) as exc:
            log.warning("Zip %s could not be read (%s) -> skipped", archive.name, exc)
        return kept

    # -- layout: a Hugging Face Dataset (parquet-backed, possibly nested) ----------------

    def _hf_dataset(self, repo_id: str, wanted: Dict[str, int], revision: Optional[str],
                    annotations: Optional[Dict[str, Annotation]] = None,
                    split: str = "train") -> Dict[str, int]:
        """Stream a `datasets`-loadable mirror and materialise the clips we need.

        Written against the `liuhuanjim013/kinetics400` schema - video_id / video_path /
        metadata / clips[] / frames[] - which has no label column and stores *paths* rather than
        inline bytes, but it also handles the simpler flat layouts. Two things are resolved per
        row: the action class (`resolve_row_label`) and the best clip to take (`rank_clips`,
        which uses the mirror's own quality metrics).

        Rows are streamed, so the 241k-row table is never materialised locally, and the loop
        stops as soon as every label quota is met.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                f"{repo_id} is a Hugging Face Dataset, which needs the `datasets` package:\n"
                f"    pip install datasets\n"
                f"Alternatively set generation.kinetics.local_root to an extracted tree, or "
                f"point hf_repo at a mirror that ships plain video files.") from exc

        counts = {label: self._count_on_disk(label) for label in wanted}
        log.info("Streaming dataset %s (split=%s) for %d label(s)", repo_id, split, len(wanted))
        ds = load_dataset(repo_id, split=split, streaming=True, revision=revision)

        target = sum(max(0, v - counts.get(k, 0)) for k, v in wanted.items())
        scanned = written = unlabelled = 0
        logged_keys = False
        ds_bar = progress.bar(target, "materialising clips", "clip", log_every=100)
        pbar = ds_bar.__enter__()
        for row in ds:
            scanned += 1
            if not logged_keys:
                log.info("Row schema: %s", sorted(row)[:12])
                logged_keys = True
            if scanned % 20000 == 0:
                log.info("  scanned %d row(s) | pool %d | %d unlabelled",
                         scanned, sum(counts.values()), unlabelled)

            label = resolve_row_label(row, wanted, annotations)
            if label is None:
                unlabelled += 1
                continue
            need = wanted.get(label, 0) - counts.get(label, 0)
            if need <= 0:
                continue

            dest_dir = self.clips_dir / label.replace("/", "_")
            dest_dir.mkdir(parents=True, exist_ok=True)
            for clip in rank_clips(row):
                if need <= 0:
                    break
                name = _first_str(clip.get("clip_name")) or _first_str(row.get("video_id")) \
                    or f"{label}_{counts.get(label, 0):06d}"
                dest = dest_dir / f"{_safe_name(name)}.mp4"
                if dest.exists():
                    continue
                if self._materialise(repo_id, clip, dest, revision):
                    counts[label] = counts.get(label, 0) + 1
                    written += 1
                    need -= 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"{label[:24]} | scanned {scanned:,}")

            if written and written % 200 == 0:
                self.ledger["counts"] = counts
                self._save_ledger()
                log.info("  %d clip(s) written | pool %d", written, sum(counts.values()))
            if all(counts.get(k, 0) >= v for k, v in wanted.items()):
                log.info("All label quotas satisfied after %d row(s)", scanned)
                break

        ds_bar.__exit__(None, None, None)
        self.ledger["counts"] = counts
        self._save_ledger()
        log.info("Dataset ingest done: scanned %d row(s), wrote %d clip(s), %d row(s) had no "
                 "resolvable label", scanned, written, unlabelled)
        if written == 0:
            raise RuntimeError(
                f"Streamed {scanned} row(s) from {repo_id} but could not resolve any of the "
                f"{len(wanted)} Kinetics classes the spec needs. If the mirror records classes "
                f"somewhere this code does not look, extend resolve_row_label(); if its ids are "
                f"YouTube ids, make sure generation.kinetics.mirror_base is set so the official "
                f"annotation CSVs can be joined on them.")
        return counts

    def _materialise(self, repo_id: str, clip: Dict[str, object], dest: Path,
                     revision: Optional[str]) -> bool:
        """Write one clip to `dest`, whether the row holds bytes or a path into the repo."""
        payload = None
        for key in VIDEO_KEYS:
            if key in clip and clip[key] is not None:
                payload = clip[key]
                break
        if payload is None:
            return False

        # inline bytes (datasets' Video/Audio features decode to a dict)
        data = payload.get("bytes") if isinstance(payload, dict) else payload
        if isinstance(data, (bytes, bytearray)) and data:
            dest.write_bytes(data)
            return True

        path = _first_str(payload.get("path") if isinstance(payload, dict) else payload)
        if not path:
            return False
        local = Path(path)
        if local.exists():                      # already on disk (local_files / cached)
            try:
                os.link(local, dest)
            except OSError:
                shutil.copy2(local, dest)
            return True

        from huggingface_hub import hf_hub_download
        repo_rel = path.lstrip("./")
        try:
            got = hf_hub_download(repo_id, repo_rel, repo_type="dataset", revision=revision)
            shutil.copy2(got, dest)
            return True
        except Exception as exc:                # noqa: BLE001 - one bad clip is not fatal
            log.debug("Could not fetch %s from %s: %s", repo_rel, repo_id, str(exc)[:160])
            return False

    # ---------------- pool table ----------------

    def write_pool(self, labels: Iterable[str], probe: bool = True) -> Path:
        """Probe every clip on disk and write `source_pool.csv` (the sampler's input)."""
        rows = []
        on_disk = 0
        empty_files = 0
        unreadable: List[Path] = []
        all_clips = []
        for label in sorted(set(labels)):
            d = self.clips_dir / label.replace("/", "_")
            if d.exists():
                all_clips.extend((label, c) for c in sorted(d.glob("*.mp4")))
        log.info("Probing %d clip(s)%s", len(all_clips),
                 " (ffprobe + sha256 per file)" if probe else "")
        with progress.bar(len(all_clips), "probing clips", "clip", log_every=250) as pbar:
            for label, clip in all_clips:
                pbar.update(1)
                on_disk += 1
                if clip.stat().st_size == 0:
                    empty_files += 1
                    continue
                row = {"clip_id": clip.stem, "label": label, "path": str(clip.resolve())}
                if probe:
                    meta = probe_video(clip)
                    if meta is None:
                        unreadable.append(clip)
                        continue
                    row.update(meta)
                    row["sha256"] = sha256_file(clip)
                rows.append(row)

        if unreadable:
            log.warning("%d of %d clip(s) could not be decoded and were left out of the pool. "
                        "First few: %s", len(unreadable), on_disk,
                        [str(p) for p in unreadable[:3]])
        if not rows:
            # Distinguish the two very different causes - an empty directory means the download
            # never happened, whereas files that all fail to decode means the probe is broken,
            # and telling someone to "run the kinetics stage" when it just ran and left 267
            # clips on disk sends them looking in the wrong place entirely.
            if on_disk == 0:
                raise RuntimeError(
                    f"No clips were downloaded to {self.clips_dir}. Check "
                    f"generation.kinetics.hf_repo / local_root / mirror_base and your Hub login, "
                    f"then re-run the 'kinetics' stage.")
            raise RuntimeError(
                f"{on_disk} clip(s) are present under {self.clips_dir} but none could be read "
                f"({len(unreadable)} failed to decode, {empty_files} were zero bytes), so the "
                f"source pool is empty.\n"
                f"Most often this means neither ffprobe nor OpenCV can open them:\n"
                f"  * install ffmpeg (Linux: sudo apt install ffmpeg, or "
                f"conda install -c conda-forge ffmpeg)\n"
                f"  * check a file by hand: ffprobe {unreadable[0] if unreadable else '<clip>'}\n"
                f"  * if the files are Git LFS pointers rather than video, delete "
                f"{self.clips_dir} and re-run so they are fetched properly\n"
                f"Set generation.kinetics.probe_clips=false to skip probing entirely (the pool "
                f"then carries no container metadata).")
        log.info("Source pool: %d clip(s) on disk -> %d usable", on_disk, len(rows))
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
    """Fill the source pool to `wanted` clips per label using the configured sources.

    Order: an already-extracted local tree (free), then the Hugging Face mirror (the default -
    it ships only the clips, needs no annotation CSVs and reuses the Hub client's auth and
    retries), then the CVDF S3 mirror as a fallback.
    """
    pool = SourcePool(cache_dir)
    kin = gen_cfg.kinetics
    have = pool.refresh_counts(wanted)
    missing = {k: v for k, v in wanted.items() if have.get(k, 0) < v}
    log.info("Source pool: %d/%d clips present; %d label(s) still short",
             sum(have.values()), sum(wanted.values()), len(missing))

    def shortfall(counts: Dict[str, int]) -> Dict[str, int]:
        return {k: v for k, v in wanted.items() if counts.get(k, 0) < v}

    if missing and kin.local_root:
        have = pool.ingest_local(Path(kin.local_root), wanted)
        missing = shortfall(have)

    if missing and kin.hf_repo:
        # Some mirrors - including liuhuanjim013/kinetics400 - carry no label column, only
        # YouTube ids. Load the official annotation table first so those ids can be joined to a
        # class; it is two small CSVs and it is what makes such a mirror usable at all.
        annotations: Optional[Dict[str, Annotation]] = None
        if kin.mirror_base:
            try:
                annotations = load_annotations(pool.root, kin.mirror_base, tuple(kin.splits))
            except Exception as exc:                          # noqa: BLE001 - not fatal
                log.warning("Could not load the Kinetics annotation CSVs (%s). Labels will have "
                            "to come from the mirror's own fields or its paths.", str(exc)[:200])

        log.info("Fetching Kinetics-400 clips from the Hugging Face mirror %s", kin.hf_repo)
        try:
            have = pool.ingest_hf(kin.hf_repo, wanted, revision=kin.hf_revision,
                                  annotations=annotations)
            missing = shortfall(have)
        except Exception as exc:                              # noqa: BLE001 - fall back to S3
            log.error("Hugging Face mirror %s failed: %s", kin.hf_repo, str(exc)[:400])
            if not kin.mirror_base:
                raise

    if missing and kin.mirror_base:
        log.info("Falling back to the CVDF S3 mirror for %d short label(s)", len(missing))
        annotations = load_annotations(pool.root, kin.mirror_base, tuple(kin.splits))
        have = pool.stream_mirror(kin.mirror_base, wanted, annotations,
                                  max_shards=kin.max_shards, splits=tuple(kin.splits))
        missing = shortfall(have)

    if missing:
        short = sorted(missing.items(), key=lambda kv: have.get(kv[0], 0) - kv[1])[:10]
        log.warning("%d label(s) below quota after all sources; worst: %s. The sampler will "
                    "redistribute within each source group.", len(missing),
                    [(k, have.get(k, 0), v) for k, v in short])
    if sum(have.values()) == 0:
        raise RuntimeError(
            "No Kinetics clips could be acquired from any configured source. Check "
            "generation.kinetics.hf_repo / local_root / mirror_base and your Hub login.")
    return pool.write_pool(wanted.keys(), probe=kin.probe_clips)
