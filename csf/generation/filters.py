"""
Source-clip qualification.

Each family needs source clips that its manipulation can actually act on. The document sets
this out most explicitly for lip-sync ("mouth visibility -> face size -> pose -> temporal
stability -> category balance, in that priority order"), and implies it for the face families
("Kinetics-400 filtered for visible faces") and the object families ("objects visible enough
frames for reliable tracking").

This module scores every clip in the source pool once, caches the result, and exposes
`qualifies(row, filter_name)` so the sampler can pick clips that suit each family.

Detectors are chosen by what is installed, best first:
    InsightFace (buffalo_l)  - accurate boxes + yaw/pitch/roll + embeddings, GPU
    MediaPipe FaceMesh       - landmarks incl. mouth, CPU
    OpenCV Haar cascades     - always available fallback, boxes only
Object presence uses frame-differencing + edge-density statistics, which needs no weights and
is enough to reject static or empty scenes.

Input : source_pool.csv rows.
Output: <cache>/kinetics/clip_features.csv with one row per clip, plus boolean filter verdicts.
"""

from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from csf.logging_utils import get_logger

log = get_logger("generation.filters")

FILTER_NAMES = ("any", "face", "mouth", "object")


@dataclass
class ClipFeatures:
    clip_id: str
    label: str
    path: str
    frames_scanned: int = 0
    face_frames: int = 0          # frames with >=1 detected face
    face_ratio: float = 0.0       # face_frames / frames_scanned
    mean_face_size: float = 0.0   # face box side / min(frame side)
    max_faces: int = 0
    mouth_frames: int = 0         # frames with visible mouth landmarks
    mouth_ratio: float = 0.0
    mean_yaw: float = 0.0
    mean_abs_yaw: float = 0.0
    motion_score: float = 0.0     # mean abs frame difference, 0-1
    edge_density: float = 0.0     # mean Canny edge fraction, proxy for object structure
    temporal_stability: float = 0.0
    detector: str = "none"
    error: str = ""

    # ---- qualification rules ----
    def qualifies(self, name: str) -> bool:
        """Return whether the clip satisfies a named source group."""
        if self.error:
            return False
        if name == "any":
            return self.frames_scanned > 0 and self.motion_score > 0.002
        if name == "face":
            return self.face_ratio >= 0.5 and self.mean_face_size >= 0.08
        if name == "mouth":
            return (self.mouth_ratio >= 0.6 and self.mean_face_size >= 0.10
                    and self.mean_abs_yaw <= 35.0 and self.temporal_stability >= 0.5)
        if name == "object":
            return self.motion_score > 0.004 and self.edge_density >= 0.02
        raise ValueError(f"unknown filter {name!r}; choose from {FILTER_NAMES}")


# --------------------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------------------


class _Detector:
    """Lazily-initialised face detector, chosen by what the worker process can import."""

    def __init__(self, prefer_gpu: bool = False):
        """Initialize the lazy detector with the preferred device policy."""
        self.kind = "none"
        self._impl = None
        self._prefer_gpu = prefer_gpu

    def _init(self):
        """Load available face and object detector backends."""
        if self._impl is not None or self.kind == "failed":
            return
        try:
            from insightface.app import FaceAnalysis
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._prefer_gpu \
                else ["CPUExecutionProvider"]
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0 if self._prefer_gpu else -1, det_size=(320, 320))
            self._impl, self.kind = app, "insightface"
            return
        except Exception:                                    # noqa: BLE001 - optional dependency
            pass
        try:
            import mediapipe as mp
            self._impl = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=4,
                                                         refine_landmarks=False,
                                                         min_detection_confidence=0.5)
            self.kind = "mediapipe"
            return
        except Exception:                                    # noqa: BLE001
            pass
        try:
            import cv2
            cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            clf = cv2.CascadeClassifier(str(cascade))
            if clf.empty():
                raise RuntimeError("cascade did not load")
            self._impl, self.kind = clf, "haar"
            return
        except Exception as exc:                             # noqa: BLE001
            log.debug("No face detector available: %s", exc)
            self.kind = "failed"

    def detect(self, frame) -> List[Dict[str, float]]:
        """Return [{x, y, w, h, yaw, mouth}] in pixels; `mouth` is 1.0 when mouth landmarks exist."""
        self._init()
        if self.kind in ("none", "failed"):
            return []
        import cv2
        import numpy as np
        h, w = frame.shape[:2]
        if self.kind == "insightface":
            faces = self._impl.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out = []
            for f in faces:
                x1, y1, x2, y2 = [float(v) for v in f.bbox]
                yaw = float(f.pose[1]) if getattr(f, "pose", None) is not None else 0.0
                kps = getattr(f, "kps", None)
                mouth = 1.0 if kps is not None and len(kps) >= 5 else 0.0
                out.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "yaw": yaw, "mouth": mouth})
            return out
        if self.kind == "mediapipe":
            res = self._impl.process(frame)
            if not res.multi_face_landmarks:
                return []
            out = []
            for lm in res.multi_face_landmarks:
                xs = [p.x * w for p in lm.landmark]
                ys = [p.y * h for p in lm.landmark]
                # landmark 13/14 are the inner lips; their separation proxies mouth visibility
                mouth = 1.0 if len(lm.landmark) > 14 else 0.0
                # yaw proxy: nose (1) offset relative to the face box centre
                nose_x = lm.landmark[1].x * w
                cx = (min(xs) + max(xs)) / 2.0
                span = max(1.0, max(xs) - min(xs))
                yaw = float(math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (nose_x - cx) / span)))))
                out.append({"x": min(xs), "y": min(ys), "w": max(xs) - min(xs),
                            "h": max(ys) - min(ys), "yaw": yaw, "mouth": mouth})
            return out
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        boxes = self._impl.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(32, 32))
        return [{"x": float(x), "y": float(y), "w": float(bw), "h": float(bh), "yaw": 0.0, "mouth": 0.0}
                for x, y, bw, bh in boxes]


# --------------------------------------------------------------------------------------
# per-clip scoring
# --------------------------------------------------------------------------------------

_DETECTOR: Optional[_Detector] = None


def _detector() -> _Detector:
    """Return the process-wide lazily initialized detector."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = _Detector(prefer_gpu=os.environ.get("CSF_FILTER_GPU") == "1")
    return _DETECTOR


def score_clip(clip_id: str, label: str, path: str, num_frames: int = 12,
               need_faces: bool = True) -> ClipFeatures:
    """Sample `num_frames` frames and compute the qualification features for one clip."""
    import cv2
    import numpy as np

    feat = ClipFeatures(clip_id=clip_id, label=label, path=path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        feat.error = "unreadable"
        return feat
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
            feat.error = "no frames"
            return feat
        wanted = sorted({int(round(i)) for i in np.linspace(0, total - 1, num_frames)})
        frames = []
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if len(frames) < 2:
        feat.error = "too few decodable frames"
        return feat

    feat.frames_scanned = len(frames)
    small = [cv2.resize(f, (160, 160), interpolation=cv2.INTER_AREA) for f in frames]
    diffs = [float(np.mean(np.abs(small[i].astype(np.int16) - small[i - 1].astype(np.int16))) / 255.0)
             for i in range(1, len(small))]
    feat.motion_score = round(float(np.mean(diffs)), 5)
    feat.temporal_stability = round(float(1.0 / (1.0 + 8.0 * np.std(diffs))), 4)
    feat.edge_density = round(float(np.mean([np.count_nonzero(cv2.Canny(f, 80, 180)) / f[:, :, 0].size
                                             for f in small])), 5)

    if not need_faces:
        return feat

    det = _detector()
    sizes, yaws, mouths, face_frames, max_faces = [], [], 0, 0, 0
    for frame in frames:
        try:
            faces = det.detect(frame)
        except Exception as exc:                             # noqa: BLE001 - detector can be flaky
            feat.error = f"detector: {type(exc).__name__}"
            break
        if not faces:
            continue
        face_frames += 1
        max_faces = max(max_faces, len(faces))
        biggest = max(faces, key=lambda f: f["w"] * f["h"])
        ref = min(frame.shape[0], frame.shape[1])
        sizes.append(max(biggest["w"], biggest["h"]) / max(1.0, ref))
        yaws.append(biggest["yaw"])
        if biggest["mouth"] > 0:
            mouths += 1
    feat.detector = det.kind
    feat.face_frames = face_frames
    feat.max_faces = max_faces
    feat.face_ratio = round(face_frames / max(1, feat.frames_scanned), 4)
    feat.mean_face_size = round(float(sum(sizes) / len(sizes)) if sizes else 0.0, 4)
    feat.mean_yaw = round(float(sum(yaws) / len(yaws)) if yaws else 0.0, 3)
    feat.mean_abs_yaw = round(float(sum(abs(y) for y in yaws) / len(yaws)) if yaws else 0.0, 3)
    # Haar gives no landmarks; treat a solidly-detected frontal face as mouth-visible
    if det.kind == "haar":
        mouths = face_frames if feat.mean_face_size >= 0.10 else 0
    feat.mouth_frames = mouths
    feat.mouth_ratio = round(mouths / max(1, feat.frames_scanned), 4)
    return feat


def _score_one(args) -> Dict[str, object]:
    """Extract filter features for one source clip."""
    clip_id, label, path, num_frames, need_faces = args
    try:
        return asdict(score_clip(clip_id, label, path, num_frames, need_faces))
    except Exception as exc:                                 # noqa: BLE001 - never kill the pool
        return asdict(ClipFeatures(clip_id=clip_id, label=label, path=path,
                                   error=f"{type(exc).__name__}: {exc}"[:200]))


def score_pool(pool_csv: Path, out_csv: Path, workers: int = 8, num_frames: int = 12,
               need_faces: bool = True, force: bool = False) -> Path:
    """Score every clip in the pool, reusing any already-scored rows (resumable)."""
    pool_csv, out_csv = Path(pool_csv), Path(out_csv)
    with open(pool_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    done: Dict[str, Dict[str, object]] = {}
    if out_csv.exists() and not force:
        with open(out_csv, newline="", encoding="utf-8") as fh:
            done = {r["clip_id"]: r for r in csv.DictReader(fh)}
        log.info("Reusing %d already-scored clips from %s", len(done), out_csv)

    todo = [(r["clip_id"], r["label"], r["path"], num_frames, need_faces)
            for r in rows if r["clip_id"] not in done]
    log.info("Scoring %d clip(s) with %d worker(s) (%d cached)", len(todo), workers, len(done))

    results: List[Dict[str, object]] = list(done.values())
    if todo:
        if workers <= 1:
            for i, args in enumerate(todo, 1):
                results.append(_score_one(args))
                if i % 500 == 0:
                    log.info("  scored %d/%d", i, len(todo))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_score_one, a) for a in todo]
                for i, fut in enumerate(as_completed(futures), 1):
                    results.append(fut.result())
                    if i % 500 == 0:
                        log.info("  scored %d/%d", i, len(todo))

    fields = [f for f in ClipFeatures.__dataclass_fields__]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})

    counts = {name: sum(1 for r in results if _row_qualifies(r, name)) for name in FILTER_NAMES}
    detectors = {}
    for r in results:
        detectors[r.get("detector", "none")] = detectors.get(r.get("detector", "none"), 0) + 1
    log.info("Clip features written to %s | qualifying: %s | detectors: %s",
             out_csv, json.dumps(counts), json.dumps(detectors))
    if counts["face"] == 0 and need_faces:
        log.warning("No clip passed the 'face' filter - the face detector is probably missing. "
                    "Install insightface or mediapipe in the driver environment.")
    return out_csv


def _row_qualifies(row: Dict[str, object], name: str) -> bool:
    """Return whether a serialized feature row qualifies for a source group."""
    def num(key, default=0.0):
        """Coerce a row value to float, falling back to the supplied default."""
        try:
            return float(row.get(key) or default)
        except (TypeError, ValueError):
            return default

    feat = ClipFeatures(clip_id=str(row.get("clip_id", "")), label=str(row.get("label", "")),
                        path=str(row.get("path", "")), frames_scanned=int(num("frames_scanned")),
                        face_ratio=num("face_ratio"), mean_face_size=num("mean_face_size"),
                        mouth_ratio=num("mouth_ratio"), mean_abs_yaw=num("mean_abs_yaw"),
                        motion_score=num("motion_score"), edge_density=num("edge_density"),
                        temporal_stability=num("temporal_stability"),
                        error=str(row.get("error") or ""))
    return feat.qualifies(name)


def load_features(path: Path) -> List[Dict[str, object]]:
    """Load clip feature rows from a JSON Lines file."""
    with open(Path(path), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def qualifying(features: Sequence[Dict[str, object]], name: str) -> List[Dict[str, object]]:
    """Index qualifying clips by source group."""
    return [r for r in features if _row_qualifies(r, name)]
