"""
First Order Motion Model - facial reenactment (pipeline B).

Learned keypoints + local affine motion: the target's identity is animated by the motion of a
*separate* driving clip, which is exactly the separation the document asks for ("Kinetics-400
supplies target videos only; the driving motion should come from a separate source").

FOMM's released vox checkpoint operates on 256x256 aligned crops, so the worker crops the target
face, reenacts it, and composites the result back into the original frame. That keeps the output
a full-resolution video rather than a face thumbnail, and leaves a realistic blend boundary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from _common import (env_root, feather, has_audio, note, read_video, repo_path, require, serve,
                     write_video)

SOURCE_SCAN = 12          # frames searched for the source face
CROP = 256
MAX_FRAMES = 120


class State:
    def __init__(self, generator, kp_detector, detector):
        """Store the reusable components required by this model worker."""
        self.generator = generator
        self.kp_detector = kp_detector
        self.detector = detector


def load() -> State:
    """Load the model and return its reusable worker state."""
    import torch
    import yaml

    repo = repo_path("fomm")
    sys.path.insert(0, str(repo))
    ckpt = require(env_root() / "weights" / "vox-adv-cpk.pth.tar", "FOMM vox-adv checkpoint")
    cfg_path = require(repo / "config" / "vox-adv-256.yaml", "FOMM config")

    from modules.generator import OcclusionAwareGenerator
    from modules.keypoint_detector import KPDetector

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    generator = OcclusionAwareGenerator(**cfg["model_params"]["generator_params"],
                                        **cfg["model_params"]["common_params"]).cuda()
    kp_detector = KPDetector(**cfg["model_params"]["kp_detector_params"],
                            **cfg["model_params"]["common_params"]).cuda()
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    generator.load_state_dict(state["generator"])
    kp_detector.load_state_dict(state["kp_detector"])
    generator.eval()
    kp_detector.eval()

    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) /
                                        "haarcascade_frontalface_default.xml"))
    note("fomm: generator + kp_detector loaded")
    return State(generator, kp_detector, cascade)


def _sample(frames, count: int):
    """Up to `count` frames spread evenly across the clip."""
    if len(frames) <= count:
        return list(frames)
    step = len(frames) / float(count)
    return [frames[min(len(frames) - 1, int(i * step))] for i in range(count)]


def _face_box(state: State, frame):
    """Detect the primary face and return its bounding box.

    The parameters are deliberately looser than the OpenCV defaults. The clip pool was
    qualified with InsightFace's SCRFD, which finds faces this Haar cascade does not - profile
    views, motion blur, anything under 64 px - so a strict cascade rejects clips the planner
    has already promised to this family.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes = state.detector.detectMultiScale(gray, 1.05, 3, minSize=(32, 32))
    if len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    pad = int(0.35 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad)
    y1 = min(frame.shape[0], y + h + pad)
    return x0, y0, x1 - x0, y1 - y0


def render(state: State, payload: dict) -> dict:
    """Render one generation job with the loaded worker state."""
    import torch

    src = payload["source_path"]
    driving = payload.get("driving_path") or ""
    if not driving or not Path(driving).exists() or driving == src:
        raise RuntimeError("reenactment job needs a driving clip distinct from the target")

    target, fps = read_video(src, max_frames=1, max_side=1280)
    full, _ = read_video(src, max_frames=MAX_FRAMES, max_side=1280)
    drive, _ = read_video(driving, max_frames=MAX_FRAMES, max_side=512)
    if len(drive) < 4:
        raise RuntimeError("driving clip is too short to reenact")

    # Search for the source frame rather than insisting on frame 0. A clip qualifies for this
    # family on `face_ratio >= 0.5` - a face in half its frames, not necessarily the first one -
    # so demanding one in frame 0 threw away clips that were correctly selected. FOMM conditions
    # on a single source frame, so any frame with a clear face will do, and the biggest
    # detection is the best-posed one available.
    source_frame, box = None, None
    for frame in _sample(full or target, SOURCE_SCAN):
        found = _face_box(state, frame)
        if found is not None and (box is None or found[2] * found[3] > box[2] * box[3]):
            source_frame, box = frame, found
    if box is None:
        raise RuntimeError(
            f"no face found in any of {SOURCE_SCAN} frames sampled from the target clip")
    x, y, w, h = box
    crop = cv2.resize(source_frame[y:y + h, x:x + w], (CROP, CROP),
                      interpolation=cv2.INTER_AREA)

    dboxes = [_face_box(state, f) for f in drive]
    dcrops = []
    for frame, db in zip(drive, dboxes):
        if db is None:
            continue
        dx, dy, dw, dh = db
        dcrops.append(cv2.resize(frame[dy:dy + dh, dx:dx + dw], (CROP, CROP),
                                 interpolation=cv2.INTER_AREA))
    if len(dcrops) < 4:
        raise RuntimeError("no usable face track in the driving clip")

    def to_tensor(img):
        """Convert an image array to a normalized model tensor."""
        return torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()

    out_crops = []
    with torch.inference_mode():
        source_t = to_tensor(crop)
        kp_source = state.kp_detector(source_t)
        kp_initial = state.kp_detector(to_tensor(dcrops[0]))
        for dc in dcrops:
            kp_driving = state.kp_detector(to_tensor(dc))
            # relative motion transfer: keeps the target's identity, borrows the driver's motion
            kp_norm = {k: v.clone() for k, v in kp_driving.items()}
            kp_norm["value"] = kp_driving["value"] - kp_initial["value"] + kp_source["value"]
            if "jacobian" in kp_driving and kp_driving["jacobian"] is not None:
                jac = torch.matmul(kp_driving["jacobian"],
                                   torch.inverse(kp_initial["jacobian"]))
                kp_norm["jacobian"] = torch.matmul(jac, kp_source["jacobian"])
            pred = state.generator(source_t, kp_source=kp_source, kp_driving=kp_norm)["prediction"]
            out_crops.append((pred[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255)
                             .astype(np.uint8))

    # composite the reenacted face back into the (static) target frame
    blend = np.zeros((h, w), np.uint8)
    cv2.ellipse(blend, (w // 2, h // 2), (int(w * 0.42), int(h * 0.46)), 0, 0, 360, 255, -1)
    alpha = feather(blend, radius=31)
    base = target[0]
    result = []
    for oc in out_crops:
        frame = base.copy()
        patch = cv2.resize(oc, (w, h), interpolation=cv2.INTER_CUBIC)
        region = frame[y:y + h, x:x + w].astype(np.float32)
        frame[y:y + h, x:x + w] = np.clip(patch.astype(np.float32) * alpha
                                          + region * (1.0 - alpha), 0, 255).astype(np.uint8)
        result.append(frame)

    write_video(result, payload["output_path"], fps=fps,
                audio_from=src if has_audio(src) else None)
    detected = sum(1 for b in dboxes if b is not None)
    return {"reenactment_model": "fomm", "target_video_id": Path(src).stem,
            "driving_video_id": Path(driving).stem,
            "target_identity": Path(src).stem, "driving_identity": Path(driving).stem,
            "face_size": round(max(w, h) / max(1, min(base.shape[:2])), 4),
            # frames of the driving clip with no detectable face
            "occlusion_level": round(1.0 - detected / max(1, len(drive)), 4),
            "frames": len(result)}


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="fomm"))
