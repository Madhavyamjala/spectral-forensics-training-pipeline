"""
INSwapper face swap (InsightFace buffalo_l + inswapper_128).

Identity-conditioned latent swap, per the spec's pipeline D. This is the high-throughput
baseline of the face-swap family: detection, recognition and swapping all run in ONNX Runtime on
the GPU, so it comfortably clears the throughput the one-week budget needs.

Pipeline, matching the family's shared recipe:
    face detect/align (buffalo_l) -> source identity embedding -> swap -> blend
    -> temporal consistency (identity locked to one source face for the whole clip)
    -> audio restored by remuxing the original track

Per-video metadata recorded: source/target identity, detected face quality, visibility and pose,
which is what the document asks for so artifact attribution can separate "learned the swap" from
"learned the model fingerprint".
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from _common import (emit, env_root, has_audio, note, read_video, require, serve, write_video)


class State:
    def __init__(self, app, swapper):
        """Store the reusable components required by this model worker."""
        self.app = app
        self.swapper = swapper
        self._identity_cache = {}

    def identity(self, path: str):
        """Best-quality face from a clip, cached - the identity donor is reused across jobs."""
        if path in self._identity_cache:
            return self._identity_cache[path]
        frames, _ = read_video(path, max_frames=24, max_side=1024)
        best, best_score = None, -1.0
        for frame in frames[::3]:
            for face in self.app.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                x1, y1, x2, y2 = face.bbox
                score = float(face.det_score) * float((x2 - x1) * (y2 - y1)) ** 0.5
                if score > best_score:
                    best, best_score = face, score
        if len(self._identity_cache) > 64:
            self._identity_cache.clear()
        self._identity_cache[path] = best
        return best


def load() -> State:
    """Load the model and return its reusable worker state."""
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    model_path = require(env_root() / "weights" / "inswapper_128.onnx", "inswapper_128.onnx")
    swapper = get_model(str(model_path), providers=providers)
    note("inswapper: buffalo_l + inswapper_128 loaded")
    return State(app, swapper)


def render(state: State, payload: dict) -> dict:
    """Render one generation job with the loaded worker state."""
    target_path = payload["source_path"]
    donor_path = payload.get("driving_path") or target_path
    frames, fps = read_video(target_path, max_side=1280)

    source_face = state.identity(donor_path)
    if source_face is None and donor_path != target_path:
        source_face = state.identity(target_path)
    if source_face is None:
        raise RuntimeError("no usable source identity face found in the donor clip")

    swapped, hit, qualities, sizes = [], 0, [], []
    yaws, pitches, rolls = [], [], []
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = state.app.get(bgr)
        if faces:
            hit += 1
            # swap every detected face so multi-person clips are fully manipulated
            for face in faces:
                bgr = state.swapper.get(bgr, face, source_face, paste_back=True)
            biggest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = biggest.bbox
            qualities.append(float(biggest.det_score))
            sizes.append(float(max(x2 - x1, y2 - y1)) / max(1.0, min(frame.shape[:2])))
            # buffalo_l returns a full (pitch, yaw, roll) pose, so record all three rather
            # than just yaw - the specification asks for all of them per video
            pose = getattr(biggest, "pose", None)
            if pose is not None and len(pose) >= 3:
                pitches.append(float(pose[0]))
                yaws.append(float(pose[1]))
                rolls.append(float(pose[2]))
        swapped.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if hit == 0:
        raise RuntimeError("no face detected in any frame of the target clip")

    write_video(swapped, payload["output_path"], fps=fps,
                audio_from=target_path if has_audio(target_path) else None)
    return {
        "face_swap_model": "inswapper",
        "source_identity_id": Path(donor_path).stem,
        "target_identity_id": Path(target_path).stem,
        "face_visibility": round(hit / max(1, len(frames)), 4),
        "target_face_quality": round(float(np.mean(qualities)), 4) if qualities else 0.0,
        "source_face_quality": round(float(source_face.det_score), 4),
        "face_size": round(float(np.mean(sizes)), 4) if sizes else 0.0,
        "yaw": round(float(np.mean(yaws)), 3) if yaws else "",
        "pitch": round(float(np.mean(pitches)), 3) if pitches else "",
        "roll": round(float(np.mean(rolls)), 3) if rolls else "",
        # frames where no face was found at all - the closest honest proxy for occlusion this
        # pipeline can measure without a dedicated occlusion model
        "occlusion_level": round(1.0 - hit / max(1, len(frames)), 4),
        "frames": len(swapped),
    }


if __name__ == "__main__":
    raise SystemExit(serve(load, render, model_name="inswapper"))
