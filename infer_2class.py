"""
Batch inference for a two-class CSF export: Real vs Fake (AI-Generated).

Note:
    This wrapper loads one CSFDetector and applies the exported agentic model to several
    selected video files, printing a compact verdict table while optionally saving the
    complete JSON result and evidence graph for each video.

TODO:
    Add CSV output and confidence-threshold controls once the two-class calibration run is finalized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from csf.inference import CSFDetector


def _expand_videos(inputs: Iterable[str]) -> list[Path]:
    """Expand explicit files and directories into a deterministic list of video paths."""
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    paths: list[Path] = []
    for value in inputs:
        p = Path(value)
        if p.is_dir():
            paths.extend(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts)
        elif p.is_file():
            paths.append(p)
        else:
            paths.extend(Path(".").glob(value))
    unique = sorted({p.resolve() for p in paths})
    if not unique:
        raise FileNotFoundError("No video files matched the supplied paths.")
    return unique


def main() -> int:
    """Run the exported two-class model on selected videos."""
    ap = argparse.ArgumentParser(description="Run CSF two-class inference on selected videos.")
    ap.add_argument("videos", nargs="+", help="Video files, directories, or glob patterns.")
    ap.add_argument("--model-dir", required=True, help="Export folder, e.g. runs/full_2class/export")
    ap.add_argument("--profile", default="balanced", choices=["ultra_fast", "balanced", "max_security"])
    ap.add_argument("--mode", default="agentic", choices=["agentic", "static", "scanner"])
    ap.add_argument("--save-dir", default=None, help="Directory for one JSON result per video.")
    args = ap.parse_args()

    det = CSFDetector(args.model_dir)
    available = sorted(det.cfg.get("classes", []))
    expected = ["real", "ai_generated"]
    if available and available != expected:
        raise ValueError(
            f"This helper expects a Real/AI-Generated export, but the bundle declares classes={available!r}."
        )

    videos = _expand_videos(args.videos)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_dir}")
    print(f"Mode: {args.mode} | Profile: {args.profile}")
    print("")
    print(f"{'VIDEO':60s}  {'VERDICT':8s}  {'LABEL':15s}  {'CONF':>7s}  {'ACTION':20s}")
    print("-" * 120)

    for video in videos:
        result = det.predict(str(video), mode=args.mode, profile=args.profile)
        label = result["label"]
        confidence = float(result["probs"][label])
        verdict = "REAL" if not result["is_fake"] else "FAKE"
        action = str(result.get("action") or "-")
        print(f"{video.name[:60]:60s}  {verdict:8s}  {label:15s}  {confidence:7.3f}  {action:20s}")

        if save_dir:
            payload = dict(result)
            out = save_dir / f"{video.stem}.json"
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
