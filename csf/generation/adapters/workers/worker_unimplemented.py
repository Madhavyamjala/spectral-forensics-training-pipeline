"""
Placeholder worker for spec models with no runnable public release.

It refuses every job. That is the point: a model we cannot actually run must never quietly emit
a copied or lightly-perturbed source clip, because such a video would be labelled AI-Edited in
the manifest and would teach the detector that "AI-Edited" means "unchanged video".

The scheduler records these as failures with a clear reason, and `regen_manifest` leaves their
rows out of the dataset, so the class counts reported at the end are the videos that genuinely
got edited.
"""

from __future__ import annotations

import os

from _common import emit, note

REASON = ("no runnable public release wired up for this model - see the registry note in "
          "csf/generation/adapters/__init__.py and docs/REGENERATION.md")


def main() -> int:
    """Report that the requested model worker is not implemented."""
    model = os.environ.get("CSF_ADAPTER", "unimplemented")
    note(f"worker_unimplemented: refusing all jobs for '{model}' ({REASON})")
    emit({"event": "ready", "model": model, "device": os.environ.get("CSF_GPU", "?"),
          "implemented": False})
    import sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        import json
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("cmd") == "shutdown":
            break
        emit({"event": "result", "job_id": payload.get("job_id"), "ok": False,
              "error": f"NotImplemented[{payload.get('model')}]: {REASON}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
