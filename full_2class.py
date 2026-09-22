"""
Full Real vs AI-Generated CSF pipeline entry point.

Note:
    This wrapper uses configs/full_2class.yaml and delegates to the normal resumable main.py
    driver, so all stages and checkpoint/export behavior remain identical to the standard full run.

TODO:
    Add a dedicated CLI alias once the two-class configuration stabilizes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import main as pipeline


def _argv() -> list[str]:
    """Inject the two-class config unless the caller explicitly supplied another config."""
    argv = sys.argv[1:]
    has_config = any(arg == "--config" or arg.startswith("--config=") for arg in argv)
    if has_config:
        return argv
    return ["--config", str(Path(__file__).resolve().parent / "configs" / "full_2class.yaml"), *argv]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_argv()]
    raise SystemExit(pipeline.main())
