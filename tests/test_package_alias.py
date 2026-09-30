"""
The package was renamed from `csf` to `safer`. Old imports, `python -m csf...` and exported bundles that
import `csf` must keep working, and must get the very same module objects (not a second copy).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_csf_submodules_are_the_safer_modules():
    import csf.eval.metrics as old
    import safer.eval.metrics as new
    import csf
    import safer
    assert old is new
    assert csf is safer
    from csf.eval.paper import Table as A
    from safer.eval.paper import Table as B
    assert A is B


def test_python_dash_m_through_the_alias():
    for mod in ("csf.eval.paper", "csf.data.normalize"):
        res = subprocess.run([sys.executable, "-m", mod, "--help"], cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert "usage:" in res.stdout


def test_unknown_csf_submodule_still_raises_module_not_found():
    import pytest
    with pytest.raises(ModuleNotFoundError):
        __import__("csf.does_not_exist")
