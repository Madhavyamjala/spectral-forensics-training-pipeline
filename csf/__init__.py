"""
Compatibility alias: this package was renamed to `safer`.

`import csf`, `import csf.inference`, `from csf.eval.metrics import ...` and `python -m csf.<module>` all keep
working. Every `csf.X` resolves to the *same module object* as `safer.X` (not a second copy), so classes,
registries and module-level state are shared and `isinstance` checks behave. New code should import `safer`.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

import safer as _safer

_PREFIX = __name__ + "."


class _SaferAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves csf.X to safer.X. Also serves runpy (`python -m csf.X`) through get_code."""

    @staticmethod
    def _target(fullname: str) -> str:
        return "safer." + fullname[len(_PREFIX):]

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        real = importlib.util.find_spec(self._target(fullname))
        if real is None:
            return None
        # origin is the real file: runpy uses it for sys.argv[0] and __file__ under `python -m csf.X`
        spec = importlib.util.spec_from_loader(fullname, self, origin=real.origin,
                                               is_package=real.submodule_search_locations is not None)
        spec.has_location = real.has_location
        if real.submodule_search_locations is not None:
            spec.submodule_search_locations = list(real.submodule_search_locations)
        return spec

    def create_module(self, spec):
        return importlib.import_module(self._target(spec.name))

    def exec_module(self, module):
        pass                                           # already executed as safer.X

    # runpy (`python -m csf.X`) executes the target's code as __main__
    def get_code(self, fullname):
        real = importlib.util.find_spec(self._target(fullname))
        return real.loader.get_code(real.name)

    def is_package(self, fullname):
        real = importlib.util.find_spec(self._target(fullname))
        return real is not None and real.submodule_search_locations is not None


if not any(isinstance(f, _SaferAlias) for f in sys.meta_path):
    sys.meta_path.insert(0, _SaferAlias())

sys.modules[__name__] = _safer
