"""
Every module an upstream entry point imports, checked in one pass.

Finding missing dependencies by running the model discovers exactly one per attempt: the
first `import` Python reaches. Each fix reveals the next, and each round costs an env
rebuild and a render. E2FGVI alone took four rounds - matplotlib, then mmcv, then the
mmcv.runner that 2.x had deleted - and none of those needed a GPU to discover.

So walk the import graph instead. Parse the entry point, follow the imports that resolve
inside the repo, and for everything else ask the environment's own interpreter whether it
can find the module. The answer is the complete list, before anything is rendered.

Two distinctions matter for the answer to be worth acting on:

  * `mmcv.runner` is not `mmcv`. A repo that imports the submodule needs the release that
    still has it, so the dotted name is what gets reported.
  * An import inside `try: ... except ImportError:` is optional by construction - upstream
    ships a fallback. Wan2.1 guards flash-attn that way and runs fine without it. Those are
    reported separately and never fail the check.

`python -m csf.generation.envs --scan all` prints the report.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from csf.logging_utils import get_logger

log = get_logger("generation.importscan")

#: Runs inside the env's interpreter: it is the only one that can answer "is this importable
#: here", and it is also where the repos are laid out.
SCAN_SOURCE = r'''
import ast, importlib.util, json, sys
from pathlib import Path

args = json.loads(sys.argv[1])
roots = [Path(r) for r in args["roots"]]
entries = [Path(e) for e in args["entries"]]

STDLIB = set(getattr(sys, "stdlib_module_names", ()))
OPTIONAL_EXC = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}

def local_path(dotted):
    """The file a dotted name resolves to inside the repos, or None."""
    rel = dotted.replace(".", "/")
    for root in roots:
        for candidate in (root / (rel + ".py"), root / rel / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None

def optional_handlers(node):
    """Names of exceptions a try/except catches, flattened."""
    names = set()
    for handler in node.handlers:
        exc = handler.type
        parts = exc.elts if isinstance(exc, ast.Tuple) else [exc]
        for part in parts:
            if isinstance(part, ast.Name):
                names.add(part.id)
            elif isinstance(part, ast.Attribute):
                names.add(part.attr)
            elif part is None:
                names.add("BaseException")
    return names

seen_files, queue = set(), list(entries)
external = {}          # dotted name -> {"by": file, "optional": bool}
unparsed = []

def record(dotted, source_file, optional):
    top = dotted.split(".")[0]
    if not top or top in STDLIB or top == "__future__":
        return
    if local_path(dotted) is not None or local_path(top) is not None:
        target = local_path(dotted) or local_path(top)
        if target not in seen_files:
            queue.append(target)
        return
    entry = external.setdefault(dotted, {"by": str(source_file), "optional": optional})
    # an import that appears both guarded and unguarded is genuinely required
    if not optional:
        entry["optional"] = False

while queue:
    path = queue.pop()
    if path in seen_files or not path.is_file():
        continue
    seen_files.add(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as exc:
        unparsed.append({"file": str(path), "error": str(exc)})
        continue

    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and optional_handlers(node) & OPTIONAL_EXC:
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    guarded.add(id(child))

    package_dir = path.parent
    for node in ast.walk(tree):
        optional = id(node) in guarded
        if isinstance(node, ast.Import):
            for alias in node.names:
                record(alias.name, path, optional)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative: resolve against this file's package
                base = package_dir
                for _ in range(node.level - 1):
                    base = base.parent
                parts = [base / (node.module.replace(".", "/") if node.module else "")]
                for candidate in parts:
                    for target in (Path(str(candidate) + ".py"), candidate / "__init__.py"):
                        if target.is_file() and target not in seen_files:
                            queue.append(target)
                    for alias in node.names:     # `from . import x` where x is a module
                        for target in (candidate / (alias.name + ".py"),
                                       candidate / alias.name / "__init__.py"):
                            if target.is_file() and target not in seen_files:
                                queue.append(target)
            elif node.module:
                record(node.module, path, optional)

missing, optional_missing = [], []
for dotted in sorted(external):
    try:
        found = importlib.util.find_spec(dotted) is not None
        reason = ""
    except Exception as exc:                     # a parent that raises is as good as absent
        found, reason = False, f"{type(exc).__name__}: {exc}"
    if found:
        continue
    row = {"module": dotted, "imported_by": external[dotted]["by"], "reason": reason}
    (optional_missing if external[dotted]["optional"] else missing).append(row)

print(json.dumps({"files_scanned": len(seen_files), "external": len(external),
                  "missing": missing, "optional_missing": optional_missing,
                  "unparsed": unparsed}))
'''


def scan(python: Path, roots: Sequence[Path], entries: Sequence[Path],
         timeout: int = 600) -> Dict[str, object]:
    """Run the scan inside `python` and return its report."""
    # Python puts a script's own directory on sys.path, and several of these entry points sit
    # a level down from the repo root (VACE runs vace/vace_wan_inference.py, whose `annotators`
    # package lives beside it). Mirror that, or those imports look external and unresolvable.
    all_roots = list(dict.fromkeys([Path(r) for r in roots] + [Path(e).parent for e in entries]))
    payload = json.dumps({"roots": [str(r) for r in all_roots],
                          "entries": [str(e) for e in entries]})
    try:
        proc = subprocess.run([str(python), "-c", SCAN_SOURCE, payload],
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        # one unusable interpreter must not end a sweep over every environment
        return {"error": f"could not run {python}: {exc}",
                "missing": [], "optional_missing": [], "unparsed": []}
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "").strip()[-1500:],
                "missing": [], "optional_missing": [], "unparsed": []}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": f"unreadable scan output: {proc.stdout[-500:]}",
                "missing": [], "optional_missing": [], "unparsed": []}


def format_report(name: str, report: Dict[str, object], root: Optional[Path] = None) -> List[str]:
    """Human-readable lines for one environment's scan."""
    def short(path: str) -> str:
        return str(Path(path).relative_to(root)) if root and str(path).startswith(str(root)) \
            else path

    if report.get("error"):
        return [f"{name:<16} SCAN FAILED: {report['error']}"]
    missing = report.get("missing") or []
    lines = [f"{name:<16} {len(missing)} missing | "
             f"{report.get('external', 0)} external module(s) across "
             f"{report.get('files_scanned', 0)} file(s)"]
    for row in missing:
        detail = f"  ({row['reason']})" if row.get("reason") else ""
        lines.append(f"    MISSING  {row['module']:<28} imported by {short(row['imported_by'])}"
                     f"{detail}")
    for row in report.get("optional_missing") or []:
        lines.append(f"    optional {row['module']:<28} guarded in {short(row['imported_by'])}")
    for row in report.get("unparsed") or []:
        lines.append(f"    unparsed {short(row['file'])}: {row['error']}")
    return lines
