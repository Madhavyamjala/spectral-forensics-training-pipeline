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
    ships a fallback. Wan2.1 guards flash-attn that way and runs fine without it.
  * An import inside a function body only runs if something calls that function, and the
    files it in turn imports are deferred with it. Wav2Lip imports `lws` inside
    `_lws_processor()`, which its default config never calls - and Wav2Lip renders. Reporting
    that as missing is how a list of 41 findings hides the six that matter.

  Only the imports that execute when the entry point is imported are counted as missing;
  the other two are listed under it so nothing is silently dropped.

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
DEFER_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

def local_path(dotted):
    # the file a dotted name resolves to inside the repos, or None
    rel = dotted.replace(".", "/")
    for root in roots:
        for candidate in (root / (rel + ".py"), root / rel / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None

def catches_import(node):
    # whether a try/except swallows an import failure
    for handler in node.handlers:
        exc = handler.type
        if exc is None:
            return True
        for part in (exc.elts if isinstance(exc, ast.Tuple) else [exc]):
            name = getattr(part, "id", None) or getattr(part, "attr", None)
            if name in OPTIONAL_EXC:
                return True
    return False

def is_type_checking(test):
    # `if TYPE_CHECKING:` - a block that never executes at runtime
    return (getattr(test, "id", None) == "TYPE_CHECKING"
            or getattr(test, "attr", None) == "TYPE_CHECKING")

seen = {}                      # path -> reached only through a deferred import
queue = [(e, False) for e in entries]
external = {}
unparsed = []

def enqueue(target, deferred):
    if target not in seen or (seen[target] and not deferred):
        queue.append((target, deferred))

def record(dotted, source_file, deferred, guarded):
    top = dotted.split(".")[0]
    if not top or top in STDLIB or top == "__future__":
        return
    target = local_path(dotted) or local_path(top)
    if target is not None:
        enqueue(target, deferred)
        return
    row = external.setdefault(dotted, {"by": str(source_file), "deferred": True,
                                       "optional": True})
    # the worst case across every path that reaches it is what matters
    if not deferred:
        row["deferred"] = False
    if not guarded:
        row["optional"] = False
    if not deferred and not guarded:
        row["by"] = str(source_file)

def enqueue_relative(node, package_dir, deferred):
    # `from . import x` / `from .mod import y`, resolved against the importing file
    base = package_dir
    for _ in range(node.level - 1):
        base = base.parent
    candidate = base / (node.module.replace(".", "/") if node.module else "")
    targets = [Path(str(candidate) + ".py"), candidate / "__init__.py"]
    for alias in node.names:
        targets += [candidate / (alias.name + ".py"), candidate / alias.name / "__init__.py"]
    for target in targets:
        if target.is_file():
            enqueue(target, deferred)

def walk(node, path, package_dir, deferred, guarded):
    if isinstance(node, ast.Import):
        for alias in node.names:
            record(alias.name, path, deferred, guarded)
        return
    if isinstance(node, ast.ImportFrom):
        if node.level:
            enqueue_relative(node, package_dir, deferred)
        elif node.module:
            record(node.module, path, deferred, guarded)
        return
    if isinstance(node, DEFER_SCOPES):
        deferred = True            # a body that runs only when something calls it
    if isinstance(node, ast.If) and is_type_checking(node.test):
        deferred = True
    if isinstance(node, ast.Try) and catches_import(node):
        for child in node.body:
            walk(child, path, package_dir, deferred, True)
        for handler in node.handlers:
            for child in handler.body:
                walk(child, path, package_dir, deferred, guarded)
        for child in list(node.orelse) + list(node.finalbody):
            walk(child, path, package_dir, deferred, guarded)
        return
    for child in ast.iter_child_nodes(node):
        walk(child, path, package_dir, deferred, guarded)

while queue:
    path, deferred = queue.pop()
    if not path.is_file():
        continue
    if path in seen and (seen[path] == deferred or not seen[path]):
        continue                   # already walked, and not newly reachable eagerly
    seen[path] = deferred
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as exc:
        unparsed.append({"file": str(path), "error": str(exc)})
        continue
    walk(tree, path, path.parent, deferred, False)

missing, deferred_missing, optional_missing = [], [], []
for dotted in sorted(external):
    try:
        found = importlib.util.find_spec(dotted) is not None
        reason = ""
    except Exception as exc:       # a parent that raises is as good as absent
        found, reason = False, "%s: %s" % (type(exc).__name__, exc)
    if found:
        continue
    info = external[dotted]
    row = {"module": dotted, "imported_by": info["by"], "reason": reason}
    if info["optional"]:
        optional_missing.append(row)
    elif info["deferred"]:
        deferred_missing.append(row)
    else:
        missing.append(row)

print(json.dumps({"files_scanned": len(seen), "external": len(external),
                  "missing": missing, "deferred_missing": deferred_missing,
                  "optional_missing": optional_missing, "unparsed": unparsed}))
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
                "missing": [], "deferred_missing": [], "optional_missing": [], "unparsed": []}
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "").strip()[-1500:],
                "missing": [], "deferred_missing": [], "optional_missing": [], "unparsed": []}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": f"unreadable scan output: {proc.stdout[-500:]}",
                "missing": [], "deferred_missing": [], "optional_missing": [], "unparsed": []}


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
    for row in report.get("deferred_missing") or []:
        lines.append(f"    deferred {row['module']:<28} inside a function in "
                     f"{short(row['imported_by'])}")
    for row in report.get("optional_missing") or []:
        lines.append(f"    optional {row['module']:<28} guarded in {short(row['imported_by'])}")
    for row in report.get("unparsed") or []:
        lines.append(f"    unparsed {short(row['file'])}: {row['error']}")
    return lines
