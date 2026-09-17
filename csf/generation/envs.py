"""
Per-model virtual environments.

The manipulation models in the spec cannot share one Python environment: SimSwap and GANimation
want old torch, MuseTalk and FLUX want new diffusers, InsightFace wants onnxruntime, ProPainter
pins its own mmcv. Rather than fight that, every adapter declares an `EnvSpec` and gets its own
venv under `<cache>/generation/envs/<name>/`, created on first use and reused afterwards.

An env is built in four steps, each skipped when already done:
    1. `python -m venv` (or a named conda env when `conda` is configured)
    2. pip install torch from the CUDA wheel index
    3. pip install the adapter's requirements
    4. clone the upstream git repo(s) and fetch weights

Readiness is recorded in `<env>/.csf_ready.json`, keyed by a hash of the spec, so changing a
requirement rebuilds the env instead of silently running against a stale one.

Input : `EnvSpec`, cache dir.
Output: `ReadyEnv` with the interpreter path and repo locations; raises `EnvBuildError` on failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from csf.generation.progress import Heartbeat
from csf.logging_utils import get_logger

log = get_logger("generation.envs")

DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"


class EnvBuildError(RuntimeError):
    """Raised when an adapter's environment cannot be built."""


@dataclass
class GitRepo:
    url: str
    commit: Optional[str] = None      # pin for reproducibility; None = default branch tip
    name: Optional[str] = None

    @property
    def folder(self) -> str:
        """Return the checkout directory name for this repository."""
        return self.name or Path(self.url.rstrip("/")).stem.replace(".git", "")


@dataclass
class WeightFile:
    """A checkpoint to place inside the repo before the model can run."""
    dest: str                         # path relative to the repo/env root
    url: Optional[str] = None         # direct download
    hf_repo: Optional[str] = None     # or a Hugging Face repo id
    hf_file: Optional[str] = None
    hf_type: str = "model"


@dataclass
class EnvSpec:
    name: str
    python: str = ""                              # interpreter to build the venv from
    torch: str = "torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"
    torch_index: str = DEFAULT_TORCH_INDEX
    requirements: Sequence[str] = ()
    repos: Sequence[GitRepo] = ()
    weights: Sequence[WeightFile] = ()
    env_vars: Dict[str, str] = field(default_factory=dict)
    post_install: Sequence[Sequence[str]] = ()    # extra commands run inside the env
    #: Hub repos a post-install hook pulls. Declared explicitly so the prefetch stage can check
    #: access to them up front - parsing them out of the hook's command line is fragile.
    hub_repos: Sequence[str] = ()
    note: str = ""

    def digest(self) -> str:
        """Return a stable digest of inputs that define this environment."""
        payload = json.dumps({
            "python": self.python, "torch": self.torch, "torch_index": self.torch_index,
            "requirements": list(self.requirements),
            "repos": [[r.url, r.commit] for r in self.repos],
            "hub_repos": list(self.hub_repos),
            "weights": [[w.dest, w.url, w.hf_repo, w.hf_file] for w in self.weights],
            "post_install": [list(c) for c in self.post_install],
        }, sort_keys=True)
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


@dataclass
class ReadyEnv:
    spec: EnvSpec
    root: Path
    python: Path
    repos: Dict[str, Path]

    def environ(self) -> Dict[str, str]:
        """Return environment variables for commands in the ready environment."""
        env = os.environ.copy()
        env.update(self.spec.env_vars)
        # make the cloned repos importable without each worker hard-coding paths
        extra = os.pathsep.join(str(p) for p in self.repos.values())
        if extra:
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (extra, env.get("PYTHONPATH", ""))))
        env.setdefault("CSF_ENV_ROOT", str(self.root))
        return env


# --------------------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------------------


def _run(cmd: Sequence[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
         timeout: int = 3600, what: str = "") -> None:
    """Run a build step, streaming its output so a long install is visibly alive.

    Environment builds spend most of their time inside a single `pip install` that can run for
    fifteen minutes. Buffering its output until it exits makes the terminal look frozen, which is
    indistinguishable from a hang, so each line is logged as it arrives and a heartbeat reports
    elapsed time plus whatever pip last printed.
    """
    from csf.generation.progress import run_streaming

    label = what or " ".join(str(c) for c in cmd[:3])
    log.debug("$ %s", " ".join(str(c) for c in cmd))
    started = time.monotonic()
    try:
        proc = run_streaming(cmd, cwd=cwd, env=env, timeout=timeout, desc=label)
    except subprocess.TimeoutExpired as exc:
        raise EnvBuildError(f"{label} timed out after {timeout}s") from exc
    except OSError as exc:
        raise EnvBuildError(f"{label} could not be started: {exc}") from exc
    if proc.returncode != 0:
        raise EnvBuildError(f"{label} failed (exit {proc.returncode}):\n"
                            f"{(proc.stdout or '')[-2000:]}")
    log.info("  %s done in %.0fs", label, time.monotonic() - started)


def _venv_python(root: Path) -> Path:
    """Return the Python executable path for a virtual environment."""
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _fetch(url: str, dest: Path) -> None:
    """Download an asset atomically when it is not already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        _run(["curl", "-fsSL", "--retry", "5", "--retry-delay", "5", "-o", str(dest), url],
             timeout=7200, what=f"download {url}")
    elif shutil.which("wget"):
        _run(["wget", "-q", "--tries=5", "-O", str(dest), url], timeout=7200, what=f"download {url}")
    else:
        import urllib.request
        with urllib.request.urlopen(url, timeout=300) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)


def build_env(spec: EnvSpec, envs_root: Path, force: bool = False,
              offline: bool = False) -> ReadyEnv:
    """Create (or reuse) the venv, repos and weights for one adapter."""
    root = Path(envs_root) / spec.name
    venv_dir = root / "venv"
    marker = root / ".csf_ready.json"
    repos_dir = root / "repos"
    py = _venv_python(venv_dir)

    want = spec.digest()
    if marker.exists() and not force:
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
        if state.get("digest") == want and py.exists():
            return ReadyEnv(spec, root, py,
                            {r.folder: repos_dir / r.folder for r in spec.repos})
        if state.get("digest") != want and state:
            log.info("Env '%s' spec changed (%s -> %s) -> rebuilding", spec.name,
                     state.get("digest"), want)

    if offline:
        raise EnvBuildError(f"Env '{spec.name}' is not built and generation.offline is set. "
                            f"Pre-build it with: python -m csf.generation.envs --build {spec.name}")

    steps = (2 + (1 if spec.torch else 0) + (1 if spec.requirements else 0)
             + len(spec.repos) + len(spec.weights) + len(spec.post_install))
    log.info("Building environment '%s' at %s | %d step(s). The torch install alone usually "
             "takes 5-20 minutes; each step streams its output below.", spec.name, root, steps)
    build_started = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    step = [0]

    def announce(label: str) -> str:
        step[0] += 1
        log.info("[%s  step %d/%d] %s", spec.name, step[0], steps, label)
        return label

    if not py.exists():
        base = spec.python or sys.executable
        _run([base, "-m", "venv", str(venv_dir)],
             what=announce("creating the virtual environment"), timeout=600)
    else:
        announce("virtual environment already present")
    _run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"],
         what=announce("upgrading pip"), timeout=1800)

    if spec.torch:
        cmd = [py, "-m", "pip", "install", *spec.torch.split()]
        if spec.torch_index:
            cmd += ["--index-url", spec.torch_index]
        _run(cmd, what=announce(f"installing {spec.torch.split()[0]} (several GB)"), timeout=7200)

    if spec.requirements:
        _run([py, "-m", "pip", "install", *spec.requirements],
             what=announce(f"installing {len(spec.requirements)} requirement(s)"), timeout=7200)

    repos: Dict[str, Path] = {}
    for repo in spec.repos:
        target = repos_dir / repo.folder
        if not (target / ".git").exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", "--recursive", repo.url, str(target)],
                 what=announce(f"cloning {repo.folder}"), timeout=3600)
        else:
            announce(f"{repo.folder} already cloned")
        if repo.commit:
            _run(["git", "checkout", repo.commit], cwd=target, what=f"checkout {repo.commit}")
        repos[repo.folder] = target

    for weight in spec.weights:
        dest = root / weight.dest
        if dest.exists() and dest.stat().st_size > 0:
            announce(f"{Path(weight.dest).name} already present")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if weight.url:
            announce(f"downloading {Path(weight.dest).name}")
            with Heartbeat(f"downloading {Path(weight.dest).name}"):
                _fetch(weight.url, dest)
        elif weight.hf_repo:
            announce(f"downloading {weight.hf_file} from {weight.hf_repo}")
            code = ("from huggingface_hub import hf_hub_download; import shutil; "
                    f"p=hf_hub_download({weight.hf_repo!r}, {weight.hf_file!r}, "
                    f"repo_type={weight.hf_type!r}); shutil.copy2(p, {str(dest)!r})")
            _run([py, "-c", code], what=f"fetching {weight.hf_file}", timeout=7200)
        else:
            raise EnvBuildError(f"Weight {weight.dest} for env {spec.name} has no url or hf_repo")

    # post-install hooks (upstream downloaders, huggingface-cli pulls) locate the env through
    # CSF_ENV_ROOT, which is otherwise only injected at worker launch - set it here too, and put
    # the venv's bin dir first so `huggingface-cli` resolves to this env's copy.
    if spec.post_install:
        hook_env = os.environ.copy()
        hook_env.update(spec.env_vars)
        hook_env["CSF_ENV_ROOT"] = str(root)
        hook_env["PATH"] = os.pathsep.join(
            [str(py.parent), hook_env.get("PATH", "")]).rstrip(os.pathsep)
        for cmd in spec.post_install:
            _run([py, *cmd], cwd=root, env=hook_env,
                 what=announce("running the post-install hook (may download weights)"),
                 timeout=7200)

    marker.write_text(json.dumps({"digest": want, "name": spec.name,
                                  "repos": {k: str(v) for k, v in repos.items()}}, indent=2),
                      encoding="utf-8")
    log.info("Environment '%s' ready in %.0fs", spec.name, time.monotonic() - build_started)
    return ReadyEnv(spec, root, py, repos)


def env_status(specs: Sequence[EnvSpec], envs_root: Path) -> Dict[str, str]:
    """"ready" / "stale" / "missing" per env, without building anything."""
    out: Dict[str, str] = {}
    for spec in specs:
        marker = Path(envs_root) / spec.name / ".csf_ready.json"
        if not marker.exists():
            out[spec.name] = "missing"
            continue
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out[spec.name] = "stale"
            continue
        out[spec.name] = "ready" if state.get("digest") == spec.digest() else "stale"
    return out


def _main() -> int:
    """`python -m csf.generation.envs --build <name|all>` pre-builds envs before a run."""
    import argparse

    from csf.generation.adapters import ADAPTERS, env_specs

    ap = argparse.ArgumentParser(description="Build the per-model environments")
    ap.add_argument("--build", default="", help="env or adapter name, or 'all'")
    ap.add_argument("--envs-root", default="./cache/generation/envs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    specs = env_specs()
    root = Path(args.envs_root)
    if args.status or not args.build:
        status = env_status(list(specs.values()), root)
        for name in sorted(status):
            print(f"{status[name]:>8}  {name}")
        implemented = sum(1 for a in ADAPTERS.values() if a.implemented)
        print(f"\n{implemented}/{len(ADAPTERS)} adapters implemented; {len(specs)} environments")
        return 0

    wanted = list(specs.values()) if args.build == "all" else \
        [specs[k] for k in specs if k == args.build] or \
        ([specs[ADAPTERS[args.build].env_name]] if args.build in ADAPTERS else [])
    if not wanted:
        print(f"Unknown env/adapter {args.build!r}. Known envs: {sorted(specs)}")
        return 2
    failures = []
    for spec in wanted:
        try:
            build_env(spec, root, force=args.force)
        except EnvBuildError as exc:
            failures.append((spec.name, str(exc)[:400]))
            log.error("Env '%s' failed to build: %s", spec.name, exc)
    for name, err in failures:
        print(f"FAILED {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
