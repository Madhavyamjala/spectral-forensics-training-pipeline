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
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from csf.generation.progress import Heartbeat
from csf.logging_utils import get_logger

log = get_logger("generation.envs")

DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"


#: requirement -> module name, for the derived import check. Only unambiguous, directly
#: installed packages belong here; anything installed transitively or through a git URL is
#: left out so the check cannot fail on something the env never promised.
_IMPORT_NAMES: Dict[str, str] = {
    "opencv-python": "cv2", "opencv-python-headless": "cv2", "opencv-contrib-python": "cv2",
    "numpy": "numpy", "pillow": "PIL", "scipy": "scipy", "scikit-image": "skimage",
    "imageio": "imageio", "diffusers": "diffusers", "transformers": "transformers",
    "accelerate": "accelerate", "safetensors": "safetensors", "insightface": "insightface",
    "onnxruntime": "onnxruntime", "onnxruntime-gpu": "onnxruntime", "librosa": "librosa",
    "huggingface-hub": "huggingface_hub", "huggingface_hub": "huggingface_hub", "einops": "einops", "omegaconf": "omegaconf",
}


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
    #: Filename to look for in the staging directory (generation.staged_weights_dir) instead of
    #: downloading. Several upstream projects host their checkpoints on Google Drive, Tsinghua
    #: Cloud or OneDrive, none of which can be fetched unattended - you download them once by
    #: hand, drop them in one folder, and the build copies them into place.
    staged_name: Optional[str] = None
    where: str = ""                   # where to obtain it, quoted in the error when it is absent


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
    #: Modules the worker imports at startup. Checked inside the env before it is marked ready,
    #: so a half-installed env fails at build time (where the error names the env and can be
    #: retried) instead of at job time as a bare ModuleNotFoundError from 200 workers.
    verify_imports: Sequence[str] = ()
    #: The upstream files a worker actually executes, relative to the env root. Their import
    #: graphs are what `--scan` walks, so a missing dependency is found by reading the repo
    #: rather than by rendering a video and waiting for the first ImportError. Deliberately
    #: NOT part of `digest()`: naming an entry point changes nothing about what is installed,
    #: and no env should rebuild because we pointed the scanner somewhere new.
    entry_points: Sequence[str] = ()
    note: str = ""

    def needs_hub(self) -> bool:
        """Whether building this env talks to the Hugging Face Hub."""
        return bool(self.hub_repos) or any(w.hf_repo for w in self.weights)

    def pip_requirements(self) -> Tuple[str, ...]:
        """Requirements as installed, with `huggingface_hub` added when the build needs it.

        The weight step and the post-install hooks import `huggingface_hub` inside the env, so
        it has to be installed there. Most envs get it transitively through diffusers or
        transformers - but insightface, fomm, liveportrait and wav2lip pull neither, and would
        only discover that at the weight step, after a full torch install. Deriving it from what
        the spec declares means a new env cannot forget it.
        """
        reqs = tuple(self.requirements)
        if self.needs_hub() and not any("huggingface" in r.lower() for r in reqs):
            reqs += ("huggingface_hub",)
        return reqs

    def checks(self) -> Tuple[str, ...]:
        """Modules this env must be able to import.

        Declared explicitly via `verify_imports`, or derived from the requirements when it is
        not. The derived set is deliberately small - packages that are installed directly and
        whose import name is unambiguous - because a false failure here would rebuild a good
        env for no reason.
        """
        if self.verify_imports:
            return tuple(self.verify_imports)
        mods: List[str] = ["torch"] if self.torch else []
        for req in self.pip_requirements():
            name = re.split(r"[=<>!\[ ]", req.strip(), 1)[0].lower()
            mod = _IMPORT_NAMES.get(name)
            if mod and mod not in mods:
                mods.append(mod)
        return tuple(mods)

    def digest(self) -> str:
        """Return a stable digest of inputs that define this environment."""
        payload = json.dumps({
            "python": self.python, "torch": self.torch, "torch_index": self.torch_index,
            "requirements": list(self.pip_requirements()),
            "repos": [[r.url, r.commit] for r in self.repos],
            "hub_repos": list(self.hub_repos),
            "weights": [[w.dest, w.url, w.hf_repo, w.hf_file, w.staged_name]
                        for w in self.weights],
            "post_install": [list(c) for c in self.post_install],
            "verify_imports": list(self.verify_imports),
        }, sort_keys=True)
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


@dataclass
class ReadyEnv:
    spec: EnvSpec
    root: Path
    python: Path
    repos: Dict[str, Path]

    def scan_imports(self) -> Dict[str, object]:
        """Report every module this env's entry points import but cannot import."""
        from csf.generation.importscan import scan

        entries = [self.root / e for e in self.spec.entry_points]
        present = [e for e in entries if e.is_file()]
        if not present:
            return {"missing": [], "optional_missing": [], "unparsed": [], "external": 0,
                    "files_scanned": 0,
                    "skipped": "no entry points declared" if not entries
                               else f"entry point(s) not present: "
                                    f"{[str(e) for e in entries if not e.is_file()]}"}
        return scan(self.python, list(self.repos.values()), present)

    def environ(self) -> Dict[str, str]:
        """Environment variables for commands run inside this env, isolated from the driver's.

        The whole point of a per-model env is that its torch, numpy and opencv are the ones the
        model was written against. An inherited `PYTHONPATH` quietly defeats that - the driver
        runs from its own venv (often inside conda on a cluster), and anything on its
        PYTHONPATH lands ahead of the env's own site-packages, so the worker imports the
        driver's torch 2.14 / numpy 2 instead of the 2.4.1 / numpy<2 it pinned. The symptom is
        not an ImportError but a crash inside the model, which is far harder to trace back.

        So PYTHONPATH is rebuilt from the cloned repos alone, `PYTHONHOME` (which some module
        systems and conda setups export, and which breaks a venv interpreter outright) is
        dropped, and user site-packages are switched off.
        """
        env = _clean_environ()
        venv_bin = Path(self.python).parent
        _ffmpeg_shim(venv_bin)
        env["VIRTUAL_ENV"] = str(venv_bin.parent)
        env["PATH"] = os.pathsep.join([str(venv_bin), env.get("PATH", "")]).rstrip(os.pathsep)
        # make the cloned repos importable without each worker hard-coding paths. This
        # *replaces* any inherited PYTHONPATH rather than prepending to it.
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in self.repos.values())
        if not env["PYTHONPATH"]:
            env.pop("PYTHONPATH")
        env.update(self.spec.env_vars)
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
        tail = (proc.stdout or "")[-2000:]
        raise EnvBuildError(f"{label} failed (exit {proc.returncode}):\n{tail}{_network_hint(tail)}")
    log.info("  %s done in %.0fs", label, time.monotonic() - started)


NETWORK_MARKERS = ("NameResolutionError", "Temporary failure in name resolution",
                   "Network is unreachable", "Could not find a version",
                   "Failed to establish a new connection", "ProxyError", "Connection refused")


def _network_hint(output: str) -> str:
    """Extra guidance when a build step failed because the machine could not reach an index."""
    if not any(marker in output for marker in NETWORK_MARKERS):
        return ""
    return ("\n\nThis looks like a network failure rather than a packaging one. Either this "
            "node cannot reach the package index (build the envs where it can, with "
            "`python -m csf.generation.envs --build all --envs-root <envs_root>`, then run "
            "generation with generation.offline=true so it never tries), or pip is configured "
            "with an index this node cannot resolve - check PIP_INDEX_URL / PIP_EXTRA_INDEX_URL "
            "and pip.conf, since an unreachable extra index fails the install even when PyPI "
            "itself is reachable.")


def _venv_python(root: Path) -> Path:
    """Absolute path to a venv's interpreter, WITHOUT resolving symlinks.

    `bin/python` inside a venv is a symlink to the base interpreter, so `Path.resolve()` walks
    straight out of the environment and hands back `/usr/bin/python3.x`. Running that binary
    skips the venv entirely: none of its packages are importable and it has no pip, which is
    exactly the "No module named pip" / "No module named 'cv2'" pair this produced on every
    job. `abspath` normalises the path (absolute, no `..`) without following the link, so the
    interpreter keeps running as its own venv.

    It only bites after the first build: before the venv exists there is no symlink to follow,
    so `resolve()` returned the right path and the env built perfectly - then every later run
    used the system Python.
    """
    name = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return Path(os.path.abspath(Path(root) / name))


def torch_pins(spec: "EnvSpec") -> List[str]:
    """The exact `name==version` pins in an env's torch line."""
    return [tok for tok in spec.torch.split() if "==" in tok]


def _write_constraints(spec: "EnvSpec", root: Path) -> Optional[Path]:
    """Write a pip constraints file pinning torch for every later install in this env.

    Without it, any requirement that declares a newer torch - SAM2 asks for >=2.5.1 - makes pip
    quietly uninstall the CUDA-matched build we just placed and pull a multi-gigabyte wheel from
    PyPI instead. On the next rebuild the torch step puts the pinned version back and the
    requirements step swaps it out again: the env never settles, and whichever torch wins is not
    the one the adapter was pinned against. As a constraint the same conflict becomes a
    resolution error at build time, naming the package that wants to move torch.
    """
    pins = list(torch_pins(spec))
    # numpy rides along: these envs pin numpy<2 for InsightFace and ONNX Runtime, and a later
    # install (opencv-python-headless 5.x wants numpy>=2) would otherwise pull it forward.
    pins += [r for r in spec.pip_requirements()
             if r.split("<")[0].split(">")[0].split("=")[0].strip() == "numpy" and
             any(op in r for op in ("<", ">", "="))]
    if not pins:
        return None
    path = Path(root) / "constraints.txt"
    path.write_text("\n".join(pins) + "\n", encoding="utf-8")
    return path


def _ffmpeg_shim(venv_bin: Path) -> None:
    """Expose imageio-ffmpeg's bundled binary as a plain `ffmpeg` on the env's PATH.

    Several upstream repos shell out to bare `ffmpeg` - Wav2Lip muxes its result that way, and
    ignores the return code, so on a machine without a system ffmpeg it exits 0 having written
    nothing at all. Every env already installs imageio-ffmpeg, which ships a binary under a
    version-stamped name; linking it under the name those repos call makes them work, and
    `environ()` already puts this directory first on PATH.

    Done here rather than at build time so envs built before this existed pick it up too,
    without a rebuild. Best-effort: an env with no imageio-ffmpeg, or a filesystem that refuses
    the link, is left exactly as it was.
    """
    target = venv_bin / "ffmpeg"
    if target.exists():
        return
    binaries = sorted((venv_bin.parent).glob(
        "lib*/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"))
    usable = [b for b in binaries if b.is_file() and os.access(b, os.X_OK)]
    if not usable:
        return
    try:
        target.symlink_to(usable[-1])
        log.debug("Linked %s -> %s", target, usable[-1])
    except OSError as exc:                                   # noqa: BLE001 - never fatal
        log.debug("Could not link ffmpeg into %s: %s", venv_bin, exc)


def _clean_environ() -> Dict[str, str]:
    """os.environ minus the interpreter state that would leak the driver's packages in.

    `pip install` decides "Requirement already satisfied" from the target interpreter's
    sys.path, so an inherited PYTHONPATH pointing at the driver's site-packages makes pip skip
    packages that are not in the venv at all - the env then looks built and fails at import.
    """
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONSTARTUP", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _probe(py: Path, code: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a one-liner inside an env's interpreter and return the completed process."""
    return subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=timeout,
                          check=False, env=_clean_environ())


def _venv_is_sane(py: Path, venv_dir: Path) -> bool:
    """Whether `py` really runs as this venv rather than as the base interpreter.

    A venv whose `pyvenv.cfg` is missing or points at a moved interpreter still has a working
    `bin/python` symlink - it simply resolves to the system Python, with none of the packages
    that were installed into the venv. That is how an env that "built fine" ends up raising
    `No module named 'cv2'` for every job, and why the error names /usr/bin/python3.x.
    """
    if not py.exists():
        return False
    try:
        proc = _probe(py, "import sys; print(sys.prefix)")
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    try:
        return Path(proc.stdout.strip()).resolve() == Path(venv_dir).resolve()
    except (OSError, ValueError):
        return False


def _ensure_pip(py: Path, label: str) -> None:
    """Make sure `py -m pip` works, bootstrapping it when the venv was built without it.

    Several distributions ship `venv` without `ensurepip` (Debian's python3-venv, RHEL's
    platform-python), and `python -m venv` then produces a venv with no pip at all. The
    subsequent install fails with a message that names the *base* interpreter, which sends you
    looking at the wrong Python entirely.
    """
    try:
        if _probe(py, "import pip").returncode == 0:
            return
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvBuildError(f"{label}: cannot run {py}: {exc}") from exc

    log.warning("%s: the virtual environment has no pip (this Python's venv module was built "
                "without ensurepip) - bootstrapping it", label)
    if _probe(py, "import ensurepip; ensurepip.bootstrap(upgrade=True)", timeout=600).returncode == 0 \
            and _probe(py, "import pip").returncode == 0:
        return

    get_pip = Path(py).parent.parent / "get-pip.py"
    try:
        _fetch("https://bootstrap.pypa.io/get-pip.py", get_pip)
        _run([py, str(get_pip)], what=f"{label}: bootstrapping pip", timeout=900)
    except EnvBuildError as exc:
        raise EnvBuildError(
            f"{label}: the virtual environment has no pip and it could not be bootstrapped "
            f"({exc}). Install the venv/ensurepip package for this Python "
            f"(Debian/Ubuntu: python3-venv, RHEL: python3-pip), or point "
            f"generation.envs at a Python that has it.") from exc
    if _probe(py, "import pip").returncode != 0:
        raise EnvBuildError(f"{label}: pip is still missing after bootstrapping")


def _verify_torch(py: Path, spec: "EnvSpec") -> None:
    """Confirm the env ended up with the torch it pinned, not one a dependency dragged in."""
    want = dict(tok.split("==", 1) for tok in torch_pins(spec)).get("torch")
    if not want:
        return
    try:
        proc = _probe(py, "import torch; print(torch.__version__)", timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvBuildError(f"env '{spec.name}': torch check could not run: {exc}") from exc
    if proc.returncode != 0:
        raise EnvBuildError(f"env '{spec.name}': torch is not importable after the build")
    got = proc.stdout.strip().split("+")[0]
    if got != want:
        raise EnvBuildError(
            f"env '{spec.name}': pinned torch=={want} but the built environment has {got}. "
            f"A requirement pulled a different build in - check which one asks for a newer "
            f"torch; the pin is in the env's constraints.txt.")


FRAMEWORK_PROBE = """
import transformers, torch
ok = transformers.utils.is_torch_available()
print('transformers %s | torch %s | torch enabled: %s'
      % (transformers.__version__, torch.__version__, ok))
raise SystemExit(0 if ok else 3)
"""


def _verify_framework(py: Path, spec: "EnvSpec") -> None:
    """Confirm transformers can actually use the torch in this env.

    `import transformers` succeeding proves nothing: transformers 5 requires torch>=2.5 and,
    below that, prints "Disabling PyTorch" and carries on with tokenizers only. Every model
    class is then unavailable, so an env passes an import check and cannot load a single model -
    the failure only surfaces when a worker tries to render, hours into a run.
    """
    if "transformers" not in spec.checks() or not spec.torch:
        return
    try:
        proc = _probe(py, FRAMEWORK_PROBE, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvBuildError(f"env '{spec.name}': framework check could not run: {exc}") from exc
    detail = (proc.stdout or proc.stderr or "").strip().splitlines()
    if proc.returncode != 0:
        raise EnvBuildError(
            f"env '{spec.name}': transformers is installed but has PyTorch disabled "
            f"({detail[-1] if detail else 'no detail'}). It will load tokenizers and no models. "
            f"This happens when transformers outruns the env's torch pin - cap it below the "
            f"major version that requires a newer torch.")
    log.info("  env '%s': %s", spec.name, detail[-1] if detail else "framework ok")


def _verify_imports(py: Path, modules: Sequence[str], label: str) -> None:
    """Fail the build if the env cannot import what its worker needs."""
    if not modules:
        return
    code = "import " + ", ".join(modules)
    try:
        proc = _probe(py, code, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvBuildError(f"{label}: import check could not run: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise EnvBuildError(
            f"{label}: the environment is built but cannot import {', '.join(modules)} - "
            f"{tail[-1] if tail else 'unknown error'}. The requirements install did not take "
            f"effect; rebuild with --force.")


def _is_lfs_pointer(path: Path) -> bool:
    """Whether a file is a Git LFS pointer rather than the content it stands for.

    Staging checkpoints through LFS is a sensible way to share them, but a clone without
    `git lfs pull` leaves ~130 bytes of text where the weights should be. Copied into place it
    fails much later, inside torch.load, with an error about a corrupt archive.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _staged_file(weight: WeightFile, staged_dir: Optional[Path], env_name: str) -> Path:
    """Locate a manually downloaded checkpoint, or explain precisely how to supply it."""
    name = weight.staged_name or Path(weight.dest).name
    where = f" Get it from: {weight.where}" if weight.where else ""
    if staged_dir is None:
        raise EnvBuildError(
            f"Env '{env_name}' needs the checkpoint '{name}', which cannot be downloaded "
            f"unattended, but no staging directory is configured. Set "
            f"generation.staged_weights_dir to a folder holding it.{where}")
    staged_dir = Path(staged_dir)
    # accept the file at the top level or one level down, so model_paths/ may be organised
    candidates = [staged_dir / name, *sorted(staged_dir.glob(f"*/{name}"))]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            if _is_lfs_pointer(candidate):
                raise EnvBuildError(
                    f"Env '{env_name}': {candidate} is a Git LFS pointer, not the checkpoint - "
                    f"it is {candidate.stat().st_size} bytes of text. Fetch the real file with "
                    f"`git lfs install && git lfs pull` in the repository, then build again.")
            return candidate
    present = sorted(p.name for p in staged_dir.glob("*") if p.is_file()) \
        if staged_dir.is_dir() else []
    have = ", ".join(present) if present else "none (the directory is empty or missing)"
    raise EnvBuildError(
        f"Env '{env_name}' needs '{name}' in the staging directory {staged_dir}, and it is not "
        f"there.{where} Files currently staged: {have}")


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


def build_env(spec: EnvSpec, envs_root: Path, force: bool = False, offline: bool = False,
              staged_dir: Optional[Path] = None) -> ReadyEnv:
    """Create (or reuse) the venv, repos and weights for one adapter."""
    # Absolute, always. The worker is launched with cwd set to this directory, so a relative
    # interpreter path would resolve against the env itself and disappear - the env builds
    # perfectly, then every worker dies with FileNotFoundError on venv/bin/python.
    root = (Path(envs_root) / spec.name).resolve()
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
            # the marker says "built"; confirm it still *works* before handing workers an env
            # that would fail on every job. Cheap (two subprocess imports) next to a job.
            if not _venv_is_sane(py, venv_dir):
                log.warning("Env '%s' is marked ready but %s does not run as its own venv "
                            "(moved or half-created) -> rebuilding", spec.name, py)
            else:
                try:
                    _verify_imports(py, spec.checks(), f"env '{spec.name}'")
                    _verify_torch(py, spec)
                    _verify_framework(py, spec)
                    return ReadyEnv(spec, root, py,
                                    {r.folder: repos_dir / r.folder for r in spec.repos})
                except EnvBuildError as exc:
                    log.warning("%s -> rebuilding", exc)
            force = True
            shutil.rmtree(venv_dir, ignore_errors=True)
        if state.get("digest") != want and state:
            log.info("Env '%s' spec changed (%s -> %s) -> rebuilding", spec.name,
                     state.get("digest"), want)

    if offline:
        raise EnvBuildError(f"Env '{spec.name}' is not built and generation.offline is set. "
                            f"Pre-build it with: python -m csf.generation.envs --build {spec.name}")

    steps = (2 + (1 if spec.torch else 0) + (1 if spec.pip_requirements() else 0)
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

    if py.exists() and not _venv_is_sane(py, venv_dir):
        log.warning("Env '%s': %s exists but does not run as its own venv - recreating it",
                    spec.name, py)
        shutil.rmtree(venv_dir, ignore_errors=True)
    if not py.exists():
        base = spec.python or sys.executable
        _run([base, "-m", "venv", str(venv_dir)],
             what=announce("creating the virtual environment"), timeout=600)
        if not _venv_is_sane(py, venv_dir):
            raise EnvBuildError(
                f"Env '{spec.name}': `{base} -m venv {venv_dir}` did not produce a usable "
                f"environment. Check that the venv module works on this machine "
                f"(some distributions need python3-venv / ensurepip installed).")
    else:
        announce("virtual environment already present")
    _ensure_pip(py, f"env '{spec.name}'")
    build_env_vars = _clean_environ()
    _run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"],
         what=announce("upgrading pip"), timeout=1800, env=build_env_vars)

    if spec.torch:
        cmd = [py, "-m", "pip", "install", *spec.torch.split()]
        if spec.torch_index:
            cmd += ["--index-url", spec.torch_index]
        _run(cmd, what=announce(f"installing {spec.torch.split()[0]} (several GB)"),
             timeout=7200, env=build_env_vars)

    constraints = _write_constraints(spec, root)
    if constraints:
        # pip reads this for every install below, post-install hooks included
        build_env_vars["PIP_CONSTRAINT"] = str(constraints)

    requirements = spec.pip_requirements()
    if requirements:
        _run([py, "-m", "pip", "install", *requirements],
             what=announce(f"installing {len(requirements)} requirement(s)"), timeout=7200,
             env=build_env_vars)

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
        if weight.staged_name:
            src = _staged_file(weight, staged_dir, spec.name)
            announce(f"copying {weight.staged_name} from the staging directory")
            shutil.copy2(src, dest)
        elif weight.url:
            announce(f"downloading {Path(weight.dest).name}")
            with Heartbeat(f"downloading {Path(weight.dest).name}"):
                _fetch(weight.url, dest)
        elif weight.hf_repo:
            announce(f"downloading {weight.hf_file} from {weight.hf_repo}")
            code = ("from huggingface_hub import hf_hub_download; import shutil; "
                    f"p=hf_hub_download({weight.hf_repo!r}, {weight.hf_file!r}, "
                    f"repo_type={weight.hf_type!r}); shutil.copy2(p, {str(dest)!r})")
            _run([py, "-c", code], what=f"fetching {weight.hf_file}", timeout=7200,
                 env=build_env_vars)
        else:
            raise EnvBuildError(f"Weight {weight.dest} for env {spec.name} has no url or hf_repo")

    # post-install hooks (upstream downloaders, Hub snapshot pulls) locate the env through
    # CSF_ENV_ROOT, which is otherwise only injected at worker launch - set it here too, and put
    # the venv's bin dir first so any console script resolves to this env's copy.
    if spec.post_install:
        hook_env = ReadyEnv(spec, root, py, repos).environ()
        # hooks that pip-install must honour the same pins as the requirements step, or one
        # of them quietly pulls numpy 2 into an env built entirely against numpy<2
        if constraints:
            hook_env["PIP_CONSTRAINT"] = str(constraints)
        for cmd in spec.post_install:
            _run([py, *cmd], cwd=root, env=hook_env,
                 what=announce("running the post-install hook (may download weights)"),
                 timeout=7200)

    if not py.exists():
        raise EnvBuildError(
            f"Environment '{spec.name}' finished building but its interpreter is missing at "
            f"{py}. The venv step probably failed silently - check that `python -m venv` works "
            f"on this machine (some distributions need python3-venv / ensurepip installed).")
    _verify_imports(py, spec.checks(), f"env '{spec.name}'")
    _verify_torch(py, spec)
    _verify_framework(py, spec)
    marker.write_text(json.dumps({"digest": want, "name": spec.name, "python": str(py),
                                  "root": str(root),
                                  "repos": {k: str(v) for k, v in repos.items()}}, indent=2),
                      encoding="utf-8")
    log.info("Environment '%s' ready in %.0fs", spec.name, time.monotonic() - build_started)
    return ReadyEnv(spec, root, py, repos)


def burn_envs(specs: Sequence[EnvSpec], envs_root: Path,
              keep: Sequence[str] = ()) -> Dict[str, float]:
    """Delete the environments under `envs_root` so the next build recreates them from scratch.

    Rebuilding in place is usually enough - the readiness marker is keyed by the spec digest, and
    a broken venv is detected and recreated. This is for the case where that is not enough:
    half-installed packages from an interrupted build, an env whose torch was swapped underneath
    it, or simply wanting to prove the whole install path works end to end on a fresh machine.

    Only directories named after a registered environment are removed, so anything else living
    under `envs_root` is left alone, and the returned sizes let the caller report what the
    rebuild will have to download again.
    """
    envs_root = Path(envs_root)
    known = {spec.name for spec in specs} - set(keep)
    removed: Dict[str, float] = {}
    if not envs_root.is_dir():
        log.info("Nothing to burn: %s does not exist", envs_root)
        return removed
    for child in sorted(envs_root.iterdir()):
        if not child.is_dir() or child.name not in known:
            continue
        gb = sum(f.stat().st_size for f in child.rglob("*") if f.is_file()) / 2 ** 30
        log.warning("Burning environment '%s' (%.1f GiB) at %s", child.name, gb, child)
        shutil.rmtree(child, ignore_errors=True)
        removed[child.name] = round(gb, 2)
    if removed:
        log.warning("Burned %d environment(s), freeing %.1f GiB. They will be rebuilt on demand, "
                    "which re-downloads torch, the requirements and every checkpoint.",
                    len(removed), sum(removed.values()))
    else:
        log.info("Nothing to burn under %s", envs_root)
    return removed


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


def diagnose(specs: Sequence[EnvSpec], envs_root: Path) -> Dict[str, Dict[str, object]]:
    """Report what is actually wrong with each env, without building anything.

    `env_status` only reads the readiness marker, which is exactly the thing that lies when a
    build half-succeeded. This runs the interpreter.
    """
    out: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        root = (Path(envs_root) / spec.name).resolve()
        venv_dir = root / "venv"
        py = _venv_python(venv_dir)
        info: Dict[str, object] = {"root": str(root), "python": str(py),
                                   "marker": (root / ".csf_ready.json").exists(),
                                   "interpreter": py.exists()}
        if not py.exists():
            info["verdict"] = "missing"
            out[spec.name] = info
            continue
        info["own_venv"] = _venv_is_sane(py, venv_dir)
        try:
            pip = _probe(py, "import pip; print(pip.__version__)")
            info["pip"] = pip.stdout.strip() if pip.returncode == 0 else "MISSING"
        except (OSError, subprocess.SubprocessError) as exc:
            info["pip"] = f"error: {exc}"
        missing = []
        for mod in spec.checks():
            try:
                if _probe(py, f"import {mod}").returncode != 0:
                    missing.append(mod)
            except (OSError, subprocess.SubprocessError):
                missing.append(mod)
        info["missing_imports"] = missing
        want = dict(tok.split("==", 1) for tok in torch_pins(spec)).get("torch")
        if want and "torch" not in missing:
            try:
                proc = _probe(py, "import torch; print(torch.__version__)", timeout=300)
                info["torch"] = proc.stdout.strip() if proc.returncode == 0 else "?"
            except (OSError, subprocess.SubprocessError):
                info["torch"] = "?"
            info["torch_pinned"] = want
            if str(info["torch"]).split("+")[0] != want:
                info["torch_drifted"] = True
        # importable is not usable: transformers past its torch floor disables PyTorch and
        # loads tokenizers only, which every other check in here would call healthy
        if "transformers" in spec.checks() and spec.torch and "transformers" not in missing:
            try:
                proc = _probe(py, FRAMEWORK_PROBE, timeout=300)
                info["torch_enabled"] = proc.returncode == 0
                line = (proc.stdout or "").strip().splitlines()
                info["frameworks"] = line[-1] if line else ""
            except (OSError, subprocess.SubprocessError):
                info["torch_enabled"] = False
        info["verdict"] = ("ok" if info["own_venv"] and info["pip"] != "MISSING" and not missing
                           and not info.get("torch_drifted")
                           and info.get("torch_enabled", True) else "broken")
        out[spec.name] = info
    return out


def _main() -> int:
    """`python -m csf.generation.envs --build <name|all>` pre-builds envs before a run."""
    import argparse
    import logging as _logging

    from csf.generation.adapters import ADAPTERS, env_specs

    # Nothing configures logging when this module is run directly, so every step this command
    # narrates - which env, which of its steps, the pip output - went to a logger with no
    # handler. A thirteen-env rebuild taking twenty minutes looked identical to one that did
    # nothing at all, twice, while we were trying to work out whether it had run.
    if not _logging.getLogger().handlers:
        _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                             datefmt="%H:%M:%S")

    ap = argparse.ArgumentParser(description="Build the per-model environments")
    ap.add_argument("--build", default="", help="env or adapter name, or 'all'")
    ap.add_argument("--config", default="configs/regen.yaml",
                    help="config to read generation.envs_root and staged_weights_dir from")
    ap.add_argument("--envs-root", default=None,
                    help="override the root from --config")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--staged-weights-dir", default=None,
                    help="folder holding checkpoints that cannot be downloaded unattended")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--scan", default="", metavar="NAME",
                    help="env or adapter name, or 'all': read the import graph of each "
                         "upstream entry point and report every module it cannot import")
    ap.add_argument("--doctor", default="", help="env or adapter name, or 'all': check what is "
                                                 "actually installed in each built env")
    ap.add_argument("--burn", default="", help="env or adapter name, or 'all': delete the built "
                                               "environment(s) so they are recreated from "
                                               "scratch. Combine with --build to rebuild now.")
    args = ap.parse_args()

    # The run stage builds its envs under generation.envs_root. This command used to default
    # somewhere else entirely, so a rebuild landed in a tree nothing reads, the pipeline
    # quietly rebuilt the real one on its next run, and the two drifted apart while looking
    # like they agreed. Read the same config unless told otherwise.
    envs_root, staged = args.envs_root, args.staged_weights_dir
    source = "--envs-root" if envs_root else args.config
    if envs_root is None or staged is None:
        try:
            from csf.config import load_config
            cfg = load_config(args.config, [])
            envs_root = envs_root or cfg.generation.envs_root
            staged = staged or cfg.generation.staged_weights_dir
        except Exception as exc:                     # noqa: BLE001 - a missing config is fine
            log.warning("Could not read %s (%s); falling back to the built-in defaults",
                        args.config, exc)
            envs_root = envs_root or "./cache/regen/envs"
            staged = staged or "./model_paths"
    args.envs_root, args.staged_weights_dir = envs_root, staged
    log.info("Environments root: %s (from %s)", envs_root, source)

    specs = env_specs()
    root = Path(args.envs_root)

    def _select(name: str) -> List[EnvSpec]:
        """Resolve a name to env specs. Accepts a comma-separated list, an adapter name, 'all'."""
        if name == "all":
            # 'all' means every env a runnable model needs - not every env in the registry.
            # An env whose only adapter is unwired (vid2vid, whose torch 1.13.1 has no wheel for
            # Python 3.12 anyway) can never be used by a run, so building it only manufactures a
            # failure that means nothing.
            live = {a.env_name for a in ADAPTERS.values() if a.implemented}
            skipped = sorted(set(specs) - live)
            if skipped:
                log.info("Skipping %s: no implemented adapter uses %s",
                         ", ".join(skipped), "them" if len(skipped) > 1 else "it")
            return [spec for key, spec in specs.items() if key in live]
        picked: List[EnvSpec] = []
        for part in (p.strip() for p in name.split(",") if p.strip()):
            if part in specs:
                picked.append(specs[part])
            elif part in ADAPTERS:
                picked.append(specs[ADAPTERS[part].env_name])
            else:
                # name the offending entry: a list of thirteen with one typo in it used to
                # come back empty, and the command then did nothing that looked like nothing
                raise ValueError(
                    f"unknown env/adapter {part!r} in {name!r}. Known envs: {sorted(specs)}")
        # de-duplicate while preserving order: several adapters can share one env
        return list({spec.name: spec for spec in picked}.values())

    if args.burn:
        try:
            wanted = _select(args.burn)
        except ValueError as exc:
            print(f"Nothing was burned: {exc}")
            return 2
        if not wanted:
            print(f"Unknown env/adapter {args.burn!r}. Known envs: {sorted(specs)}")
            return 2
        freed = burn_envs(wanted, root)
        for name, gb in sorted(freed.items()):
            print(f"burned  {name}  ({gb} GiB)")
        if not args.build:
            return 0

    if args.doctor:
        try:
            wanted = _select(args.doctor)
        except ValueError as exc:
            print(f"Nothing was checked: {exc}")
            return 2
        if not wanted:
            print(f"Unknown env/adapter {args.doctor!r}. Known envs: {sorted(specs)}")
            return 2
        report = diagnose(wanted, root)
        bad = 0
        for name in sorted(report):
            info = report[name]
            print(f"{str(info['verdict']):>7}  {name}")
            print(f"         python: {info['python']}"
                  f"{'' if info.get('interpreter') else '   (does not exist)'}")
            if info.get("interpreter"):
                if not info.get("own_venv"):
                    print("         WARNING: this interpreter does not run as its own venv - "
                          "packages installed into it are invisible")
                print(f"         pip: {info.get('pip')}")
                missing = info.get("missing_imports") or []
                print(f"         cannot import: {', '.join(missing) if missing else '-'}")
                if info.get("torch_pinned"):
                    drift = "  <- NOT the pinned build" if info.get("torch_drifted") else ""
                    print(f"         torch: {info.get('torch')} "
                          f"(pinned {info['torch_pinned']}){drift}")
                if "torch_enabled" in info:
                    note = "" if info["torch_enabled"] else \
                        "  <- transformers has PyTorch DISABLED; it can load no models"
                    print(f"         {info.get('frameworks') or 'frameworks'}{note}")
            if info["verdict"] != "ok":
                bad += 1
                print(f"         fix: python -m csf.generation.envs --build {name} --force "
                      f"--envs-root {root}")
        return 1 if bad else 0

    if args.scan:
        try:
            wanted = _select(args.scan)
        except ValueError as exc:
            print(f"Nothing was scanned: {exc}")
            return 2
        from csf.generation.importscan import format_report
        total = 0
        for spec in wanted:
            env_root = root / spec.name
            py = _venv_python(env_root / "venv")
            if not py.exists():
                print(f"{spec.name:<16} not built - nothing to scan")
                continue
            ready = ReadyEnv(spec, env_root, py,
                             {r.folder: env_root / "repos" / r.folder for r in spec.repos})
            report = ready.scan_imports()
            if report.get("skipped"):
                print(f"{spec.name:<16} skipped: {report['skipped']}")
                continue
            total += len(report.get("missing") or [])
            print("\n".join(format_report(spec.name, report, env_root)))
        print(f"\n{total} missing module(s) across {len(wanted)} environment(s)")
        return 1 if total else 0

    if args.status or not args.build:
        status = env_status(_select("all"), root)
        for name in sorted(status):
            print(f"{status[name]:>8}  {name}")
        implemented = sum(1 for a in ADAPTERS.values() if a.implemented)
        print(f"\n{implemented}/{len(ADAPTERS)} adapters implemented; "
              f"{len(status)} environment(s) a runnable model needs")
        return 0

    try:
        wanted = _select(args.build)
    except ValueError as exc:
        print(f"Nothing was built: {exc}")
        return 2
    if not wanted:
        print(f"Unknown env/adapter {args.build!r}. Known envs: {sorted(specs)}")
        return 2
    failures = []
    for spec in wanted:
        try:
            build_env(spec, root, force=args.force,
                      staged_dir=Path(args.staged_weights_dir))
        except EnvBuildError as exc:
            failures.append((spec.name, str(exc)[:400]))
            log.error("Env '%s' failed to build: %s", spec.name, exc)
    for name, err in failures:
        print(f"FAILED {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
