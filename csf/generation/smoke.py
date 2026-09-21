"""
One video per model, before committing to a run that takes days.

Every generation failure this pipeline has hit so far - a missing `matplotlib`, a librosa
signature that went keyword-only, an entry point that upstream renamed - is the kind a single
rendered video would have caught in minutes. Instead they surfaced hours into a 33,333-video
run, as thousands of identical tracebacks in a per-worker log, and with `fail_fast` cascading
one broken model into a whole abandoned group.

So: plan the real jobs, take the first one for each model asked for, and render it into a
scratch directory. Real clips, real donors, real payloads, the same worker protocol and the
same environments - only the destination differs, so nothing here can reach the dataset tree.

    python -m csf.generation.smoke --model wav2lip,tokenflow
    python -m csf.generation.smoke --family lip_sync --gpu 3
    python -m csf.generation.smoke --all

The exit status is the number of models that failed, and every failure prints the worker's own
stderr tail underneath it. `--list` names what can be asked for without running anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from csf.generation.adapters import ADAPTERS, WorkerPool, env_specs
from csf.generation.adapters.base import AdapterError
from csf.generation.envs import EnvBuildError
from csf.logging_utils import get_logger

log = get_logger("generation.smoke")


@dataclass
class Result:
    """The outcome of rendering one model's sample video."""
    model: str
    runs_as: str
    env: str
    ok: bool
    seconds: float = 0.0
    output: str = ""
    size_kb: float = 0.0
    frames: int = 0
    error: str = ""
    stderr: str = ""

    @property
    def status(self) -> str:
        """A short word for the summary table."""
        return "ok" if self.ok else "FAILED"


def selectable() -> List[str]:
    """Model slots a smoke test can run: the wired-up ones, in spec order."""
    return [key for key, adapter in ADAPTERS.items() if adapter.implemented]


def _resolve(models: Sequence[str], family: str, everything: bool) -> List[str]:
    """Turn the CLI's selection into a list of model slots, or raise with what was valid."""
    wired = selectable()
    if everything:
        return wired
    if family:
        chosen = [k for k in wired if ADAPTERS[k].family == family]
        if not chosen:
            families = sorted({a.family for a in ADAPTERS.values() if a.implemented})
            raise SystemExit(f"No wired-up model in family {family!r}. Families: {families}")
        return chosen
    names: List[str] = []
    for entry in models:
        names.extend(part.strip() for part in entry.split(",") if part.strip())
    unknown = [n for n in names if n not in ADAPTERS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Choose from: {wired}")
    scaffold = [n for n in names if not ADAPTERS[n].implemented]
    if scaffold:
        raise SystemExit(f"Model(s) {scaffold} are tier-2 scaffolds with no runnable "
                         f"implementation - there is nothing to smoke-test.")
    return names


def _first_jobs(cfg, models: Sequence[str]) -> Dict[str, object]:
    """The first planned job for each requested model, from the real generation plan."""
    from csf.generation.run import plan_jobs

    wanted = set(models)
    picked: Dict[str, object] = {}
    for job in plan_jobs(cfg):
        if job.model in wanted and job.model not in picked:
            picked[job.model] = job
            if len(picked) == len(wanted):
                break
    return picked


def run_smoke(cfg, models: Sequence[str], gpu: int, out_dir: Path,
              timeout: int = 1800) -> List[Result]:
    """Render one video per model and return a result per model, in the order asked for."""
    gen = cfg.generation
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = _first_jobs(cfg, models)

    pool = WorkerPool(
        envs_root=Path(gen.envs_root), log_dir=out_dir / "logs",
        # one resident worker: the next model's env evicts the previous one, so a smoke run
        # over every model needs no more VRAM than the largest single model does
        max_resident=1, job_timeout=timeout, offline=gen.offline,
        staged_dir=Path(gen.staged_weights_dir) if gen.staged_weights_dir else None)
    results: List[Result] = []
    try:
        for name in models:
            adapter = ADAPTERS[name]
            result = Result(model=name, runs_as=adapter.runs, env=adapter.env_name, ok=False)
            job = jobs.get(name)
            if job is None:
                result.error = ("no planned job for this model - it may be excluded by "
                                "only_models / skip_models, or the plan may predate it")
                results.append(result)
                continue
            out = out_dir / f"{name}.mp4"
            started = time.monotonic()
            log.info("Smoke: %s (%s) on GPU %d -> %s", name, adapter.runs, gpu, out)
            try:
                worker = pool.get(adapter, env_specs()[adapter.env_name], gpu)
                msg = worker.run(adapter.payload(job, out, gpu))
            except (AdapterError, EnvBuildError, OSError) as exc:
                result.error = str(exc)[:800]
            else:
                if msg.get("ok"):
                    result.ok = True
                    result.output = str(msg.get("output_path") or out)
                else:
                    result.error = str(msg.get("error") or "unknown")[:800]
                    result.stderr = worker.stderr_tail(30)
            result.seconds = time.monotonic() - started
            if result.ok and out.exists():
                result.size_kb = out.stat().st_size / 1024
                # a frame count, because a plausible-looking file size is not evidence that a
                # model rendered anything - a one-frame or duration-zero mp4 weighs the same
                result.frames = int((msg.get("metadata") or {}).get("frames") or 0)
                if not result.size_kb:
                    result.ok, result.error = False, "the worker reported success but wrote an "\
                                                     "empty file"
            results.append(result)
    finally:
        pool.shutdown()
    return results


def report(results: Sequence[Result]) -> int:
    """Print the summary table and return the number of failures."""
    print(f"\n{'model':<22}{'runs as':<18}{'env':<14}{'status':<9}"
          f"{'secs':>7}{'KB':>9}{'frames':>8}")
    print("-" * 85)
    for r in results:
        print(f"{r.model:<22}{r.runs_as:<18}{r.env:<14}{r.status:<9}"
              f"{r.seconds:>7.0f}{r.size_kb:>9.0f}{r.frames:>8}")
    failed = [r for r in results if not r.ok]
    for r in failed:
        print(f"\n--- {r.model} ({r.env}) ---\n{r.error}")
        if r.stderr:
            print(f"worker stderr tail:\n{r.stderr}")
    print(f"\n{len(results) - len(failed)}/{len(results)} model(s) rendered a video")
    return len(failed)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for `python -m csf.generation.smoke`."""
    ap = argparse.ArgumentParser(
        description="Render one video per manipulation model, to validate it before a full run")
    ap.add_argument("--config", default="configs/regen.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="config override, repeatable")
    ap.add_argument("--model", action="append", default=[], metavar="NAME[,NAME...]",
                    help="model slot(s) to smoke-test; repeatable and comma-separated")
    ap.add_argument("--family", default="", help="every wired-up model in one family")
    ap.add_argument("--all", action="store_true", help="every wired-up model")
    ap.add_argument("--list", action="store_true", help="print what can be selected and exit")
    ap.add_argument("--gpu", type=int, default=None, help="GPU index (default: the first "
                                                          "configured one)")
    ap.add_argument("--out", default="", help="output directory (default: <work_dir>/smoke)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-video timeout in seconds")
    ap.add_argument("--json", default="", help="also write the results to this JSON file")
    args = ap.parse_args(argv)

    if args.list:
        for key in selectable():
            a = ADAPTERS[key]
            runs = a.runs if a.substituted else "(itself)"
            print(f"{key:<22}{runs:<18}{a.family:<30}{a.env_name}")
        return 0
    if not (args.model or args.family or args.all):
        ap.error("choose what to test: --model NAME, --family NAME, --all, or --list")

    from csf.config import load_config
    cfg = load_config(args.config, args.overrides)
    models = _resolve(args.model, args.family, args.all)
    gpu = args.gpu if args.gpu is not None else (list(cfg.generation.gpus) or [0])[0]
    out_dir = Path(args.out) if args.out else Path(cfg.paths.work_dir) / "smoke"

    results = run_smoke(cfg, models, gpu, out_dir, timeout=args.timeout)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps([r.__dict__ for r in results], indent=2),
                                   encoding="utf-8")
    return report(results)


if __name__ == "__main__":
    sys.exit(_main())
