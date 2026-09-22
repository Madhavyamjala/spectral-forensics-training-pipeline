"""
Fetching every Hugging Face asset the pipeline needs, before the run starts.

A full run touches roughly twenty Hub repositories: the two training VLMs, the SVD VAE, and the
generator checkpoints behind SAM2, FLUX, LatentSync, VACE, DreamID-V, MuseTalk and the rest.
Discovering a gated repo or a typo'd id three days into generation is expensive, so this module
pulls and verifies everything up front, in one place, with one login.

What it does that a bare `hf_hub_download` loop does not:

  * **fails fast on access** - gated repos (Llama, and some generator weights) are checked with a
    cheap metadata call before any bytes move, and the error says exactly which licence to accept,
  * **resumes** - `snapshot_download` reuses the Hub cache, so an interrupted prefetch continues,
  * **retries** - 429s and connection drops back off exponentially instead of aborting,
  * **reports size** - so you know whether the disk will hold it before committing,
  * **scopes** - `--stage generate` pulls only the generator weights, `--stage train` only the
    training models, so a training-only run does not download 80 GB of diffusion checkpoints.

    python -m csf.generation.prefetch --stage all --config configs/regen.yaml
    python -m csf.generation.prefetch --stage train --dry-run

Input : the adapter registry + the run config.
Output: a populated Hub cache, and a report of what was fetched, skipped or refused.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from csf.generation import progress
from csf.logging_utils import get_logger

log = get_logger("generation.prefetch")

RETRY_STATUS = (429, 500, 502, 503, 504)


@dataclass
class HubAsset:
    """One Hub repository (or single file) the pipeline needs."""
    repo_id: str
    kind: str = "model"                    # "model" or "dataset"
    files: Sequence[str] = ()              # empty => whole-repo snapshot
    gated: bool = False
    stage: str = "generate"                # "generate" | "train" | "both"
    used_by: str = ""
    allow_patterns: Optional[Sequence[str]] = None
    note: str = ""


def training_assets(cfg) -> List[HubAsset]:
    """Base models the training stages load."""
    return [
        HubAsset(cfg.models.qwen_id, stage="train", used_by="Phase 1 scanner",
                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"]),
        HubAsset(cfg.models.llama_id, stage="train", gated=True, used_by="Phase 4 arbiter",
                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"],
                 note="Accept the Llama licence on its model page before running."),
        HubAsset(cfg.models.vae_id, stage="train", used_by="DIRE latent tool",
                 allow_patterns=[f"{cfg.models.vae_subfolder}/*", "*.json"]),
        HubAsset(cfg.models.vae_fallback_id, stage="train", used_by="DIRE latent tool (fallback)"),
    ]


def generation_assets() -> List[HubAsset]:
    """Every Hub repo referenced by a wired adapter's environment.

    Derived from the registry rather than hand-listed, so adding an adapter cannot leave a
    weight out of the prefetch.
    """
    from csf.generation.adapters import ADAPTERS, env_specs

    wired_envs = {a.env_name for a in ADAPTERS.values() if a.implemented}
    users: Dict[str, List[str]] = {}
    for a in ADAPTERS.values():
        if a.implemented:
            users.setdefault(a.env_name, []).append(a.runs)

    seen: Dict[str, HubAsset] = {}
    for name, spec in env_specs().items():
        if name not in wired_envs:
            continue
        used_by = ", ".join(sorted(set(users.get(name, []))))
        for w in spec.weights:
            if not w.hf_repo:
                continue
            asset = seen.setdefault(w.hf_repo, HubAsset(
                w.hf_repo, kind=w.hf_type, stage="generate", used_by=used_by))
            if w.hf_file:
                asset.files = tuple(sorted(set(asset.files) | {w.hf_file}))
        # repos a post-install hook pulls, declared explicitly on the EnvSpec
        for repo in spec.hub_repos:
            seen.setdefault(repo, HubAsset(repo, stage="generate", used_by=used_by,
                                           note="pulled by the env's post-install hook"))

    # base diffusion models the workers load by id at runtime
    for repo, used in (("black-forest-labs/FLUX.1-schnell", "bg_flux_image, bg_svd_video"),
                       ("stabilityai/stable-video-diffusion-img2vid-xt", "bg_svd_video"),
                       ("Manojb/stable-diffusion-2-1-base", "tokenflow")):
        seen.setdefault(repo, HubAsset(repo, stage="generate", used_by=used))
    return sorted(seen.values(), key=lambda a: a.repo_id)


def dataset_assets(cfg) -> List[HubAsset]:
    out = [HubAsset(cfg.data.repo_id, kind="dataset", stage="train",
                    used_by="real / ai_generated classes", files=("manifest.csv",),
                    note="manifest only; videos stream during the features stage")]
    hf_repo = getattr(cfg.generation.kinetics, "hf_repo", None)
    if hf_repo:
        out.append(HubAsset(hf_repo, kind="dataset", stage="generate",
                            used_by="Kinetics-400 source clips",
                            note="listed only; clips stream during the kinetics stage"))
    return out


def all_assets(cfg, stage: str = "all") -> List[HubAsset]:
    """Every asset for `stage`, deduplicated.

    A repo can be needed by both halves of the pipeline - SVD backs the DIRE latent tool during
    training and the moving-background generator - so merge duplicates and mark them "both"
    rather than downloading them twice.
    """
    merged: Dict[str, HubAsset] = {}
    candidates = training_assets(cfg) + generation_assets() + dataset_assets(cfg)
    # Filter by stage BEFORE merging. The merge widens a shared repo's download to the union of
    # what each side wants, and the SVD video generator wants the whole repo - so a train-only
    # prefetch that merged first inherited it and pulled ~20 GB of UNet and image encoder, when the
    # DIRE tool needs only the VAE subfolder.
    if stage != "all":
        candidates = [a for a in candidates if a.stage == stage]
    for a in candidates:
        prior = merged.get(a.repo_id)
        if prior is None:
            merged[a.repo_id] = a
            continue
        if prior.stage != a.stage:
            prior.stage = "both"
        prior.gated = prior.gated or a.gated
        prior.files = tuple(sorted(set(prior.files) | set(a.files)))
        used = [u for u in (prior.used_by, a.used_by) if u]
        prior.used_by = "; ".join(dict.fromkeys(used))
        if prior.allow_patterns and a.allow_patterns:
            prior.allow_patterns = tuple(dict.fromkeys(
                list(prior.allow_patterns) + list(a.allow_patterns)))
        else:
            prior.allow_patterns = None          # one side wants everything
    assets = sorted(merged.values(), key=lambda x: (x.kind, x.repo_id))
    if stage == "all":
        return assets
    return [a for a in assets if a.stage in (stage, "both")]


# --------------------------------------------------------------------------------------
# access + download
# --------------------------------------------------------------------------------------


def _token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None


def check_access(asset: HubAsset, token: Optional[str] = None) -> Optional[str]:
    """Cheap metadata probe. Returns None if reachable, else a human-readable reason."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import (GatedRepoError, RepositoryNotFoundError)

    api = HfApi(token=token or _token())
    try:
        if asset.kind == "dataset":
            api.dataset_info(asset.repo_id)
        else:
            api.model_info(asset.repo_id)
        return None
    except GatedRepoError:
        return (f"GATED: accept the licence at https://huggingface.co/{asset.repo_id} "
                f"then `hf auth login` (or set HF_TOKEN)")
    except RepositoryNotFoundError:
        return (f"NOT FOUND (or private): https://huggingface.co/{asset.repo_id} - check the id, "
                f"or log in if it is private")
    except Exception as exc:                                   # noqa: BLE001 - network etc.
        return f"{type(exc).__name__}: {str(exc)[:200]}"


def fetch(asset: HubAsset, token: Optional[str] = None, retries: int = 5) -> Dict[str, object]:
    """Download one asset, resuming and retrying. Returns a small report."""
    from huggingface_hub import hf_hub_download, snapshot_download

    token = token or _token()
    started = time.monotonic()
    for attempt in range(1, retries + 1):
        try:
            if asset.files:
                paths = [hf_hub_download(asset.repo_id, f, repo_type=asset.kind, token=token)
                         for f in asset.files]
                local = str(Path(paths[0]).parent)
            else:
                local = snapshot_download(asset.repo_id, repo_type=asset.kind, token=token,
                                          allow_patterns=list(asset.allow_patterns)
                                          if asset.allow_patterns else None)
            size = _dir_size(Path(local))
            return {"repo_id": asset.repo_id, "ok": True, "path": local,
                    "size_gb": round(size / 2 ** 30, 2),
                    "seconds": round(time.monotonic() - started, 1)}
        except Exception as exc:                               # noqa: BLE001
            msg = str(exc)
            transient = any(str(c) in msg for c in RETRY_STATUS) or \
                "timed out" in msg.lower() or "connection" in msg.lower()
            if not transient or attempt == retries:
                return {"repo_id": asset.repo_id, "ok": False,
                        "error": f"{type(exc).__name__}: {msg[:300]}"}
            wait = min(10 * 2 ** (attempt - 1), 300)
            log.warning("Transient error on %s (attempt %d/%d): %s -> retry in %.0fs",
                        asset.repo_id, attempt, retries, msg[:160], wait)
            time.sleep(wait)
    return {"repo_id": asset.repo_id, "ok": False, "error": "exhausted retries"}


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def prefetch(cfg, stage: str = "all", dry_run: bool = False,
             token: Optional[str] = None) -> Dict[str, object]:
    """Check access to every asset, then download the reachable ones."""
    assets = all_assets(cfg, stage)
    log.info("Prefetch: %d Hub asset(s) for stage %r", len(assets), stage)

    blocked: Dict[str, str] = {}
    for a in progress.track(assets, "checking access", unit="repo", log_every=5):
        reason = check_access(a, token)
        if reason:
            blocked[a.repo_id] = reason
            level = log.error if a.gated else log.warning
            level("  %-52s %s", a.repo_id, reason)
        else:
            log.info("  %-52s OK   (%s)", a.repo_id, a.used_by or a.stage)

    gated_blocked = [a.repo_id for a in assets if a.gated and a.repo_id in blocked]
    if gated_blocked:
        raise RuntimeError(
            "Cannot reach gated repo(s): " + ", ".join(gated_blocked) + ".\n"
            + "\n".join(f"  {r}: {blocked[r]}" for r in gated_blocked) +
            "\nAccept the licence on each model page, then `hf auth login`.")

    if dry_run:
        return {"assets": len(assets), "blocked": blocked, "dry_run": True}

    todo = [a for a in assets if a.repo_id not in blocked]
    results, failed, total_gb = [], [], 0.0
    with progress.bar(len(todo), "downloading repos", "repo", log_every=1) as pbar:
        for a in todo:
            pbar.set_postfix_str(a.repo_id[:40])
            log.info("Fetching %s ...", a.repo_id)
            with progress.Heartbeat(f"downloading {a.repo_id}"):
                r = fetch(a, token)
            pbar.update(1)
            results.append(r)
            if r.get("ok"):
                total_gb += float(r.get("size_gb") or 0.0)
                log.info("  %s -> %.2f GB in %.0fs", a.repo_id, r["size_gb"], r["seconds"])
            else:
                failed.append(r)
                log.error("  %s FAILED: %s", a.repo_id, r.get("error"))

    report = {"assets": len(assets), "fetched": len(results) - len(failed),
              "failed": [f["repo_id"] for f in failed], "blocked": blocked,
              "total_gb": round(total_gb, 2)}
    log.info("Prefetch complete: %s", json.dumps(report))
    if failed:
        raise RuntimeError(f"{len(failed)} Hub asset(s) could not be fetched: "
                           f"{[f['repo_id'] for f in failed]}. See the log for each error.")
    return report


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Prefetch every Hugging Face asset the run needs")
    ap.add_argument("--config", default="configs/regen.yaml")
    ap.add_argument("--stage", default="all", choices=["all", "generate", "train"])
    ap.add_argument("--dry-run", action="store_true", help="check access, download nothing")
    ap.add_argument("--list", action="store_true", help="print the asset table and exit")
    args = ap.parse_args()

    from csf.config import load_config
    cfg = load_config(args.config)
    assets = all_assets(cfg, args.stage)

    if args.list:
        print(f"{'repo':<52}{'kind':<9}{'stage':<10}used by")
        for a in assets:
            mark = "*" if a.gated else " "
            print(f"{mark}{a.repo_id:<51}{a.kind:<9}{a.stage:<10}{a.used_by}")
        print(f"\n* = gated: accept the licence on the model page first")
        print(f"{len(assets)} asset(s) for stage {args.stage!r}")
        return 0

    try:
        prefetch(cfg, args.stage, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
