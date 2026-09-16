"""
Chrono-Spectral Forensics - end-to-end training / evaluation / publishing driver.

Runs the proposal's pipeline on Chrono-TriClass-100k as a sequence of resumable stages:

    prepare          manifest -> run subset (5 000 rows in test mode, everything in full mode)
    features         distributed download + decode + toolpool feature cache (disk-bounded)
    train_qwen       Phase 1 scanner: Qwen2.5-VL-3B + LoRA (DDP)
    train_llama      Phase 4 arbiter: Llama-3.2-11B-Vision + LoRA with tool dropout (DDP)
    predict_scanner  scanner probabilities on valid / test
    outcomes         shared-backbone arbiter outcome tables on valid / test (distributed)
    train_dispatcher Phase 2 GRPO dispatchers, one per ablation profile
    evaluate         ablation study of the three model types + baselines, metrics, plots, report
    export           model card + adapters + dispatchers + metrics + code -> <work_dir>/export
    latency          live raw-video latency benchmark through the exported bundle (CSFDetector)
    push             asks for a Hugging Face token + repo id and uploads the export (rank 0)

Usage (single GPU):
    python main.py --config configs/test.yaml
    python main.py --config configs/full.yaml
Multi-GPU:
    torchrun --nproc_per_node=<N> main.py --config configs/full.yaml
Options:
    --stage all | <stage>[,<stage>...]   run a subset (default all)
    --force <stage>[,<stage>...]         re-run stages even if marked complete in <work_dir>/state.json
    --set section.key=value              override any config value (repeatable)

Input : YAML config (configs/*.yaml).
Output: <work_dir>/ {logs/, checkpoints/, predictions/, metrics/, export/, state.json, resolved_config.json}
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if platform.system() == "Windows":
    os.environ.setdefault("USE_LIBUV", "0")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

STAGES = ["prepare", "features", "train_qwen", "train_llama", "predict_scanner", "outcomes", "train_dispatcher",
          "evaluate", "export", "latency", "push"]


def parse_args():
    ap = argparse.ArgumentParser(description="Chrono-Spectral Forensics pipeline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--force", default="")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preflight(cfg, dist_info, log) -> None:
    import torch
    import transformers
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "transformers": transformers.__version__,
            "world_size": dist_info.world_size}
    try:
        import peft
        info["peft"] = peft.__version__
    except ImportError:
        raise RuntimeError("peft is not installed - run the setup script (setup_env.ps1 / setup_env.sh).")
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except ImportError as exc:
        index = f"https://download.pytorch.org/whl/cu{torch.version.cuda.replace('.', '')}" \
            if torch.version.cuda else "https://download.pytorch.org/whl/cpu"
        raise RuntimeError(
            f"torchvision is not installed ({exc}), but the Qwen2.5-VL video processor requires it. "
            f"Install the build matching torch {torch.__version__}:\n"
            f"    pip install torchvision --index-url {index}") from exc
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(dist_info.device)
        info.update(gpu=props.name, vram_gib=round(props.total_memory / 2**30, 1),
                    compute_capability=f"{props.major}.{props.minor}", bf16=torch.cuda.is_bf16_supported(),
                    cuda_runtime=torch.version.cuda)
        try:
            import bitsandbytes
            info["bitsandbytes"] = bitsandbytes.__version__
        except Exception as exc:
            info["bitsandbytes"] = f"UNAVAILABLE ({exc})"
            if "4bit" in (cfg.train.qwen.quantization, cfg.train.llama.quantization):
                raise RuntimeError("bitsandbytes is required for 4-bit training but failed to import: "
                                   f"{exc}. Re-run the setup script.")
        need = 11 if cfg.mode == "test" else 22
        if info["vram_gib"] < need:
            log.warning("GPU has %.1f GiB VRAM; the %s profile expects >= %d GiB. Expect OOM - lower "
                        "data.num_frames / use 4bit quantization.", info["vram_gib"], cfg.mode, need)
    else:
        log.warning("CUDA is NOT available - training will run on CPU (only sensible for tiny smoke tests).")
    free_gb = shutil.disk_usage(Path(cfg.paths.cache_dir).resolve().anchor).free / 2**30
    info["disk_free_gib"] = round(free_gb, 1)
    log.info("Environment: %s", json.dumps(info))
    if free_gb < (20 if cfg.mode == "test" else 120):
        log.warning("Only %.1f GiB free disk space; the feature cache + model downloads may not fit.", free_gb)


def check_gated_access(model_id: str, log) -> None:
    """Fail fast (before hours of feature extraction) if a gated base model is not accessible."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    try:
        hf_hub_download(model_id, "config.json")
        log.info("Access OK: %s", model_id)
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        if sys.stdin and sys.stdin.isatty() and int(os.environ.get("WORLD_SIZE", 1)) == 1:
            import getpass
            print(f"\n{model_id} is gated. Accept its licence on huggingface.co, then paste a READ token.")
            token = getpass.getpass("Hugging Face READ token (input hidden, used for this session only): ").strip()
            os.environ["HF_TOKEN"] = token
            hf_hub_download(model_id, "config.json", token=token)
            log.info("Access OK with provided token: %s", model_id)
        else:
            raise RuntimeError(f"No access to gated model {model_id}: {exc}. Accept the licence on the Hub and run "
                               f"`huggingface-cli login` (or set HF_TOKEN) before launching.") from exc


def main() -> int:
    args = parse_args()
    from csf.config import load_config
    cfg = load_config(args.config, args.overrides)

    from csf.distributed import barrier, cleanup, init_distributed
    from csf.logging_utils import RunState, setup_logging, stage

    dist_info = init_distributed()
    work_dir = cfg.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(work_dir, dist_info.rank, cfg.debug)
    log.info("Run '%s' | mode=%s | config=%s | rank %d/%d | device %s", cfg.run_name, cfg.mode, args.config,
             dist_info.rank, dist_info.world_size, dist_info.device)
    if dist_info.is_main:
        cfg.save(work_dir / "resolved_config.json")

    import pandas as pd
    import torch
    if cfg.train.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(cfg.seed + dist_info.rank)

    selected = STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s) {unknown}; choose from {STAGES}")
    forced = {s.strip() for s in args.force.split(",") if s.strip()}
    state = RunState(work_dir)

    def should_run(name: str) -> bool:
        state.reload()
        if name not in selected:
            return False
        if name in forced:
            return True
        if state.done(name):
            log.info("Stage %s already completed (state.json) -> skipping. Use --force %s to redo.", name, name)
            return False
        return True

    def mark(name: str, **info) -> None:
        if dist_info.is_main:
            state.mark(name, info)
        barrier()

    with stage("preflight", work_dir, dist_info.rank):
        if dist_info.is_main:
            preflight(cfg, dist_info, log)
            if any(s in selected for s in ("train_llama", "outcomes", "latency")):
                check_gated_access(cfg.models.llama_id, log)
        barrier()

    from csf.data.manifest import load_run_manifest
    run_manifest = work_dir / "run_manifest.csv"
    if should_run("prepare") or not run_manifest.exists():
        with stage("prepare", work_dir, dist_info.rank):
            if dist_info.is_main:
                df = load_run_manifest(cfg.data, cfg.cache_dir, cfg.seed)
                df.to_csv(run_manifest, index=False)
            barrier()
            mark("prepare", rows=int(pd.read_csv(run_manifest).shape[0]))
    df = pd.read_csv(run_manifest, dtype={"video_id": str})

    index_file = cfg.cache_dir / "index.parquet"
    if should_run("features") or not index_file.exists():
        with stage("features", work_dir, dist_info.rank):
            from csf.data.feature_cache import build_feature_cache
            index = build_feature_cache(df, cfg, dist_info)
            mark("features", cached=int(len(index)))
    index = pd.read_parquet(index_file)
    index = index[index["video_id"].isin(set(df["video_id"]))].reset_index(drop=True)
    stats = json.loads((cfg.cache_dir / "feature_stats.json").read_text(encoding="utf-8"))
    tool_costs = json.loads((cfg.cache_dir / "tool_costs.json").read_text(encoding="utf-8"))
    log.info("Cached dataset: %d videos | %s", len(index), index["split"].value_counts().to_dict())

    ckpt = {k: work_dir / "checkpoints" / k / "best" for k in ("qwen", "llama")}
    from csf.train.classifier_trainer import train_classifier
    for kind in ("qwen", "llama"):
        name = f"train_{kind}"
        missing = name in selected and not (ckpt[kind] / "head.pt").exists()
        if missing and state.done(name):
            log.warning("%s is marked complete but %s is missing -> re-running it", name, ckpt[kind])
        if missing or should_run(name):
            with stage(name, work_dir, dist_info.rank):
                if name in forced and dist_info.is_main and ckpt[kind].parent.exists():
                    log.info("--force %s: removing previous checkpoints in %s (fresh training)", name, ckpt[kind].parent)
                    shutil.rmtree(ckpt[kind].parent)
                barrier()
                train_classifier(kind, cfg, dist_info, index, stats)
                if not (ckpt[kind] / "head.pt").exists():
                    raise RuntimeError(f"{name} finished without producing {ckpt[kind]}")
                mark(name)

    from csf.pipeline import build_outcomes, load_npz, predict_scanner, save_npz
    pred_dir = work_dir / "predictions"
    if should_run("predict_scanner"):
        with stage("predict_scanner", work_dir, dist_info.rank):
            from csf.models.classifier import load_classifier
            bundle = load_classifier(ckpt["qwen"], dist_info.device, attn_implementation=cfg.models.attn_implementation)
            for split in ("valid", "test"):
                res = predict_scanner(cfg, dist_info, index, stats, ckpt["qwen"], split, model_bundle=bundle)
                if dist_info.is_main:
                    save_npz(pred_dir / f"scanner_{split}.npz", res)
            del bundle
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            mark("predict_scanner")

    if should_run("outcomes"):
        with stage("outcomes", work_dir, dist_info.rank):
            scanner = {s: load_npz(pred_dir / f"scanner_{s}.npz") for s in ("valid", "test")}
            outs = build_outcomes(cfg, dist_info, index, stats, ckpt["llama"], scanner, ["valid", "test"])
            if dist_info.is_main:
                for split, o in outs.items():
                    save_npz(pred_dir / f"outcomes_{split}.npz", o)
            del outs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            mark("outcomes")

    from csf.eval.ablation import evaluate_ablation, train_dispatchers, write_report
    disp_dir = work_dir / "checkpoints" / "dispatcher"
    if should_run("train_dispatcher"):
        with stage("train_dispatcher", work_dir, dist_info.rank):
            if dist_info.is_main:
                train_dispatchers(cfg, load_npz(pred_dir / "outcomes_valid.npz"), tool_costs, dist_info.device)
            mark("train_dispatcher")

    if should_run("evaluate"):
        with stage("evaluate", work_dir, dist_info.rank):
            if dist_info.is_main:
                o_test = load_npz(pred_dir / "outcomes_test.npz")
                paths = {p: disp_dir / f"{p}.pt" for p in cfg.eval.profiles}
                results, raw = evaluate_ablation(cfg, index, load_npz(pred_dir / "scanner_test.npz"), o_test, paths,
                                                 dist_info.device)
                extra = {"tool_costs_median_s": tool_costs, "vision_state_sharing": bool(o_test["vision_cache_used"]),
                         "train_summaries": {k: json.loads((work_dir / "metrics" / f"train_{k}.json").read_text())
                                             for k in ("qwen", "llama")
                                             if (work_dir / "metrics" / f"train_{k}.json").exists()}}
                write_report(cfg, results, raw, o_test["labels"], extra)
            mark("evaluate")

    from csf.hub import export_bundle, push_interactive
    export_dir = work_dir / "export"
    if should_run("export"):
        with stage("export", work_dir, dist_info.rank):
            if dist_info.is_main:
                export_bundle(cfg, ckpt["qwen"], ckpt["llama"], disp_dir)
            mark("export")

    if should_run("latency"):
        with stage("latency", work_dir, dist_info.rank):
            if dist_info.is_main:
                from csf import PRETTY_LABELS
                from csf.eval.latency import live_latency_benchmark
                test = index[index["split"] == "test"].sort_values("video_id")
                vids, labs = [], []
                for cls, repo_path in zip(test["class"], test["repo_path"]):
                    p = Path(cfg.paths.video_dir) / repo_path
                    if p.exists():
                        vids.append(p)
                        labs.append(PRETTY_LABELS[cls])
                vids, labs = vids[:cfg.eval.latency_samples], labs[:cfg.eval.latency_samples]
                live_latency_benchmark(cfg, export_dir, vids, labs, resident=cfg.mode == "full")
                shutil.copytree(work_dir / "metrics", export_dir / "metrics", dirs_exist_ok=True)
            mark("latency")

    cleanup()
    if dist_info.is_main and "push" in selected:
        with stage("push", work_dir, dist_info.rank):
            url = push_interactive(cfg, export_dir)
            if url:
                state.mark("push", {"url": url})
    log.info("All requested stages finished. Outputs in %s", work_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
