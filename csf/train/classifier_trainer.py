"""
Distributed (DDP) fine-tuning loop for a `VLMClassifier` (Qwen scanner or Llama arbiter).

Features: DistributedSampler sharding, gradient accumulation with `no_sync`, bf16/fp16 autocast
(GradScaler on fp16-only GPUs), 8-bit paged AdamW, cosine schedule with warm-up, gradient clipping,
periodic validation (macro-F1 selects the best checkpoint), early stopping, NaN/Inf guard that
names the offending batch, resumable "last" checkpoints and a JSONL log of every optimiser step.

Input : kind ("qwen" | "llama"), Config, DistInfo, cached index DataFrame, feature stats.
Output: <work_dir>/checkpoints/<kind>/best  (adapter/, head.pt, classifier_config.json)
        <work_dir>/checkpoints/<kind>/last  (+ trainer_state.pt for resuming)
        <work_dir>/metrics/train_<kind>.json and <work_dir>/logs/train_<kind>.jsonl
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from csf import LABELS
from csf.config import Config
from csf.data.datasets import CachedVideoDataset, LlamaCollator, QwenCollator
from csf.distributed import DistInfo, all_gather_objects, all_reduce_mean, barrier
from csf.logging_utils import Throughput, get_logger, set_context
from csf.models.classifier import NON_MODEL_KEYS, build_classifier, compute_dtype_for, save_classifier

log = get_logger("train.classifier")


def make_collator(kind: str, processor, cfg: Config, train: bool):
    if kind == "qwen":
        return QwenCollator(processor)
    tool_dropout = cfg.train.llama.tool_dropout if train else 0.0
    return LlamaCollator(processor, cfg.data.mosaic_frames, cfg.data.mosaic_size, tool_dropout=tool_dropout)


def _move(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device, non_blocking=True)
            if v.is_floating_point() and k.startswith("pixel_values"):
                v = v.to(dtype)
        out[k] = v
    return out


def _optimizer(model: torch.nn.Module, tcfg, device: torch.device):
    lora = [p for n, p in model.named_parameters() if p.requires_grad and ".head." not in f".{n}"]
    head = [p for n, p in model.named_parameters() if p.requires_grad and ".head." in f".{n}"]
    groups = [{"params": lora, "lr": tcfg.lr, "weight_decay": tcfg.weight_decay},
              {"params": head, "lr": tcfg.head_lr, "weight_decay": 0.0}]
    if tcfg.optimizer == "adamw_8bit" and device.type == "cuda":
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(groups)
        except Exception as exc:
            log.warning("bitsandbytes 8-bit AdamW unavailable (%s) -> torch AdamW", exc)
    return torch.optim.AdamW(groups, fused=device.type == "cuda")


def _scheduler(opt, total_steps: int, warmup_ratio: float):
    warm = max(1, int(total_steps * warmup_ratio))

    def fn(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def run_inference(model, loader: DataLoader, device: torch.device, dtype: torch.dtype, dist_info: DistInfo,
                  max_batches=None, return_pooled: bool = False, desc: str = "infer",
                  active_ids: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:
    """Returns {key: {"probs": np.ndarray[3], "label": int, "latency": sec/sample, ("pooled")}} gathered from all ranks.

    `probs` always has one column per label in LABELS so everything downstream keeps its shape. When
    `active_ids` excludes a class, the softmax is taken over the active columns only and the excluded
    ones are reported as exactly 0 - renormalising over an untrained logit would invent a probability."""
    from tqdm import tqdm
    was_training = model.training
    model.eval()
    results: Dict[str, Dict[str, Any]] = {}
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    for bi, batch in enumerate(tqdm(loader, total=total, desc=desc, disable=not dist_info.is_main,
                                    dynamic_ncols=True, leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        set_context(batch_index=bi, batch_keys=batch["keys"])
        batch = _move(batch, device, dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits, pooled = model(**{k: v for k, v in batch.items() if k not in NON_MODEL_KEYS})
        if active_ids is not None and len(active_ids) < logits.shape[-1]:
            idx = torch.as_tensor(active_ids, device=logits.device)
            probs = torch.zeros_like(logits, dtype=torch.float32)
            probs[:, idx] = torch.softmax(logits.float()[:, idx], dim=-1)
        else:
            probs = torch.softmax(logits.float(), dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_sample = (time.perf_counter() - t0) / len(batch["keys"])
        for i, key in enumerate(batch["keys"]):
            rec = {"probs": probs[i].cpu().numpy(), "label": int(batch["labels"][i]), "latency": per_sample}
            if return_pooled:
                rec["pooled"] = pooled[i].cpu().numpy().astype(np.float16)
            results[key] = rec
    if was_training:
        model.train()
    merged: Dict[str, Dict[str, Any]] = {}
    for part in all_gather_objects(results):
        merged.update(part)
    return merged


def _val_metrics(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    y = np.array([r["label"] for r in results.values()])
    p = np.stack([r["probs"] for r in results.values()])
    pred = p.argmax(1)
    loss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())
    return {"val_acc": float(accuracy_score(y, pred)), "val_macro_f1": float(f1_score(y, pred, average="macro")),
            "val_loss": loss, "val_n": int(len(y))}


def train_classifier(kind: str, cfg: Config, dist_info: DistInfo, index: pd.DataFrame,
                     stats: Dict[str, Any]) -> Path:
    from tqdm import tqdm

    tcfg = cfg.train.qwen if kind == "qwen" else cfg.train.llama
    model_id = cfg.models.qwen_id if kind == "qwen" else cfg.models.llama_id
    device = dist_info.device
    dtype = compute_dtype_for(device)
    ckpt_root = cfg.work_dir / "checkpoints" / kind
    best_dir, last_dir = ckpt_root / "best", ckpt_root / "last"
    jsonl = cfg.work_dir / "logs" / f"train_{kind}.jsonl"

    active_ids = cfg.data.active_label_ids()
    # Slice the logits to the active classes rather than masking the others to -inf: with
    # label_smoothing > 0 an excluded class still carries a non-zero target, and -log(0) is inf.
    active_idx = torch.tensor(active_ids, device=device)
    label_remap = torch.full((len(LABELS),), -1, dtype=torch.long, device=device)
    label_remap[active_idx] = torch.arange(len(active_ids), device=device)
    if len(active_ids) < len(LABELS):
        log.warning("[%s] training on %d of %d classes (%s); the excluded head row(s) stay at "
                    "initialisation instead of being trained to never fire.", kind, len(active_ids),
                    len(LABELS), [LABELS[i] for i in active_ids])

    model, processor = build_classifier(kind, model_id, tcfg, device, cfg.models.attn_implementation)
    if device.type == "cuda":
        log.info("[%s] GPU memory after load: %.2f GiB", kind, torch.cuda.memory_allocated() / 2**30)

    train_ds = CachedVideoDataset(index[index["split"] == "train"], cfg.cache_dir, stats)
    val_ds = CachedVideoDataset(index[index["split"] == "valid"], cfg.cache_dir, stats)
    train_sampler = DistributedSampler(train_ds, dist_info.world_size, dist_info.rank, shuffle=True, seed=cfg.seed) \
        if dist_info.distributed else None
    val_sampler = DistributedSampler(val_ds, dist_info.world_size, dist_info.rank, shuffle=False) \
        if dist_info.distributed else None
    nw = min(cfg.train.num_workers, os.cpu_count() or 1)
    if nw < cfg.train.num_workers:
        log.warning("num_workers %d > %d available CPU(s) -> using %d (over-subscribing starves the GPU)",
                    cfg.train.num_workers, os.cpu_count(), nw)
    loader_kw = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    if nw > 0:
        loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, sampler=train_sampler,
                              shuffle=train_sampler is None, drop_last=True,
                              collate_fn=make_collator(kind, processor, cfg, train=True), **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=tcfg.eval_batch_size, sampler=val_sampler, shuffle=False,
                            collate_fn=make_collator(kind, processor, cfg, train=False), **loader_kw)
    if len(train_loader) == 0:
        raise RuntimeError(f"[{kind}] empty training loader: {len(train_ds)} samples, batch {tcfg.batch_size}, "
                           f"world {dist_info.world_size}")

    steps_per_epoch = max(1, len(train_loader) // tcfg.grad_accum)
    total_steps = int(math.ceil(tcfg.epochs * steps_per_epoch))
    if tcfg.max_steps:
        total_steps = min(total_steps, tcfg.max_steps)
    opt = _optimizer(model, tcfg, device)
    sched = _scheduler(opt, total_steps, tcfg.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    step, best_f1, bad_evals, history = 0, -1.0, 0, []
    state_file = last_dir / "trainer_state.pt"
    if state_file.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        st = torch.load(state_file, map_location="cpu", weights_only=False)
        adapter_file = last_dir / "adapter" / "adapter_model.safetensors"
        set_peft_model_state_dict(model.backbone, load_file(str(adapter_file)))
        model.head.load_state_dict(torch.load(last_dir / "head.pt", map_location="cpu"))
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step, best_f1, bad_evals, history = st["step"], st["best_f1"], st["bad_evals"], st["history"]
        log.info("[%s] RESUMED from %s at optimiser step %d (best val macro-F1 %.4f)", kind, last_dir, step, best_f1)

    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    model._ddp_params_and_buffers_to_ignore = [n for n, _ in model.named_parameters() if n not in trainable_names]
    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=tcfg.ddp_find_unused_parameters, broadcast_buffers=False) \
        if dist_info.distributed else model

    log.info("[%s] training: %d train / %d valid samples | batch %d x accum %d x world %d = %d effective | "
             "%d steps/epoch | total optimiser steps %d | lr %.2e (head %.2e) | dtype %s",
             kind, len(train_ds), len(val_ds), tcfg.batch_size, tcfg.grad_accum, dist_info.world_size,
             tcfg.batch_size * tcfg.grad_accum * dist_info.world_size, steps_per_epoch, total_steps,
             tcfg.lr, tcfg.head_lr, dtype)

    def evaluate_and_checkpoint(force_save: bool = False) -> bool:
        nonlocal best_f1, bad_evals
        res = run_inference(ddp_model.module if dist_info.distributed else ddp_model, val_loader, device, dtype,
                            dist_info, max_batches=tcfg.max_eval_batches, desc=f"val {kind}",
                            active_ids=active_ids)
        m = _val_metrics(res)
        m.update(step=step, time=time.time())
        history.append(m)
        improved = m["val_macro_f1"] > best_f1 or force_save
        log.info("[%s] step %d VALIDATION: acc=%.4f macro_f1=%.4f loss=%.4f (n=%d) %s", kind, step, m["val_acc"],
                 m["val_macro_f1"], m["val_loss"], m["val_n"], "<- new best" if improved else "")
        if improved:
            best_f1, bad_evals = max(best_f1, m["val_macro_f1"]), 0
            if dist_info.is_main:
                save_classifier(model, best_dir, {"kind": kind, "model_id": model_id, "quantization": tcfg.quantization,
                                                  "skip_quant_modules": list(tcfg.skip_quant_modules), "step": step,
                                                  "val": m, "num_frames": cfg.data.num_frames,
                                                  "frame_size": cfg.data.frame_size,
                                                  "mosaic_frames": cfg.data.mosaic_frames,
                                                  "mosaic_size": cfg.data.mosaic_size})
        else:
            bad_evals += 1
        barrier()
        return bad_evals >= tcfg.early_stop_patience

    def save_last():
        if dist_info.is_main:
            tmp = ckpt_root / "last_tmp"
            save_classifier(model, tmp, {"kind": kind, "model_id": model_id, "quantization": tcfg.quantization,
                                         "skip_quant_modules": list(tcfg.skip_quant_modules), "step": step})
            torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step,
                        "best_f1": best_f1, "bad_evals": bad_evals, "history": history}, tmp / "trainer_state.pt")
            if last_dir.exists():
                shutil.rmtree(last_dir)
            tmp.rename(last_dir)
            log.info("[%s] saved resumable checkpoint at step %d -> %s", kind, step, last_dir)
        barrier()

    ddp_model.train()
    t_start = time.time()
    tp = Throughput(total_steps)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    pbar = tqdm(total=total_steps, initial=step, desc=f"train {kind}", disable=not dist_info.is_main,
                dynamic_ncols=True)
    stop, epoch = step >= total_steps, step // steps_per_epoch
    running: List[float] = []
    while not stop:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        micro = 0
        for batch in train_loader:
            set_context(epoch=epoch, step=step, micro=micro, batch_keys=batch["keys"])
            batch = _move(batch, device, dtype)
            sync = (micro + 1) % tcfg.grad_accum == 0
            ctx = ddp_model.no_sync() if (dist_info.distributed and not sync) else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                    logits, _ = ddp_model(**{k: v for k, v in batch.items() if k not in NON_MODEL_KEYS})
                loss = F.cross_entropy(logits.float()[:, active_idx], label_remap[batch["labels"]],
                                       label_smoothing=tcfg.label_smoothing)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"[{kind}] non-finite loss {loss.item()} at step {step} micro {micro}; "
                                             f"batch keys={batch['keys']}")
                scaler.scale(loss / tcfg.grad_accum).backward()
            running.append(loss.item())
            micro += 1
            if not sync:
                continue

            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            loss_avg = all_reduce_mean(float(np.mean(running)), device)
            running.clear()
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            rec = {"step": step, "epoch": epoch, "loss": loss_avg, "grad_norm": float(gnorm),
                   "lr": sched.get_last_lr()[0], "peak_mem_gib": mem, "elapsed_s": time.time() - t_start}
            if dist_info.is_main:
                with open(jsonl, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_avg:.4f}", lr=f"{rec['lr']:.1e}", mem=f"{mem:.1f}G")
                if step % 10 == 0 or step == 1:
                    log.info("[%s] step %s | loss %.4f | grad_norm %.3f | lr %.2e | peak_mem %.2f GiB",
                             kind, tp.line(step), loss_avg, rec["grad_norm"], rec["lr"], mem)
            if step % tcfg.eval_every == 0 or step >= total_steps:
                if evaluate_and_checkpoint():
                    log.info("[%s] early stopping: no val improvement for %d evaluations", kind,
                             tcfg.early_stop_patience)
                    stop = True
            if not stop and step % tcfg.save_every == 0:
                save_last()
            if step >= total_steps:
                stop = True
            if stop:
                break
        epoch += 1
    pbar.close()

    if not best_dir.exists():
        evaluate_and_checkpoint(force_save=True)
    summary = {"kind": kind, "model_id": model_id, "optimizer_steps": step, "epochs_seen": epoch,
               "best_val_macro_f1": best_f1, "train_time_s": time.time() - t_start,
               "peak_train_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
               "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "total_params": sum(p.numel() for p in model.parameters()),
               "effective_batch": tcfg.batch_size * tcfg.grad_accum * dist_info.world_size,
               "history": history}
    if dist_info.is_main:
        (cfg.work_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (cfg.work_dir / "metrics" / f"train_{kind}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("[%s] training finished: %d steps in %.1f min, best val macro-F1 %.4f", kind, step,
             summary["train_time_s"] / 60, best_f1)

    ddp_model = model = opt = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    barrier()
    return best_dir
