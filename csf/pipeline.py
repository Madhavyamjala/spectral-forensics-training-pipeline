"""
CSF orchestration over the feature cache: scanner predictions and arbiter outcome tables.

`predict_scanner`  - Phase 1 Qwen2.5-VL scanner probabilities (+ per-sample latency) for a split.
`build_outcomes`   - For every video in a split, runs the shared Llama-3.2-Vision backbone:
                        pass 0      pixels + "no tools executed" prompt -> dispatcher state + no-tool probs
                        static full pixels + all tools                -> "Llama arbiter (static)" model type
                        per action  evidence prompt for the action's tool subset, REUSING the vision
                                    cross-attention states captured in pass 0 (the proposal's shared
                                    dispatcher/arbiter cache: the 11B vision encoder runs once per video)
                     The early-exit row is the mean of scanner and no-tool probabilities.
                     Work is sharded across ranks and gathered on every rank.

Input : Config, DistInfo, index DataFrame, feature stats, checkpoint dirs.
Output: dicts of numpy arrays saved to <work_dir>/predictions/{scanner,outcomes}_<split>.npz
        keys, labels, methods, probs[...], latency[...], state (float16) ...
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler

from csf import LABELS, PRETTY_LABELS
from csf.config import Config
from csf.data.datasets import CachedVideoDataset, LlamaCollator, QwenCollator
from csf.distributed import DistInfo, all_gather_objects
from csf.logging_utils import Throughput, get_logger, set_context
from csf.models.classifier import NON_MODEL_KEYS, VLMClassifier, compute_dtype_for, load_classifier
from csf.models.dispatcher import ACTIONS, action_mask_array
from csf.train.classifier_trainer import _move, run_inference

log = get_logger("pipeline")
VISION_ONLY_KEYS = ("pixel_values", "aspect_ratio_ids", "aspect_ratio_mask")


def _active_probs_from_logits(logits: torch.Tensor, active_ids: List[int]) -> torch.Tensor:
    """Softmax only across trained classes, then scatter probabilities into canonical label slots.

    Note:
        Two-class runs keep a canonical three-slot head so downstream artefacts remain shape-stable,
        but the excluded class must not participate in the softmax because its head row was not trained.

    TODO:
        Remove the compatibility path once all published bundles carry an explicit head schema.
    """
    active = list(active_ids)
    if not active:
        raise ValueError("active_ids must contain at least one class")
    if logits.shape[-1] == len(active):
        active_probs = torch.softmax(logits.float(), dim=-1)
    elif logits.shape[-1] == len(LABELS):
        idx = torch.as_tensor(active, device=logits.device, dtype=torch.long)
        active_probs = torch.softmax(logits.float()[:, idx], dim=-1)
    else:
        raise ValueError(
            f"Unexpected classifier head width {logits.shape[-1]}; expected {len(active)} or {len(LABELS)}."
        )
    out = torch.zeros((logits.shape[0], len(LABELS)), device=logits.device, dtype=torch.float32)
    idx = torch.as_tensor(active, device=logits.device, dtype=torch.long)
    out[:, idx] = active_probs
    return out


def save_npz(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in data.items():
        a = np.asarray(v)
        arrays[k] = a.astype(str) if a.dtype == object else a
    np.savez(path, **arrays)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _ordered(index: pd.DataFrame, results: Dict[str, Dict[str, Any]]):
    keys = [f"{c}/{v}" for c, v in zip(index["class"], index["video_id"])]
    missing = [k for k in keys if k not in results]
    if missing:
        raise RuntimeError(f"{len(missing)} prediction(s) missing after gather, e.g. {missing[:5]}")
    return keys


def predict_scanner(cfg: Config, dist_info: DistInfo, index: pd.DataFrame, stats, ckpt: Path, split: str,
                    model_bundle=None) -> Dict[str, np.ndarray]:
    device = dist_info.device
    dtype = compute_dtype_for(device)
    model, processor, _ = model_bundle or load_classifier(ckpt, device, attn_implementation=cfg.models.attn_implementation)
    sub = index[index["split"] == split].reset_index(drop=True)
    ds = CachedVideoDataset(sub, cfg.cache_dir, stats)
    sampler = DistributedSampler(ds, dist_info.world_size, dist_info.rank, shuffle=False) if dist_info.distributed else None
    loader = DataLoader(ds, batch_size=cfg.train.qwen.eval_batch_size, sampler=sampler, shuffle=False,
                        num_workers=cfg.train.num_workers, collate_fn=QwenCollator(processor, [PRETTY_LABELS[LABELS[i]] for i in cfg.data.active_label_ids()]))
    log.info("Scanner inference on %s split: %d videos", split, len(sub))
    res = run_inference(model, loader, device, dtype, dist_info, desc=f"scanner {split}",
                        active_ids=cfg.data.active_label_ids())
    keys = _ordered(sub, res)
    return {"keys": np.array(keys), "labels": np.array([res[k]["label"] for k in keys]),
            "methods": sub["method"].astype(str).to_numpy(), "probs": np.stack([res[k]["probs"] for k in keys]),
            "latency": np.array([res[k]["latency"] for k in keys]),
            "t_decode": sub["t_decode"].to_numpy()}


class ArbiterRunner:
    """Llama-3.2-Vision classifier with optional reuse of cross-attention (vision) states across prompts."""

    def __init__(self, model: VLMClassifier, processor, cfg: Config, device: torch.device,
                 active_ids: List[int] | None = None):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.device = device
        self.dtype = compute_dtype_for(device)
        self.active_ids = list(active_ids if active_ids is not None else range(len(LABELS)))
        self.class_names = [PRETTY_LABELS[LABELS[i]] for i in self.active_ids]
        self.hidden = model.backbone.get_base_model().config.text_config.hidden_size \
            if hasattr(model.backbone.get_base_model().config, "text_config") else None
        self._vision_out = None
        self.cache_ok = False
        projector = None
        for name, module in model.backbone.named_modules():
            if name.endswith("multi_modal_projector"):
                projector = module
        if projector is not None and self.hidden is not None:
            projector.register_forward_hook(self._capture)
            self.cache_ok = True
        else:
            log.warning("No multi_modal_projector found -> vision-state sharing disabled (full pixel passes).")

    def _capture(self, module, inputs, output):
        self._vision_out = output

    def collator(self, mask=None) -> LlamaCollator:
        return LlamaCollator(self.processor, self.cfg.data.mosaic_frames, self.cfg.data.mosaic_size,
                             tool_dropout=0.0, force_mask=mask, class_names=self.class_names)

    def _forward(self, batch: Dict[str, Any]):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            logits, pooled = self.model(**{k: v for k, v in batch.items() if k not in NON_MODEL_KEYS})
        probs = _active_probs_from_logits(logits, self.active_ids)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return probs.cpu().numpy(), pooled, (time.perf_counter() - t0) / probs.shape[0]

    def pixel_pass(self, samples: List[Dict[str, Any]], mask: np.ndarray):
        self._vision_out = None
        batch = _move(self.collator(mask)(samples), self.device, self.dtype)
        probs, pooled, t = self._forward(batch)
        cache = None
        if self.cache_ok and self._vision_out is not None:
            v = self._vision_out
            cache = v.reshape(-1, v.shape[-2], self.hidden)
        return probs, pooled.cpu().numpy().astype(np.float16), t, cache

    def cached_pass(self, samples: List[Dict[str, Any]], mask: np.ndarray, cache):
        batch = _move(self.collator(mask)(samples), self.device, self.dtype)
        if cache is not None and self.cache_ok:
            fast = {k: v for k, v in batch.items() if k not in VISION_ONLY_KEYS}
            fast["cross_attention_states"] = cache
            try:
                probs, _, t = self._forward(fast)
                return probs, t
            except Exception as exc:
                log.warning("Shared vision-state pass failed (%s: %s) -> falling back to full pixel passes "
                            "for the rest of the run.", type(exc).__name__, str(exc)[:300])
                self.cache_ok = False
        probs, _, t = self._forward(batch)
        return probs, t


def build_outcomes(cfg: Config, dist_info: DistInfo, index: pd.DataFrame, stats, llama_ckpt: Path,
                   scanner: Dict[str, Dict[str, np.ndarray]], splits: List[str], model_bundle=None) -> Dict[str, Dict[str, np.ndarray]]:
    from tqdm import tqdm
    device = dist_info.device
    model, processor, _ = model_bundle or load_classifier(llama_ckpt, device, attn_implementation=cfg.models.attn_implementation)
    runner = ArbiterRunner(model, processor, cfg, device, active_ids=cfg.data.active_label_ids())
    bs = cfg.train.llama.eval_batch_size
    none_mask = np.zeros(3, bool)
    all_mask = np.ones(3, bool)
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for split in splits:
        sub = index[index["split"] == split].reset_index(drop=True)
        if split == "valid" and cfg.train.dispatcher.train_videos and len(sub) > cfg.train.dispatcher.train_videos:
            sub = sub.sample(n=cfg.train.dispatcher.train_videos, random_state=cfg.seed).reset_index(drop=True)
        ds = CachedVideoDataset(sub, cfg.cache_dir, stats)
        mine = list(range(dist_info.rank, len(ds), dist_info.world_size))
        log.info("Outcome table for %s: %d videos (%d on this rank), batch %d, %d arbiter passes/video",
                 split, len(ds), len(mine), bs, 2 + len(ACTIONS) - 1)
        res: Dict[str, Dict[str, Any]] = {}
        tp = Throughput(len(mine))
        last = time.time()
        for bi in tqdm(range(0, len(mine), bs), desc=f"outcomes {split}", disable=not dist_info.is_main,
                       dynamic_ncols=True):
            samples = [ds[i] for i in mine[bi:bi + bs]]
            set_context(split=split, batch_keys=[s["key"] for s in samples])
            p0, state, t0, cache = runner.pixel_pass(samples, none_mask)
            p_static, _, t_static, _ = runner.pixel_pass(samples, all_mask)
            per_action, t_action = {}, {}
            for a in ACTIONS[1:]:
                per_action[a], t_action[a] = runner.cached_pass(samples, action_mask_array(a), cache)
            for j, s in enumerate(samples):
                res[s["key"]] = {"label": s["label"], "notool": p0[j], "state": state[j], "t_state": t0,
                                 "static": p_static[j], "t_static": t_static,
                                 "actions": np.stack([per_action[a][j] for a in ACTIONS[1:]]),
                                 "t_actions": np.array([t_action[a] for a in ACTIONS[1:]])}
            if time.time() - last > 30:
                log.info("outcomes %s: %s", split, tp.line(min(bi + bs, len(mine))))
                last = time.time()

        merged: Dict[str, Dict[str, Any]] = {}
        for part in all_gather_objects(res):
            merged.update(part)
        keys = _ordered(sub, merged)
        sc = scanner[split]
        sc_pos = {k: i for i, k in enumerate(sc["keys"].tolist())}
        scanner_probs = np.stack([sc["probs"][sc_pos[k]] for k in keys])
        notool = np.stack([merged[k]["notool"] for k in keys])
        early = (scanner_probs + notool) / 2.0
        action_probs = np.concatenate([early[:, None], np.stack([merged[k]["actions"] for k in keys])], axis=1)
        static = np.stack([merged[k]["static"] for k in keys])
        diff = float(np.abs(static - action_probs[:, ACTIONS.index("full_tri_domain")]).max())
        log.info("[%s] sanity: max |p(static full, pixel pass) - p(full_tri_domain, shared vision states)| = %.2e %s",
                 split, diff, "(OK)" if diff < 0.05 else "(LARGE - vision-state sharing may be inexact)")
        out[split] = {
            "keys": np.array(keys), "labels": np.array([merged[k]["label"] for k in keys]),
            "methods": sub["method"].astype(str).to_numpy(), "scanner_probs": scanner_probs,
            "scanner_latency": np.array([sc["latency"][sc_pos[k]] for k in keys]),
            "notool_probs": notool, "state": np.stack([merged[k]["state"] for k in keys]),
            "t_state": np.array([merged[k]["t_state"] for k in keys]),
            "static_probs": static, "t_static": np.array([merged[k]["t_static"] for k in keys]),
            "action_probs": action_probs,
            "t_actions": np.concatenate([np.zeros((len(keys), 1)), np.stack([merged[k]["t_actions"] for k in keys])], 1),
            "t_tools": sub[["t_decode", "t_proposal", "t_spatial", "t_spectral", "t_latent"]].to_numpy(),
            "vision_cache_used": np.array(runner.cache_ok),
        }
    return out
