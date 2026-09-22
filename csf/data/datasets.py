"""
Torch datasets and model-specific collators over the feature cache.

`CachedVideoDataset` reads one <cache_dir>/items/<class>/<video_id>.npz per sample and returns
    frames (T,S,S,3 uint8), z (normalised toolpool features), mask (tool groups run), label, key, method.

Collators turn a list of samples into processor inputs:
  * `QwenCollator`  - Phase 1 scanner prompt with the frames as a Qwen2.5-VL *video* input.
  * `LlamaCollator` - Phase 4 arbiter prompt: one 2x2 frame mosaic image + evidence-graph text.
                      During training, whole tool groups are randomly dropped (`tool_dropout`) so the
                      arbiter learns to reason with whatever subset the dispatcher decides to run.
                      `force_mask` overrides the mask (used by the dispatcher / CSF evaluation).

Input : index DataFrame (from the feature cache), feature stats, HF processor.
Output: dict of tensors accepted by the VLM forward + "labels" (LongTensor) + "keys" (list[str]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from csf.data.feature_cache import item_path
from csf.data.video_io import decode_jpegs, make_mosaic
from csf.graph import evidence_text, normalize_features

QWEN_SYSTEM = ("You are a forensic video pre-scanner. Inspect faces, lighting boundaries, textures and "
               "motion for signs of synthesis or local manipulation.")
QWEN_QUESTION = "Is this video {classes}? Answer:"
LLAMA_QUESTION = ("You are a forensic arbiter. The image is a 2x2 mosaic of frames sampled from one video. "
                  "Using the frames and the evidence graph below, decide which candidate class best explains the video: "
                  "{classes}.\n\n{evidence}\n\nAnswer:")


class CachedVideoDataset(Dataset):
    def __init__(self, index: pd.DataFrame, cache_dir: Path, stats: Dict[str, Any]):
        self.index = index.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self.index.iloc[i]
        path = item_path(self.cache_dir, row["class"], row["video_id"])
        try:
            with np.load(path) as z:
                frames = decode_jpegs(z["frames_buf"], z["frames_off"])
                raw = z["features"]
                mask = z["mask"].astype(bool)
                times = z["times"]
        except Exception as exc:
            raise RuntimeError(f"Failed to read cached item {path} (row {i}, {row['class']}/{row['video_id']}): "
                               f"{type(exc).__name__}: {exc}") from exc
        return {"frames": frames, "z": normalize_features(raw, self.stats), "mask": mask,
                "label": int(row["label"]), "key": f"{row['class']}/{row['video_id']}",
                "method": row.get("method", "unknown"), "times": times}


def _chat(processor, messages, fallback: str) -> str:
    if getattr(processor, "chat_template", None) is None:
        return fallback
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _pad_token_fix(processor) -> None:
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"


class QwenCollator:
    def __init__(self, processor, class_names: Optional[List[str]] = None):
        self.processor = processor
        _pad_token_fix(processor)
        self.class_names = list(class_names or ["Real", "AI-Generated", "AI-Edited"])
        question = QWEN_QUESTION.format(classes=", ".join(self.class_names))
        messages = [{"role": "system", "content": [{"type": "text", "text": QWEN_SYSTEM}]},
                    {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
        video_token = getattr(processor, "video_token", "<|video_pad|>")
        self.prompt = _chat(processor, messages,
                            f"{QWEN_SYSTEM}\n<|vision_start|>{video_token}<|vision_end|>{question}")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        videos = [list(s["frames"]) for s in batch]
        enc = self.processor(text=[self.prompt] * len(batch), videos=videos, padding=True, return_tensors="pt")
        out = dict(enc)
        out["labels"] = torch.tensor([s["label"] for s in batch], dtype=torch.long)
        out["keys"] = [s["key"] for s in batch]
        return out


class LlamaCollator:
    def __init__(self, processor, mosaic_frames: int, mosaic_size: int, tool_dropout: float = 0.0,
                 force_mask: Optional[np.ndarray] = None, class_names: Optional[List[str]] = None):
        self.processor = processor
        _pad_token_fix(processor)
        self.mosaic_frames = mosaic_frames
        self.mosaic_size = mosaic_size
        self.tool_dropout = tool_dropout
        self.force_mask = force_mask
        self.class_names = list(class_names or ["Real", "AI-Generated", "AI-Edited"])

    def prompt_for(self, z: np.ndarray, mask: np.ndarray) -> str:
        classes = ", ".join(self.class_names)
        text = LLAMA_QUESTION.format(classes=classes, evidence=evidence_text(z, mask, candidate_labels=self.class_names))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        bos = getattr(self.processor.tokenizer, "bos_token", "") or ""
        return _chat(self.processor, messages, f"{bos}{getattr(self.processor, 'image_token', '<|image|>')}{text}")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts, images, masks = [], [], []
        for s in batch:
            mask = s["mask"].copy()
            if self.force_mask is not None:
                mask = mask & self.force_mask
            elif self.tool_dropout > 0:
                mask = mask & (torch.rand(len(mask)).numpy() >= self.tool_dropout)
            masks.append(mask)
            texts.append(self.prompt_for(s["z"], mask))
            images.append([make_mosaic(s["frames"], self.mosaic_frames, self.mosaic_size)])
        enc = self.processor(images=images, text=texts, padding=True, return_tensors="pt", add_special_tokens=False)
        out = dict(enc)
        out["labels"] = torch.tensor([s["label"] for s in batch], dtype=torch.long)
        out["keys"] = [s["key"] for s in batch]
        out["tool_masks"] = torch.tensor(np.stack(masks))
        return out
