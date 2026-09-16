"""
VLM tri-class classifier: (quantised) vision-language backbone + LoRA + linear head.

Used for both VLMs in the pipeline:
  kind="qwen"  -> Qwen2.5-VL-3B-Instruct            (Phase 1 semantic scanner, video input)
  kind="llama" -> Llama-3.2-11B-Vision-Instruct     (Phase 2 dispatcher state + Phase 4 arbiter)

Instead of generating "Real"/"AI-Generated"/"AI-Edited" tokens, the final-norm hidden state of the
last prompt token is captured with a forward hook and fed to a small fp32 linear head. This gives
calibrated class probabilities, a single forward pass per video, and lets `logits_to_keep=1` skip
the full-vocabulary LM head (large memory saving). LoRA adapters are attached only to the language
model's attention / MLP projections (vision towers stay frozen).

Memory profile is controlled by `quantization`: "4bit" (NF4 + double quant, QLoRA),
"8bit", or "none" (bf16/fp16 weights, fastest on >=24GB GPUs).

Input : model id, ClassifierTrainConfig, torch device / compute dtype.
Output: `VLMClassifier` whose forward(**processor_inputs) returns (logits[B,3], pooled[B,H]);
        `save_classifier` / `load_classifier` persist adapter + head + classifier_config.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from csf import LABELS
from csf.logging_utils import get_logger

log = get_logger("models.classifier")

LORA_TARGET_REGEX = r"^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
NON_MODEL_KEYS = ("labels", "keys", "tool_masks")


def compute_dtype_for(device: torch.device) -> torch.dtype:
    """bf16 only on Ampere or newer. `torch.cuda.is_bf16_supported()` also returns True on Turing
    (T4, sm_75), where bf16 is emulated: slow, and unsupported by several bitsandbytes kernels.
    fp16 + GradScaler (wired up in the trainer) is the correct path there."""
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if (major >= 8 and torch.cuda.is_bf16_supported()) else torch.float16


def find_final_norm(model: nn.Module) -> nn.Module:
    """Final norm of the text decoder = sibling `norm` of the largest `layers` ModuleList."""
    best, best_len = None, -1
    for _, module in model.named_modules():
        layers = getattr(module, "layers", None)
        norm = getattr(module, "norm", None)
        if isinstance(layers, nn.ModuleList) and isinstance(norm, nn.Module) and len(layers) > best_len:
            best, best_len = norm, len(layers)
    if best is None:
        raise RuntimeError("Could not locate the text decoder's final norm layer for hidden-state pooling.")
    return best


def hidden_size_of(model: nn.Module) -> int:
    cfg = model.config
    for sub in ("text_config", "language_config"):
        if hasattr(cfg, sub) and getattr(getattr(cfg, sub), "hidden_size", None):
            return int(getattr(cfg, sub).hidden_size)
    return int(cfg.hidden_size)


def _quant_config(quantization: str, compute_dtype: torch.dtype, skip_modules):
    if quantization == "none":
        return None
    from transformers import BitsAndBytesConfig
    if quantization == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=compute_dtype, llm_int8_skip_modules=list(skip_modules))
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=list(skip_modules))
    raise ValueError(f"Unknown quantization {quantization!r} (use 4bit / 8bit / none)")


def load_processor(model_id: str):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id)


def load_backbone(model_id: str, quantization: str, skip_modules, device: torch.device,
                  attn_implementation: str = "sdpa"):
    from transformers import AutoModelForImageTextToText
    if device.type != "cuda" and quantization != "none":
        log.warning("bitsandbytes quantisation needs CUDA; loading %s unquantised on %s", model_id, device)
        quantization = "none"
    dtype = compute_dtype_for(device)
    kwargs: Dict[str, Any] = {"attn_implementation": attn_implementation, "low_cpu_mem_usage": True}
    qc = _quant_config(quantization, dtype, skip_modules)
    if qc is not None:
        kwargs["quantization_config"] = qc
    if device.type == "cuda":
        kwargs["device_map"] = {"": device.index or 0}
    log.info("Loading backbone %s | quantization=%s | dtype=%s | attn=%s | device=%s",
             model_id, quantization, dtype, attn_implementation, device)
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    if device.type != "cuda":
        model.to(device)
    model.config.use_cache = False
    return model


class VLMClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int = len(LABELS), dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_labels))
        self._captured: Optional[torch.Tensor] = None
        self._logits_to_keep_ok = True
        find_final_norm(backbone).register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._captured = output[0] if isinstance(output, tuple) else output

    def forward(self, **inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        model_inputs = {k: v for k, v in inputs.items() if k not in NON_MODEL_KEYS}
        if self._logits_to_keep_ok:
            try:
                self.backbone(**model_inputs, use_cache=False, logits_to_keep=1)
            except TypeError as exc:
                if "logits_to_keep" not in str(exc):
                    raise
                self._logits_to_keep_ok = False
                self.backbone(**model_inputs, use_cache=False)
        else:
            self.backbone(**model_inputs, use_cache=False)
        hidden, self._captured = self._captured, None
        if hidden is None:
            raise RuntimeError("Final-norm hook did not fire; backbone architecture not supported.")
        mask = model_inputs["attention_mask"]
        last = mask.shape[1] - 1 - torch.argmax(mask.flip(dims=[1]).int(), dim=1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
        return self.head(pooled), pooled


def build_classifier(kind: str, model_id: str, tcfg, device: torch.device, attn_implementation: str,
                     for_training: bool = True) -> Tuple[VLMClassifier, Any]:
    from peft import LoraConfig, get_peft_model

    processor = load_processor(model_id)
    backbone = load_backbone(model_id, tcfg.quantization, tcfg.skip_quant_modules, device, attn_implementation)
    if for_training and tcfg.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in backbone.parameters():
        p.requires_grad_(False)
    lora = LoraConfig(r=tcfg.lora_r, lora_alpha=tcfg.lora_alpha, lora_dropout=tcfg.lora_dropout,
                      target_modules=LORA_TARGET_REGEX, bias="none")
    backbone = get_peft_model(backbone, lora)
    model = VLMClassifier(backbone, hidden_size_of(backbone))
    model.head.to(device=device, dtype=torch.float32)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("[%s] classifier ready: trainable params %.2fM / total %.2fM (%.3f%%)",
             kind, trainable / 1e6, total / 1e6, 100 * trainable / max(total, 1))
    return model, processor


def save_classifier(model: VLMClassifier, out_dir: Path, meta: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(out_dir / "adapter"))
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    (out_dir / "classifier_config.json").write_text(json.dumps({"labels": LABELS, **meta}, indent=2),
                                                    encoding="utf-8")


def load_classifier(ckpt_dir: Path, device: torch.device, quantization: Optional[str] = None,
                    skip_modules=("multi_modal_projector", "lm_head"), attn_implementation: str = "sdpa",
                    model_id: Optional[str] = None) -> Tuple[VLMClassifier, Any, Dict[str, Any]]:
    from peft import PeftModel

    meta = json.loads((ckpt_dir / "classifier_config.json").read_text(encoding="utf-8"))
    model_id = model_id or meta["model_id"]
    processor = load_processor(model_id)
    backbone = load_backbone(model_id, quantization or meta.get("quantization", "4bit"),
                             meta.get("skip_quant_modules", list(skip_modules)), device, attn_implementation)
    backbone = PeftModel.from_pretrained(backbone, str(ckpt_dir / "adapter"), is_trainable=False)
    model = VLMClassifier(backbone, hidden_size_of(backbone))
    model.head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))
    model.head.to(device=device, dtype=torch.float32)
    model.eval()
    return model, processor, meta
