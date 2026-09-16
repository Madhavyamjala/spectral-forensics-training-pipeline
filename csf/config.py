"""
Run configuration.

Loads a YAML file (configs/test.yaml or configs/full.yaml) into nested dataclasses and
applies command-line overrides of the form `section.sub.key=value`.

Input : path to a YAML file, optional list of "dotted.key=value" override strings.
Output: `Config` object; `Config.to_dict()` gives the resolved, JSON-serialisable view
        that is written to <work_dir>/resolved_config.json for reproducibility.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PathsConfig:
    work_dir: str = "./runs/test"
    cache_dir: str = "./cache/test"
    video_dir: str = "./cache/test/videos"


@dataclass
class DataConfig:
    repo_id: str = "madhav-yrc/Chrono-TriClass-100k"
    revision: Optional[str] = None
    manifest: str = "manifest.csv"
    max_rows: Optional[int] = 5000
    num_frames: int = 8
    frame_size: int = 224
    mosaic_frames: int = 4
    mosaic_size: int = 560
    patch_size: int = 64
    num_patches: int = 4
    tool_max_side: int = 1024
    jpeg_quality: int = 95
    delete_videos_after_cache: bool = True
    keep_videos_for_latency: int = 20
    download_workers: int = 8
    prefetch: int = 16
    download_fail_fast: int = 10
    extract_fail_fast: int = 50


@dataclass
class ModelsConfig:
    qwen_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    llama_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    vae_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    vae_subfolder: str = "vae"
    vae_fallback_id: str = "stabilityai/sd-vae-ft-mse"
    attn_implementation: str = "sdpa"


@dataclass
class ClassifierTrainConfig:
    quantization: str = "4bit"
    skip_quant_modules: List[str] = field(default_factory=lambda: ["multi_modal_projector", "lm_head"])
    epochs: float = 1.0
    max_steps: Optional[int] = None
    batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    label_smoothing: float = 0.05
    tool_dropout: float = 0.3
    gradient_checkpointing: bool = True
    optimizer: str = "adamw_8bit"
    eval_every: int = 200
    save_every: int = 200
    max_eval_batches: Optional[int] = None
    early_stop_patience: int = 3
    ddp_find_unused_parameters: bool = False


@dataclass
class DispatcherTrainConfig:
    train_videos: int = 300
    iterations: int = 300
    batch_videos: int = 64
    group_size: int = 8
    inner_epochs: int = 2
    lr: float = 3e-4
    clip_eps: float = 0.2
    kl_beta: float = 0.02
    entropy_coef: float = 0.01
    hidden_dim: int = 256
    alpha_attr: float = 0.0


@dataclass
class TrainConfig:
    qwen: ClassifierTrainConfig = field(default_factory=ClassifierTrainConfig)
    llama: ClassifierTrainConfig = field(default_factory=ClassifierTrainConfig)
    dispatcher: DispatcherTrainConfig = field(default_factory=DispatcherTrainConfig)
    num_workers: int = 4
    tf32: bool = True


@dataclass
class EvalConfig:
    profiles: List[str] = field(default_factory=lambda: ["ultra_fast", "balanced", "max_security"])
    latency_samples: int = 20
    latency_warmup: int = 3
    ece_bins: int = 15


@dataclass
class HubConfig:
    private: bool = True
    prompt_for_push: bool = True
    repo_id: Optional[str] = None


@dataclass
class Config:
    run_name: str = "test"
    mode: str = "test"
    seed: int = 42
    debug: bool = False
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @property
    def work_dir(self) -> Path:
        return Path(self.paths.work_dir)

    @property
    def cache_dir(self) -> Path:
        return Path(self.paths.cache_dir)


def _build(cls, values: Dict[str, Any]):
    """Recursively instantiate dataclass `cls` from a (possibly partial) dict, rejecting unknown keys."""
    if values is None:
        return cls()
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in values:
            continue
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), values[name])
        else:
            kwargs[name] = values[name]
    return cls(**kwargs)


def _apply_override(raw: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must look like section.key=value, got: {override!r}")
    key, value = override.split("=", 1)
    node = raw
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(value)


def load_config(path: str, overrides: Optional[List[str]] = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for ov in overrides or []:
        _apply_override(raw, ov)
    cfg = _build(Config, raw)
    if cfg.mode not in ("test", "full"):
        raise ValueError(f"mode must be 'test' or 'full', got {cfg.mode!r}")
    return cfg
