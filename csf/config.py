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
    classes: Optional[List[str]] = None   # train on a subset of LABELS; None = all three
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

    def active_label_ids(self) -> List[int]:
        """Label ids this run actually trains on. `classes` narrows the manifest to a subset - used
        while a class is unusable (being regenerated, or mislabelled) so the run is not blocked by it."""
        from csf import LABEL2ID
        if not self.classes:
            return list(range(len(LABEL2ID)))
        unknown = [c for c in self.classes if c not in LABEL2ID]
        if unknown:
            raise ValueError(f"data.classes contains unknown label(s) {unknown}; valid: {list(LABEL2ID)}")
        return sorted(LABEL2ID[c] for c in self.classes)


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
class KineticsConfig:
    """Where Kinetics-400 source clips come from, and how many to keep.

    The Hugging Face mirror is the default: it needs no annotation CSVs, reuses the Hub client's
    auth/retry/caching, and the loader detects its layout at runtime. The CVDF S3 shards remain
    as a fallback for labels the mirror cannot satisfy.
    """
    local_root: Optional[str] = None            # an already-extracted tree on the cluster
    hf_repo: Optional[str] = "liuhuanjim013/kinetics400"
    hf_revision: Optional[str] = None
    mirror_base: Optional[str] = "https://s3.amazonaws.com/kinetics/400"   # fallback
    splits: List[str] = field(default_factory=lambda: ["train", "val"])
    max_shards: Optional[int] = None            # cap the shard downloads (smoke runs)
    probe_clips: bool = True
    demand_margin: float = 1.6                  # oversample: clips are lost to the filters
    score_workers: int = 8
    score_frames: int = 12
    rescore: bool = False


@dataclass
class GenerationPushConfig:
    """Publishing the regenerated class back to the dataset repo (destructive - opt in)."""
    enabled: bool = False
    repo_id: Optional[str] = None               # defaults to data.repo_id
    delete_old: bool = True
    private: bool = True
    dry_run: bool = False


@dataclass
class GenerationConfig:
    enabled: bool = False
    total_videos: int = 33333
    gpus: List[int] = field(default_factory=lambda: [2, 3, 4])
    #: Which GPU the driver process itself binds. These are physical ids, the same numbering
    #: nvidia-smi uses. null => the first entry of `gpus`, so the driver never lands on a card
    #: the run was told to avoid.
    driver_gpu: Optional[int] = None
    video_root: str = "./cache/regen/videos"
    envs_root: str = "./cache/regen/envs"
    jobs_csv: str = "./cache/regen/jobs.csv"
    ledger: str = "./cache/regen/ledger.jsonl"
    manifest_out: str = "manifest_regen.csv"
    #: Per-video metadata for artifact-attribution analysis, written alongside the manifest.
    #: Empty => metadata.csv next to manifest_out.
    metadata_out: str = ""
    job_timeout_s: int = 1800
    fail_fast: int = 8
    min_free_gb: float = 50.0
    offline: bool = False                       # fail instead of building envs on the fly
    deadline_hours: Optional[float] = None      # stop scheduling new groups after this long
    # Retry jobs that failed on an earlier run. On by default: nearly every failure seen in
    # practice is environmental (an adapter env still building, a checkpoint not yet staged,
    # a busy GPU), and treating the first failure as final leaves the run permanently stuck
    # with nothing pending and nothing produced. `max_attempts` caps the retries per job.
    retry_failed: bool = True
    max_attempts: int = 3
    # Re-plan jobs.csv from the current source pool instead of reusing the existing plan.
    # Needed after the pool changes (new labels, more clips): the old plan still points at the
    # clips that existed when it was written. Job ids are derived from the assignment, so
    # re-planning starts the ledger fresh for anything that moved.
    rebuild_jobs: bool = False
    #: Where to find checkpoints that cannot be downloaded unattended (Google Drive, Tsinghua
    #: Cloud, OneDrive). Download them once by hand, drop them in this folder under the exact
    #: filename the adapter expects, and the env build copies them into place.
    staged_weights_dir: str = "./model_paths"
    #: Delete every built per-model environment before generating, so they are recreated from
    #: scratch. Destructive and expensive - it re-downloads torch, every requirement and every
    #: checkpoint - so it is meant as `--set generation.burn_envs=true` for a one-off clean
    #: rebuild, not as a standing setting in a config file.
    burn_envs: bool = False
    keep_old_edited: bool = False
    only_models: List[str] = field(default_factory=list)      # restrict the run to these models
    skip_models: List[str] = field(default_factory=list)
    # Choose the model set up front to fit a wall-clock budget, instead of letting
    # deadline_hours cut the run off mid-group (which biases the dataset toward whichever
    # models sort earliest). null disables the planner and runs everything wired.
    budget_wall_clock_hours: Optional[float] = None
    budget_diversity_floor: int = 2                           # distinct renderers per family
    budget_pin_models: List[str] = field(default_factory=list)
    # Six of the document's models have no runnable release. With this set, each family's
    # target is re-apportioned across the models that DO run, so every family still reaches the
    # size the document specifies instead of coming up short.
    reallocate_unfillable: bool = True
    # Per-GPU concurrency. One worker per card leaves a 143 GB H200 almost idle on a 3 GB model,
    # and the small nets are latency-bound (video decode, face detection) rather than
    # compute-bound, so several in parallel scale nearly linearly.
    gpu_vram_gb: float = 143.0
    max_workers_per_gpu: int = 4
    # REFace's checkpoint is trained on CelebAMask-HQ: non-commercial research only. Its
    # adapter refuses to start unless this is set explicitly.
    accept_noncommercial: bool = False
    kinetics: KineticsConfig = field(default_factory=KineticsConfig)
    push: GenerationPushConfig = field(default_factory=GenerationPushConfig)


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
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete configuration to a dictionary."""
        return dataclasses.asdict(self)

    def save(self, path: Path) -> None:
        """Write the resolved configuration as formatted JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @property
    def work_dir(self) -> Path:
        """Return the configured working directory as a path."""
        return Path(self.paths.work_dir)

    @property
    def cache_dir(self) -> Path:
        """Return the configured cache directory as a path."""
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
    """Apply one dotted command-line override to a raw configuration."""
    if "=" not in override:
        raise ValueError(f"Override must look like section.key=value, got: {override!r}")
    key, value = override.split("=", 1)
    node = raw
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(value)


def load_config(path: str, overrides: Optional[List[str]] = None) -> Config:
    """Load, override, validate, and instantiate a pipeline configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for ov in overrides or []:
        _apply_override(raw, ov)
    cfg = _build(Config, raw)
    if cfg.mode not in ("test", "full"):
        raise ValueError(f"mode must be 'test' or 'full', got {cfg.mode!r}")
    return cfg
