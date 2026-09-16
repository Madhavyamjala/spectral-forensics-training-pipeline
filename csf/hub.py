"""
Export trained artefacts and push them (with evaluation metrics) to a Hugging Face model repo.

`export_bundle` assembles <work_dir>/export/:
    README.md (model card with usage guidelines + headline metrics), csf_config.json,
    feature_stats.json, tool_costs.json, qwen_scanner/, llama_arbiter/, dispatchers/*.pt,
    metrics/ (ablation json / csv / md / plots, training summaries, latency benchmark),
    code/ (the `csf` package, main.py, requirements.txt) so the repo is self-contained.
Base-model weights are NOT re-uploaded: adapters load on top of the original Qwen / Llama repos,
which also keeps the Llama 3.2 Community License gating intact.

`push_interactive` asks for the Hugging Face token (hidden input) and the target repo id, validates
the token with `whoami`, creates the repo and uploads the export folder. Non-interactive sessions
(no TTY) skip the prompt and print the exact command to push later.

Input : Config, paths of the best checkpoints / dispatchers / metrics.
Output: export folder; URL of the pushed repo (or None).
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from csf.config import Config
from csf.logging_utils import get_logger

log = get_logger("hub")
ROOT = Path(__file__).resolve().parent.parent


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _metric_rows(metrics: Dict[str, Any]) -> str:
    rows = ["| Model | Accuracy | Macro-F1 | ROC-AUC | Miss rate | p50 latency (ms) |", "|---|---|---|---|---|---|"]
    for name, m in metrics.get("models", {}).items():
        if not name.startswith(("A_", "B_", "C_")):
            continue
        auc = m.get("roc_auc_ovr_macro")
        rows.append(f"| {name} | {100 * m['accuracy']:.2f}% | {100 * m['macro_f1']:.2f}% | "
                    f"{auc:.3f} | {100 * m['binary_fake_detection']['miss_rate']:.2f}% | {m['latency_ms_p50']:.1f} |"
                    if auc is not None else
                    f"| {name} | {100 * m['accuracy']:.2f}% | {100 * m['macro_f1']:.2f}% | - | "
                    f"{100 * m['binary_fake_detection']['miss_rate']:.2f}% | {m['latency_ms_p50']:.1f} |")
    return "\n".join(rows)


def model_card(cfg: Config, metrics: Dict[str, Any], repo_id: str) -> str:
    return f"""---
license: other
license_name: llama3.2-and-qwen-research
tags: [video-forensics, deepfake-detection, ai-generated-video, qwen2.5-vl, llama-3.2-vision, lora, grpo]
datasets: [{cfg.data.repo_id}]
base_model: [{cfg.models.qwen_id}, {cfg.models.llama_id}]
pipeline_tag: video-classification
---

# Chrono-Spectral Forensics (CSF) - tri-class video attribution

Classifies a video as **Real**, **AI-Generated** or **AI-Edited** with the agentic pipeline from
*Agentic Forensic Systems for AI-Generated Video Detection*:

1. **Phase 1 - scanner**: `{cfg.models.qwen_id}` + LoRA reads the frames as video.
2. **Phase 2 - dispatcher**: a GRPO-trained policy on the `{cfg.models.llama_id}` state chooses which forensic
   tools to run (or exits early) under a compute budget. One dispatcher per profile: `ultra_fast`, `balanced`, `max_security`.
3. **Phase 3 - toolpool**: spatial (saturation, lighting, edges, optical flow), spectral (FFT / 3D-DCT / phase
   correlation / noise residual) and latent (DIRE with a video-diffusion VAE) tools on sparse 64x64 patches.
4. **Phase 4 - arbiter**: the same Llama-3.2-Vision backbone (+ LoRA) reads a frame mosaic plus the evidence
   graph and outputs calibrated probabilities, reusing the dispatcher's vision states.

Trained on `{cfg.data.repo_id}` (run `{cfg.run_name}`, mode `{cfg.mode}`, {cfg.data.num_frames} frames/video).

## Results (held-out test split)

{_metric_rows(metrics)}

The full metric set (per-class P/R/F1, confusion matrices, ROC/PR-AUC, calibration, per-edit-method accuracy,
latency percentiles, routing statistics, shortcut baselines) is in `metrics/ablation_metrics.json`,
`metrics/ablation_report.md` and `metrics/plots/`.

## Usage

```bash
git clone https://huggingface.co/{repo_id} csf-model && cd csf-model/code
pip install -r requirements.txt        # plus a CUDA build of torch, see README of the training repo
huggingface-cli login                  # Llama 3.2 Vision is gated: accept its license first
python -m csf.inference /path/to/video.mp4 --model_dir .. --mode agentic --profile balanced
```

```python
import sys; sys.path.insert(0, "csf-model/code")
from csf.inference import CSFDetector
det = CSFDetector("csf-model")                # or the Hub id "{repo_id}"
print(det.predict("video.mp4", mode="agentic", profile="balanced"))
# modes: "scanner" (Qwen only, fastest) | "static" (all tools + Llama) | "agentic" (routed CSF)
```

Low-VRAM GPUs (12-16 GB): pass `quantization="4bit"` (default) and, if needed, `components=("qwen",)`
or `components=("llama", "vae")` to load only what a mode needs.

## Guidelines and limitations

- The toolpool is computed on native-resolution frames; do not pre-resize videos before inference.
- Profiles trade quality for latency: `ultra_fast` (early exits, no latent tool) for streaming triage,
  `balanced` for general use, `max_security` for audits where missing a fake is costly.
- Check the `baseline_metadata_shortcut` row in the metrics: it shows how much of the dataset can be solved from
  container metadata alone. Scores on other video sources may be lower than the numbers above.
- The evidence graph encodes hypothesised mechanisms, not established causal relations.
- Outputs are probabilistic forensic evidence, not proof; keep a human in the loop for consequential decisions.
- Base-model licences apply (Llama 3.2 Community License, Qwen licence).
"""


def export_bundle(cfg: Config, qwen_ckpt: Path, llama_ckpt: Path, dispatcher_dir: Path) -> Path:
    out = cfg.work_dir / "export"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _copy(qwen_ckpt, out / "qwen_scanner")
    _copy(llama_ckpt, out / "llama_arbiter")
    _copy(dispatcher_dir, out / "dispatchers")
    for name in ("feature_stats.json", "tool_costs.json"):
        _copy(cfg.cache_dir / name, out / name)
    d = cfg.data
    csf_cfg = {"num_frames": d.num_frames, "frame_size": d.frame_size, "mosaic_frames": d.mosaic_frames,
               "mosaic_size": d.mosaic_size, "patch_size": d.patch_size, "num_patches": d.num_patches,
               "tool_max_side": d.tool_max_side, "qwen_id": cfg.models.qwen_id, "llama_id": cfg.models.llama_id,
               "vae_id": cfg.models.vae_id, "vae_subfolder": cfg.models.vae_subfolder,
               "vae_fallback_id": cfg.models.vae_fallback_id, "dataset": d.repo_id, "run_name": cfg.run_name}
    (out / "csf_config.json").write_text(json.dumps(csf_cfg, indent=2), encoding="utf-8")
    _copy(cfg.work_dir / "metrics", out / "metrics")
    _copy(cfg.work_dir / "resolved_config.json", out / "metrics" / "resolved_config.json")
    _copy(ROOT / "csf", out / "code" / "csf")
    for f in ("main.py", "requirements.txt", "README.md"):
        _copy(ROOT / f, out / "code" / f)
    metrics_file = cfg.work_dir / "metrics" / "ablation_metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    (out / "README.md").write_text(model_card(cfg, metrics, cfg.hub.repo_id or "<your-namespace>/<repo>"),
                                   encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log.info("Export bundle ready: %s (%.1f MB)", out, size / 2**20)
    return out


def push_folder(folder: Path, repo_id: str, token: str, private: bool, cfg: Config) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    user = api.whoami()
    log.info("Authenticated to Hugging Face as %s", user.get("name"))
    url = api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    metrics_file = cfg.work_dir / "metrics" / "ablation_metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    (folder / "README.md").write_text(model_card(cfg, metrics, repo_id), encoding="utf-8")
    log.info("Uploading %s -> %s (private=%s)...", folder, repo_id, private)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                      commit_message=f"CSF {cfg.run_name}: adapters, dispatchers, metrics")
    log.info("Push complete: %s", url)
    return str(url)


def push_interactive(cfg: Config, folder: Path) -> Optional[str]:
    if not cfg.hub.prompt_for_push:
        log.info("hub.prompt_for_push=false -> not pushing. Export is at %s", folder)
        return None
    if not sys.stdin or not sys.stdin.isatty():
        log.warning("No interactive terminal: skipping the Hub push prompt. Push later with:\n"
                    "  python main.py --config <same config> --stage push")
        return None
    print("\n" + "=" * 78)
    print(f"Training finished. Weights + metrics are exported at:\n  {folder}")
    print("Push them to the Hugging Face Hub now? A WRITE token is needed "
          "(https://huggingface.co/settings/tokens).")
    print("=" * 78)
    for attempt in range(3):
        repo_id = (input(f"Target repo id [namespace/name]{f' (default {cfg.hub.repo_id})' if cfg.hub.repo_id else ''} "
                         "(leave empty to skip): ").strip() or (cfg.hub.repo_id or ""))
        if not repo_id:
            log.info("Push skipped by user. Push later with: python main.py --config <cfg> --stage push")
            return None
        if repo_id.count("/") != 1:
            print("Repo id must look like 'namespace/name'.")
            continue
        token = os.environ.get("HF_WRITE_TOKEN") or getpass.getpass("Hugging Face WRITE token (input hidden): ").strip()
        private = input(f"Private repo? [Y/n] (default {'Y' if cfg.hub.private else 'n'}): ").strip().lower()
        is_private = cfg.hub.private if private == "" else private.startswith("y")
        try:
            cfg.hub.repo_id = repo_id
            return push_folder(folder, repo_id, token, is_private, cfg)
        except Exception as exc:
            log.error("Push attempt %d failed: %s: %s", attempt + 1, type(exc).__name__, exc)
            print("Push failed (see log). Check the token has WRITE scope and the namespace is yours.")
    log.error("Giving up after 3 attempts. Push later with: python main.py --config <cfg> --stage push")
    return None
