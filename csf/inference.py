"""
Standalone inference for trained / published CSF models (raw video file -> verdict).

`CSFDetector(model_dir)` loads the exported artefacts (the folder pushed to the Hub):
    csf_config.json, feature_stats.json, tool_costs.json,
    qwen_scanner/ , llama_arbiter/ (LoRA adapter + head), dispatchers/<profile>.pt
and exposes
    predict(video_path, mode="agentic", profile="balanced") -> dict
        mode "scanner" : Qwen2.5-VL scanner only                         (model type A)
        mode "static"  : all tools + Llama-3.2-Vision arbiter             (model type B)
        mode "agentic" : scanner -> GRPO dispatcher -> chosen tools -> arbiter reusing the
                         dispatcher's vision states                        (model type C)
    The result holds label, class probabilities, routing action, tools executed, a per-component
    latency breakdown (ms) and the evidence graph (node-link JSON) for explainability.

`components` lets low-VRAM machines load only what a mode needs (e.g. ("qwen",) then ("llama", "vae")).

Input : model directory (local path or Hub repo id), torch device, optional quantization override.
Output: python dict per video (see above).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from csf import LABELS, PRETTY_LABELS
from csf.data.datasets import QwenCollator
from csf.data.video_io import decode_frames, to_vlm_frames
from csf.graph import build_evidence_graph, normalize_features
from csf.logging_utils import get_logger
from csf.models.classifier import compute_dtype_for, load_classifier
from csf.models.dispatcher import (ACTION_GROUPS, ACTIONS, PROFILES, DispatcherPolicy, action_mask_array,
                                   allowed_actions, dispatcher_scalars)
from csf.tools.toolpool import TOOL_GROUPS, LatentTool, run_toolpool

log = get_logger("inference")


def _resolve_dir(model_dir: str) -> Path:
    p = Path(model_dir)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_dir))


class _Cfg:
    """Minimal attribute view used by ArbiterRunner (cfg.data.mosaic_frames / mosaic_size)."""

    def __init__(self, d: Dict[str, Any]):
        self.data = type("D", (), d)()


class CSFDetector:
    def __init__(self, model_dir: str, device: Optional[str] = None, quantization: Optional[str] = None,
                 components: Iterable[str] = ("qwen", "llama", "vae"), attn_implementation: str = "sdpa"):
        self.dir = _resolve_dir(model_dir)
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")
        self.dtype = compute_dtype_for(self.device)
        self.cfg = json.loads((self.dir / "csf_config.json").read_text(encoding="utf-8"))
        self.stats = json.loads((self.dir / "feature_stats.json").read_text(encoding="utf-8"))
        self.active_ids = list(self.cfg.get("active_label_ids", range(len(LABELS))))
        self.active_classes = list(self.cfg.get("classes", LABELS))
        components = set(components)
        self.qwen = self.qwen_proc = self.llama = self.runner = self.vae = None
        if "qwen" in components:
            self.qwen, self.qwen_proc, _ = load_classifier(self.dir / "qwen_scanner", self.device, quantization,
                                                           attn_implementation=attn_implementation)
            self.qwen_collate = QwenCollator(
                self.qwen_proc,
                [PRETTY_LABELS[LABELS[i]] for i in self.active_ids],
            )
        if "llama" in components:
            from csf.pipeline import ArbiterRunner
            self.llama, proc, _ = load_classifier(self.dir / "llama_arbiter", self.device, quantization,
                                                  attn_implementation=attn_implementation)
            self.runner = ArbiterRunner(self.llama, proc, _Cfg({"mosaic_frames": self.cfg["mosaic_frames"],
                                                                "mosaic_size": self.cfg["mosaic_size"]}), self.device,
                                         active_ids=self.active_ids)
        if "vae" in components:
            self.vae = LatentTool(self.cfg["vae_id"], self.cfg.get("vae_subfolder"), self.cfg["vae_fallback_id"],
                                  self.device, torch.float16 if self.device.type == "cuda" else torch.float32)
        self.policies: Dict[str, DispatcherPolicy] = {}
        for p in (self.dir / "dispatchers").glob("*.pt"):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            c = ck["config"]
            pol = DispatcherPolicy(c["state_dim"], c["n_scalar"], c["hidden"], c["n_actions"])
            pol.load_state_dict(ck["state_dict"])
            self.policies[p.stem] = pol.to(self.device).eval()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def load_video(self, video_path: str):
        t = time.perf_counter()
        native = decode_frames(Path(video_path), self.cfg["num_frames"], self.cfg["tool_max_side"])
        vlm = to_vlm_frames(native, self.cfg["frame_size"])
        return native, vlm, time.perf_counter() - t

    def scan(self, vlm_frames: np.ndarray):
        batch = self.qwen_collate([{"frames": vlm_frames, "label": 0, "key": "video"}])
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        if "pixel_values_videos" in batch:
            batch["pixel_values_videos"] = batch["pixel_values_videos"].to(self.dtype)
        self._sync()
        t = time.perf_counter()
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            logits, _ = self.qwen(**{k: v for k, v in batch.items() if k not in ("labels", "keys")})
        from csf.pipeline import _active_probs_from_logits
        probs = _active_probs_from_logits(logits, self.active_ids)[0].cpu().numpy()
        self._sync()
        return probs, time.perf_counter() - t

    def tools(self, native, groups):
        res = run_toolpool(native, self.cfg["patch_size"], self.cfg["num_patches"], self.vae, groups)
        return res, sum(v for k, v in res.times.items())

    def predict(self, video_path: str, mode: str = "agentic", profile: str = "balanced",
                scanner_probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        lat: Dict[str, float] = {}
        native, vlm, lat["decode"] = self.load_video(video_path)
        out: Dict[str, Any] = {"video": str(video_path), "mode": mode}

        if mode in ("scanner", "agentic") and scanner_probs is None:
            scanner_probs, lat["scanner"] = self.scan(vlm)
        if mode == "scanner":
            probs, action, groups, z, mask = scanner_probs, None, [], None, None
        elif mode == "static":
            res, lat["tools"] = self.tools(native, TOOL_GROUPS)
            z, mask = normalize_features(res.features, self.stats), res.mask
            p, _, lat["arbiter"], _ = self.runner.pixel_pass([{"frames": vlm, "z": z, "mask": mask, "label": 0,
                                                               "key": "video"}], mask)
            probs, action, groups = p[0], "full_tri_domain", list(TOOL_GROUPS)
        elif mode == "agentic":
            if profile not in self.policies:
                raise ValueError(f"No dispatcher for profile {profile!r}; available: {sorted(self.policies)}")
            empty = {"frames": vlm, "z": np.zeros(len(self.stats["mean"]), np.float32), "mask": np.zeros(3, bool),
                     "label": 0, "key": "video"}
            p0, state, lat["dispatcher_state"], cache = self.runner.pixel_pass([empty], np.zeros(3, bool))
            scalars = dispatcher_scalars(scanner_probs[None], p0)
            with torch.no_grad():
                logits = self.policies[profile](torch.tensor(state.astype(np.float32), device=self.device),
                                                torch.tensor(scalars, device=self.device),
                                                allowed_actions(PROFILES[profile]))
            action = ACTIONS[int(logits.argmax(-1)[0])]
            groups = ACTION_GROUPS[action]
            if action == "early_exit":
                probs, z, mask = (scanner_probs + p0[0]) / 2.0, None, None
            else:
                res, lat["tools"] = self.tools(native, groups)
                z, mask = normalize_features(res.features, self.stats), action_mask_array(action) & res.mask
                p, lat["arbiter"] = self.runner.cached_pass([{"frames": vlm, "z": z, "mask": mask, "label": 0,
                                                              "key": "video"}], mask, cache)
                probs = p[0]
        else:
            raise ValueError(f"mode must be scanner / static / agentic, got {mode!r}")

        out.update(label=PRETTY_LABELS[LABELS[int(np.argmax(probs))]],
                   probs={PRETTY_LABELS[l]: float(v) for l, v in zip(LABELS, probs)},
                   action=action, tools_run=groups, profile=profile if mode == "agentic" else None,
                   latency_ms={k: round(v * 1000, 2) for k, v in lat.items()},
                   total_latency_ms=round(sum(lat.values()) * 1000, 2),
                   is_fake=bool(LABELS[int(np.argmax(probs))] != "real"))
        if z is not None:
            from networkx.readwrite import json_graph
            out["evidence_graph"] = json_graph.node_link_data(
                build_evidence_graph(z, mask,
                                     candidate_labels=[PRETTY_LABELS[x] for x in self.active_classes])
            )
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run Chrono-Spectral Forensics on a video file")
    ap.add_argument("video")
    ap.add_argument("--model_dir", required=True, help="exported folder or Hub repo id")
    ap.add_argument("--mode", default="agentic", choices=["agentic", "static", "scanner"])
    ap.add_argument("--profile", default="balanced", choices=list(PROFILES))
    ap.add_argument("--quantization", default=None, choices=[None, "4bit", "8bit", "none"])
    a = ap.parse_args()
    det = CSFDetector(a.model_dir, quantization=a.quantization)
    r = det.predict(a.video, a.mode, a.profile)
    r.pop("evidence_graph", None)
    print(json.dumps(r, indent=2))
