"""
Phase 4 - Spatio-Spectral Evidence Graph.

Builds the proposal's hypothesised-mechanism DAG
    video -> class hypothesis -> {global synthesis, local edit} mechanisms -> tool evidence
from the (z-normalised) toolpool features and serialises it to the compact text block the
Llama-3.2-Vision arbiter reads. Edges are labelled as hypotheses: a single forward pass
cannot establish causality, the graph only structures which evidence supports which mechanism.

Input : raw feature vector (csf.tools.ALL_FEATURES order), tool-group mask, feature stats.
Output: `normalize_features` -> z-scores (NaN -> 0 for tools that were not run);
        `build_evidence_graph` -> networkx.DiGraph (exported by inference for explainability);
        `evidence_text` -> prompt string for the arbiter.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from csf.tools.toolpool import FEATURE_NAMES, GROUP_SLICES, TOOL_GROUPS

GLOBAL_SYNTHESIS_EVIDENCE = ["fft_hf_global", "spectrum_slope", "dct3d_hf_global", "dire_global", "sat_mean"]
LOCAL_EDIT_EVIDENCE = ["phase_shift_spike", "patch_context_color_gap", "fft_hf_patch_std",
                       "dire_patch_global_ratio", "noise_residual_cv", "edge_lapvar_patch_ratio"]
TEMPORAL_EVIDENCE = ["flow_inconsistency", "lum_jump_std", "phase_corr_global"]


def normalize_features(raw: np.ndarray, stats: Dict[str, List[float]]) -> np.ndarray:
    z = (raw - np.asarray(stats["mean"], np.float32)) / np.asarray(stats["std"], np.float32)
    return np.clip(np.nan_to_num(z, nan=0.0), -8.0, 8.0).astype(np.float32)


def _level(z: float) -> str:
    if z > 1.5:
        return "very high"
    if z > 0.75:
        return "high"
    if z < -1.5:
        return "very low"
    if z < -0.75:
        return "low"
    return "typical"


def build_evidence_graph(z: np.ndarray, mask: np.ndarray, candidate_labels: Optional[List[str]] = None) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("video", kind="root")
    candidates = list(candidate_labels) if candidate_labels is not None else ["Real", "AI-Generated", "AI-Edited"]
    g.add_node("class_hypothesis", kind="latent_cause", candidates=candidates)
    g.add_edge("video", "class_hypothesis")
    mechanisms = {"global_synthesis": GLOBAL_SYNTHESIS_EVIDENCE, "local_edit": LOCAL_EDIT_EVIDENCE,
                  "temporal_incoherence": TEMPORAL_EVIDENCE}
    for mech in mechanisms:
        g.add_node(mech, kind="mechanism")
        g.add_edge("class_hypothesis", mech, relation="hypothesized_mechanism")
    for gi, group in enumerate(TOOL_GROUPS):
        g.add_node(f"tool:{group}", kind="tool", executed=bool(mask[gi]))
        if not mask[gi]:
            continue
        for j, name in enumerate(FEATURE_NAMES[group]):
            val = float(z[GROUP_SLICES[group]][j])
            g.add_node(name, kind="evidence", tool=group, z=round(val, 3), level=_level(val))
            g.add_edge(f"tool:{group}", name, relation="measured")
            for mech, names in mechanisms.items():
                if name in names:
                    g.add_edge(mech, name, relation="predicts")
    return g


def evidence_text(z: np.ndarray, mask: np.ndarray, scanner_note: Optional[str] = None,
                  candidate_labels: Optional[List[str]] = None) -> str:
    """Serialize measured tool evidence for the arbiter prompt.

    Note:
        The default wording remains tri-class for backwards compatibility; two-class runs pass
        their active class names explicitly.

    TODO:
        Store prompt-template version in the export manifest for experiment reproducibility.
    """
    candidates = list(candidate_labels) if candidate_labels else ["Real", "AI-Generated", "AI-Edited"]
    lines = ["Forensic evidence graph (z-scores vs. the training distribution):",
             "Candidate classes: " + ", ".join(candidates)]
    for gi, group in enumerate(TOOL_GROUPS):
        if not mask[gi]:
            lines.append(f"[{group}] not executed")
            continue
        vals = z[GROUP_SLICES[group]]
        items = [f"{n}={v:+.2f}({_level(float(v))})" for n, v in zip(FEATURE_NAMES[group], vals)]
        lines.append(f"[{group}] " + ", ".join(items))
    executed = [n for gi, grp in enumerate(TOOL_GROUPS) if mask[gi] for n in FEATURE_NAMES[grp]]
    for mech, names in (("global_synthesis", GLOBAL_SYNTHESIS_EVIDENCE), ("local_edit", LOCAL_EDIT_EVIDENCE),
                        ("temporal_incoherence", TEMPORAL_EVIDENCE)):
        support = [n for n in names if n in executed]
        if support:
            lines.append(f"mechanism {mech} <- " + ", ".join(support))
    if scanner_note:
        lines.append(scanner_note)
    if "AI-Edited" in candidates:
        lines.append("Rule of thumb: low DIRE = fits a generative latent manifold (AI-Generated); "
                     "authentic global statistics with localized phase/colour spikes = AI-Edited.")
    else:
        lines.append("Rule of thumb: lower DIRE can support a generative explanation; "
                     "interpret the evidence jointly with the class hypotheses above.")
    return "\n".join(lines)
