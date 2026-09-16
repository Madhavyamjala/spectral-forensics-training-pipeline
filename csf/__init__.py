"""
Chrono-Spectral Forensics (CSF) package.

Tri-class (Real / AI-Generated / AI-Edited) video attribution pipeline built on the
Chrono-TriClass-100k dataset, following the "Agentic Forensic Systems for AI-Generated
Video Detection" proposal:

    Phase 1  Qwen2.5-VL-3B semantic scanner            -> csf.models.classifier (kind="qwen")
    Phase 2  GRPO budget-constrained dispatcher         -> csf.models.dispatcher
    Phase 3  Spatial / Spectral / Latent toolpool       -> csf.tools
    Phase 4  Evidence graph + Llama-3.2-Vision arbiter  -> csf.graph, csf.models.classifier (kind="llama")

Entry point: main.py at the repository root.
"""

__version__ = "1.0.0"

LABELS = ["real", "ai_generated", "ai_edited"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
PRETTY_LABELS = {"real": "Real", "ai_generated": "AI-Generated", "ai_edited": "AI-Edited"}
