"""
SAFER: cost-aware forensic evidence acquisition for AI-generated video detection.

The package keeps the project's working name `csf` (Chrono-Spectral Forensics). The paper's setting is binary
(Real vs AI-Generated, Chrono-66k: the two-class part of the Chrono-TriClass-100k manifest, selected with
`data.classes`); the code also supports the full three-class Real / AI-Generated / AI-Edited setting.

    Frontline   Qwen2.5-VL-3B scanner + Llama-3.2-Vision no-tool pass  -> csf.models.classifier
    Dispatcher  GRPO cost-aware router (exit without tools, or pay for tools) -> csf.models.dispatcher
    Toolpool    spatial / spectral / latent forensic features             -> csf.tools
    Arbiter     evidence graph + the same Llama-3.2-Vision weights       -> csf.graph, csf.models.classifier

The no-tool arbiter pass is the same-reasoner baseline against which the value of every tool subset is measured.

Entry point: main.py at the repository root (full_2class.py for the paper's configuration).
"""

__version__ = "1.0.0"

LABELS = ["real", "ai_generated", "ai_edited"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
PRETTY_LABELS = {"real": "Real", "ai_generated": "AI-Generated", "ai_edited": "AI-Edited"}
