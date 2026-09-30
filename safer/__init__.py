"""
SAFER: cost-aware forensic evidence acquisition for AI-generated video detection.

This package was called `csf` (Chrono-Spectral Forensics, the project's working name); `import csf...` still works
through a compatibility alias that maps every `csf.X` onto the same `safer.X` module. The paper's setting is binary
(Real vs AI-Generated, Chrono-66k: the two-class part of the Chrono-TriClass-100k manifest, selected with
`data.classes`); the code also supports the full three-class Real / AI-Generated / AI-Edited setting.

    Frontline   Qwen2.5-VL-3B scanner + Llama-3.2-Vision no-tool pass  -> safer.models.classifier
    Dispatcher  GRPO cost-aware router (exit without tools, or pay for tools) -> safer.models.dispatcher
    Toolpool    spatial / spectral / latent forensic features             -> safer.tools
    Arbiter     evidence graph + the same Llama-3.2-Vision weights       -> safer.graph, safer.models.classifier

The no-tool arbiter pass is the same-reasoner baseline against which the value of every tool subset is measured.

Entry point: main.py at the repository root (full_2class.py for the paper's configuration).
"""

__version__ = "1.0.0"

LABELS = ["real", "ai_generated", "ai_edited"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
PRETTY_LABELS = {"real": "Real", "ai_generated": "AI-Generated", "ai_edited": "AI-Edited"}
