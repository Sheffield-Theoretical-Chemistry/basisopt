"""Auto-basis pipeline: YAML-driven, resumable, step-selectable generation of
basis sets (steps 1-6: primitives -> reduction -> contraction -> uncontraction
-> purification -> pruning). Polarisation (step 7) is handled separately.

Typical use:

    from basisopt.autobasis import run_pipeline
    run_pipeline("my-run.yaml")

or on the command line:

    python -m basisopt.autobasis run my-run.yaml
"""

from .config import CANONICAL_STEPS, ConfigError, PipelineConfig, load_config
from .manifest import Manifest
from .pipeline import load_basis, run_config, run_pipeline, save_basis
from .state import RunState, StepResult
from .steps import STEP_REGISTRY, get_step, register_step

# Registers the concrete steps (1-6) in STEP_REGISTRY on import.
from . import chemistry  # noqa: E402,F401  (import for side effect; must follow the above)

__all__ = [
    "CANONICAL_STEPS",
    "ConfigError",
    "PipelineConfig",
    "load_config",
    "Manifest",
    "RunState",
    "StepResult",
    "STEP_REGISTRY",
    "get_step",
    "register_step",
    "run_pipeline",
    "run_config",
    "load_basis",
    "save_basis",
]
