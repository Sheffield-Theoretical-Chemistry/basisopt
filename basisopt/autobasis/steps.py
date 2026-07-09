"""Step registry for the auto-basis pipeline.

Each step is registered by name and has the uniform signature
``run(state, step_cfg) -> StepResult``. The concrete chemistry wrappers live in
``chemistry.py`` and register themselves on import; keeping the registry separate
lets the driver be tested with lightweight dummy steps.
"""

from __future__ import annotations

from typing import Callable, Optional

from .state import RunState, StepResult

StepFn = Callable[[RunState, dict], StepResult]

STEP_REGISTRY: dict[str, StepFn] = {}


def register_step(name: str) -> Callable[[StepFn], StepFn]:
    """Decorator registering a step function under ``name``."""

    def decorator(fn: StepFn) -> StepFn:
        STEP_REGISTRY[name] = fn
        return fn

    return decorator


def get_step(name: str, registry: Optional[dict[str, StepFn]] = None) -> StepFn:
    """Look up a step function, from an override registry if given."""
    reg = STEP_REGISTRY if registry is None else registry
    if name not in reg:
        raise KeyError(f"No step registered under '{name}'. Registered: {sorted(reg)}")
    return reg[name]
