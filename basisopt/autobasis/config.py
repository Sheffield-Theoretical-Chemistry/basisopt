"""Configuration model for the auto-basis pipeline.

A run is described by a YAML file that optionally ``extends`` a bundled tier
preset (min/fast/mid/accu). The preset supplies the per-step target policy and
method defaults; the user config supplies the run-specifics (element, CBS limit,
target accuracy, geometry, workdir, ...). The two are deep-merged, user winning.

See docs (AUTOBASIS_PIPELINE) for the schema. This module only loads, merges and
validates config; it runs nothing.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Canonical step order. A config may run any subset, in this order.
CANONICAL_STEPS = (
    "primitives",
    "reduction",
    "contraction",
    "uncontraction",
    "purification",
    "pruning",
)

# Presets (the min/fast/mid/accu tier definitions) are NOT shipped with basisopt
# - they are user-owned config. Point at a directory of them with the
# BASISOPT_AUTOBASIS_PRESETS env var to use bare-name `extends: fast`; otherwise
# `extends:` takes a path to a preset YAML (relative to the config file).
PRESETS_DIR = os.environ.get("BASISOPT_AUTOBASIS_PRESETS")


class ConfigError(ValueError):
    """Raised when a pipeline config is malformed."""


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (override wins).

    Nested dicts are merged key-by-key; any non-dict value (including lists)
    replaces the base value wholesale.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class ReferenceConfig:
    """Per-run reference data (user-supplied)."""

    cbs_limit: Optional[float] = None
    target: Optional[float] = None
    geometry: Optional[str] = None
    multiplicity: Optional[int] = None
    charge: int = 0
    wf_cards: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendConfig:
    """Default backend + scratch directory (steps may override the backend)."""

    default: str = "psi4"
    tmp_dir: str = "./tmp"


@dataclass
class PipelineConfig:
    """A fully-resolved pipeline configuration for one element.

    ``step_configs`` holds the merged per-step config dicts (each step
    interprets its own keys); ``defaults`` holds shared defaults such as
    ``method`` params keyed by backend. ``raw`` is the whole merged mapping,
    persisted for provenance.
    """

    name: str
    workdir: str
    element: str
    steps: list[str]
    reference: ReferenceConfig
    backend: BackendConfig
    step_configs: dict[str, dict]
    defaults: dict[str, Any]
    raw: dict[str, Any]
    tier: Optional[str] = None

    def step_config(self, name: str) -> dict:
        """Return the merged config dict for a step (empty if none given)."""
        return self.step_configs.get(name, {})

    def step_backend(self, name: str) -> str:
        """Backend for a step: its own ``backend`` key, else the global default."""
        return self.step_config(name).get("backend") or self.backend.default

    def method_params(self, backend: str, name: Optional[str] = None) -> dict:
        """Method params for a backend: ``defaults.method[backend]`` merged with
        a step's own ``params`` (step wins)."""
        base = copy.deepcopy(self.defaults.get("method", {}).get(backend, {}))
        if name is not None:
            base = deep_merge(base, self.step_config(name).get("params", {}))
        return base


def _load_yaml(path: Path) -> dict:
    with open(path) as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping, got {type(data).__name__}")
    return data


def _presets_dir() -> Optional[Path]:
    """The user's presets directory (from ``BASISOPT_AUTOBASIS_PRESETS``), or None."""
    return Path(PRESETS_DIR) if PRESETS_DIR else None


def list_presets() -> list[str]:
    """Names of presets in the configured presets directory (empty if none)."""
    directory = _presets_dir()
    return [] if directory is None else sorted(p.stem for p in directory.glob("*.yaml"))


def load_preset(name: str) -> dict:
    """Load a tier preset by bare name from the configured presets directory."""
    directory = _presets_dir()
    if directory is None:
        raise ConfigError(
            f"Cannot resolve preset '{name}': no presets directory configured. Set the "
            "BASISOPT_AUTOBASIS_PRESETS env var, or give a path in 'extends:'."
        )
    path = directory / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(list_presets()) or "(none)"
        raise ConfigError(f"Unknown preset '{name}' in {directory}. Available: {available}")
    return _load_yaml(path)


def _resolve_extends(value: str, config_path: Path) -> dict:
    """Resolve an ``extends`` value: a path (relative to the config file, or
    absolute) if it points at a file, otherwise a bare preset name."""
    for candidate in (config_path.parent / value, Path(value)):
        if candidate.suffix and candidate.exists():
            return _load_yaml(candidate)
    return load_preset(value)


def _resolve_raw(path: str | Path) -> dict:
    """Load a user config and merge its ``extends`` preset underneath it."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    user = _load_yaml(path)
    extends = user.get("extends")
    if extends is None:
        return user
    merged = deep_merge(_resolve_extends(extends, path), user)
    merged.pop("extends", None)
    return merged


def parse_config(raw: dict) -> PipelineConfig:
    """Validate and structure a already-merged raw config mapping."""
    for required in ("name", "workdir", "element", "steps"):
        if required not in raw:
            raise ConfigError(f"Missing required config key: '{required}'")

    steps = raw["steps"]
    if not isinstance(steps, list) or not steps:
        raise ConfigError("'steps' must be a non-empty list")
    unknown = [s for s in steps if s not in CANONICAL_STEPS]
    if unknown:
        raise ConfigError(
            f"Unknown step(s): {unknown}. Valid steps: {', '.join(CANONICAL_STEPS)}"
        )

    ref_raw = raw.get("reference", {}) or {}
    known_ref = {f for f in ReferenceConfig.__dataclass_fields__}
    reference = ReferenceConfig(**{k: v for k, v in ref_raw.items() if k in known_ref})

    backend_raw = raw.get("backend", {}) or {}
    known_backend = {f for f in BackendConfig.__dataclass_fields__}
    backend = BackendConfig(**{k: v for k, v in backend_raw.items() if k in known_backend})

    step_configs = {name: (raw.get(name) or {}) for name in CANONICAL_STEPS if name in raw}

    return PipelineConfig(
        name=raw["name"],
        workdir=raw["workdir"],
        element=raw["element"],
        steps=list(steps),
        reference=reference,
        backend=backend,
        step_configs=step_configs,
        defaults=raw.get("defaults", {}) or {},
        raw=raw,
        tier=raw.get("tier"),
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Load, merge (``extends`` preset) and validate a pipeline config file."""
    return parse_config(_resolve_raw(path))
