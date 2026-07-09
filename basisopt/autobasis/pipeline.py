"""The auto-basis pipeline driver.

Resolves a config, then for each requested step: skip if already done, resolve
and load the step's input basis (``auto`` -> most recent completed prior step
from the manifest, or an explicit path), run the step, and persist its output
(canonical ``basis.json`` + any exports + ``record.json``) while updating the
manifest. Backend/scratch setup is the step wrappers' responsibility, so this
module never imports a backend and is testable with dummy steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml
from monty.json import MontyDecoder, MontyEncoder

from .config import CANONICAL_STEPS, PipelineConfig, load_config
from .manifest import Manifest
from .state import RunState
from .steps import StepFn, get_step


def _step_dirname(name: str) -> str:
    """Stable, canonically-numbered folder name, e.g. '04_uncontraction'."""
    return f"{CANONICAL_STEPS.index(name) + 1:02d}_{name}"


def save_basis(basis, path: str | Path) -> None:
    """Write an internal basis to the canonical JSON artifact (MSONable Shells
    + numpy arrays via MontyEncoder)."""
    with open(path, "w") as handle:
        json.dump(basis, handle, cls=MontyEncoder)


def load_basis(path: str | Path, fmt: Optional[str] = None):
    """Load an internal basis. ``fmt`` defaults to 'json' for .json files and
    'molpro' otherwise (external NAO/contracted bases are Molpro-formatted)."""
    path = Path(path)
    if fmt is None:
        fmt = "json" if path.suffix == ".json" else "molpro"
    if fmt == "json":
        with open(path) as handle:
            return json.load(handle, cls=MontyDecoder)
    # delegate other formats to the BSE bridge (lazy import: keeps the json path
    # and the dummy-step tests free of basis_set_exchange)
    from basis_set_exchange.readers import read_formatted_basis_str

    from basisopt.bse_wrapper import bse_to_internal

    text = path.read_text().replace(";", "")
    return bse_to_internal(read_formatted_basis_str(text, fmt))


def _resolve_input(cfg: PipelineConfig, name: str, manifest: Manifest):
    """Resolve and load a step's input basis (or None if it generates its own)."""
    spec = cfg.step_config(name).get("input", "auto")
    if spec not in ("auto", None):
        return load_basis(spec, cfg.step_config(name).get("input_format"))
    prior = manifest.latest_completed_before(name, CANONICAL_STEPS)
    if prior is None:
        return None
    return load_basis(manifest.basis_path(prior))


def _write_resolved_config(cfg: PipelineConfig) -> None:
    with open(Path(cfg.workdir) / "config.resolved.yaml", "w") as handle:
        yaml.safe_dump(cfg.raw, handle, sort_keys=False)


def run_config(
    cfg: PipelineConfig,
    *,
    registry: Optional[dict[str, StepFn]] = None,
    force: bool = False,
    timestamp: str = "",
) -> Manifest:
    """Run the steps listed in ``cfg`` against its workdir, returning the manifest.

    ``timestamp`` is stamped into each step record (pass one in; the driver does
    not read the clock). ``force`` re-runs steps even if already recorded.
    """
    from basisopt import bo_logger

    workdir = Path(cfg.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(cfg)
    manifest = Manifest.load_or_create(workdir, cfg.element)

    for name in cfg.steps:
        step_cfg = cfg.step_config(name)
        if manifest.has(name) and not force and not step_cfg.get("force"):
            bo_logger.info("[auto-basis] skipping '%s' (already done); use force to rerun", name)
            continue

        bo_logger.info("[auto-basis] running step '%s' for %s", name, cfg.element)
        state = RunState(config=cfg, element=cfg.element, input_basis=_resolve_input(cfg, name, manifest))
        result = get_step(name, registry)(state, step_cfg)

        step_dir = workdir / _step_dirname(name)
        step_dir.mkdir(parents=True, exist_ok=True)
        save_basis(result.basis, step_dir / "basis.json")
        for filename, text in result.exports.items():
            (step_dir / filename).write_text(text)
        with open(step_dir / "record.json", "w") as handle:
            json.dump(result.record, handle, indent=2)

        manifest.record_step(
            name,
            basis_relpath=str(Path(_step_dirname(name)) / "basis.json"),
            record=result.record,
            backend=cfg.step_backend(name),
            timestamp=timestamp,
            exports={fn: str(Path(_step_dirname(name)) / fn) for fn in result.exports},
        )
    return manifest


def run_pipeline(
    config_path: str | Path,
    *,
    registry: Optional[dict[str, StepFn]] = None,
    force: bool = False,
    timestamp: str = "",
) -> Manifest:
    """Load a config file and run it (see :func:`run_config`)."""
    return run_config(load_config(config_path), registry=registry, force=force, timestamp=timestamp)
