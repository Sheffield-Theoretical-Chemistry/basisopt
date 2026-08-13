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
import time
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


DEFAULT_EXPORT_FORMAT = "molpro"


def export_basis(basis, path: str | Path, fmt: str = DEFAULT_EXPORT_FORMAT) -> None:
    """Write an internal basis to ``path`` in ``fmt`` using the built-in exporter.

    ``fmt`` is a basis_set_exchange writer name ('molpro', 'psi4', 'gaussian94',
    'nwchem', ...) or 'json'/'internal' for the pipeline's own JSON -- the format
    ``load_basis`` reads back, so a JSON (or molpro/...) export can seed a later
    resume via a step's ``input:``. Parent directories are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("json", "internal"):
        save_basis(basis, path)
        return
    from basis_set_exchange.writers import write_formatted_basis_str

    from basisopt.bse_wrapper import internal_to_bse

    path.write_text(write_formatted_basis_str(internal_to_bse(basis), fmt))


def _normalize_export(value) -> Optional[dict]:
    """Normalise an ``export`` config (a str path, or a {path, format} mapping)
    to a dict, or None when unset."""
    if value is None:
        return None
    if isinstance(value, str):
        return {"path": value}
    return dict(value)


def _export_format(step_export: Optional[dict], global_export: Optional[dict]) -> str:
    """Format for an export: the step's own, else the global export's, else the
    built-in default."""
    return (
        (step_export or {}).get("format")
        or (global_export or {}).get("format")
        or DEFAULT_EXPORT_FORMAT
    )


def _run_export(
    basis, export_cfg: dict, global_export: Optional[dict], label: str
) -> Optional[dict]:
    """Perform one configured export; return a provenance dict, or None. Export
    failures are logged but never abort the run (a completed basis must not be
    lost to an export typo)."""
    from basisopt.api import ab_logger

    path = export_cfg.get("path")
    if not path:
        return None
    fmt = _export_format(export_cfg, global_export)
    try:
        export_basis(basis, path, fmt)
    except Exception:  # noqa: BLE001 - export is a convenience, never fatal
        ab_logger.warning("export of %s to %s (%s) failed", label, path, fmt, exc_info=True)
        return None
    ab_logger.info("exported %s basis -> %s (%s)", label, path, fmt)
    return {"path": str(path), "format": fmt}


def _summarise_record(record: dict) -> str:
    """One-line human summary of a step's ``record`` for the run log."""
    from basisopt.util import format_with_prefix

    parts: list[str] = []
    if record.get("composition"):
        parts.append(str(record["composition"]))
    if record.get("dE_CBS") is not None:
        parts.append(f"dE_CBS={format_with_prefix(record['dE_CBS'], 'Eh')}")
    if record.get("target_met") is not None:
        parts.append("target met" if record["target_met"] else "TARGET NOT MET")
    if record.get("decontract_error_percent") is not None:
        parts.append(f"decontract err {record['decontract_error_percent']:.3g}%")
    if record.get("final_loss") is not None:
        parts.append(f"loss={record['final_loss']:.3e}")
    if record.get("stop_reason"):
        parts.append(f"stop={record['stop_reason']}")
    return " · ".join(parts) if parts else "done"


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
    from basisopt.api import ab_logger

    workdir = Path(cfg.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(cfg)
    manifest = Manifest.load_or_create(workdir, cfg.element)
    # optional global end-of-flow export (independent of any per-step export)
    global_export = _normalize_export(cfg.raw.get("export"))

    for name in cfg.steps:
        step_cfg = cfg.step_config(name)
        step_export = _normalize_export(step_cfg.get("export"))

        if manifest.has(name) and not force and not step_cfg.get("force"):
            ab_logger.info("skipping '%s' (already done); use force to rerun", name)
            # still honour a configured per-step export, from the recorded basis
            if step_export and step_export.get("path"):
                _run_export(load_basis(manifest.basis_path(name)), step_export, global_export, name)
            continue

        ab_logger.info(
            "running step '%s' for %s (backend=%s)", name, cfg.element, cfg.step_backend(name)
        )
        state = RunState(
            config=cfg, element=cfg.element, input_basis=_resolve_input(cfg, name, manifest)
        )
        started = time.perf_counter()
        result = get_step(name, registry)(state, step_cfg)
        ab_logger.info(
            "step '%s' done in %.1fs — %s",
            name,
            time.perf_counter() - started,
            _summarise_record(result.record),
        )

        step_dir = workdir / _step_dirname(name)
        step_dir.mkdir(parents=True, exist_ok=True)
        save_basis(result.basis, step_dir / "basis.json")
        for filename, text in result.exports.items():
            (step_dir / filename).write_text(text)
        # user-configured per-step export (own path + format), recorded for provenance
        if step_export and step_export.get("path"):
            info = _run_export(result.basis, step_export, global_export, name)
            if info:
                result.record["export"] = info
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

    # global end-of-flow export of the final basis
    if global_export and global_export.get("path"):
        final = cfg.steps[-1] if cfg.steps else None
        if final and manifest.has(final):
            _run_export(
                load_basis(manifest.basis_path(final)), global_export, global_export, "final"
            )
        else:
            ab_logger.warning(
                "global export requested but the final step '%s' did not complete", final
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
