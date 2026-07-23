"""Concrete auto-basis steps (1-6), wrapping the functions the manual scripts
already used. Importing this module registers all six steps.

Each step sets up its own backend/scratch/occupation-cards from config, so the
driver stays backend-agnostic. Heavy/optional imports (a specific backend, the
basis_set_exchange bridge) are done lazily inside the steps.
"""

from __future__ import annotations

import copy
from typing import Optional

from basisopt.util import get_composition

from .state import RunState, StepResult
from .steps import register_step

# Default molecular "method" string per backend (what the old scripts set on
# ``mol.method``). Overridable per step via a ``method:`` config key.
_DEFAULT_METHODS = {"psi4": "dft", "molpro": "rks", "dummy": "linear"}


def _activate_backend(state: RunState, step_name: str) -> str:
    """Set the active backend + scratch dir for a step; return the backend name."""
    from basisopt import api

    backend = state.config.step_backend(step_name)
    api.set_backend(backend, verbose=False)
    api.set_tmp_dir(state.config.backend.tmp_dir, verbose=False)
    return backend


def _method_name(step_cfg: dict, backend: str) -> str:
    return step_cfg.get("method") or _DEFAULT_METHODS.get(backend, "scf")


def _method_params(state: RunState, step_name: str, backend: str, method: str,
                   wf_key: Optional[str] = None) -> dict:
    """Merged method params for a step, with the Molpro occupation card applied
    (``<method>-params``) when the backend is Molpro and the card is present."""
    params = state.config.method_params(backend, step_name)
    if backend == "molpro" and wf_key:
        card = state.reference.wf_cards.get(wf_key)
        if card:
            params = dict(params)
            params[f"{method}-params"] = card
    return params


def _require_cbs(state: RunState, step_name: str) -> float:
    if state.reference.cbs_limit is None:
        raise ValueError(f"Step '{step_name}' needs reference.cbs_limit; none set")
    return state.reference.cbs_limit


def _target(state: RunState, step_cfg: dict, step_name: str) -> float:
    target = step_cfg.get("target", state.reference.target)
    if target is None:
        raise ValueError(f"Step '{step_name}' needs a target (step config or reference.target)")
    return target


def _run_energy(mol, params) -> Optional[float]:
    """Run a single-point calc and return the energy (None on failure)."""
    from basisopt import api

    if api.run_calculation(mol=mol, params=params) != 0:
        return None
    # run_calculation stores the result under the evaluate key ("energy"), not
    # under the method name; get_value(mol.method) always missed and returned None.
    return api.get_backend().get_value("energy")


def _to_molpro(basis) -> str:
    """Export an internal basis to a Molpro-format string (lazy BSE import)."""
    from basis_set_exchange.writers import write_formatted_basis_str

    from basisopt.bse_wrapper import internal_to_bse

    return write_formatted_basis_str(internal_to_bse(basis), "molpro")


def _molpro_export(basis) -> dict:
    """Best-effort Molpro export as a step artifact (skipped if BSE unavailable)."""
    try:
        return {"basis.molpro.txt": _to_molpro(basis)}
    except Exception:  # noqa: BLE001 - export is a convenience, never fatal
        return {}


# --------------------------------------------------------------------------- #
# Step 1: primitives (Legendre)
# --------------------------------------------------------------------------- #
@register_step("primitives")
def step_primitives(state: RunState, step_cfg: dict) -> StepResult:
    from basisopt import opt
    from basisopt.opt.auto_basis import AutoBasisLegendre

    cbs_limit = _require_cbs(state, "primitives")
    target = _target(state, step_cfg, "primitives")
    backend = _activate_backend(state, "primitives")
    method = _method_name(step_cfg, backend)

    mol = state.build_atom(method)
    strategy = AutoBasisLegendre()
    if "n_prim" in step_cfg:
        strategy.n_prim = tuple(step_cfg["n_prim"])
    if step_cfg.get("legendre_params") is not None:
        strategy.legendre_params = step_cfg["legendre_params"]
    strategy.set_cbs_limit(cbs_limit)
    strategy.target = target
    strategy.params = _method_params(state, "primitives", backend, method)

    opt.atom_auto(
        molecule=mol,
        strategy=strategy,
        algorithm=step_cfg.get("algorithm", "Nelder-Mead"),
        opt_params=step_cfg.get("opt_params", {}),
    )

    energy = strategy.last_objective
    record = {
        "atomic_energy": energy,
        "cbs_limit": cbs_limit,
        "dE_CBS": None if energy is None else energy - cbs_limit,
        "target": target,
        "composition": get_composition(mol.basis, state.element),
    }
    return StepResult(basis=mol.basis, record=record)


# --------------------------------------------------------------------------- #
# Step 2: reduction
# --------------------------------------------------------------------------- #
@register_step("reduction")
def step_reduction(state: RunState, step_cfg: dict) -> StepResult:
    from basisopt import opt
    from basisopt.opt.auto_basis import AutoBasisReduceStrategy

    basis = state.require_input("reduction")
    cbs_limit = _require_cbs(state, "reduction")
    target = _target(state, step_cfg, "reduction")
    backend = _activate_backend(state, "reduction")
    method = _method_name(step_cfg, backend)

    mol = state.build_atom(method)
    mol.basis = basis

    strategy = AutoBasisReduceStrategy()
    strategy.set_cbs_limit(cbs_limit)
    strategy.target = target
    strategy.params = _method_params(state, "reduction", backend, method)

    opt.atom_auto_reduce(
        molecule=mol,
        strategy=strategy,
        algorithm=step_cfg.get("algorithm", "Nelder-Mead"),
        opt_params=step_cfg.get("opt_params", {}),
    )

    energy = strategy.last_objective
    record = {
        "atomic_energy": energy,
        "cbs_limit": cbs_limit,
        "dE_CBS": None if energy is None else energy - cbs_limit,
        "target": target,
        "composition": get_composition(mol.basis, state.element),
    }
    return StepResult(basis=mol.basis, record=record)


# --------------------------------------------------------------------------- #
# Step 3: contraction (natural atomic orbitals - external Molpro for now)
# --------------------------------------------------------------------------- #
@register_step("contraction")
def step_contraction(state: RunState, step_cfg: dict) -> StepResult:
    # NAO contraction is currently produced in Molpro and dropped in via
    # `contraction.input: <file>`; the driver has already loaded it, so this
    # step just adopts it. (Replicating NAOs in Psi4 is a science to-do.)
    basis = state.require_input("contraction")
    record = {
        "note": "adopted externally-contracted (NAO) basis",
        "composition": get_composition(basis, state.element),
    }
    return StepResult(basis=basis, record=record, exports=_molpro_export(basis))


# --------------------------------------------------------------------------- #
# Step 4: uncontraction (weighted toward lower angular momenta)
# --------------------------------------------------------------------------- #
@register_step("uncontraction")
def step_uncontraction(state: RunState, step_cfg: dict) -> StepResult:
    from basisopt.basis.basis import uncontract
    from basisopt.uncontract import uncontract_percentage

    contracted_basis = state.require_input("uncontraction")
    percent = step_cfg.get("decontract_error_percent")
    if percent is None:
        raise ValueError("uncontraction needs 'decontract_error_percent'")
    backend = _activate_backend(state, "uncontraction")
    method = _method_name(step_cfg, backend)

    mol = state.build_geometry_molecule(method)
    mol.basis = {
        el.lower(): contracted_basis[el.lower()] for el in mol.unique_atoms()
    }
    params = _method_params(state, "uncontraction", backend, method, wf_key="molpro_diatomic")

    # contracted vs fully-uncontracted reference energies (as the scripts logged)
    contracted = copy.deepcopy(mol.basis)
    uncontract(mol.basis)
    uncontracted_energy = _run_energy(mol, params)
    mol.basis = contracted
    contracted_energy = _run_energy(mol, params)
    if uncontracted_energy is None or contracted_energy is None:
        from basisopt.exceptions import FailedCalculation

        raise FailedCalculation(
            "uncontraction reference calculation failed "
            f"(uncontracted={uncontracted_energy}, contracted={contracted_energy})"
        )
    # uncontract_percentage reads these off the molecule (via get_result); without
    # storing them it saw 0.0/0.0 and raised ZeroDivisionError.
    mol.add_result("uncontracted_energy", uncontracted_energy)
    mol.add_result("contracted_energy", contracted_energy)

    results, uncontracted_mol = uncontract_percentage(mol, state.element, percent, params)

    record = {
        "contracted_energy": contracted_energy,
        "uncontracted_energy": uncontracted_energy,
        "decontract_error_percent": percent,
        "log": results,
        "composition": get_composition(uncontracted_mol.basis, state.element),
    }
    return StepResult(
        basis=uncontracted_mol.basis, record=record, exports=_molpro_export(uncontracted_mol.basis)
    )


# --------------------------------------------------------------------------- #
# Step 5: purification (extended Davidson)
# --------------------------------------------------------------------------- #
@register_step("purification")
def step_purification(state: RunState, step_cfg: dict) -> StepResult:
    from basisopt.util import davidson_purify_extended

    basis = state.require_input("purification")
    purified = davidson_purify_extended(basis)

    record = {"composition": get_composition(purified, state.element)}

    # optional purified single-point energy (mirrors the scripts) when a backend
    # + occupation card are configured
    if step_cfg.get("evaluate_energy", True):
        backend = _activate_backend(state, "purification")
        method = _method_name(step_cfg, backend)
        params = _method_params(state, "purification", backend, method, wf_key="molpro_atomic")
        mol = state.build_atom(method)
        mol.basis = purified
        record["purified_energy"] = _run_energy(mol, params)

    return StepResult(basis=purified, record=record, exports=_molpro_export(purified))


# --------------------------------------------------------------------------- #
# Step 6: pruning
# --------------------------------------------------------------------------- #
@register_step("pruning")
def step_pruning(state: RunState, step_cfg: dict) -> StepResult:
    from basisopt.prune import prune_element

    basis = state.require_input("pruning")
    energy_target = step_cfg.get("energy_target")
    if energy_target is None:
        raise ValueError("pruning needs 'energy_target'")
    target = energy_target * step_cfg.get("fraction", 1.0)
    backend = _activate_backend(state, "pruning")
    method = _method_name(step_cfg, backend)
    params = _method_params(state, "pruning", backend, method, wf_key="molpro_atomic")
    mol = state.build_atom(method)
    mol.basis = basis

    pruned = prune_element(mol, state.element, target, params)
    record = {
        "energy_target": energy_target,
        "fraction": step_cfg.get("fraction", 1.0),
        "prune_target": target,
        "composition": get_composition(pruned.basis, state.element),
    }
    return StepResult(basis=pruned.basis, record=record, exports=_molpro_export(pruned.basis))
