"""Concrete auto-basis steps (1-6), wrapping the functions the manual scripts
already used. Importing this module registers all six steps.

Each step sets up its own backend/scratch/occupation-cards from config, so the
driver stays backend-agnostic. Heavy/optional imports (a specific backend, the
basis_set_exchange bridge) are done lazily inside the steps.
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import numpy as np

from basisopt.data import AM_DICT
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


def _method_params(
    state: RunState, step_name: str, backend: str, method: str, wf_key: Optional[str] = None
) -> dict:
    """Merged method params for a step, with the Molpro occupation card applied
    (``<method>-params``) when the backend is Molpro and the card is present."""
    params = state.config.method_params(backend, step_name)
    if backend == "molpro" and wf_key:
        card = state.reference.wf_cards.get(wf_key)
        if card:
            params = dict(params)
            params[f"{method}-params"] = card
    return params


def _num(value):
    """Coerce a config numeric to float (None passes through). PyYAML parses
    scientific notation without a decimal point (e.g. ``2e-4``) as a str."""
    return None if value is None else float(value)


def _mult_charge(step_cfg: dict):
    """Per-step ``(multiplicity, charge)`` overrides (None -> use the reference
    defaults). Lets an atomic stage (e.g. the H doublet) and a molecular stage
    (e.g. the H2 singlet) run at different spin states."""
    mult = step_cfg.get("multiplicity")
    charge = step_cfg.get("charge")
    return (None if mult is None else int(mult), None if charge is None else int(charge))


def _require_cbs(state: RunState, step_name: str) -> float:
    if state.reference.cbs_limit is None:
        raise ValueError(f"Step '{step_name}' needs reference.cbs_limit; none set")
    return float(state.reference.cbs_limit)


def _target(state: RunState, step_cfg: dict, step_name: str) -> float:
    target = step_cfg.get("target", state.reference.target)
    if target is None:
        raise ValueError(f"Step '{step_name}' needs a target (step config or reference.target)")
    return float(target)


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

    mol = state.build_atom(method, *_mult_charge(step_cfg))
    strategy = AutoBasisLegendre()
    if "n_prim" in step_cfg:
        strategy.n_prim = tuple(step_cfg["n_prim"])
    if step_cfg.get("legendre_params") is not None:
        strategy.legendre_params = step_cfg["legendre_params"]
    strategy.set_cbs_limit(cbs_limit)
    strategy.target = target
    # Optional growth cutoffs (all off unless set): a hard cap on primitives per
    # shell, a global iteration cap, and a saturation (stall) tolerance. The
    # signed CBS target is always the primary stop.
    for key in ("max_n", "max_its", "stall_tol"):
        if step_cfg.get(key) is not None:
            setattr(strategy, key, step_cfg[key])
    strategy.params = _method_params(state, "primitives", backend, method)

    sampling_cfg = step_cfg.get("sampling")
    if sampling_cfg:
        # Parallel multi-start: run n_starts Legendre optimisations from perturbed
        # coefficient seeds (start 0 = the default) and keep the lowest energy.
        from basisopt.opt import sampling
        from basisopt.opt.auto_basis import _ATOMIC_LEGENDRE_COEFFS

        base_leg = strategy.legendre_params or _ATOMIC_LEGENDRE_COEFFS.get(
            state.element.capitalize()
        )
        ray_params = {
            "backend": backend,
            "tmp_dir": state.config.backend.tmp_dir,
            "threads_per_job": sampling_cfg.get("threads_per_job", 1),
        }
        obj, stop_reason, mol.basis, n_starts = sampling.multistart_primitives(
            mol,
            strategy,
            base_leg,
            step_cfg.get("algorithm", "Nelder-Mead"),
            step_cfg.get("opt_params", {}),
            sampling_cfg,
            ray_params,
        )
        energy = None if obj is None else float(obj)
    else:
        opt.atom_auto(
            molecule=mol,
            strategy=strategy,
            algorithm=step_cfg.get("algorithm", "Nelder-Mead"),
            opt_params=step_cfg.get("opt_params", {}),
        )
        # native float so the record JSON-serializes (backends may return numpy scalars)
        energy = strategy.last_objective
        energy = None if energy is None else float(energy)
        stop_reason = getattr(strategy, "stop_reason", None)
        n_starts = 1

    record = {
        "atomic_energy": energy,
        "cbs_limit": cbs_limit,
        "dE_CBS": None if energy is None else energy - cbs_limit,
        "target": target,
        # why the growth stopped: 'target' (CBS target met) or a cutoff
        # ('stall'/'max_n'/'max_its'); target_met is the at-a-glance check.
        "stop_reason": stop_reason,
        "target_met": None if energy is None else bool((energy - cbs_limit) < target),
        "n_starts": n_starts,
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

    mol = state.build_atom(method, *_mult_charge(step_cfg))
    mol.basis = basis

    strategy = AutoBasisReduceStrategy()
    strategy.set_cbs_limit(cbs_limit)
    strategy.target = target
    strategy.params = _method_params(state, "reduction", backend, method)
    # optional Ray parallelism: fan the per-exponent ranking trials across the pool
    strategy.parallel, strategy.ray_params = _parallel_settings(state, backend, step_cfg)

    opt.atom_auto_reduce(
        molecule=mol,
        strategy=strategy,
        algorithm=step_cfg.get("algorithm", "Nelder-Mead"),
        opt_params=step_cfg.get("opt_params", {}),
    )

    # Read the FINALISED energy, not strategy.last_objective. The reduce strategy's
    # finalize() runs a load-bearing calculation on the *restored* basis and stores
    # it on the molecule; last_objective is the last trial the optimiser evaluated,
    # which -- when the final removal was rejected and rolled back -- is that rejected
    # (worse) basis, not the one we return. Reading it left atomic_energy/dE_CBS
    # describing the discarded trial while composition described the kept basis.
    # native float so the record JSON-serializes (backends may return numpy scalars)
    energy = mol.get_result(strategy.eval_type)
    energy = None if energy is None else float(energy)
    record = {
        "atomic_energy": energy,
        "cbs_limit": cbs_limit,
        "dE_CBS": None if energy is None else energy - cbs_limit,
        "target": target,
        "composition": get_composition(mol.basis, state.element),
    }
    return StepResult(basis=mol.basis, record=record)


# --------------------------------------------------------------------------- #
# Step 3: contraction (natural atomic orbitals)
# --------------------------------------------------------------------------- #
def _apply_naos(basis, element: str, nao_data: dict, n_keep: dict):
    """Replace each shell's coefficients with its leading natural orbitals.

    ``nao_data`` is the backend's ``{l: (occupations, coefficients)}``; ``n_keep``
    maps angular-momentum letter -> number of NAOs to keep. Returns the contracted
    basis and the kept occupations per shell (for the record).
    """
    contracted = copy.deepcopy(basis)
    kept_occupations = {}
    for shell in contracted[element.lower()]:
        am = AM_DICT[shell.l]
        if am not in nao_data:
            continue
        occupations, coefficients = nao_data[am]
        if coefficients.shape[0] != len(shell.exps):
            raise ValueError(
                f"NAO coefficient count ({coefficients.shape[0]}) does not match the "
                f"number of {shell.l} primitives ({len(shell.exps)}) for {element}"
            )
        keep = min(int(n_keep.get(shell.l, 0)), coefficients.shape[1])
        if keep == 0:
            raise ValueError(
                f"contraction.n_keep gives no count for the {shell.l} shell of {element}"
            )
        shell.coefs = [np.asarray(coefficients[:, k], dtype=float) for k in range(keep)]
        kept_occupations[shell.l] = [float(o) for o in occupations[:keep]]
    return contracted, kept_occupations


@register_step("contraction")
def step_contraction(state: RunState, step_cfg: dict) -> StepResult:
    """Natural-orbital (NAO) contraction.

    Two modes:
      - ``generate: true`` -- generate the NAOs natively with the configured
        backend (per-backend: e.g. Psi4's density-average route) from the
        uncontracted input primitives, keeping ``n_keep`` orbitals per shell.
      - default -- adopt an externally-produced (e.g. Molpro) NAO basis dropped
        in via ``contraction.input`` (the driver has already loaded it).
    """
    basis = state.require_input("contraction")

    if not step_cfg.get("generate", False):
        record = {
            "note": "adopted externally-contracted (NAO) basis",
            "composition": get_composition(basis, state.element),
        }
        return StepResult(basis=basis, record=record, exports=_molpro_export(basis))

    from basisopt import api
    from basisopt.basis.basis import uncontract

    n_keep = step_cfg.get("n_keep")
    if not n_keep:
        raise ValueError(
            "contraction.generate needs 'n_keep' (natural orbitals to keep per "
            "shell, e.g. {s: 2, p: 1})"
        )

    backend = _activate_backend(state, "contraction")
    method = _method_name(step_cfg, backend)
    params = _method_params(state, "contraction", backend, method, wf_key="molpro_atomic")

    mol = state.build_atom(method, *_mult_charge(step_cfg))
    # the natural orbitals live in the primitive space, so build them from the
    # fully uncontracted input basis
    mol.basis = uncontract(copy.deepcopy(basis))
    nao_data = api.get_backend().natural_orbitals(mol, params)
    contracted, occupations = _apply_naos(basis, state.element, nao_data, n_keep)

    record = {
        "note": f"natural-orbital contraction generated with the {backend} backend",
        "n_keep": n_keep,
        "multiplicity": mol.multiplicity,  # the atom's spin state (e.g. H doublet)
        "occupations": occupations,
        "composition": get_composition(contracted, state.element),
    }

    # optional contraction-error diagnostic (uncontracted vs contracted energy)
    if step_cfg.get("evaluate_energy", True):
        mol.basis = uncontract(copy.deepcopy(basis))
        e_uncontracted = _run_energy(mol, params)
        mol.basis = contracted
        e_contracted = _run_energy(mol, params)
        record["uncontracted_energy"] = e_uncontracted
        record["contracted_energy"] = e_contracted
        if e_uncontracted is not None and e_contracted is not None:
            record["contraction_error_mEh"] = (e_contracted - e_uncontracted) * 1e3

    return StepResult(basis=contracted, record=record, exports=_molpro_export(contracted))


# --------------------------------------------------------------------------- #
# Step 4: uncontraction (weighted toward lower angular momenta)
# --------------------------------------------------------------------------- #
@register_step("uncontraction")
def step_uncontraction(state: RunState, step_cfg: dict) -> StepResult:
    from basisopt.basis.basis import uncontract
    from basisopt.uncontract import uncontract_percentage

    contracted_basis = state.require_input("uncontraction")
    percent = _num(step_cfg.get("decontract_error_percent"))
    if percent is None:
        raise ValueError("uncontraction needs 'decontract_error_percent'")
    backend = _activate_backend(state, "uncontraction")
    method = _method_name(step_cfg, backend)

    mol = state.build_geometry_molecule(method, *_mult_charge(step_cfg))
    mol.basis = {el.lower(): contracted_basis[el.lower()] for el in mol.unique_atoms()}
    params = _method_params(state, "uncontraction", backend, method, wf_key="molpro_diatomic")

    # contracted vs fully-uncontracted reference energies (as the scripts logged).
    # uncontract() returns a NEW basis and does not mutate in place, so its result
    # must be assigned back -- otherwise both energies are computed on the same
    # (contracted) basis, contraction_error is 0 and uncontract_percentage divides
    # by zero.
    contracted = copy.deepcopy(mol.basis)
    mol.basis = uncontract(mol.basis)
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

    # optional Ray parallelism: fan the per-function ranking trials across the pool
    parallel, ray_params = _parallel_settings(state, backend, step_cfg)
    results, uncontracted_mol = uncontract_percentage(
        mol, state.element, percent, params, parallel=parallel, ray_params=ray_params
    )

    record = {
        "contracted_energy": contracted_energy,
        "uncontracted_energy": uncontracted_energy,
        "decontract_error_percent": percent,
        "multiplicity": mol.multiplicity,  # the molecule's spin state (e.g. H2 singlet)
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
        mol = state.build_atom(method, *_mult_charge(step_cfg))
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
    energy_target = _num(step_cfg.get("energy_target"))
    if energy_target is None:
        raise ValueError("pruning needs 'energy_target'")
    target = energy_target * float(step_cfg.get("fraction", 1.0))
    backend = _activate_backend(state, "pruning")
    method = _method_name(step_cfg, backend)
    params = _method_params(state, "pruning", backend, method, wf_key="molpro_atomic")
    mol = state.build_atom(method, *_mult_charge(step_cfg))
    mol.basis = basis

    # optional Ray parallelism: fan each re-ranking pass's per-coefficient trials
    parallel, ray_params = _parallel_settings(state, backend, step_cfg)
    pruned = prune_element(
        mol, state.element, target, params, parallel=parallel, ray_params=ray_params
    )
    record = {
        "energy_target": energy_target,
        "fraction": step_cfg.get("fraction", 1.0),
        "prune_target": target,
        "composition": get_composition(pruned.basis, state.element),
    }
    return StepResult(basis=pruned.basis, record=record, exports=_molpro_export(pruned.basis))


# --------------------------------------------------------------------------- #
# Step 7: polarisation (multi-molecule, cross-element)
# --------------------------------------------------------------------------- #
def _normalise_keys(basis: dict) -> dict:
    """Lower-case the element keys of an internal basis (BSE/file bases may be
    title-cased, while the pipeline keys everything lower-case)."""
    return {k.lower(): v for k, v in basis.items()}


def _spectator_basis(spectator_cfg: dict) -> tuple[str, list]:
    """Return ``(element_lower, shells)`` for a spectator atom, from either a
    published basis name (default pc-seg-4) or a prior-built basis file."""
    element = spectator_cfg.get("element")
    if not element:
        raise ValueError("polarisation spectator needs an 'element'")
    el = element.lower()

    if spectator_cfg.get("basis_file"):
        from .pipeline import load_basis

        loaded = _normalise_keys(
            load_basis(spectator_cfg["basis_file"], spectator_cfg.get("basis_format"))
        )
    else:
        from basisopt.bse_wrapper import fetch_basis

        loaded = _normalise_keys(fetch_basis(spectator_cfg.get("basis", "pcseg-4"), element))

    if el not in loaded:
        raise ValueError(f"spectator basis has no entry for element '{element}'")
    return el, loaded[el]


def _build_reference_molecule(mol_cfg: dict, method: str, index: int, defaults):
    """Build one reference Molecule for the polarisation set: geometry, charge,
    multiplicity and the molecular CBS limit (read by collective_polarize)."""
    from basisopt.molecule import Molecule

    geometry = mol_cfg.get("geometry")
    if not geometry:
        raise ValueError("each polarisation molecule needs a 'geometry' path")
    if "cbs_limit" not in mol_cfg:
        raise ValueError(f"polarisation molecule '{geometry}' needs a 'cbs_limit'")

    mol = Molecule.from_xyz(geometry)
    base = os.path.splitext(os.path.basename(geometry))[0]
    mol.name = mol_cfg.get("name") or f"ref{index}_{base}"  # unique -> run_all key
    mol.method = method
    mol.charge = mol_cfg.get("charge", defaults.charge or 0)
    mult = mol_cfg.get("multiplicity", defaults.multiplicity)
    if mult is not None:
        mol.multiplicity = mult
    mol.cbs_limit = float(mol_cfg["cbs_limit"])
    return mol


def _parallel_settings(state: RunState, backend: str, step_cfg: Optional[dict] = None):
    """Translate a ``parallel`` config into ``(parallel, ray_params)``.

    A step may carry its own ``parallel`` block, which takes precedence over the
    global ``backend.parallel``; either accepts ``n_cores`` (total cores for Ray),
    ``threads_per_job`` (backend threads per calc) and an optional ``n_workers``.
    Returns ``(False, None)`` when neither is set. Otherwise turns Ray on with the
    requested core count and builds the ray_params the warm actor pool needs."""
    pcfg = (step_cfg or {}).get("parallel") or state.config.backend.parallel
    if not pcfg:
        return False, None
    from basisopt import api

    api.set_parallel(True, pcfg.get("n_cores", 2))
    ray_params = {
        "backend": backend,
        "tmp_dir": state.config.backend.tmp_dir,
        "threads_per_job": pcfg.get("threads_per_job", 1),
    }
    if pcfg.get("n_workers"):
        ray_params["n_workers"] = pcfg["n_workers"]
    return True, ray_params


def _reference_basis(reference_cfg, elements) -> dict:
    """Resolve a *complete* reference basis covering every element in ``elements``.

    ``reference_cfg`` is a published basis name (str) or a
    ``{basis_file: ..., [basis_format]}`` mapping. It represents a whole prior set
    applied to *all* atoms, so a missing element is an error, not a silent
    fallback (that would benchmark against a different basis than intended)."""
    from basisopt.bse_wrapper import fetch_basis

    from .pipeline import load_basis

    if isinstance(reference_cfg, dict) and reference_cfg.get("basis_file"):
        loaded = _normalise_keys(
            load_basis(reference_cfg["basis_file"], reference_cfg.get("basis_format"))
        )
        source = reference_cfg["basis_file"]
        resolved = {}
        for el in elements:
            if el not in loaded:
                raise ValueError(
                    f"reference_basis '{source}' has no entry for element '{el}'; a "
                    f"reference basis must cover every atom in the reference molecules."
                )
            resolved[el] = loaded[el]
        return resolved

    name = reference_cfg["basis"] if isinstance(reference_cfg, dict) else reference_cfg
    resolved = {}
    for el in elements:
        loaded = _normalise_keys(fetch_basis(name, el))
        if el not in loaded:
            raise ValueError(
                f"reference_basis '{name}' has no entry for element '{el}'; a reference "
                f"basis must cover every atom in the reference molecules."
            )
        resolved[el] = loaded[el]
    return resolved


def _reference_loss(molecules, ref_basis, params, loss, parallel, ray_params) -> float:
    """Evaluate the reference molecules with ``ref_basis`` on ALL atoms and return
    the aggregated basis-set-incompleteness loss -- the benchmark a relative
    ``target_ratio`` scales."""
    from basisopt import api
    from basisopt.opt.optimizers import POLARISATION_LOSSES

    results = api.run_all(
        evaluate="energy",
        mols=molecules,
        params=params,
        parallel=parallel,
        ray_params=ray_params,
        shared_basis=ref_basis,
    )
    bsies = [max(0.0, float(results[m.name]) - m.cbs_limit) for m in molecules]
    nelec = [m.nelectrons() for m in molecules]
    nvalence = [m.nvalence_electrons() for m in molecules]
    return POLARISATION_LOSSES[loss](bsies, nelec, nvalence)


def _resolve_polarisation_target(step_cfg, molecules, params, loss, parallel, ray_params):
    """Resolve the polarisation loss target plus a provenance record.

    Absolute ``target``; or ``target_ratio`` times a reference given either
    explicitly (``reference_loss``) or computed from a complete ``reference_basis``
    applied to all atoms of the reference molecules (a benchmark of the prior set)."""
    if step_cfg.get("target") is not None:
        return float(_num(step_cfg["target"])), {"mode": "absolute"}

    ratio = step_cfg.get("target_ratio")
    if ratio is None:
        return 1e-4, {"mode": "default"}  # historical default when nothing is set
    ratio = float(_num(ratio))

    if step_cfg.get("reference_loss") is not None:
        reference = float(_num(step_cfg["reference_loss"]))
        source = "reference_loss"
    elif step_cfg.get("reference_basis") is not None:
        elements = set()
        for mol in molecules:
            elements.update(a.lower() for a in mol.unique_atoms())
        ref_basis = _reference_basis(step_cfg["reference_basis"], elements)
        reference = _reference_loss(molecules, ref_basis, params, loss, parallel, ray_params)
        source = "reference_basis"
    else:
        raise ValueError(
            "polarisation 'target_ratio' needs a 'reference_loss' (number) or a "
            "'reference_basis' (name or file) to scale."
        )
    record = {"mode": "relative", "ratio": ratio, "reference_loss": reference, "source": source}
    return ratio * reference, record


@register_step("polarisation")
def step_polarisation(state: RunState, step_cfg: dict) -> StepResult:
    """Grow polarisation shells (d/f/g; p for H) onto element X's sp basis by
    optimising against a set of reference molecules. Each heteronuclear molecule
    gives its non-optimised (spectator) atom a large fixed basis (pc-seg-4 by
    default, or a prior-built basis file) so the energy lowering is attributable
    to X. Uses the AutoBasisPolarisation strategy inside collective_polarize."""
    from basisopt.opt import collective_polarize
    from basisopt.opt.optimizers import POLARISATION_LOSSES
    from basisopt.opt.polarisation import AutoBasisPolarisation

    element = state.element
    el = element.lower()
    working = _normalise_keys(state.require_input("polarisation"))
    if el not in working:
        raise ValueError(f"input basis has no entry for element '{element}'")

    backend = _activate_backend(state, "polarisation")
    method = _method_name(step_cfg, backend)
    params = _method_params(state, "polarisation", backend, method, wf_key="molpro_diatomic")

    mol_cfgs = step_cfg.get("molecules")
    if not mol_cfgs:
        raise ValueError("polarisation step needs a non-empty 'molecules' list")

    # Combined basis shared across the molecule set: element X (the working basis
    # the strategy grows) plus a fixed spectator basis on every OTHER atom (pc-seg-4
    # by default) so the energy lowering is attributable to X. A step-level
    # `spectator_basis` sets the default; a per-molecule `spectator` overrides it.
    # collective_polarize assigns this whole dict to every molecule; each backend
    # picks out the elements it contains.
    default_spectator = step_cfg.get("spectator_basis")
    combined = {el: copy.deepcopy(working[el])}
    molecules = []
    for i, mol_cfg in enumerate(mol_cfgs):
        mol = _build_reference_molecule(mol_cfg, method, i, state.reference)
        molecules.append(mol)
        spectator = mol_cfg.get("spectator")
        if spectator:  # per-molecule override wins
            spec_el, spec_shells = _spectator_basis(spectator)
            if spec_el != el:
                combined[spec_el] = spec_shells
        if default_spectator:  # step-level default fills any remaining non-X atom
            for mol_el in {a.lower() for a in mol.unique_atoms()}:
                if mol_el != el and mol_el not in combined:
                    _, spec_shells = _spectator_basis(
                        {"element": mol_el, "basis": default_spectator}
                    )
                    combined[mol_el] = spec_shells
    # every atom in every molecule must have a basis (X grows; the rest are spectators)
    for mol in molecules:
        for mol_el in {a.lower() for a in mol.unique_atoms()}:
            if mol_el not in combined:
                raise ValueError(
                    f"molecule '{mol.name}' contains element '{mol_el}' with no basis. "
                    f"Set a step-level 'spectator_basis' (e.g. pcseg-4) or a per-molecule "
                    f"'spectator' block."
                )

    loss = step_cfg.get("loss", "mean_per_electron")
    if loss not in POLARISATION_LOSSES:
        raise ValueError(
            f"unknown polarisation 'loss' '{loss}'; choose from {sorted(POLARISATION_LOSSES)}"
        )

    # Ray settings for the reference-loss evaluation and the single-start optimise.
    parallel, ray_params = _parallel_settings(state, backend, step_cfg)

    # Target: absolute, or target_ratio x (explicit reference_loss | reference_basis
    # evaluated with that basis on ALL atoms of the reference molecules).
    target, target_source = _resolve_polarisation_target(
        step_cfg, molecules, params, loss, parallel, ray_params
    )

    min_l = int(step_cfg.get("min_l", 2))
    max_l = int(step_cfg.get("max_l", 3))
    seed_exponent = float(step_cfg.get("seed_exponent", 1.0))
    npass = step_cfg.get("npass", 1)
    algorithm = step_cfg.get("algorithm", "Nelder-Mead")
    opt_params = step_cfg.get("opt_params", {})
    result_key = "energy_" + element.title()
    # 'greedy' (default): AutoBasisPolarisation saturates one l then advances.
    # 'config_search': compare competing d/f/g configurations at equal budget
    # (the row-2 method; see opt/polarisation_search.py).
    mode = step_cfg.get("mode", "greedy")
    trace = None

    if mode == "config_search":
        from basisopt.opt.polarisation_search import config_search_polarisation

        max_l = int(step_cfg.get("max_l", 4))  # d..g by default for the search
        max_n = None if step_cfg.get("max_n") is None else int(step_cfg["max_n"])
        max_total = None if step_cfg.get("max_total") is None else int(step_cfg["max_total"])
        min_improvement = _num(step_cfg.get("min_improvement", 0.0))
        non_increasing = bool(step_cfg.get("non_increasing", True))
        pcfg = (step_cfg.get("parallel") or state.config.backend.parallel) or {}
        dispatch_cfg = (
            {
                "n_cores": pcfg.get("n_cores", 2),
                "threads_per_job": pcfg.get("threads_per_job", 1),
            }
            if pcfg
            else {}
        )
        final_loss, stop_reason, combined, final_config, trace = config_search_polarisation(
            molecules,
            combined,
            el,
            algorithm,
            opt_params,
            params,
            min_l=min_l,
            max_l=max_l,
            seed_exponent=seed_exponent,
            target=target,
            loss=loss,
            max_n=max_n,
            max_total=max_total,
            min_improvement=min_improvement,
            non_increasing=non_increasing,
            npass=npass,
            parallel=parallel,
            ray_params=ray_params,
            dispatch_cfg=dispatch_cfg,
        )
        final_loss = _opt_float(final_loss)
        per_molecule = None  # per-molecule diagnostics live in the trace instead
        n_starts = 1
        convergence = {
            "mode": "config_search",
            "target": _opt_float(target),
            "min_improvement": min_improvement,
            "max_n": max_n,
            "max_l": max_l,
            "max_total": max_total,
            "non_increasing": non_increasing,
            "final_config": final_config,
        }
    else:
        strategy = AutoBasisPolarisation(
            target=target,
            min_l=min_l,
            max_l=max_l,
            seed_exponent=seed_exponent,
            max_n=None if step_cfg.get("max_n") is None else int(step_cfg["max_n"]),
            max_its=None if step_cfg.get("max_its") is None else int(step_cfg["max_its"]),
            stall_tol=_num(step_cfg.get("stall_tol", 1e-5)),
            delta_e=None if step_cfg.get("delta_e") is None else _num(step_cfg["delta_e"]),
        )
        strategy.params = params

        sampling_cfg = step_cfg.get("sampling")
        if sampling_cfg:
            # Parallel multi-start (perturb the seed exponent); each start runs its
            # molecule set serially, so the starts -- not the molecules -- get Ray.
            from basisopt.opt import sampling

            sampling_ray = {
                "backend": backend,
                "tmp_dir": state.config.backend.tmp_dir,
                "threads_per_job": sampling_cfg.get("threads_per_job", 1),
            }
            obj, stop_reason, combined, n_starts = sampling.multistart_polarisation(
                molecules,
                combined,
                strategy,
                el,
                algorithm,
                opt_params,
                npass,
                sampling_cfg,
                sampling_ray,
                loss=loss,
            )
            final_loss = _opt_float(obj)
            per_molecule = None  # each start's per-molecule errors live on its own copies
        else:
            # Single start; the reference-molecule calcs in each objective evaluation
            # fan out across the warm actor pool if a parallel block is set (Level A).
            opt_data = [(el, algorithm, strategy, (lambda x: 0), opt_params)]
            collective_polarize(
                molecules,
                combined,
                opt_data=opt_data,
                npass=npass,
                parallel=parallel,
                ray_params=ray_params,
                loss=loss,
            )
            stop_reason = getattr(strategy, "stop_reason", None)
            final_loss = _opt_float(strategy.last_objective)
            per_molecule = {mol.name: _opt_float(mol.get_result(result_key)) for mol in molecules}
            n_starts = 1
        convergence = {
            "mode": "greedy",
            "target": _opt_float(strategy.target),
            "delta_e": strategy.delta_e,
            "stall_tol": strategy.stall_tol,
            "max_n": strategy.max_n,
            "max_l": strategy.max_l,
            "max_its": strategy.max_its,
        }

    result_basis = {el: combined[el]}
    pol_shells = [sh.l for sh in combined[el] if AM_DICT[sh.l] >= min_l]
    record = {
        "note": f"polarisation shells grown with the {backend} backend",
        "parallel": parallel,
        "n_starts": n_starts,
        "loss": loss,
        "mode": mode,
        "target": _opt_float(target),
        "target_source": target_source,
        "spectator_basis": default_spectator,
        "stop_reason": stop_reason,
        "polarisation_shells": pol_shells,
        "final_loss": final_loss,
        "per_molecule_error": per_molecule,
        # the energy/convergence criteria that governed termination
        "convergence": convergence,
        "composition": get_composition(result_basis, element),
    }
    if trace is not None:
        record["trace"] = trace
    return StepResult(basis=result_basis, record=record, exports=_molpro_export(result_basis))


def _opt_float(value) -> Optional[float]:
    """None-safe float cast so records JSON-serialize across backends."""
    return None if value is None else float(value)
