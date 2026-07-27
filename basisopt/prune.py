import copy

import numpy as np

from . import api, bo_logger
from .util import rank_shell_contractions


def argsort_inhomogeneous_3d_array(array):
    flat_array = []
    index_mapping = []

    for i, outer_list in enumerate(array):
        for j, middle_list in enumerate(outer_list):
            for k, element in enumerate(middle_list):
                flat_array.append(element)
                index_mapping.append((i, j, k))

    sorted_indices = np.argsort(flat_array)

    ranked_indices = [index_mapping[idx] for idx in sorted_indices]
    sorted_values = [flat_array[idx] for idx in sorted_indices]

    return ranked_indices, sorted_values


def rank_basis(mol, element, params, parallel=False, ray_params=None):
    """Rank every contraction coefficient in ``element``'s basis by importance.

    Distinct from ``testing.rank``'s exponent-dropping ranking: this path zeroes
    contraction *coefficients* and is used by ``prune_element``. Returns
    ``(energies, errors, ranked_idx, sorted_errors)``; ``prune_element`` consumes
    only the last two.

    Serial (default) ranks shell-by-shell via ``util.rank_shell_contractions``.
    When ``parallel`` is set, the independent per-coefficient trials across all
    shells are built here and fanned through ``api.run_all`` (warm actor pool,
    ``robust=True``); the resulting (energies, errors) have the identical jagged
    ``[shell][contraction][kept-primitive]`` structure, so the ranking matches.
    """
    el = element.lower()
    # one reference for all shells in this pass
    api.run_calculation(mol=mol, params=params)
    ref_energy = api.get_backend().get_value('energy')

    if not parallel:
        energies = []
        errors = []
        for shell in mol.basis[el]:
            en, er, ra, sr = rank_shell_contractions(mol, shell, params, ref_energy=ref_energy)
            energies.append(en)
            errors.append(er)
        ranked_idx, sorted_errors = argsort_inhomogeneous_3d_array(errors)
        return energies, errors, ranked_idx, sorted_errors

    # parallel: build one trial molecule per (shell, contraction, primitive) that
    # rank_shell_contractions would evaluate (skip a coefficient that is the only
    # non-zero in its contraction -- can't zero the last one), preserving order.
    shells = mol.basis[el]
    energies = [[[] for _ in sh.coefs] for sh in shells]
    errors = [[[] for _ in sh.coefs] for sh in shells]
    trials = []  # (shell idx, contraction idx, trial molecule) in build order
    for s, shell in enumerate(shells):
        for c_idx, coeffs in enumerate(shell.coefs):
            for i in range(len(coeffs)):
                if np.count_nonzero(shell.coefs[c_idx]) == 1:
                    continue
                trial = copy.deepcopy(mol)
                trial.basis[el][s].coefs[c_idx][i] = 0.0
                trial.name = f"{mol.name}__prune_s{s}_c{c_idx}_p{i}"
                trials.append((s, c_idx, trial))

    values = api.run_all(
        evaluate='energy',
        mols=[t[2] for t in trials],
        params=params,
        parallel=True,
        ray_params=ray_params,
        robust=True,
    )
    for s, c_idx, trial in trials:
        value = values.get(trial.name)
        if value is None:
            # a failed calc must NOT look like a zero-cost removal -> rank last
            energies[s][c_idx].append(np.nan)
            errors[s][c_idx].append(np.inf)
        else:
            energies[s][c_idx].append(value)
            errors[s][c_idx].append(abs(value - ref_energy))
    ranked_idx, sorted_errors = argsort_inhomogeneous_3d_array(errors)
    return energies, errors, ranked_idx, sorted_errors


def prune_element(mol, element, target, params, parallel=False, ray_params=None):
    """Prunes contraction coefficients to zero, least-important first, until the
    energy rises more than ``target`` above the reference, then reverts the last
    (over-aggressive) prune. ``parallel``/``ray_params`` fan each re-ranking pass's
    per-coefficient trials across the Ray actor pool.
    """
    bo_logger.info(f'Pruning {element} to {target}')
    api.run_calculation(mol=mol, params=params)
    reference_energy = api.get_backend().get_value('energy')
    energy = reference_energy

    shell = None
    old_coefs = None
    idx = exp_idx = None
    while energy < reference_energy + target:
        _, _, ranked_idx, sorted_errors = rank_basis(
            mol, element, params, parallel=parallel, ray_params=ray_params
        )

        # find the least-important coefficient that is not already zeroed
        ang_idx = None
        while ranked_idx:
            ang_idx, idx, exp_idx = ranked_idx.pop(0)
            if sorted_errors.pop(0) != 0.0:
                break
            ang_idx = None
        if ang_idx is None:
            # nothing left to prune
            break

        shell = mol.basis[element.lower()][ang_idx]
        old_coefs = copy.deepcopy(shell.coefs)
        shell.coefs[idx][exp_idx] = 0.0
        bo_logger.info(f'Pruned {shell.l} {idx} {exp_idx}')
        api.run_calculation(mol=mol, params=params)
        energy = api.get_backend().get_value('energy')
        bo_logger.info(f'Energy: {energy}')
        bo_logger.info(f'Target: {reference_energy + target}')
        bo_logger.info(f'Diff: {energy - reference_energy}')

    # revert the last prune that pushed the energy over target, if any was made
    if shell is not None and old_coefs is not None:
        shell.coefs = old_coefs
        bo_logger.info(f'Reverted Prune of {shell.l} {idx} {exp_idx}')
    return mol
