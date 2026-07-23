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


def rank_basis(mol, element, params):
    """Rank every contraction coefficient in ``element``'s basis by importance.

    Distinct from ``testing.rank``'s exponent-dropping ranking: this path zeroes
    contraction *coefficients* (via ``util.rank_shell_contractions``) and is used
    by ``prune_element``. Returns ``(energies, errors, ranked_idx, sorted_errors)``;
    ``prune_element`` consumes only the last two.
    """
    # one reference for all shells in this pass (each trial restores coefs)
    api.run_calculation(mol=mol, params=params)
    ref_energy = api.get_backend().get_value('energy')
    energies = []
    errors = []
    for shell in mol.basis[element.lower()]:
        en, er, ra, sr = rank_shell_contractions(mol, shell, params, ref_energy=ref_energy)
        energies.append(en)
        errors.append(er)
    ranked_idx, sorted_errors = argsort_inhomogeneous_3d_array(errors)
    return energies, errors, ranked_idx, sorted_errors


def prune_element(mol, element, target, params):
    """Prunes contraction coefficients to zero, least-important first, until the
    energy rises more than ``target`` above the reference, then reverts the last
    (over-aggressive) prune.
    """
    bo_logger.info(f'Pruning {element} to {target}')
    api.run_calculation(mol=mol, params=params)
    reference_energy = api.get_backend().get_value('energy')
    energy = reference_energy

    shell = None
    old_coefs = None
    idx = exp_idx = None
    while energy < reference_energy + target:
        _, _, ranked_idx, sorted_errors = rank_basis(mol, element, params)

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
