from . import api
from . import bo_logger
from .util import rank_shell_contractions
import copy
import numpy as np

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
    energies = []
    errors = []
    for shell in mol.basis[element.lower()]:
        en, er, ra, sr = rank_shell_contractions(mol, shell, params)
        energies.append(en)
        errors.append(er)
    ranked_idx, sorted_errors = argsort_inhomogeneous_3d_array(errors)
    return energies, errors, ranked_idx, sorted_errors


def prune_element(mol, element, target, params):
    bo_logger.info(f'Pruning {element} to {target}')
    api.run_calculation(mol=mol, params=params)
    reference_energy = api.get_backend().get_value('energy')
    energy = reference_energy
    while energy < reference_energy+target:
        energies, errors, ranked_idx, sorted_errors = rank_basis(mol, element, params)
        ang_idx, idx, exp_idx = ranked_idx.pop(0)
        current_err = sorted_errors.pop(0)
        shell = mol.basis[element.lower()][ang_idx]
        #while shell.coefs[idx][exp_idx] == 0.0:
        while current_err == 0.0:
            ang_idx, idx, exp_idx = ranked_idx.pop(0)
            current_err = sorted_errors.pop(0)
            shell = mol.basis[element.lower()][ang_idx]
        else:
            old_coefs = copy.deepcopy(shell.coefs)
            shell.coefs[idx][exp_idx] = 0.0
            bo_logger.info(f'Pruned {shell.l} {idx} {exp_idx}')
            api.run_calculation(mol=mol, params=params)
            energy = api.get_backend().get_value('energy')
            bo_logger.info(f'Energy: {energy}')
            bo_logger.info(f'Target: {reference_energy+target}')
            bo_logger.info(f'Diff: {energy - reference_energy}')
    else:
        shell.coefs = old_coefs
        bo_logger.info(f'Reverted Prune of {shell.l} {idx} {exp_idx}')
        energy = api.get_backend().get_value('energy')
        bo_logger.info(f'Energy: {energy}')
        bo_logger.info(f'Target: {reference_energy+target}')
        bo_logger.info(f'Diff: {energy - reference_energy}')
    return mol
