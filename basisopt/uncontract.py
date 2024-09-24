from . import api
#from .api import _CURRENT_BACKEND as wrapper
from .api import bo_logger

import numpy as np
import copy


def argsort_inhomogeneous_array(array):
    """
    Argsorts an inhomogeneous array globally while keeping dimensional information.

    Parameters:
    array: list of lists
        A 2D inhomogeneous array where each sublist can have a different length.

    Returns:
    ranked_indices: list of tuples
        A list of tuples where each tuple represents (sublist index, element index)
        sorted in ascending order of the array values.
    sorted_values: list
        The corresponding values of the array in ascending order.
    """
    # Flatten the array while keeping track of original indices
    flat_array = []
    index_mapping = []

    # Loop through each sublist and element to build flattened array and index map
    for i, sublist in enumerate(array):
        for j, element in enumerate(sublist):
            flat_array.append(element)
            index_mapping.append((i, j))

    # Use numpy argsort to sort the flat array
    sorted_indices = np.argsort(flat_array)

    # Generate the ranked indices and sorted values
    ranked_indices = [index_mapping[idx] for idx in sorted_indices]
    sorted_values = [flat_array[idx] for idx in sorted_indices]

    return ranked_indices, sorted_values


def rank_uncontract_element(mol, element, params, verbose=False):
    """Rank the uncontracted functions of an element by energy contribution.
    This will fail if uncontracting a function results in a linear dependency.

    Args:
        mol (Molecule): BasisOpt Molecule object
        element (str): Element to rank in basis set
        verbose (bool, optional): Print rankings. Defaults to False.
    """
    wrapper = api.get_backend()
    def rank_uncontract_angular_momentum(mol, shell, verbose=True):
        energies = []
        errors = []
        n_exps = len(shell.exps)
        api.run_calculation(mol=mol, params=params)
        ref_energy = wrapper.get_value('energy')
        for i in range(n_exps):
            old_coefs = copy.deepcopy(shell.coefs)
            new_coefs = np.zeros(n_exps)
            new_coefs[i] = 1.0
            if any(np.array_equal(new_coefs, coef) for coef in shell.coefs):
                energies.append(0)
                errors.append(0)
                continue
            shell.coefs.append(new_coefs)
            # print(shell.coefs)
            api.run_calculation(mol=mol, params=params)
            energies.append(wrapper.get_value('energy'))
            errors.append(abs(energies[-1] - ref_energy))
            shell.coefs = old_coefs
            if verbose:
                bo_logger.info(
                    f'Ranking {shell.l} {i}:\n Energy {energies[-1]}\n Delta: {errors[-1]}'
                )
        ranks = np.argsort(errors)
        return energies, errors, ranks

    energies = []
    errors = []
    ranks = []
    for shell in mol.basis[element.lower()]:
        en, er, ra = rank_uncontract_angular_momentum(mol, shell, verbose)
        energies.append(en)
        errors.append(er)
        ranks.append(ra)
    ranked_idx, sorted_errors = argsort_inhomogeneous_array(errors)
    return energies, errors, ranks, ranked_idx, sorted_errors


def rank_uncontract_element_robust(mol, element, params, verbose=False):
    """Rank the uncontracted functions of an element by energy contribution.
    This one is more robust than the previous one, as it will not fail if uncontracting a function results in a linear dependency.

    Args:
        mol (Molecule): BasisOpt Molecule object
        element (str): Element to rank in basis set
        verbose (bool, optional): Print rankings. Defaults to False.
    """
    wrapper = api.get_backend()
    def rank_uncontract_angular_momentum_robust(mol, shell, verbose=True):
        energies = []
        errors = []
        n_exps = len(shell.exps)
        api.run_calculation(mol=mol, params=params)
        ref_energy = wrapper.get_value('energy')
        for i in range(n_exps):
            old_coefs = copy.deepcopy(shell.coefs)
            new_coefs = np.zeros(n_exps)
            new_coefs[i] = 1.0
            if any(np.array_equal(new_coefs, coef) for coef in shell.coefs):
                energies.append(0)
                errors.append(0)
                continue
            shell.coefs.append(new_coefs)
            # print(shell.coefs)
            try:
                api.run_calculation(mol=mol, params=params)
                energies.append(wrapper.get_value('energy'))
                errors.append(abs(energies[-1] - ref_energy))
            except Exception as e:
                bo_logger.error(f'Failed to calculate {shell.l} {i}: {e}')
                energies.append(-1)
                errors.append(-1)
            shell.coefs = old_coefs
            if verbose:
                bo_logger.info(
                    f'Ranking {shell.l} {i}:\n Energy {energies[-1]}\n Delta: {errors[-1]}'
                )
        ranks = np.argsort(errors)
        return energies, errors, ranks

    energies = []
    errors = []
    ranks = []
    for shell in mol.basis[element.lower()]:
        en, er, ra = rank_uncontract_angular_momentum_robust(mol, shell, verbose)
        energies.append(en)
        errors.append(er)
        ranks.append(ra)
    ranked_idx, sorted_errors = argsort_inhomogeneous_array(errors)
    return energies, errors, ranks, ranked_idx, sorted_errors


def add_uncontracted_functions(mol, element, params, target, verbose=False):
    """Add uncontracted functions to a basis set element until the energy difference is below a target.

    Args:
        mol (Molecule): BasisOpt Molecule object.
        element (str): Element to uncontract in basis set.
        target (float): Maximum energy difference to reach.
    """
    wrapper = api.get_backend()
    def sort_by_length_and_nonzero_index(array_list):
        def sort_key(array):
            length = len(array)
            non_zero_indices = np.nonzero(array)[0]
            first_non_zero_index = (
                non_zero_indices[0] if non_zero_indices.size > 0 else float('inf')
            )
            return (length, first_non_zero_index)

        sorted_list = sorted(array_list, key=sort_key)
        return sorted_list

    api.run_calculation(mol=mol, params=params)
    energy = wrapper.get_value('energy')
    reference_energy = energy
    uncontracted_functions = []
    while energy > reference_energy - target:
        energies, errors, ranks, ranked_idx, sorted_errors = rank_uncontract_element(
            mol, element, params, verbose
        )
        angular_momentum, exp_idx = ranked_idx.pop(-1)
        while (angular_momentum, exp_idx) in uncontracted_functions:
            angular_momentum, exp_idx = ranked_idx.pop(-1)
        uncontracted_functions.append((angular_momentum, exp_idx))
        shell = mol.basis[element.lower()][angular_momentum]
        shell.coefs.append(np.zeros(len(shell.exps)))
        shell.coefs[-1][exp_idx] = 1.0
        shell.coefs = sort_by_length_and_nonzero_index(shell.coefs)
        # print(f'{shell.l}: {shell.coefs}')
        api.run_calculation(mol=mol, params=params)
        energy = wrapper.get_value('energy')
    return uncontracted_functions


def add_uncontracted_functions_cutoff(mol, element, params, cutoff, verbose=False):
    """Uncontracts all functions that have a contribution above a cutoff.
    This performs inplace modification of the basis set and returns the uncontracted functions.

    Args:
        mol (Molecule): BasisOpt Molecule object.
        element (str): Element to uncontract in basis set.
        cutoff (float): Maximum energy contribution to keep the function contracted.
    Returns:
        list: List of tuples with the angular momentum and exponent index of the uncontracted functions.
    """
    wrapper = api.get_backend()
    def sort_by_length_and_nonzero_index(array_list):
        def sort_key(array):
            length = len(array)
            non_zero_indices = np.nonzero(array)[0]
            first_non_zero_index = (
                non_zero_indices[0] if non_zero_indices.size > 0 else float('inf')
            )
            return (length, first_non_zero_index)

        sorted_list = sorted(array_list, key=sort_key)
        return sorted_list

    bo_logger.info(f'Uncontracting coefficients for {element} with cutoff {cutoff}.')

    api.run_calculation(mol=mol, params=params)
    energy = wrapper.get_value('energy')
    reference_energy = energy
    uncontracted_functions = []
    uncontract = True
    while uncontract:
        energies, errors, ranks, ranked_idx, sorted_errors = rank_uncontract_element_robust(
            mol,
            element,
            params,
            verbose,
        )
        angular_momentum, exp_idx = ranked_idx.pop(-1)
        contribution = sorted_errors.pop(-1)
        # if (angular_momentum, exp_idx) in uncontracted_functions:
        #     angular_momentum, exp_idx = ranked_idx.pop(-1)
        #     contribution = sorted_errors.pop(-1)
        if contribution > cutoff:
            bo_logger.info(f'Uncontracting {element} {angular_momentum} {exp_idx}')
            uncontracted_functions.append((angular_momentum, exp_idx))
            shell = mol.basis[element.lower()][angular_momentum]
            shell.coefs.append(np.zeros(len(shell.exps)))
            shell.coefs[-1][exp_idx] = 1.0
            shell.coefs = sort_by_length_and_nonzero_index(shell.coefs)
            api.run_calculation(mol=mol, params=params)
            energy = wrapper.get_value('energy')
        else:
            uncontract = False
    return uncontracted_functions
