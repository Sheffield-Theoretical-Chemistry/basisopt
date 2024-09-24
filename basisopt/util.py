# utility functions
import json
import logging
from typing import Any

import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable

bo_logger = logging.getLogger("basisopt")  # internal logging object


def read_json(filename: str) -> MSONable:
    """Reads an MSONable object from file

    Arguments:
         filename (str): path to JSON file

    Returns:
         object
    """
    with open(filename, "r", encoding="utf-8") as f:
        obj = json.load(f, cls=MontyDecoder)
    bo_logger.info("Read %s from %s", type(obj).__name__, filename)
    return obj


def write_json(filename: str, obj: MSONable):
    """Writes an MSONable object to file

    Arguments:
         filename (str): path to JSON file
         obj (MSONable): object to be written
    """
    obj_type = type(obj).__name__
    if isinstance(obj, MSONable):
        bo_logger.info(f"Writing {obj_type} to {filename}")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(obj, f, cls=MontyEncoder)
    else:
        bo_logger.error("%s cannot be converted to JSON format", obj_type)


def dict_decode(d: dict[str, Any]) -> dict[str, Any]:
    decoder = MontyDecoder()
    return {k: decoder.process_decoded(v) for k, v in d.items()}


def fit_poly(
    x: np.ndarray, y: np.ndarray, n: int = 6
) -> tuple[np.poly1d, float, float, list[float]]:
    """Fits a polynomial of order n to the set of (x [Bohr], y [Hartree]) coordinates given,
    and calculates data necessary for a Dunham analysis.

    Arguments:
         x (numpy array): atomic separations in Bohr
         y (numpy array): energies at each point in Hartree
         n (int): order of polynomial to fit

    Returns:
         poly1d object, reference separation (Bohr), equilibrium separation (Bohr),
         first (n+1) Taylor series coefficients at eq. sep.
    """
    # Find best guess at minimum and shift coordinates
    xref = x[np.argmin(y)]
    xshift = x - xref

    # Fit polynomial to shifted system
    z = np.polyfit(xshift, y, n)
    p = np.poly1d(z)

    # Find the true minimum by interpolation, if possible
    xmin = min(xshift) - 0.1
    xmax = max(xshift) + 0.1
    crit_points = [x.real for x in p.deriv().r if np.abs(x.imag) < 1e-8 and xmin < x.real < xmax]
    if len(crit_points) == 0:
        bo_logger.warning("Minimum not found in polynomial fit")
        # Set outputs to default values
        re = xref
        pt = [0.0] * (n + 1)
    else:
        dx = crit_points[0]
        re = xref + dx  # Equilibrium geometry
        # Calculate 0th - nth Taylor series coefficients at true minimum
        pt = [p.deriv(i)(dx) / np.math.factorial(i) for i in range(n + 1)]

    # Return fitted polynomial, x-shift, equilibrium bond length,
    # and Taylor series coefficients
    return p, xref, re, pt


def format_with_prefix(value: float, unit: str, dp: int = 3) -> str:
    """Utility function for converting a float to scientific notation with units"""
    prefixes = [
        (1e24, 'Y'),
        (1e21, 'Z'),
        (1e18, 'E'),
        (1e15, 'P'),
        (1e12, 'T'),
        (1e9, 'G'),
        (1e6, 'M'),
        (1e3, 'k'),
        (1, ''),
        (1e-3, 'm'),
        (1e-6, 'µ'),
        (1e-9, 'n'),
        (1e-12, 'p'),
        (1e-15, 'f'),
        (1e-18, 'a'),
        (1e-21, 'z'),
        (1e-24, 'y'),
    ]

    # Create the format string dynamically based on the number of decimal places
    format_string = f"{{:.{dp}f}}"

    for factor, prefix in prefixes:
        if abs(value) >= factor:
            formatted_value = value / factor
            return format_string.format(formatted_value) + f" {prefix}{unit}"

    # Handle very small numbers that do not fit any prefix
    return format_string.format(value) + f" {unit}"


def get_composition(basis, element):
    prim_conf = ''.join([f"{len(shell.exps)}{shell.l}" for shell in basis[element]])
    contracted_conf = ''.join([f"{len(shell.coefs)}{shell.l}" for shell in basis[element]])
    if prim_conf != contracted_conf:
        return prim_conf
    else:
        return f"({prim_conf}) -> [{contracted_conf}]"


def inside_out(basis_coefficients, inside_out=True):
        """Performs the inside-out part of the Davidson purification"""
        K = len(basis_coefficients)  # Number of contractions
        M = K - 1  # Number of zero primitives per contraction
        for m in range(M):
            for k in range(m + 1, K):
                ratio = basis_coefficients[k][m] / basis_coefficients[m][m]
                if np.isnan(ratio) or np.isinf(ratio):
                    ratio = 0
                # starting from the first coefficient
                for l in range(k, K):
                    if inside_out:
                        if sum(basis_coefficients[l]) == 1.0:
                            # Ignores any uncontracted shells when doing inside-out
                            basis_coefficients[l] = basis_coefficients[l]
                        else:
                            basis_coefficients[l] -= basis_coefficients[m] * ratio
                            basis_coefficients[l] = np.round(basis_coefficients[l], 8)
                    else:
                        basis_coefficients[l] -= basis_coefficients[m] * ratio
                        basis_coefficients[l] = np.round(basis_coefficients[l], 8)
        return np.round(basis_coefficients, 8)


def outside_in(basis_coefficients):
	"""Does the outside-in part of the Davidson purification"""
	return list(np.flip(inside_out(np.flip(basis_coefficients))))

def davidson_purification(basis_coefficients):
	"""Performs the Davidson purification"""
	basis_coefficients = outside_in(inside_out(basis_coefficients, True))
	return basis_coefficients


def davidson_purify_basis(basis):
    """A function to perform Davidson purification on a basis set
    Args:
        basis InternalBasis: Davidson purified basis set
    """

    for element in basis:
        for shell in basis[element]:
            shell.coefs = davidson_purification(shell.coefs)
    return basis

def davidson_purify_extended(basis):
    """Extended Davidson purification.
    Removes any uncontracted functions from lower contractions.
    Removes any zero coefficients, purifies the remaining coefficients and restores the zeros.

    Args:
        basis (InternalBasis): Internal basis set dictionary
    """
	
    def uncontract_contractions(coefs):
        """Remove uncontracted functions from a basis set"""
        def has_single_func(arr):
            return np.count_nonzero(arr == 1) == 1


        def get_uncontracted_index(arr):
            indices = np.where(arr == 1)[0]  # Get the indices where the value is 1
            if len(indices) == 1:
                return indices[0]  # Return the index if there's exactly one 1
            return None  # Return None if not exactly one 1
        
        coefs = coefs[::-1]
        uncontracted = []
        for coef in coefs:
            if has_single_func(coef):
                index = get_uncontracted_index(coef)
                uncontracted.append(coef.copy())
                for coef in coefs:
                    coef[index] = 0
                coefs = coefs[1:]
        return coefs[::-1], uncontracted[::-1]

    def remove_zeros(arrays):
        no_zeros_arrays = []
        zero_indices_list = []
        original_length = len(arrays[0])

        for arr in arrays:
            zero_indices = np.where(arr == 0)[0]  # Get the indices where the zeros are
            non_zero_elements = arr[arr != 0]  # Remove zeros from the array

            no_zeros_arrays.append(non_zero_elements)
            zero_indices_list.append(zero_indices)

        return no_zeros_arrays, zero_indices_list, original_length

    # Function to restore zeros to their original positions
    def restore_zeros(modified_arrays, zero_indices_list, original_length):
        restored_arrays = []

        for modified_arr, zero_indices in zip(modified_arrays, zero_indices_list):
            # Create a new array of the original length, filled with the modified non-zero values
            restored_array = np.zeros(original_length)

            # Fill the non-zero positions in the array
            non_zero_indices = np.setdiff1d(np.arange(original_length), zero_indices)
            restored_array[non_zero_indices] = modified_arr

            # Append the restored array to the list
            restored_arrays.append(restored_array)

        return restored_arrays

    def purify_reduced_coefs_new(coefs):
        # Remove zeros from the coefs
        removed_uncontracted, uncontracted_funcs = uncontract_contractions(coefs)

        no_zeros_coefs, zero_positions_list, original_length = remove_zeros(removed_uncontracted)

        # Purify the reduced coefs
        purified_coefs = davidson_purification(no_zeros_coefs)

        # Restore zeros to the purified coefs
        restored_coefs = restore_zeros(purified_coefs, zero_positions_list, original_length)

        return restored_coefs + uncontracted_funcs

    for element in basis:
        for shell in basis[element]:
            if len(shell.coefs) != len(shell.exps):
                shell.coefs = purify_reduced_coefs_new(shell.coefs)
    return basis

def rank_basis_contraction(mol, element):
    """ Rank basis functions energy contributions

    Args:
        mol (Molecule): BasisOpt Molecule object
        element (InternalBasis): Element to prune in basis set

    Returns:
        energies (list): Energy after removing each contraction coefficient
        errors (list): Energy delta after removing each contraction coefficient
        ranked_idx (list): Ranked indices of contractions
        sorted_errors (list): Sorted error contributions
    """
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
    
    energies = []
    errors = []
    for shell in mol.basis[element]:
        en, er, ra, sr = rank_shell_contractions(mol, shell)
        energies.append(en)
        errors.append(er)
    ranked_idx, sorted_errors = argsort_inhomogeneous_3d_array(errors)
    return energies, errors, ranked_idx, sorted_errors


def prune_element(mol, element, target):
    """Prune a basis set element to reach a target energy difference

    Args:
        mol (InternalBasis): BasisOpt Molecule object
        element (str): Element to prune in basis set
        target (float): Maximum 

    Returns:
        _type_: _description_
    """
    mol.name = f'{element}pruned{int(target*1000)}'
    bo.run_calculation(mol=mol, params=params)
    reference_energy = wrapper.get_value('energy')
    energy = reference_energy
    while energy < reference_energy+target:
        energies, errors, ranked_idx, sorted_errors = rank_basis(mol, element)
        ang_idx, idx, exp_idx = ranked_idx.pop(0)
        shell = mol.basis[element][ang_idx]
        bo_logger.info(f'Pruning {element} {shell.l} contractions')
        while shell.coefs[idx][exp_idx] == 0.0:
            ang_idx, idx, exp_idx = ranked_idx.pop(0)
            shell = mol.basis[element][ang_idx]
        else:
            old_coefs = copy.deepcopy(shell.coefs)
            shell.coefs[idx][exp_idx] = 0.0
            bo_logger.info(f'Pruned {shell.l} {idx} {exp_idx}')
            bo.run_calculation(mol=mol, params=params)
            energy = wrapper.get_value('energy')
            bo_logger.info(f'Energy: {energy}')
            bo_logger.info(f'Target: {reference_energy+target}')
            bo_logger.info(f'Diff: {energy - reference_energy}')
    else:
        shell.coefs = old_coefs
        bo_logger.info(f'Reverted Prune of {shell.l} {idx} {exp_idx}')
        energy = wrapper.get_value('energy')
        bo_logger.info(f'Energy: {energy}')
        bo_logger.info(f'Target: {reference_energy+target}')
        bo_logger.info(f'Diff: {energy - reference_energy}')
    return mol


