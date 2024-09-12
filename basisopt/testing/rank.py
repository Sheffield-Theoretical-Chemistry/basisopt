# funcitonality to rank basis shells
import copy
from typing import Any, Optional

import numpy as np
from mendeleev import element as md_element

from basisopt import api
from basisopt.basis import uncontract_shell
from basisopt.basis.atomic import AtomicBasis
from basisopt.containers import InternalBasis
from basisopt.exceptions import FailedCalculation
from basisopt.molecule import Molecule
from basisopt.util import bo_logger


def rank_primitives(
    atomic: AtomicBasis,
    shells: Optional[list[int]] = None,
    eval_type: str = "energy",
    basis_type: str = "orbital",
    params={},
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Systematically eliminates exponents from shells in an AtomicBasis
    to determine how much they contribute to the target property

    Arguments:
         atomic: AtomicBasis object
         shells (list): list of indices for shells in the AtomicBasis
             to be ranked. If None, will rank all shells
         eval_type (str): property to evaluate (e.g. energy)
         basis_type (str): "orbital/jfit/jkfit"
         params (dict): parameters  to pass to the backend,
                 see relevant Wrapper for options

    Returns:
         (errors, ranks), where errors is a list of numpy arrays with the
         change in target property value for each exponent in the shell,
         and ranks is a list of numpy arrays which contain the indices of
         each exponent in each shell from smallest to largest error value.
         Order of errors, ranks is same as order of shells

    Raises:
         FailedCalculation
    """
    mol = copy.copy(atomic._molecule)
    if basis_type == "jfit":
        basis = mol.jbasis[atomic._symbol]
    elif basis_type == "jkfit":
        basis = mol.jkbasis[atomic._symbol]
    else:
        basis = mol.basis[atomic._symbol]

    if not shells:
        shells = list(range(len(basis)))  # do all

    # Calculate reference value
    if api.run_calculation(evaluate=eval_type, mol=mol, params=params) != 0:
        raise FailedCalculation
    reference = api.get_backend().get_value(eval_type)
    # prefix result  as being for ranking
    atomic._molecule.add_reference("rank_" + eval_type, reference)

    errors = []
    ranks = []
    for s in shells:
        shell = basis[s]
        # copy old parameters
        exps = shell.exps.copy()
        coefs = shell.coefs.copy()
        n = len(exps)

        # make uncontracted
        shell.exps = np.zeros(n - 1)
        uncontract_shell(shell)
        err = np.zeros(n)

        # remove each exponent one at a time
        for i in range(n):
            shell.exps[:i] = exps[:i]
            shell.exps[i:] = exps[i + 1 :]
            success = api.run_calculation(evaluate=eval_type, mol=mol, params=params)
            if success != 0:
                raise FailedCalculation
            value = api.get_backend().get_value(eval_type)
            err[i] = np.abs(value - reference)

        errors.append(err)
        ranks.append(np.argsort(err))
        # reset shell to original
        shell.exps = exps
        shell.coefs = coefs

    return errors, ranks


def reduce_primitives(
    atomic: AtomicBasis,
    thresh: float = 1e-4,
    shells: Optional[list[int]] = None,
    eval_type: str = "energy",
    params: dict[str, Any] = {},
) -> tuple[InternalBasis, Any]:
    """Rank the primitive functions in an atomic basis, and remove those that contribute
    less than a threshold. TODO: add checking that does not go below minimal config

    Arguments:
         atomic: AtomicBasis object
         thresh (float): if a primitive's contribution to the target is < thresh,
         it is removed from the basis
         shells (list): list of indices of shells to be pruned; if None, does all shells
         eval_type (str): property to evaluate
         params (dict): parameters to pass to the backend

    Returns:
         (basis, delta) where basis is the pruned basis set (this is non-destructive to the
         original AtomicBasis), and delta is the change in target property with the pruned
         basis compared to the original

    Raises:
         FailedCalculation
    """
    mol = copy.copy(atomic._molecule)
    basis = mol.basis[atomic.symbol]
    if not shells:
        shells = list(range(len(basis)))  # do all
    # first rank the primitives
    errors, ranks = rank_primitives(atomic, shells=shells, eval_type=eval_type, params=params)

    # now reduce
    for s, e, r in zip(shells, errors, ranks):
        shell = basis[s]
        n = shell.exps.size
        start = 0
        value = e[r[0]]
        while (start < n - 1) and (value < thresh):
            start += 1
            bo_logger.debug("%.2e, %s, %d", e, str(r), start)
            value = e[r[start]]

        if start == (n - 1):
            bo_logger.warning("Shell %d with l=%d now empty", s, shell.l)
            shell.exps = []
            shell.coefs = []
        else:
            shell.exps = shell.exps[r[start:]]
            uncontract_shell(shell)

    success = api.run_calculation(evaluate=eval_type, mol=mol, params=params)
    if success != 0:
        raise FailedCalculation
    result = api.get_backend().get_value(eval_type)
    delta = result - atomic._molecule.get_reference("rank_" + eval_type)

    return mol.basis, delta


def rank_molecule_primitives(
    molecule: Molecule,
    shells: Optional[list[int]] = None,
    eval_type: str = "energy",
    basis_type: str = "orbital",
    params={},
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Systematically eliminates exponents from shells in an AtomicBasis
    to determine how much they contribute to the target property

    Arguments:
         atomic: AtomicBasis object
         shells (list): list of indices for shells in the AtomicBasis
             to be ranked. If None, will rank all shells
         eval_type (str): property to evaluate (e.g. energy)
         basis_type (str): "orbital/jfit/jkfit"
         params (dict): parameters  to pass to the backend,
                 see relevant Wrapper for options

    Returns:
         (errors, ranks), where errors is a list of numpy arrays with the
         change in target property value for each exponent in the shell,
         and ranks is a list of numpy arrays which contain the indices of
         each exponent in each shell from smallest to largest error value.
         Order of errors, ranks is same as order of shells

    Raises:
         FailedCalculation
    """
    mol = copy.copy(atomic._molecule)
    if basis_type == "jfit":
        basis = mol.jbasis[atomic._symbol]
    elif basis_type == "jkfit":
        basis = mol.jkbasis[atomic._symbol]
    else:
        basis = mol.basis[atomic._symbol]

    if not shells:
        shells = list(range(len(basis)))  # do all

    # Calculate reference value
    if api.run_calculation(evaluate=eval_type, mol=mol, params=params) != 0:
        raise FailedCalculation
    reference = api.get_backend().get_value(eval_type)
    # prefix result  as being for ranking
    atomic._molecule.add_reference("rank_" + eval_type, reference)

    errors = []
    ranks = []
    for s in shells:
        shell = basis[s]
        # copy old parameters
        exps = shell.exps.copy()
        coefs = shell.coefs.copy()
        n = len(exps)

        # make uncontracted
        shell.exps = np.zeros(n - 1)
        uncontract_shell(shell)
        err = np.zeros(n)

        # remove each exponent one at a time
        for i in range(n):
            shell.exps[:i] = exps[:i]
            shell.exps[i:] = exps[i + 1 :]
            success = api.run_calculation(evaluate=eval_type, mol=mol, params=params)
            if success != 0:
                raise FailedCalculation
            value = api.get_backend().get_value(eval_type)
            err[i] = np.abs(value - reference)

        errors.append(err)
        ranks.append(np.argsort(err))
        # reset shell to original
        shell.exps = exps
        shell.coefs = coefs

    return errors, ranks


def rank_mol_basis_cbs(
    mol: Molecule,
    element: str,
    cbs_limit: float,
    eval_type: str = 'energy',
    backend_params: dict = {},
):
    """Rank the primitive functions in a basis.

    Args:
        mol (Molecule): Molecule containing basis set
        element (str): Element in basis to be ranked
        cbs_limit (float): CBS limit for the molecule
        eval_type (str, optional): Molecule property to evaluate. Defaults to 'energy'.
        backend_params (dict, optional): Parameters to pass to the backend.

    Raises:
        FailedCalculation: Failed calculation, check backend parameters if this occurs.

    Returns:
        errors (list): List of difference to the DFT CBS limit for each primitive function
        ranks (list): List of ranks for each primitive function
        energies (list): List of energies for each primitive function
        dE_CBS_INITIAL (float): Initial difference to the DFT CBS limit
    """
    element = element.lower()
    if api.run_calculation(evaluate=eval_type, mol=mol, params=backend_params) != 0:
        raise Exception('Failed calculation')
    new_mol = copy.deepcopy(mol)
    reference_energy = api.get_backend().get_value(eval_type)
    dE_CBS_INITIAL = reference_energy - cbs_limit
    errors = []
    ranks = []
    energies = []

    for shell in new_mol.basis[element]:
        # copy old parameters
        exps = shell.exps.copy()
        coefs = shell.coefs.copy()
        n = len(exps)

        shell.exps = np.zeros(n - 1)
        uncontract_shell(shell)
        err = np.zeros(n)

        # remove each exponent one at a time
        ens = []
        for i in range(n):
            shell.exps[:i] = exps[:i]
            shell.exps[i:] = exps[i + 1 :]
            success = api.run_calculation(evaluate=eval_type, mol=new_mol, params=backend_params)
            if success != 0:
                raise FailedCalculation
            value = api.get_backend().get_value(eval_type)
            ens.append(value)
            err[i] = np.abs(value - cbs_limit)

        errors.append(err)
        ranks.append(np.argsort(err))
        energies.append(ens)
        # reset shell to original
        shell.exps = exps
        shell.coefs = coefs

    return errors, ranks, energies, dE_CBS_INITIAL


def rank_mol_basis(
    mol: Molecule,
    element: str,
    eval_type: str = 'energy',
    backend_params: dict = {},
):
    """Rank the primitive functions in a basis.

    Args:
        mol (Molecule): Molecule containing basis set
        element (str): Element in basis to be ranked
        eval_type (str, optional): Molecule property to evaluate. Defaults to 'energy'.
        backend_params (dict, optional): Parameters to pass to the backend.

    Raises:
        FailedCalculation: Failed calculation, check backend parameters if this occurs.

    Returns:
        errors (list): List of difference to the DFT CBS limit for each primitive function
        ranks (list): List of ranks for each primitive function
        energies (list): List of energies for each primitive function
        dE_CBS_INITIAL (float): Initial difference to the DFT CBS limit
    """
    element = element.lower()
    if api.run_calculation(evaluate=eval_type, mol=mol, params=backend_params) != 0:
        raise Exception('Failed calculation')
    new_mol = copy.deepcopy(mol)
    reference_energy = api.get_backend().get_value(eval_type)
    errors = []
    ranks = []
    energies = []

    for shell in new_mol.basis[element]:
        # copy old parameters
        exps = shell.exps.copy()
        coefs = shell.coefs.copy()
        n = len(exps)

        shell.exps = np.zeros(n - 1)
        uncontract_shell(shell)
        err = np.zeros(n)

        # remove each exponent one at a time
        ens = []
        for i in range(n):
            shell.exps[:i] = exps[:i]
            shell.exps[i:] = exps[i + 1 :]
            success = api.run_calculation(evaluate=eval_type, mol=new_mol, params=backend_params)
            if success != 0:
                raise FailedCalculation
            value = api.get_backend().get_value(eval_type)
            ens.append(value)
            err[i] = np.abs(value - reference_energy)

        errors.append(err)
        ranks.append(np.argsort(err))
        energies.append(ens)
        # reset shell to original
        shell.exps = exps
        shell.coefs = coefs

    return errors, ranks, energies


def rank_basis(
    mol: Molecule,
    eval_type: str = 'energy',
    backend_params: dict = {},
):
    """Rank the primitive functions in a basis."""
    element_ranks = {}
    for element in mol.basis:
        errors, ranks, energies = rank_mol_basis(
            mol, element, eval_type, backend_params
        )
        element_ranks[element] = {
            'errors': errors,
            'ranks': ranks,
            'energies': energies,
        }
    
    return element_ranks

def get_smallest_error_element(element_ranks):
    """Returns the element with the smallest error value"""
    smallest_error = float('inf')
    smallest_element = None
    for element, ranks in element_ranks.items():
        errors = ranks['errors']
        for error in errors:
            min_error = np.min(error)
            if min_error < smallest_error:
                smallest_error = min_error
                smallest_element = element
    return smallest_element
