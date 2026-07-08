# functionality to rank basis shells
import copy
from typing import Optional

import numpy as np

from basisopt import api
from basisopt.basis import uncontract_shell
from basisopt.basis.atomic import AtomicBasis
from basisopt.exceptions import FailedCalculation
from basisopt.molecule import Molecule


def _drop_each_exponent(mol, shell, eval_type, params, reference):
    """Remove each exponent of ``shell`` in turn and evaluate ``eval_type``.

    For each exponent i, the shell is rebuilt without it, a calculation is run
    on ``mol``, and the absolute difference of the result from ``reference`` is
    recorded. The shell's original exps/coefs are restored before returning.

    Arguments:
        mol (Molecule): the molecule to run calculations on
        shell (Shell): the shell whose exponents are ranked (mutated then restored)
        eval_type (str): property to evaluate
        params (dict): backend parameters
        reference (float): baseline to difference each result against (the
            full-basis value for rank_primitives, the CBS limit for
            rank_mol_basis_cbs)

    Returns:
        (err, energies) where err[i] = |value_without_exponent_i - reference|
        and energies[i] is the raw evaluated value

    Raises:
        FailedCalculation
    """
    exps = shell.exps.copy()
    coefs = shell.coefs.copy()
    n = len(exps)

    # make uncontracted with one fewer exponent
    shell.exps = np.zeros(n - 1)
    uncontract_shell(shell)
    err = np.zeros(n)
    energies = []

    # remove each exponent one at a time
    for i in range(n):
        shell.exps[:i] = exps[:i]
        shell.exps[i:] = exps[i + 1 :]
        if api.run_calculation(evaluate=eval_type, mol=mol, params=params) != 0:
            raise FailedCalculation
        value = api.get_backend().get_value(eval_type)
        energies.append(value)
        err[i] = np.abs(value - reference)

    # reset shell to original
    shell.exps = exps
    shell.coefs = coefs
    return err, energies


def rank_primitives(
    atomic: AtomicBasis,
    shells: Optional[list[int]] = None,
    eval_type: str = "energy",
    basis_type: str = "orbital",
    params=None,
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
    params = {} if params is None else params
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
        err, _ = _drop_each_exponent(mol, basis[s], eval_type, params, reference)
        errors.append(err)
        ranks.append(np.argsort(err))

    return errors, ranks


def rank_mol_basis_cbs(
    mol: Molecule,
    element: str,
    cbs_limit: float,
    eval_type: str = 'energy',
    backend_params: dict = None,
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
        errors (list): List of difference to the CBS limit for each primitive function
        ranks (list): List of ranks for each primitive function
        energies (list): List of energies for each primitive function
        dE_CBS_INITIAL (float): Initial difference to the CBS limit
    """
    backend_params = {} if backend_params is None else backend_params
    element = element.lower()
    if api.run_calculation(evaluate=eval_type, mol=mol, params=backend_params) != 0:
        raise FailedCalculation
    new_mol = copy.deepcopy(mol)
    reference_energy = api.get_backend().get_value(eval_type)
    dE_CBS_INITIAL = reference_energy - cbs_limit
    errors = []
    ranks = []
    energies = []

    for shell in new_mol.basis[element]:
        # rank each exponent by how close removing it leaves us to the CBS limit
        err, ens = _drop_each_exponent(new_mol, shell, eval_type, backend_params, cbs_limit)
        errors.append(err)
        ranks.append(np.argsort(err))
        energies.append(ens)

    return errors, ranks, energies, dE_CBS_INITIAL
