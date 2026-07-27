# functionality to rank basis shells
import copy
from typing import Optional

import numpy as np

from basisopt import api
from basisopt.basis import uncontract_shell
from basisopt.basis.atomic import AtomicBasis
from basisopt.exceptions import FailedCalculation
from basisopt.molecule import Molecule


def _rank_exponents(
    mol,
    element,
    shells_idx,
    eval_type,
    params,
    reference,
    parallel=False,
    ray_params=None,
    basis_attr="basis",
):
    """Rank every exponent of the given shells by how much removing it changes
    ``eval_type`` (relative to ``reference``).

    Builds one independent trial molecule per exponent (that exponent dropped
    from its shell) and evaluates them via :func:`api.run_all`: serially by
    default, or fanned across the warm actor pool when ``parallel`` is set. The
    trials are identical either way -- parallelism only changes wall-clock, not
    the ranking -- which is why the ranking steps stay behaviour-preserving.

    Arguments:
        mol (Molecule): the molecule whose basis is ranked (not mutated)
        element (str): element key in ``mol.basis``
        shells_idx (list[int]): indices of the shells to rank
        eval_type (str): property to evaluate
        params (dict): backend parameters
        reference (float): baseline to difference each result against (the
            full-basis value for rank_primitives, the CBS limit for
            rank_mol_basis_cbs)
        parallel (bool): fan the trial calcs across Ray
        ray_params (dict): Ray settings (backend/tmp_dir/threads_per_job/...)

    Returns:
        (errors, energies), each a list aligned with ``shells_idx``; ``errors``
        entries are numpy arrays with ``|value_without_exponent_i - reference|``.

    Raises:
        FailedCalculation
    """
    trials = []  # (position in shells_idx, exponent index, trial molecule)
    for pos, s in enumerate(shells_idx):
        base_exps = getattr(mol, basis_attr)[element][s].exps
        for i in range(len(base_exps)):
            trial = copy.deepcopy(mol)
            tshell = getattr(trial, basis_attr)[element][s]
            tshell.exps = np.delete(base_exps, i)
            uncontract_shell(tshell)
            trial.name = f"{mol.name}__rank_s{s}_e{i}"  # unique -> run_all key
            trials.append((pos, i, trial))

    values = api.run_all(
        evaluate=eval_type,
        mols=[t[2] for t in trials],
        params=params,
        parallel=parallel,
        ray_params=ray_params,
    )

    errors = [np.zeros(len(getattr(mol, basis_attr)[element][s].exps)) for s in shells_idx]
    energies = [np.zeros(len(getattr(mol, basis_attr)[element][s].exps)) for s in shells_idx]
    for pos, i, trial in trials:
        value = values.get(trial.name)
        if value is None:
            raise FailedCalculation
        errors[pos][i] = np.abs(value - reference)
        energies[pos][i] = value
    return errors, energies


def rank_primitives(
    atomic: AtomicBasis,
    shells: Optional[list[int]] = None,
    eval_type: str = "energy",
    basis_type: str = "orbital",
    params=None,
    parallel: bool = False,
    ray_params: dict = None,
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
    basis_attr = {"jfit": "jbasis", "jkfit": "jkbasis"}.get(basis_type, "basis")
    basis = getattr(mol, basis_attr)[atomic._symbol]

    if not shells:
        shells = list(range(len(basis)))  # do all

    # Calculate reference value (full basis)
    if api.run_calculation(evaluate=eval_type, mol=mol, params=params) != 0:
        raise FailedCalculation
    reference = api.get_backend().get_value(eval_type)
    # prefix result  as being for ranking
    atomic._molecule.add_reference("rank_" + eval_type, reference)

    errors, _ = _rank_exponents(
        mol,
        atomic._symbol,
        shells,
        eval_type,
        params,
        reference,
        parallel=parallel,
        ray_params=ray_params,
        basis_attr=basis_attr,
    )
    ranks = [np.argsort(err) for err in errors]
    return errors, ranks


def rank_mol_basis_cbs(
    mol: Molecule,
    element: str,
    cbs_limit: float,
    eval_type: str = 'energy',
    backend_params: dict = None,
    parallel: bool = False,
    ray_params: dict = None,
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

    # rank each exponent by how close removing it leaves us to the CBS limit
    shells_idx = list(range(len(new_mol.basis[element])))
    errors, energies = _rank_exponents(
        new_mol,
        element,
        shells_idx,
        eval_type,
        backend_params,
        cbs_limit,
        parallel=parallel,
        ray_params=ray_params,
    )
    ranks = [np.argsort(err) for err in errors]

    return errors, ranks, energies, dE_CBS_INITIAL
