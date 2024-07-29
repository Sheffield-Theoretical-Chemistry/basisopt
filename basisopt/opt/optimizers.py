from typing import Any, Callable, Optional

import numpy as np
from mendeleev import element as md_element
from scipy.optimize import minimize

from basisopt import api
from basisopt.containers import InternalBasis, OptCollection, OptResult
from basisopt.data import _ATOMIC_DFT_CBS, INV_AM_DICT
from basisopt.exceptions import FailedCalculation
from basisopt.molecule import Molecule
from basisopt.util import bo_logger, format_with_prefix, get_composition

from .contraction import ContractionStrategy
from .regularisers import Regulariser
from .strategies import Strategy


def _atomic_opt(
    basis: InternalBasis,
    element: str,
    algorithm: str,
    strategy: Strategy,
    opt_params: dict[str, Any],
    objective: Callable[[np.ndarray], float],
) -> OptResult:
    """Helper function to run a strategy for a single atom

    Arguments:
         basis: internal basis dictionary
         element: symbol of atom to be optimized
         algorithm (str): optimization algorithm, see scipy.optimize for options
         opt_params (dict): parameters to pass to scipy.optimize.minimize
         objective (func): function to calculate objective, must have signature
             func(x) where x is a 1D numpy array of floats

     Returns:
         a dictionary of scipy.optimize result objects for each step in the opt
    """
    bo_logger.info("Starting optimization of %s/%s", element, strategy.eval_type)
    bo_logger.info("Algorithm: %s, Strategy: %s", algorithm, strategy.name)
    objective_value = objective(strategy.get_active(basis, element))
    bo_logger.info("Initial objective value: %f", objective_value)

    # Keep going until strategy says stop
    results = {}
    ctr = 1
    while strategy.next(basis, element, objective_value):
        bo_logger.info("Doing step %d", strategy._step + 1)
        guess = strategy.get_active(basis, element)
        if len(guess) > 0:
            res = minimize(objective, guess, method=algorithm, **opt_params)
            objective_value = res.fun
            info_str = "\n".join(
                [
                    f"Parameters: {res.x}",
                    f"Objective: {objective_value}",
                    f"Delta: {objective_value - strategy.last_objective}",
                ]
            )
            results[f"atomicopt{ctr}"] = res
            ctr += 1
        else:
            info_str = "Skipping empty shell"
        bo_logger.info(info_str)
    return results


def optimize(
    molecule: Molecule,
    element: Optional[str] = None,
    algorithm: str = "l-bfgs-b",
    strategy: Strategy = Strategy(),
    reg: Regulariser = (lambda x: 0),
    opt_params: dict[str, Any] = {},
) -> OptResult:
    """General purpose optimizer for a single atomic basis

    Arguments:
        molecule: Molecule object
        element (str): symbol of atom to optimize; if None, will default to first atom in molecule
        algorithm (str): scipy.optimize algorithm to use
        strategy (Strategy): optimization strategy
        basis_type (str): which basis type to use; currently "orbital", "jfit", or "jkfit"
        reg (func): regularization function
        opt_params (dict): parameters to pass to scipy.optimize.minimize

    Returns:
        dictionary of scipy.optimize result objects for each step in the opt

    Raises:
        FailedCalculation
    """
    wrapper = api.get_backend()
    if element is None:
        element = molecule.unique_atoms()[0]
    element = element.lower()

    basis = molecule.basis
    if strategy.basis_type == "jfit":
        basis = molecule.jbasis
    elif strategy.basis_type == "jkfit":
        basis = molecule.jkbasis

    def objective(x):
        """Set exponents, run calculation, compute objective
        Currently just RMSE, need to expand via Strategy
        """
        strategy.set_active(x, basis, element)
        success = api.run_calculation(
            evaluate=strategy.eval_type, mol=molecule, params=strategy.params
        )
        if success != 0:
            raise FailedCalculation
        molecule.add_result(strategy.eval_type, wrapper.get_value(strategy.eval_type))
        result = molecule.get_delta(strategy.eval_type)
        return strategy.loss(result) + reg(x)

    # Initialise and run optimization
    strategy.initialise(basis, element)
    return _atomic_opt(basis, element, algorithm, strategy, opt_params, objective)


OptData = tuple[str, str, Strategy, Regulariser, dict[str, Any]]


def minimizer(
    molecule: Molecule,
    element: Optional[str] = None,
    algorithm: str = 'l-bfgs-b',
    strategy: Strategy = Strategy(),
    reg: Regulariser = (lambda x: 0),
    opt_params: dict[str, Any] = {},
) -> OptResult:
    """General purpose optimizer for a single atomic basis

    Arguments:
        molecule: Molecule object
        element (str): symbol of atom to optimize; if None, will default to first atom in molecule
        algorithm (str): scipy.optimize algorithm to use
        strategy (Strategy): optimization strategy
        basis_type (str): which basis type to use; currently "orbital", "jfit", or "jkfit"
        reg (func): regularization function
        opt_params (dict): parameters to pass to scipy.optimize.minimize

    Returns:
        dictionary of scipy.optimize result objects for each step in the opt

    Raises:
        FailedCalculation
    """
    wrapper = api.get_backend()
    if element is None:
        element = molecule.unique_atoms()[0]
    element = element.lower()

    basis = molecule.basis
    if strategy.basis_type == "jfit":
        basis = molecule.jbasis
    elif strategy.basis_type == "jkfit":
        basis = molecule.jkbasis

    def objective(x):
        """Set exponents, run calculation, compute objective
        Currently just RMSE, need to expand via Strategy
        """
        strategy.set_active(x, basis, element)
        success = api.run_calculation(
            evaluate=strategy.eval_type, mol=molecule, params=strategy.params
        )
        if success != 0:
            raise FailedCalculation
        molecule.add_result(strategy.eval_type, wrapper.get_value(strategy.eval_type))
        return wrapper.get_value(strategy.eval_type)

    # Initialise and run optimization
    strategy.initialise(basis, element)
    return _atomic_opt(basis, element, algorithm, strategy, opt_params, objective)


def _atomic_opt_auto(
    basis: InternalBasis,
    element: str,
    algorithm: str,
    strategy: Strategy,
    opt_params: dict[str, Any],
    objective: Callable[[np.ndarray], float],
) -> OptResult:
    """Helper function to run a strategy for a single atom

    Arguments:
         basis: internal basis dictionary
         element: symbol of atom to be optimized
         algorithm (str): optimization algorithm, see scipy.optimize for options
         opt_params (dict): parameters to pass to scipy.optimize.minimize
         objective (func): function to calculate objective, must have signature
             func(x) where x is a 1D numpy array of floats

     Returns:
         a dictionary of scipy.optimize result objects for each step in the opt
    """
    bo_logger.info("Starting optimization of %s/%s", element, strategy.eval_type)
    bo_logger.info("Algorithm: %s, Strategy: %s", algorithm, strategy.name)
    objective_value = objective(strategy.get_active(basis, element))
    init_exps = '\n\t'.join(
        [
            f"{shell.l}: " + ','.join([f"{exp:.6e}" for exp in shell.exps])
            for shell in basis[element]
        ]
    )
    bo_logger.info(f"\n\tInitial exponents:\n\t{init_exps}")
    bo_logger.info(f"CBS Limit: {strategy.cbs_limit}")
    bo_logger.info(
        "Initial difference to CBS limit: "
        + format_with_prefix(objective_value - strategy.cbs_limit, 'E\u2095')
    )
    bo_logger.info("Initial atomic energy: %f", objective_value)

    # Keep going until strategy says stop
    results = {}
    ctr = 1
    while strategy.next(basis, element, objective_value):
        bo_logger.info("Doing step %d", strategy._step + 1)
        guess = strategy.get_active(basis, element)
        if len(guess) > 0:
            res = minimize(objective, guess, method=algorithm, **opt_params)
            objective_value = res.fun
            # dE_CBS = (
            #    objective_value - _ATOMIC_DFT_CBS[md_element(element.capitalize()).atomic_number]
            # )
            dE_CBS = objective_value - strategy.cbs_limit
            info_str = "\n" + "\n".join(
                [
                    f"Parameters: {res.x}",
                    f"Objective: {objective_value}",
                    f"Delta: {objective_value - strategy.last_objective}",
                    "Difference to atomic CBS limit: " + format_with_prefix(dE_CBS, 'E\u2095'),
                ]
            )
            results[f"atomicopt{ctr}"] = res
            results[f"atomicopt{ctr}"]['dE_CBS'] = dE_CBS
            ctr += 1
        else:
            info_str = "Skipping empty shell"
        bo_logger.info(info_str)
    else:
        bo_logger.info("Optimization finished")
        bo_logger.info("Final energy: %f", objective_value)
        exps = '\n'.join(
            [
                f"{shell.l}" + ','.join([f"{exp:.6e}" for exp in shell.exps])
                for shell in basis[element]
            ]
        )
        bo_logger.info(f"\nFinal exponents:\n{exps}")
        final_leg = '\n'.join(
            [f"{shell.l}: \n" + str(shell.leg_params[0].tolist()) for shell in basis[element]]
        )
        bo_logger.info(f"\nFinal Legendre params: {final_leg}")
        bo_logger.info(
            "Difference to atomic CBS limit: "
            + format_with_prefix(
                abs(objective_value - strategy.cbs_limit),
                'E\u2095',
            )
        )
        bo_logger.info(f"Basis composition: {get_composition(basis, element)}")
    return results


def atom_auto(
    molecule: Molecule,
    element: Optional[str] = None,
    algorithm: str = 'l-bfgs-b',
    strategy: Strategy = Strategy(),
    reg: Regulariser = (lambda x: 0),
    opt_params: dict[str, Any] = {},
) -> OptResult:
    """General purpose optimizer for a single atomic basis

    Arguments:
        molecule: Molecule object
        element (str): symbol of atom to optimize; if None, will default to first atom in molecule
        algorithm (str): scipy.optimize algorithm to use
        strategy (Strategy): optimization strategy
        basis_type (str): which basis type to use; currently "orbital", "jfit", or "jkfit"
        reg (func): regularization function
        opt_params (dict): parameters to pass to scipy.optimize.minimize

    Returns:
        dictionary of scipy.optimize result objects for each step in the opt

    Raises:
        FailedCalculation
    """
    wrapper = api.get_backend()
    if element is None:
        element = molecule.unique_atoms()[0]
    element = element.lower()

    basis = molecule.basis
    if strategy.basis_type == "jfit":
        basis = molecule.jbasis
    elif strategy.basis_type == "jkfit":
        basis = molecule.jkbasis

    def objective(x):
        """Set exponents, run calculation, compute objective
        Currently just RMSE, need to expand via Strategy
        """
        strategy.set_active(x, basis, element)
        success = api.run_calculation(
            evaluate=strategy.eval_type, mol=molecule, params=strategy.params
        )
        if success != 0:
            raise FailedCalculation
        molecule.add_result(strategy.eval_type, wrapper.get_value(strategy.eval_type))
        return wrapper.get_value(strategy.eval_type)

    # Initialise and run optimization
    strategy.initialise(basis, element)
    return _atomic_opt_auto(basis, element, algorithm, strategy, opt_params, objective)


def _atomic_opt_auto_reduce(
    molecule: Molecule,
    basis: InternalBasis,
    element: str,
    algorithm: str,
    strategy: Strategy,
    opt_params: dict[str, Any],
    objective: Callable[[np.ndarray], float],
) -> OptResult:
    """Helper function to run a strategy for a single atom

    Arguments:
         basis: internal basis dictionary
         element: symbol of atom to be optimized
         algorithm (str): optimization algorithm, see scipy.optimize for options
         opt_params (dict): parameters to pass to scipy.optimize.minimize
         objective (func): function to calculate objective, must have signature
             func(x) where x is a 1D numpy array of floats

     Returns:
         a dictionary of scipy.optimize result objects for each step in the opt
    """

    bo_logger.info("Starting optimization of %s/%s", element, strategy.eval_type)
    bo_logger.info("Algorithm: %s, Strategy: %s", algorithm, strategy.name)
    objective_value = objective(strategy.get_active(basis, element))
    init_exps = '\n\t'.join(
        [
            f"{shell.l}: " + ','.join([f"{exp:.6e}" for exp in shell.exps])
            for shell in basis[element]
        ]
    )
    bo_logger.info(f"\n\tInitial exponents:\n\t{init_exps}")
    bo_logger.info(f"CBS Limit: {strategy.cbs_limit}")
    bo_logger.info("Initial atomic energy: %f", objective_value)
    bo_logger.info(
        "Initial difference to CBS limit: "
        + format_with_prefix(objective_value - strategy.cbs_limit, 'E\u2095')
    )

    # Keep going until strategy says stop
    results = {}
    ctr = 1
    while strategy.next(molecule, strategy.params, basis, element, objective_value):
        bo_logger.info("Doing step %d", strategy._step + 1)
        guess = strategy.get_active(basis, element)
        if len(guess) > 0:
            res = minimize(objective, guess, method=algorithm, **opt_params)
            objective_value = res.fun
            dE_CBS = (
                objective_value - _ATOMIC_DFT_CBS[md_element(element.capitalize()).atomic_number]
            )
            info_str = "\n" + "\n".join(
                [
                    f"Parameters: {res.x}",
                    f"Objective: {objective_value}",
                    f"Delta: {objective_value - strategy.last_objective}",
                    "Difference to atomic CBS limit: " + format_with_prefix(dE_CBS, 'E\u2095'),
                ]
            )
            results[f"atomicopt{ctr}"] = res
            results[f"atomicopt{ctr}"]['dE_CBS'] = dE_CBS
            ctr += 1
        else:
            info_str = "Skipping empty shell"
        bo_logger.info(info_str)
    else:
        bo_logger.info("Optimization finished")
        bo_logger.info("Final energy: %f", objective_value)
        exps = '\n'.join(
            [
                f"{shell.l}" + ','.join([f"{exp:.6e}" for exp in shell.exps])
                for shell in basis[element]
            ]
        )
        bo_logger.info(
            "Final difference to atomic CBS limit: "
            + format_with_prefix(
                abs(
                    objective_value
                    - _ATOMIC_DFT_CBS[md_element(element.capitalize()).atomic_number]
                ),
                'E\u2095',
            )
        )
        bo_logger.info(f"\nFinal exponents:\n{exps}")
        n_exp_removed = ''.join(
            [f'{r_exp}{INV_AM_DICT[idx]}' for idx, r_exp in enumerate(strategy.n_exps_removed)]
        )
        original_config = ''.join(
            [f'{exp}{INV_AM_DICT[idx]}' for idx, exp in enumerate(strategy.original_size)]
        )
        new_config = ''.join(
            [
                f'{o_exp-r_exp}{INV_AM_DICT[idx]}'
                for idx, (o_exp, r_exp) in enumerate(
                    zip(strategy.original_size, strategy.n_exps_removed)
                )
            ]
        )
        bo_logger.info(f"Number of exponents removed: {n_exp_removed}")
        bo_logger.info(f"Basis reduced from {original_config} to {new_config}")
        bo_logger.info(f"Basis composition: {get_composition(basis, element)}")
    return results


def atom_auto_reduce(
    molecule: Molecule,
    element: Optional[str] = None,
    algorithm: str = 'l-bfgs-b',
    strategy: Strategy = Strategy(),
    reg: Regulariser = (lambda x: 0),
    opt_params: dict[str, Any] = {},
) -> OptResult:
    """General purpose optimizer for a single atomic basis

    Arguments:
        molecule: Molecule object
        element (str): symbol of atom to optimize; if None, will default to first atom in molecule
        algorithm (str): scipy.optimize algorithm to use
        strategy (Strategy): optimization strategy
        basis_type (str): which basis type to use; currently "orbital", "jfit", or "jkfit"
        reg (func): regularization function
        opt_params (dict): parameters to pass to scipy.optimize.minimize

    Returns:
        dictionary of scipy.optimize result objects for each step in the opt

    Raises:
        FailedCalculation
    """
    wrapper = api.get_backend()
    if element is None:
        element = molecule.unique_atoms()[0]
    element = element.lower()

    basis = molecule.basis
    if strategy.basis_type == "jfit":
        basis = molecule.jbasis
    elif strategy.basis_type == "jkfit":
        basis = molecule.jkbasis

    def objective(x):
        """Set exponents, run calculation, compute objective
        Currently just RMSE, need to expand via Strategy
        """
        strategy.set_active(x, basis, element)
        success = api.run_calculation(
            evaluate=strategy.eval_type, mol=molecule, params=strategy.params
        )
        if success != 0:
            raise FailedCalculation
        molecule.add_result(strategy.eval_type, wrapper.get_value(strategy.eval_type))
        current_delta = abs(
            wrapper.get_value(strategy.eval_type)
            - _ATOMIC_DFT_CBS[md_element(element.capitalize()).atomic_number]
        )
        return wrapper.get_value(strategy.eval_type)

    # Initialise and run optimization
    strategy.initialise(basis, element)
    return _atomic_opt_auto_reduce(
        molecule, basis, element, algorithm, strategy, opt_params, objective
    )


def collective_optimize(
    molecules: list[Molecule],
    basis: InternalBasis,
    opt_data: list[OptData] = [],
    npass: int = 3,
    parallel: bool = False,
    ray_params: dict = None,
) -> OptCollection:
    """General purpose optimizer for a collection of atomic bases

     Arguments:
          molecules (list): list of Molecule objects to be included in objective
          basis: internal basis dictionary, will be used for all molecules
          opt_data (list): list of tuples, with one tuple for each atomic basis to be
              optimized, (element, algorithm, strategy, regularizer, opt_params) - see the
              signature of _atomic_opt or optimize
          npass (int): number of passes to do, i.e. it will optimize each atomic basis
              listed in opt_data in order, then loop back and iterate npass times
          parallel (bool): if True, will try to run Molecule calcs in parallel

    Returns:
          dictionary of dictionaries of scipy.optimize results for each step,
          corresponding to tuple in opt_data

    Raises:
          FailedCalculation
    """
    results = {}
    for i in range(npass):
        bo_logger.info("Collective pass %d", i + 1)
        total = 0.0

        # loop over elements in opt_data, and collect objective into total
        ctr = 1
        for el, alg, strategy, reg, params in opt_data:

            def objective(x):
                """Set exponents, compute objective for every molecule in set
                Regularisation only applied once at end
                """
                strategy.set_active(x, basis, el)
                local_total = 0.0
                for mol in molecules:
                    mol.basis = basis

                results = api.run_all(
                    evaluate=strategy.eval_type,
                    mols=molecules,
                    params=strategy.params,
                    parallel=parallel,
                    ray_params=ray_params,
                )
                for mol in molecules:
                    value = results[mol.name]
                    name = strategy.eval_type + "_" + el.title()
                    mol.add_result(name, value)
                    result = value - mol.get_reference(strategy.eval_type)
                    local_total += np.linalg.norm(result)
                return local_total + reg(x)

            strategy.initialise(basis, el)
            res = _atomic_opt(basis, el, alg, strategy, params, objective)
            total += strategy.last_objective
            results[f"pass{i}_opt{ctr}"] = res
            ctr += 1
        bo_logger.info("Collective objective: %f", total)
    return results


def collective_minimize(
    molecules: list[Molecule],
    basis: InternalBasis,
    opt_data: list[OptData] = [],
    npass: int = 3,
    parallel: bool = False,
    ray_params: dict = None,
) -> OptCollection:
    """General purpose optimizer for a collection of atomic bases

     Arguments:
          molecules (list): list of Molecule objects to be included in objective
          basis: internal basis dictionary, will be used for all molecules
          opt_data (list): list of tuples, with one tuple for each atomic basis to be
              optimized, (element, algorithm, strategy, regularizer, opt_params) - see the
              signature of _atomic_opt or optimize
          npass (int): number of passes to do, i.e. it will optimize each atomic basis
              listed in opt_data in order, then loop back and iterate npass times
          parallel (bool): if True, will try to run Molecule calcs in parallel

    Returns:
          dictionary of dictionaries of scipy.optimize results for each step,
          corresponding to tuple in opt_data

    Raises:
          FailedCalculation
    """
    results = {}
    for i in range(npass):
        bo_logger.info("Collective pass %d", i + 1)
        total = 0.0

        # loop over elements in opt_data, and collect objective into total
        ctr = 1
        for el, alg, strategy, reg, params in opt_data:

            def objective(x):
                """Set exponents, compute objective for every molecule in set
                Regularisation only applied once at end
                """
                strategy.set_active(x, basis, el)
                local_total = 0.0
                for mol in molecules:
                    mol.basis = basis

                results = api.run_all(
                    evaluate=strategy.eval_type,
                    mols=molecules,
                    params=strategy.params,
                    parallel=parallel,
                    ray_params=ray_params,
                )
                for mol in molecules:
                    value = results[mol.name]
                    name = strategy.eval_type + "_" + el.title()
                    mol.add_result(name, value)
                    result = value / mol.nelectrons()
                    local_total += result
                return local_total + reg(x)

            strategy.initialise(basis, el)
            res = _atomic_opt(basis, el, alg, strategy, params, objective)
            total = strategy.last_objective
            results[f"pass{i}_opt{ctr}"] = res
            ctr += 1
        bo_logger.info("Collective objective: %f", total)
    return results


def _atomic_contract(
    basis: InternalBasis,
    element: str,
    algorithm: str,
    strategy,
    opt_params: dict[str, Any],
    objective: Callable[[np.ndarray], float],
) -> OptResult:
    """Helper function to run a strategy for a single atom

    Arguments:
         basis: internal basis dictionary
         element: symbol of atom to be optimized
         algorithm (str): optimization algorithm, see scipy.optimize for options
         opt_params (dict): parameters to pass to scipy.optimize.minimize
         objective (func): function to calculate objective, must have signature
             func(x) where x is a 1D numpy array of floats

     Returns:
         a dictionary of scipy.optimize result objects for each step in the opt
    """
    bo_logger.info("Starting optimization of %s/%s", element, strategy.eval_type)
    bo_logger.info("Algorithm: %s, Strategy: %s", algorithm, strategy.name)
    objective_value = objective(strategy.get_active(basis, element))
    bo_logger.info("Initial objective value: %f", objective_value)

    # Keep going until strategy says stop
    results = {}
    ctr = 1
    while strategy.next(basis, element, objective_value):
        bo_logger.info(f"Doing step {strategy._step + 1}: Contraction {strategy._n_step + 1}")
        guess = strategy.get_active(basis, element)
        if len(guess) > 0:
            res = minimize(objective, guess, method=algorithm, **opt_params)
            objective_value = res.fun
            info_str = "\n".join(
                [
                    f"Parameters: {res.x}",
                    f"Objective: {objective_value}",
                    f"Delta: {objective_value - strategy.last_objective}",
                ]
            )
            results[f"atomicopt{ctr}"] = res
            ctr += 1
        else:
            info_str = "Skipping empty shell"
        bo_logger.info(info_str)
    return results


def contraction_optimize(
    molecule: Molecule,
    element: Optional[str] = None,
    algorithm: str = 'l-bfgs-b',
    strategy: Strategy = ContractionStrategy(),
    reg: Regulariser = (lambda x: 0),
    opt_params: dict[str, Any] = {},
) -> OptResult:
    """General purpose optimizer for a single atomic basis

    Arguments:
        molecule: Molecule object
        element (str): symbol of atom to optimize; if None, will default to first atom in molecule
        algorithm (str): scipy.optimize algorithm to use
        strategy (Strategy): optimization strategy
        basis_type (str): which basis type to use; currently "orbital", "jfit", or "jkfit"
        reg (func): regularization function
        opt_params (dict): parameters to pass to scipy.optimize.minimize

    Returns:
        dictionary of scipy.optimize result objects for each step in the opt

    Raises:
        FailedCalculation
    """
    wrapper = api.get_backend()
    if element is None:
        element = molecule.unique_atoms()[0]
    element = element.lower()

    basis = molecule.basis
    if strategy.basis_type == "jfit":
        basis = molecule.jbasis
    elif strategy.basis_type == "jkfit":
        basis = molecule.jkbasis

    def objective(x):
        """Set exponents, run calculation, compute objective
        Currently just RMSE, need to expand via Strategy
        """
        strategy.set_active(x, basis, element)
        success = api.run_calculation(
            evaluate=strategy.eval_type, mol=molecule, params=strategy.params
        )
        if success != 0:
            raise FailedCalculation
        molecule.add_result(strategy.eval_type, wrapper.get_value(strategy.eval_type))
        result = molecule.get_delta(strategy.eval_type)
        return wrapper.get_value(strategy.eval_type) - molecule.get_reference('uncontracted_energy')

    # Check reference energy added
    if not molecule.get_reference('uncontracted_energy'):
        raise FailedCalculation(
            "Uncontracted energy not found in molecule, please set with molecule.add_reference('uncontracted_energy', energy) before running contraction optimization"
        )

    # Initialise and run optimization
    strategy.initialise(basis, element)
    return _atomic_contract(basis, element, algorithm, strategy, opt_params, objective)
