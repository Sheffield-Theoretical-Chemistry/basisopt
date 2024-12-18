from collections.abc import Iterable
from typing import Any, Callable, Optional

import numpy as np
from scipy.optimize import minimize

from basisopt import api
from basisopt.containers import InternalBasis, OptCollection, OptResult
from basisopt.data import INV_AM_DICT
from basisopt.exceptions import FailedCalculation
from basisopt.molecule import Molecule
from basisopt.util import bo_logger, format_with_prefix, get_composition

from .objectives import default_min_loss, default_opt_loss

# from .contraction import ContractionStrategy
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
    bo_logger.info(f"CBS limit for this element: {format_with_prefix(strategy.cbs_limit, 'Eh')}")
    bo_logger.info(
        f"CBS target accuracy for this element: {format_with_prefix(strategy.target, 'Eh')}"
    )
    init_exps = '\n'.join(
        [
            f"\t{shell.l}: " + ','.join([f"{exp:.6e}" for exp in shell.exps])
            for shell in basis[element]
        ]
    )
    bo_logger.info(f"\n\tInitial exponents:\n{init_exps}")
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
            dE_CBS = objective_value - strategy.cbs_limit
            info_str = "\n" + "\n".join(
                [
                    f"\tParameters: {str(res.x.tolist())}",
                    f"\tObjective: {objective_value}",
                    f"\tDelta: {objective_value - strategy.last_objective}",
                    "\tDifference to atomic CBS limit: " + format_with_prefix(dE_CBS, 'E\u2095'),
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
                f"\t{shell.l}: " + ','.join([f"{exp:.6e}" for exp in shell.exps])
                for shell in basis[element]
            ]
        )
        bo_logger.info(f"\n\tFinal exponents:\n{exps}")
        try:
            final_leg = '\n'.join(
                [f"\t{shell.l}: " + str(shell.leg_params[0].tolist()) for shell in basis[element]]
            )
            bo_logger.info(f"\n\tFinal Legendre parameters:\n {final_leg}")
        except:
            pass
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
    bo_logger.info(f"CBS limit for this element: {format_with_prefix(strategy.cbs_limit, 'Eh')}")
    bo_logger.info(
        f"CBS target accuracy for this element: {format_with_prefix(strategy.target, 'Eh')}"
    )
    init_exps = '\n'.join(
        [
            f"\t{shell.l}: " + ','.join([f"{exp:.6e}" for exp in shell.exps])
            for shell in basis[element]
        ]
    )
    bo_logger.info(f"\n\tInitial exponents:\n{init_exps}")
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
            dE_CBS = objective_value - strategy.cbs_limit
            info_str = "\n" + "\n".join(
                [
                    f"\tParameters: {str(res.x.tolist())}",
                    f"\tObjective: {objective_value}",
                    f"\tDelta: {objective_value - strategy.last_objective}",
                    "\tDifference to atomic CBS limit: " + format_with_prefix(dE_CBS, 'E\u2095'),
                ]
            )
            results[f"atomicopt{ctr}"] = res
            results[f"atomicopt{ctr}"]['dE_CBS'] = dE_CBS
            ctr += 1
        else:
            info_str = "Skipping empty shell"
        bo_logger.info(info_str)
    else:
        wrapper = api.get_backend()
        api.run_calculation(
            evaluate=strategy.eval_type, mol=molecule, params=strategy.params
        )
        objective_value = wrapper.get_value(strategy.eval_type)
        dE_CBS = objective_value - strategy.cbs_limit
        ctr += 1
        final_energy = wrapper.get_value(strategy.eval_type)
        molecule.add_result(strategy.eval_type, wrapper.get_value(strategy.eval_type))
        bo_logger.info("Optimization finished")
        bo_logger.info("Final energy: %f", final_energy)
        exps = '\n'.join(
            [
                f"\t{shell.l}: " + ','.join([f"{exp:.6e}" for exp in shell.exps])
                for shell in basis[element]
            ]
        )

        bo_logger.info(
            "Final difference to atomic CBS limit: "
            + format_with_prefix(
                abs(final_energy - strategy.cbs_limit),
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
    strategy: Strategy,
    element: Optional[str] = None,
    algorithm: str = 'l-bfgs-b',
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
        return wrapper.get_value(strategy.eval_type) - molecule.get_reference('uncontracted_energy')

    # Check reference energy added
    if not molecule.get_reference('uncontracted_energy'):
        raise FailedCalculation(
            "Uncontracted energy not found in molecule, please set with molecule.add_reference('uncontracted_energy', energy) before running contraction optimization"
        )

    # Initialise and run optimization
    strategy.initialise(basis, element)
    return _atomic_contract(basis, element, algorithm, strategy, opt_params, objective)


def collective_polarize(
    molecules: list[Molecule],
    basis: InternalBasis,
    opt_data: list[OptData] = [],
    npass: int = 1,
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
                    value = abs(results[mol.name] - mol.cbs_limit)
                    name = strategy.eval_type + "_" + el.title()
                    mol.add_result(name, value)
                    result = value / mol.nelectrons()
                    local_total += result
                return (local_total + reg(x)) / len(molecules)

            strategy.initialise(basis, el)
            res = _atomic_opt(basis, el, alg, strategy, params, objective)
            total = strategy.last_objective
            results[f"pass{i}_opt{ctr}"] = res
            ctr += 1
        bo_logger.info("Collective objective: %f", total)
    return results


class Optimizer:
    def __init__(
        self,
        strategy,
        params,
        reference_basis=None,
        basis={},
        elements=[],
        loss=default_opt_loss,
        parallel=False,
        nprocs=2,
        parallel_params={},
    ):
        self.strategy = strategy
        self.params = params
        self.loss = loss
        self.reference_basis = reference_basis
        self.basis = basis
        self.elements = elements
        self.results = {}
        self.opt_params = {}
        self.active_element = str
        self.results = {}
        self.molecules = []
        self._initialized = False
        self.parallel = parallel
        self.parallel_params = parallel_params
        self.nprocs = nprocs

    def _objective(self, x):
        """
        Get the objective value for the current set of active exponents.
        Uses the strategy to set the active exponents, runs the calculation, and returns the loss.
        """
        self.strategy.set_active(x, self.basis, self.active_element)
        for mol in self.molecules:
            success = api.run_calculation(
                evaluate=self.strategy.eval_type, mol=mol, params=self.params
            )
            if success != 0:
                raise ValueError("Calculation failed")
            mol.add_result(self.strategy.eval_type, self.wrapper.get_value(self.strategy.eval_type))
        return self.loss(self.molecules)

    def _parallel_objective(self, x):
        self.strategy.set_active(x, self.basis, self.active_element)
        results = api.run_all(
            evaluate=self.strategy.eval_type,
            mols=self.molecules,
            params=self.params,
            parallel=self.parallel,
            ray_params=self.parallel_params,
        )
        for mol in self.molecules:
            mol.add_result(self.strategy.eval_type, results[mol.name])
        return self.loss(self.molecules)

    def _opt(self, element: str, algorithm: str):
        """
        A method to optimize the active exponents for a given element using a given algorithm.

        Args:
            element (str): Element to optimize over
            algorithm (str): Scipy optimization algorithm to use
        """
        bo_logger.info(f"Starting optimization of {self.strategy.eval_type} {element.capitalize()}")
        bo_logger.info(f"Using {algorithm} algorithm for strategy {self.strategy.name}")
        if self.parallel:
            api.set_parallel(True, self.nprocs)
            initial_objective = self._parallel_objective(
                self.strategy.get_active(self.basis, element)
            )
        else:
            initial_objective = self._objective(self.strategy.get_active(self.basis, element))
        objective_value = initial_objective
        ctr = 1
        while self.strategy.next(self.basis, element, objective_value):
            guess = self.strategy.get_active(self.basis, element)
            if len(guess) > 0:
                if self.parallel:
                    res = minimize(
                        self._parallel_objective, guess, method=algorithm, **self.opt_params
                    )
                else:
                    res = minimize(self._objective, guess, method=algorithm, **self.opt_params)
                objective_value = res.fun
                info_str = "\n".join(
                    [
                        f"Parameters: {res.x}",
                        f"Objective value: {res.fun}",
                        f"Step Delta: {objective_value - self.strategy.last_objective}",
                        f"Total Delta: {objective_value - initial_objective}",
                    ]
                )
                self.results[f"opt{ctr}"] = res
                ctr += 1
            else:
                info_str = "Skipping empty shell"
            bo_logger.info(info_str)
        bo_logger.info("Optimization complete.")

    def _initialize(self):
        """
        Initialize the optimizer.
        If given a reference basis, then that is used to calculate a reference energy.
        """
        for element in self.elements:
            """
            Set the active element and initialize the strategy
            This will set any initial exponents if the basis set is created
            as part of the strategy.
            """
            self.active_element = element
            self.strategy.initialise(self.basis, self.active_element)
        self.wrapper = api.get_backend()
        if self.molecules:
            self.molecules = self.molecules
        if not self.elements:
            for mol in self.molecules:
                for atom in mol.unique_atoms():
                    self.elements.append(atom.lower())
            self.elements = set(self.elements)
        else:
            self.elements = set(self.elements)
        for mol in self.molecules:
            if self.reference_basis:
                mol.basis = self.reference_basis
            else:
                mol.basis = self.basis
            success = api.run_calculation(
                evaluate=self.strategy.eval_type, mol=mol, params=self.params
            )
            if success != 0:
                raise ValueError("Calculation failed")
            mol.add_result("energy", self.wrapper.get_value("energy"))
            if mol.get_reference(self.strategy.eval_type) == 0.0:
                mol.add_reference(
                    self.strategy.eval_type, self.wrapper.get_value(self.strategy.eval_type)
                )
            bo_logger.info(f'Reference for {mol.name} is {mol.get_reference("energy")}')
            mol.basis = self.basis
        self._initialized = True

    def run(self, molecules: list = [], algorithm: str = "Nelder-Mead"):
        """Run the optimizer on the given molecules using the given algorithm"""
        if molecules:
            if isinstance(molecules, Iterable):
                self.molecules = molecules
            else:
                self.molecules = [molecules]
        if not self._initialized:
            self._initialize()
        if self.elements is None:
            raise ValueError("No elements to optimize")
        for element in self.elements:
            self.active_element = element
            self.strategy.initialise(self.basis, self.active_element)
            self._opt(self.active_element, algorithm)

    def set_parallel_params(self, parallel_params):
        self.parallel_params = parallel_params

    def get_results(self):
        return self.results

    def get_basis(self):
        return self.basis

    def get_strategy(self):
        return self.strategy

    def get_params(self):
        return self.params


class Minimizer(Optimizer):
    """Class to minimize the energy of a given molecule or set of molecules using a given strategy"""

    def __init__(
        self,
        strategy,
        params,
        reference_basis=None,
        basis={},
        elements=[],
        loss=default_min_loss,
        parallel=False,
        nprocs=2,
        parallel_params={},
    ):
        super().__init__(
            strategy,
            params,
            reference_basis,
            basis,
            elements,
            loss,
            parallel,
            nprocs,
            parallel_params,
        )

    def _opt(self, element: str, algorithm: str):
        """
        A method to optimize the active exponents for a given element using a given algorithm.

        Args:
            element (str): Element to optimize over
            algorithm (str): Scipy optimization algorithm to use
        """
        bo_logger.info(f"Starting optimization of {self.strategy.eval_type} {element.capitalize()}")
        bo_logger.info(f"Using {algorithm} algorithm for strategy {self.strategy.name}")
        if self.parallel:
            initial_objective = self._parallel_objective(
                self.strategy.get_active(self.basis, element)
            )
        else:
            initial_objective = self._objective(self.strategy.get_active(self.basis, element))
        objective_value = initial_objective
        ctr = 1
        while self.strategy.next(self.basis, element, objective_value):
            guess = self.strategy.get_active(self.basis, element)
            if len(guess) > 0:
                if self.parallel:
                    res = minimize(
                        self._parallel_objective, guess, method=algorithm, **self.opt_params
                    )
                else:
                    res = minimize(self._objective, guess, method=algorithm, **self.opt_params)
                objective_value = res.fun
                running_total = 0
                running_total += objective_value - self.strategy.last_objective
                info_str = "\n".join(
                    [
                        f"Parameters: {res.x}",
                        f"Objective value: {res.fun}",
                        f"Step Delta: {objective_value - self.strategy.last_objective}",
                        f"Total Delta: {running_total}",
                    ]
                )
                self.results[f"opt{ctr}"] = res
                ctr += 1
            else:
                info_str = "Skipping empty shell"
            bo_logger.info(info_str)
        bo_logger.info("Minimization complete")
