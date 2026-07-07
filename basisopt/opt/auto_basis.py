import copy
from typing import Any, Optional

import numpy as np

from basisopt.basis.basis import legendre_expansion, uncontract_shell
from basisopt.containers import InternalBasis, Shell
from basisopt.data import _ATOMIC_LEGENDRE_COEFFS, INV_AM_DICT
from basisopt.molecule import Molecule
from basisopt.testing.rank import rank_mol_basis_cbs
from basisopt.util import bo_logger, get_composition

from .preconditioners import Preconditioner, make_positive, unit
from .strategies import Strategy


class AutoBasisFree(Strategy):
    """

    Algorithm:
        Evaluate: energy (can change to any RMSE-compatible property)
        Loss: root-mean-square error
        Guess: null, uses _INITIAL_GUESS above
        Pre-conditioner: None

        Initialisation:
            - Find minimum no. of shells needed
            - max_l >= min_l
            - generate initial parameters for each shell

        First run:
            - optimize parameters for each shell once, sequentially

        Next shell in list not marked finished:
            - re-optimise
            - below threshold or n=max_n: mark finished
            - above threshold: increment n
        Repeat until all shells are marked finished.

        Uses iteration, limited by two parameters:
            max_n: max number of exponents in shell
            target: threshold for objective function

    Additional attributes:
        shells (list): list of ([A_vals], n) parameter tuples
        shell_done (list): list of flags for whether shell is finished (0) or not (1)
        target (float): threshold for optimization delta
        max_n_a (int): Maximum number of legendre values to pass as a
        n (int): number of primitives in shell expansion
        l (int): angular momentum shell to do
    """

    def __init__(
        self,
        eval_type: str = 'energy',
        target: float = 1e-6,
        max_n: int = 9,
        l: int = -1,
        max_n_a: int = 6,
        n_exp_cutoff: int = 6,
        pre: Preconditioner = make_positive,
    ):
        super().__init__(eval_type=eval_type)
        self.name = 'AutoBasisFree'
        self.shell = []
        self.shell_done = []
        self.target = target
        self.guess = None
        self.pre = pre
        self.guess_params = {}
        self.params = {}
        self.cbs_limit = None

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of object"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["target"] = self.target
        d["cbs_limit"] = self.cbs_limit
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates the strategy from an MSONable dictionary"""
        strategy = Strategy.from_dict(d)
        instance = cls(
            eval_type=d.get("eval_type", 'energy'),
            target=d.get("target", 1e-5),
        )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.cbs_limit = d.get("cbs_limit", None)
        return instance

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy (does nothing in default)

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """

        self._step = -1
        self.shells_done = [1] * len(basis[element])
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = [True] * len(basis[element])
        self.init_run = True
        self.just_added = [False] * len(basis[element])
        if not self.cbs_limit:
            raise ValueError('CBS limit not set. This can be set with the .set_cbs_limit method.')

    def set_cbs_limit(self, cbs_limit: float):
        """Sets the CBS limit for the strategy

        Arguments:
            cbs_limit: the CBS limit for the strategy
        """
        self.cbs_limit = cbs_limit

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Arguments:
             basis: internal basis dictionary
             element: symbol of the atom being optimized

        Returns:
             the set of exponents currently being optimised
        """
        elbasis = basis[element]
        x = elbasis[self._step].exps
        return self.pre(x, **self.pre.params)

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Sets the currently active exponents to the given values.

        Arguments:
            values (list): list of new exponents
            basis: internal basis dictionary
            element: symbol of atom being optimized
        """
        elbasis = basis[element]
        y = np.array(values)
        elbasis[self._step].exps = self.pre.inverse(y, **self.pre.params)

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        """Moves the strategy forward a step (see algorithm)

        Arguments:
            basis: internal basis dictionary
            element: symbol of atom being optimized
            objective: value of objective function from last steps

        Returns:
            True if there is a next step, False if strategy is finished
        """
        element_cbs_limit = self.cbs_limit

        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective

        objective_diff = np.abs(objective - element_cbs_limit)

        if self.init_run:
            if self._step == -1:
                self._step = 0
                return True
            else:
                self._step += 1
                if self._step == len(basis[element]):
                    self.init_run = False
                    self._step = 0
                else:
                    return True

        if objective_diff < self.target:
            return False

        x = self.get_active(basis, element)
        last_func, penult_func = x[-1], x[-2]
        ratio = last_func / penult_func
        x = np.append(x, last_func * ratio)
        self.set_active(x, basis, element)
        uncontract_shell(basis[element][self._step])
        self._step += 1
        if self._step == len(basis[element]):
            self._step = 0

        return True


class AutoBasisLegendre(Strategy):
    """

    Algorithm:
        Evaluate: energy (can change to any RMSE-compatible property)
        Loss: root-mean-square error
        Guess: null, uses _INITIAL_GUESS above
        Pre-conditioner: None

        Initialisation:
            - Find minimum no. of shells needed
            - max_l >= min_l
            - generate initial parameters for each shell

        First run:
            - optimize parameters for each shell once, sequentially

        Next shell in list not marked finished:
            - re-optimise
            - below threshold or n=max_n: mark finished
            - above threshold: increment n
        Repeat until all shells are marked finished.

        Uses iteration, limited by two parameters:
            max_n: max number of exponents in shell
            target: threshold for objective function

    Additional attributes:
        shells (list): list of ([A_vals], n) parameter tuples
        shell_done (list): list of flags for whether shell is finished (0) or not (1)
        target (float): threshold for optimization delta
        max_n_a (int): Maximum number of legendre values to pass as a
        n (int): number of primitives in shell expansion
        l (int): angular momentum shell to do
    """

    def __init__(
        self,
        eval_type: str = 'energy',
        target: float = 1e-6,
        max_n: int = 9,
        l: int = -1,
        max_n_a: int = 6,
        n_exp_cutoff: int = 6,
        n_coefs: Optional[tuple] = None,
    ):
        super().__init__(eval_type=eval_type, pre=unit)
        self.name = 'AutoBasisLegendre'
        self.shell = []
        self.shell_done = []
        self.target = target
        self.guess = None
        self.guess_params = {}
        self.params = {}
        self.n_prim = n_coefs
        self.cbs_limit = None

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of object"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["target"] = self.target
        d["cbs_limit"] = self.cbs_limit
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates the strategy from an MSONable dictionary"""
        strategy = Strategy.from_dict(d)
        instance = cls(
            eval_type=d.get("eval_type", 'energy'),
            target=d.get("target", 1e-5),
        )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.cbs_limit = d.get("cbs_limit", None)
        return instance

    def set_cbs_limit(self, cbs_limit: float):
        """Sets the CBS limit for the strategy

        Arguments:
            cbs_limit: the CBS limit for the strategy
        """
        self.cbs_limit = cbs_limit

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy (does nothing in default)

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """
        if self.legendre_params:
            leg_params = []
            for shell, n in zip(self.legendre_params, self.n_prim):
                leg_params.append((shell, n))
            self._shells = leg_params
        else:
            bo_logger.warning(
                'No Legendre parameters set. Using default parameters. This may result in poorly conditioned expansions.'
            )
            self._initial_guess = _ATOMIC_LEGENDRE_COEFFS[element.capitalize()]
            self._shells = [(A_vals, n) for A_vals, n in zip(self._initial_guess, self.n_prim)]
        if not isinstance(basis, dict):
            basis = {}
        self.set_basis_shell(basis, element)

        self._step = -1
        self.shells_done = [1] * len(basis[element])
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = [True] * len(basis[element])
        self.init_run = True
        self.just_added = [False] * len(basis[element])
        if self.cbs_limit is None:
            raise ValueError('CBS limit not set. This can be set with the .set_cbs_limit method.')

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Arguments:
             basis: internal basis dictionary
             element: symbol of the atom being optimized

        Returns:
             the set of exponents currently being optimised
        """
        A_vals, n = self._shells[self._step]
        return A_vals

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Sets the currently active exponents to the given values.

        Arguments:
            values (list): list of new exponents
            basis: internal basis dictionary
            element: symbol of atom being optimized
        """
        (A_vals, n) = basis[element][self._step].leg_params
        self._shells[self._step] = (values, n)
        basis[element][self._step].leg_params = (values, n)

        self.set_basis_shell(basis, element)

    def set_basis_shell(self, basis: InternalBasis, element: str):
        """Expands parameters into a basis set

        Arguments:
             basis (InternalBasis): the basis set to expand
             element (str): the atom type
        """
        basis[element] = legendre_expansion(self._shells)

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        """Moves the strategy forward a step (see algorithm)

        Arguments:
            basis: internal basis dictionary
            element: symbol of atom being optimized
            objective: value of objective function from last steps

        Returns:
            True if there is a next step, False if strategy is finished
        """
        # element_cbs_limit = _ATOMIC_DFT_CBS[el]  # Get the CBS limit for the element DFT BHHLYP
        element_cbs_limit = self.cbs_limit

        self.delta_objective = np.abs(
            objective - self.last_objective
        )  # Calculate the difference in objective function
        self.last_objective = (
            objective  # Set the last objective function to the current objective function
        )

        objective_diff = np.abs(
            objective - element_cbs_limit
        )  # Calculate the difference between the objective function and the CBS limit

        # If the strategy is in the initial run then it will optimize the
        # parameters for each shell once, sequentially to ensure the
        # A_vals are suitable for the number of primitives functions in the shell
        if self.init_run:
            if self._step == -1:
                self._step = 0
                return True
            else:
                self._step += 1
                if self._step == len(basis[element]):
                    self.init_run = False
                    self._step = 0
                else:
                    return True

        # If the difference between the objective function and the CBS limit is less than the target
        if objective_diff < self.target:
            return False

        A_vals, n = self._shells[self._step]
        if not self.just_added[self._step]:
            bo_logger.info(
                f'Increasing number of {basis[element][self._step].l} functions from {n} to {n+1}'
            )
            self._shells[self._step] = (A_vals, n + 1)
            self.set_basis_shell(basis, element)
            self.just_added[self._step] = True
            return True
        else:
            # If the shell just added and reoptimised new primitive function then set the
            # just_added flag to False and increment the step to the next shell in the basis set
            self.just_added[self._step] = False
            bo_logger.info(f'Shell exponents: {list(basis[element][self._step].exps)}')
            self._step += 1
            if self._step == len(basis[element]):
                # If the step is equal to the number of shells in the basis set then set the step to 0
                self._step = 0

        return True


class AutoBasisReduceStrategyOld(Strategy):
    """

    Algorithm:
        Evaluate: energy (can change to any RMSE-compatible property)
        Loss: root-mean-square error
        Guess: null, uses _INITIAL_GUESS above
        Pre-conditioner: None

        Initialisation:
            - Find minimum no. of shells needed
            - max_l >= min_l
            - generate initial parameters for each shell

        First run:
            - optimize parameters for each shell once, sequentially

        Next shell in list not marked finished:
            - re-optimise
            - below threshold or n=max_n: mark finished
            - above threshold: increment n
        Repeat until all shells are marked finished.

        Uses iteration, limited by two parameters:
            max_n: max number of exponents in shell
            target: threshold for objective function

    Additional attributes:
        shells (list): list of ([A_vals], n) parameter tuples
        shell_done (list): list of flags for whether shell is finished (0) or not (1)
        target (float): threshold for optimization delta
        max_n_a (int): Maximum number of legendre values to pass as a
        n (int): number of primitives in shell expansion
        l (int): angular momentum shell to do
    """

    def __init__(
        self,
        eval_type: str = 'energy',
        target: float = 1e-6,
        max_n: int = 9,
        l: int = -1,
        max_n_a: int = 6,
        n_exp_cutoff: int = 6,
    ):
        super().__init__(eval_type=eval_type, pre=unit)
        self.name = 'AutoBasisReduceOld'
        self.target = target
        self.guess = None
        self.guess_params = {}
        self.params = {}
        self.cbs_limit = None

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of object"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["target"] = self.target
        d["cbs_limit"] = self.cbs_limit
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates the strategy from an MSONable dictionary"""
        strategy = Strategy.from_dict(d)
        instance = cls(
            eval_type=d.get("eval_type", 'energy'),
            target=d.get("target", 1e-5),
        )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.cbs_limit = d.get("cbs_limit", None)
        return instance

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy (does nothing in default)

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """

        self._step = 0
        self.shells_done = [1] * len(basis[element])
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = [True] * len(basis[element])
        self.init_run = True
        self._just_removed = [False] * len(basis[element])
        self.original_shells = [copy.deepcopy(shell) for shell in basis[element]]
        self.original_size = [len(shell.exps) for shell in basis[element]]
        self.n_exps_removed = [0] * len(basis[element])
        self.old_exps = [None] * len(basis[element])
        if self.cbs_limit is None:
            raise ValueError('CBS limit not set. This can be set with the .set_cbs_limit method.')

    def set_cbs_limit(self, cbs_limit: float):
        """Sets the CBS limit for the strategy

        Arguments:
            cbs_limit: the CBS limit for the strategy
        """
        self.cbs_limit = cbs_limit

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Arguments:
             basis: internal basis dictionary
             element: symbol of the atom being optimized

        Returns:
             the set of exponents currently being optimised
        """
        elbasis = basis[element]
        x = elbasis[self._step].exps
        return self.pre(x, **self.pre.params)

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Sets the currently active exponents to the given values.

        Arguments:
            values (list): list of new exponents
            basis: internal basis dictionary
            element: symbol of atom being optimized
        """
        elbasis = basis[element]
        y = np.array(values)
        elbasis[self._step].exps = self.pre.inverse(y, **self.pre.params)

    def next(
        self,
        molecule: Molecule,
        backend_params: dict,
        basis: InternalBasis,
        element: str,
        objective: float,
    ) -> bool:
        """Moves the strategy forward a step (see algorithm)

        Arguments:
            basis: internal basis dictionary
            element: symbol of atom being optimized
            objective: value of objective function from last steps

        Returns:
            True if there is a next step, False if strategy is finished
        """
        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective

        errors, ranks, _, _ = rank_mol_basis_cbs(
            molecule,
            element,
            self.cbs_limit,
            self.eval_type,
            self.params,
        )

        while sum(self.shells_done) != 0:
            if self.shells_done[self._step] == 0:
                self._step += 1
                if self._step == len(basis[element]):
                    self._step = 0

            if not self._just_removed[self._step]:
                self.old_exps[self._step] = basis[element][self._step].exps
                new_exps = np.delete(basis[element][self._step].exps, ranks[self._step][0])
                self.set_active(new_exps, basis, element)
                uncontract_shell(basis[element][self._step])
                self._just_removed[self._step] = True

                # bo_logger.info(f"Removed exponent {basis[element][self._step].exps[ranks[self._step][0]]} from shell {basis[element][self._step].l}")
                bo_logger.info(
                    f"Removed exponent {ranks[self._step][0]} from shell {basis[element][self._step].l}"
                )

                return sum(self.shells_done) != 0
            else:
                if objective - self.cbs_limit > self.target:
                    self.set_active(self.old_exps[self._step], basis, element)
                    uncontract_shell(basis[element][self._step])
                    bo_logger.info(
                        f"Re-adding exponent {ranks[self._step][0]} to shell {basis[element][self._step].l}"
                    )
                    self._just_removed[self._step] = False
                    self.shells_done[self._step] = 0
                    self._step += 1
                    if self._step == len(basis[element]):
                        self._step = 0
                else:
                    self._just_removed[self._step] = False
                    self.n_exps_removed[self._step] += 1
                    self._step += 1
                    if self._step == len(basis[element]):
                        self._step = 0

            # if errors[self._step][ranks[self._step][0]] > self.target:
            #     self.shells_done[self._step] = 0
            #     self._step += 1
            #     if self._step == len(basis[element]):
            #         self._step = 0
            # else:
            #     new_exps = np.delete(basis[element][self._step].exps, ranks[self._step][0])
            #     self.set_active(new_exps, basis, element)
            #     uncontract_shell(basis[element][self._step])
            #     self._just_removed[self._step] = True
            #     self.n_exps_removed[self._step] += 1
            #     return sum(self.shells_done) != 0

        if sum(self.shells_done) == 0:
            return False

        return sum(self.shells_done) != 0


class AutoBasisReduceStrategy(Strategy):
    """

    Algorithm:
        Evaluate: energy (can change to any RMSE-compatible property)
        Loss: root-mean-square error
        Guess: null, uses _INITIAL_GUESS above
        Pre-conditioner: None

        Initialisation:
            - Find minimum no. of shells needed
            - max_l >= min_l
            - generate initial parameters for each shell

        First run:
            - optimize parameters for each shell once, sequentially

        Next shell in list not marked finished:
            - re-optimise
            - below threshold or n=max_n: mark finished
            - above threshold: increment n
        Repeat until all shells are marked finished.

        Uses iteration, limited by two parameters:
            max_n: max number of exponents in shell
            target: threshold for objective function

    Additional attributes:
        shells (list): list of ([A_vals], n) parameter tuples
        shell_done (list): list of flags for whether shell is finished (0) or not (1)
        target (float): threshold for optimization delta
        max_n_a (int): Maximum number of legendre values to pass as a
        n (int): number of primitives in shell expansion
        l (int): angular momentum shell to do
    """

    def __init__(
        self,
        eval_type: str = 'energy',
        target: float = 1e-6,
        max_n: int = 9,
        l: int = -1,
        max_n_a: int = 6,
        n_exp_cutoff: int = 6,
    ):
        super().__init__(eval_type=eval_type, pre=unit)
        self.name = 'AutoBasisReduce'
        self.target = target
        self.guess = None
        self.guess_params = {}
        self.params = {}
        self.cbs_limit = None
        self.skip_init = False

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of object"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["target"] = self.target
        d["cbs_limit"] = self.cbs_limit
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates the strategy from an MSONable dictionary"""
        strategy = Strategy.from_dict(d)
        instance = cls(
            eval_type=d.get("eval_type", 'energy'),
            target=d.get("target", 1e-5),
        )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.cbs_limit = d.get("cbs_limit", None)
        return instance

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy (does nothing in default)

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """

        self._step = -1
        self.shells_done = [1] * len(basis[element])
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = [True] * len(basis[element])
        self.init_run = True
        self._just_removed = False
        self.original_shells = [copy.deepcopy(shell) for shell in basis[element]]
        self.original_size = [len(shell.exps) for shell in basis[element]]
        self.n_exps_removed = [0] * len(basis[element])
        self.old_exps = [None] * len(basis[element])
        if self.cbs_limit is None:
            raise ValueError('CBS limit not set. This can be set with the .set_cbs_limit method.')

    def set_cbs_limit(self, cbs_limit: float):
        """Sets the CBS limit for the strategy

        Arguments:
            cbs_limit: the CBS limit for the strategy
        """
        self.cbs_limit = cbs_limit

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Arguments:
             basis: internal basis dictionary
             element: symbol of the atom being optimized

        Returns:
             the set of exponents currently being optimised
        """
        elbasis = basis[element]
        x = elbasis[self._step].exps
        return self.pre(x, **self.pre.params)

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Sets the currently active exponents to the given values.

        Arguments:
            values (list): list of new exponents
            basis: internal basis dictionary
            element: symbol of atom being optimized
        """
        elbasis = basis[element]
        y = np.array(values)
        elbasis[self._step].exps = self.pre.inverse(y, **self.pre.params)

    def next(
        self,
        molecule: Molecule,
        backend_params: dict,
        basis: InternalBasis,
        element: str,
        objective: float,
    ) -> bool:
        """Moves the strategy forward a step (see algorithm)

        Arguments:
            basis: internal basis dictionary
            element: symbol of atom being optimized
            objective: value of objective function from last steps

        Returns:
            True if there is a next step, False if strategy is finished
        """
        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective

        if self.init_run:
            if self._step == -1:
                self._step = 0
                return True
            else:
                self._step += 1
                if self._step == len(basis[element]):
                    self.init_run = False
                    self._step = 0
                else:
                    return True

        if not self._just_removed:
            errors, ranks, _, _ = rank_mol_basis_cbs(
                molecule,
                element,
                self.cbs_limit,
                self.eval_type,
                self.params,
            )
            min_index_np = np.unravel_index(
                np.argmin([item for sublist in errors for item in sublist]),
                (len(errors), max(len(sublist) for sublist in errors)),
            )

            self._step = min_index_np[0]
            self.old_exps[self._step] = basis[element][self._step].exps
            new_exps = np.delete(basis[element][self._step].exps, min_index_np[1])
            self.set_active(new_exps, basis, element)
            uncontract_shell(basis[element][self._step])
            bo_logger.info(
                f"Removing exponent {min_index_np[1]} from shell {basis[element][self._step].l}"
            )
            self._just_removed = True
            self.n_exps_removed[self._step] += 1
            return True
        elif objective - self.cbs_limit > self.target:
            bo_logger.info('Restoring previous basis set')
            self.set_active(self.old_exps[self._step], basis, element)
            uncontract_shell(basis[element][self._step])
            self.n_exps_removed[self._step] -= 1
            return False
        else:
            self._just_removed = False
            return True


class AutoBasisReduceStrategyAll(Strategy):
    """

    Algorithm:
        Evaluate: energy (can change to any RMSE-compatible property)
        Loss: root-mean-square error
        Guess: null, uses _INITIAL_GUESS above
        Pre-conditioner: None

        Initialisation:
            - Find minimum no. of shells needed
            - max_l >= min_l
            - generate initial parameters for each shell

        First run:
            - optimize parameters for each shell once, sequentially

        Next shell in list not marked finished:
            - re-optimise
            - below threshold or n=max_n: mark finished
            - above threshold: increment n
        Repeat until all shells are marked finished.

        Uses iteration, limited by two parameters:
            max_n: max number of exponents in shell
            target: threshold for objective function

    Additional attributes:
        shells (list): list of ([A_vals], n) parameter tuples
        shell_done (list): list of flags for whether shell is finished (0) or not (1)
        target (float): threshold for optimization delta
        max_n_a (int): Maximum number of legendre values to pass as a
        n (int): number of primitives in shell expansion
        l (int): angular momentum shell to do
    """

    def __init__(
        self,
        eval_type: str = 'energy',
        target: float = 1e-6,
        max_n: int = 9,
        l: int = -1,
        max_n_a: int = 6,
        n_exp_cutoff: int = 6,
    ):
        super().__init__(eval_type=eval_type, pre=unit)
        self.name = 'AutoBasisReduceALl'
        self.target = target
        self.guess = None
        self.guess_params = {}
        self.params = {}
        self.cbs_limit = None
        self.run_all = False

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of object"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["target"] = self.target
        d["cbs_limit"] = self.cbs_limit
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates the strategy from an MSONable dictionary"""
        strategy = Strategy.from_dict(d)
        instance = cls(
            eval_type=d.get("eval_type", 'energy'),
            target=d.get("target", 1e-5),
        )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.cbs_limit = d.get("cbs_limit", None)
        return instance

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy (does nothing in default)

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """

        self._step = 0
        self.shells_done = [1] * len(basis[element])
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = [True] * len(basis[element])
        self.init_run = True
        self._just_removed = [False] * len(basis[element])
        self.original_shells = [copy.deepcopy(shell) for shell in basis[element]]
        self.original_size = [len(shell.exps) for shell in basis[element]]
        self.n_exps_removed = [0] * len(basis[element])
        self.old_exps = [None] * len(basis[element])
        if self.cbs_limit is None:
            raise ValueError('CBS limit not set. This can be set with the .set_cbs_limit method.')

    def set_cbs_limit(self, cbs_limit: float):
        """Sets the CBS limit for the strategy

        Arguments:
            cbs_limit: the CBS limit for the strategy
        """
        self.cbs_limit = cbs_limit

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Arguments:
             basis: internal basis dictionary
             element: symbol of the atom being optimized

        Returns:
             the set of exponents currently being optimised
        """
        elbasis = basis[element]
        x = elbasis[self._step].exps
        return self.pre(x, **self.pre.params)

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Sets the currently active exponents to the given values.

        Arguments:
            values (list): list of new exponents
            basis: internal basis dictionary
            element: symbol of atom being optimized
        """
        elbasis = basis[element]
        y = np.array(values)
        elbasis[self._step].exps = self.pre.inverse(y, **self.pre.params)

    def next(
        self,
        molecule: Molecule,
        backend_params: dict,
        basis: InternalBasis,
        element: str,
        objective: float,
    ) -> bool:
        """Moves the strategy forward a step (see algorithm)

        Arguments:
            basis: internal basis dictionary
            element: symbol of atom being optimized
            objective: value of objective function from last steps

        Returns:
            True if there is a next step, False if strategy is finished
        """
        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective

        while sum(self.shells_done) != 0:
            if self.run_all:
                self._step += 1
                if self._step == len(basis[element]):
                    self._step = 0
                if self._just_removed == self._step:
                    self.run_all = False
            if self.run_all:
                return True
            # if self.shells_done[self._step] == 0:
            #     self._step += 1
            #     if self._step == len(basis[element]):
            #         self._step = 0

            if self._just_removed != self._step:
                errors, ranks, _, _ = rank_mol_basis_cbs(
                    molecule,
                    element,
                    self.cbs_limit,
                    self.eval_type,
                    self.params,
                )
                self.old_exps[self._step] = basis[element][self._step].exps
                new_exps = np.delete(basis[element][self._step].exps, ranks[self._step][0])
                self.set_active(new_exps, basis, element)
                uncontract_shell(basis[element][self._step])
                self._just_removed = self._step
                self.run_all = True
                # bo_logger.info(f"Removed exponent {basis[element][self._step].exps[ranks[self._step][0]]} from shell {basis[element][self._step].l}")
                bo_logger.info(
                    f"Removed exponent {ranks[self._step][0]} from shell {basis[element][self._step].l}"
                )

                return sum(self.shells_done) != 0
            else:
                if objective - self.cbs_limit > self.target:
                    self.set_active(self.old_exps[self._step], basis, element)
                    uncontract_shell(basis[element][self._step])
                    bo_logger.info(
                        f"Re-adding exponent {ranks[self._step][0]} to shell {basis[element][self._step].l}"
                    )
                    self.shells_done[self._step] = 0
                    self._step += 1
                    if self._step == len(basis[element]):
                        self._step = 0
                else:
                    self.n_exps_removed[self._step] += 1
                    self._step += 1
                    if self._step == len(basis[element]):
                        self._step = 0

            # if errors[self._step][ranks[self._step][0]] > self.target:
            #     self.shells_done[self._step] = 0
            #     self._step += 1
            #     if self._step == len(basis[element]):
            #         self._step = 0
            # else:
            #     new_exps = np.delete(basis[element][self._step].exps, ranks[self._step][0])
            #     self.set_active(new_exps, basis, element)
            #     uncontract_shell(basis[element][self._step])
            #     self._just_removed[self._step] = True
            #     self.n_exps_removed[self._step] += 1
            #     return sum(self.shells_done) != 0

        if sum(self.shells_done) == 0:
            return False

        return sum(self.shells_done) != 0


class AutoBasisOpt(Strategy):
    def __init__(self, eval_type: str = "energy", pre: Preconditioner = make_positive):
        self.name = "Default"
        self._eval_type = ""
        self.eval_type = eval_type
        self.params = {}
        self.guess = None
        self.guess_params = {"name": "cc-pvdz"}
        self._step = -1
        self.pre = pre
        self.pre.params = {}
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = True
        self.min_l = 2
        self.min_lh = 1

        self.basis_type = "orbital"
        self.orbital_basis = None

        # currently fixed, to be expanded later
        self.loss = np.linalg.norm

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy (does nothing in default)

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """
        if element.lower() != 'h':
            self._step = self.min_l
        else:
            self._step = self.min_lh
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = True

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        """Moves the strategy forward a step (see algorithm)

        Arguments:
            basis: internal basis dictionary
            element: symbol of atom being optimized
            objective: value of objective function from last steps

        Returns:
            True if there is a next step, False if strategy is finished
        """

        if self.first_run:
            self.first_run = False
            return True

        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective
        self._step += 1
        self.first_run = False
        maxl = len(basis[element])
        return maxl != self._step


class PolarizationStrategy(Strategy):
    """Implements a strategy for an even tempered basis set, where each angular
    momentum shell is described by three parameters: (c, x, n)
    Each exponent in that shell is then given by
        y_k = c*(x**k) for k=0,...,n

    Algorithm:
        Evaluate: energy (can change to any RMSE-compatible property)
        Loss: root-mean-square error
        Guess: null, uses _INITIAL_GUESS above
        Pre-conditioner: None

        Initialisation:
            - Find minimum no. of shells needed
            - max_l >= min_l
            - generate initial parameters for each shell

        First run:
            - optimize parameters for each shell once, sequentially

        Next shell in list not marked finished:
            - re-optimise
            - below threshold or n=max_n: mark finished
            - above threshold: increment n
        Repeat until all shells are marked finished.

        Uses iteration, limited by two parameters:
            max_n: max number of exponents in shell
            target: threshold for objective function

    Additional attributes:
        shells (list): list of (c, x, n) parameter tuples
        shell_done (list): list of flags for whether shell is finished (0) or not (1)
        target (float): threshold for optimization delta
        max_n (int): maximum number of primitives in shell expansion
        max_l (int): maximum angular momentum shell to do;
            if -1, does minimal configuration
    """

    def __init__(self, eval_type: str = "energy", pre: Preconditioner = make_positive):
        self.name = "Polarization"
        self._eval_type = ""
        self.eval_type = eval_type
        self.params = {}
        self.guess = None
        self.guess_params = {"name": "cc-pvdz"}
        self._step = -1
        self.pre = pre
        self.pre.params = {}
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = True
        self.min_l = 2
        self.target = 1e-3

        self.basis_type = "orbital"
        self.orbital_basis = None

        self.all_combinations = {'config': [], 'objective': [], 'increment': []}

        # currently fixed, to be expanded later
        self.loss = np.linalg.norm

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy by determing the initial
        parameters for each angular momentum shell for the
        given element.

        Arguments:
               basis (InternalBasis): the basis set being optimized
               element (str): the atom type of interest
        """
        # if self.max_l < 0:
        #     el = md_element(element.title())
        # l_list = [l for (n, l) in el.ec.conf.keys()]

        if len(basis[element]) < self.min_l:
            raise ValueError(
                "Basis set does not have enough shells. Minimum is {}".format(self.min_l)
            )

        self.shells = [shell.exps.tolist() for shell in basis[element][: len(basis[element])]]
        self.last_objective = 0.0
        self.delta_objective = 0.0
        self.first_run = True
        # self._step = self.min_l
        self._combination = 0
        self._possible_combinations = []
        self._testing = False

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Returns the even temper params for the current shell"""
        y = basis[element][self._step].exps
        return self.pre(y, **self.pre.params)

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Given the even temper params for a shell, expands the basis
        Checks that the smallest exponent is >= 1e-5
        and that the ratio is >= 1.01, to prevent impossible exponents
        """
        y = np.array(values)
        basis[element][self._step].exps = self.pre.inverse(y, **self.pre.params)

    def set_new_basis(self, new_basis: InternalBasis, element: str):
        """Sets the new basis (IN DEVELOPMENT)"""
        basis = new_basis

    def generate_combinations(self, basis: InternalBasis, element: str):
        """Generates all possible combinations of the current shell"""
        possible_combinations = []
        for idx, shell in enumerate(basis[element][self.min_l :]):
            possible_combinations.append((idx + self.min_l, len(shell.exps)))
        possible_combinations.append((len(basis[element]), 0))
        return possible_combinations

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        self.delta_objective = np.abs(self.last_objective - objective)
        self.last_objective = objective
        if self.first_run:
            self._step = self.min_l
            self.first_run = False
            try:
                bo_logger.info(f"Checking if shell {INV_AM_DICT[self._step]} in basis.")
                self.shells[self._step] = basis[element][self._step].exps
            except IndexError:
                shell = Shell()
                shell.l = INV_AM_DICT[self._step]
                basis[element].append(shell)
                basis[element][self._step].exps = np.array([1])
                uncontract_shell(shell)
            self.first_run_just = True
            self.first_run = False
            self.just_optimised = (self._step, len(basis[element][self._step].exps) - 1)
            return True
        elif self.first_run_just:
            self.all_combinations['config'].append(
                get_composition({element: basis[element]}, element)
            )
            self.all_combinations['objective'].append(objective)
            self.all_combinations['increment'].append(self.delta_objective)
            self.first_run_just = False
            # if self.delta_objective < self.target:
            if objective < self.target:
                self.first_run_just = False
                return False

        try:
            self._testing[self._combination] = True
            self._combinations[self._combination] = (
                self._step,
                copy.deepcopy(basis[element]),
                objective,
                objective - self.old_energy,
            )
            if all(self._testing):
                for test in self._combinations:
                    self.all_combinations['config'].append(
                        get_composition({element: test[1]}, element)
                    )
                    self.all_combinations['objective'].append(test[2])
                    self.all_combinations['increment'].append(test[3])
                energies = np.array([test[2] for test in self._combinations])
                errors = np.array([test[3] for test in self._combinations])
                bo_logger.info(f'Increments = {errors}.')
                min_idx = np.argmin(errors)
                ang, test_basis, energy, error = self._combinations[min_idx]
                basis[element] = test_basis
                bo_logger.info(
                    f'Lowest energy basis config = {" ".join([str(len(shell.exps))+shell.l for shell in basis[element]])}.'
                )
                self.last_objective = energy
                # if energy < self.target:
                if objective < self.target:
                    return False
        except:
            pass

        if self._possible_combinations:
            l, n = self._possible_combinations.pop(0)
            basis[element] = self.old_basis
            bo_logger.info(
                f"Previous basis config = {''.join([str(len(shell.exps))+shell.l for shell in basis[element]])}."
            )
            bo_logger.info("Reverting to old basis to test new combination.")
            bo_logger.info(f"Testing shell {INV_AM_DICT[l]} with {n+1} primitives.")
            try:
                exps = basis[element][l].exps.tolist()
                exps.append(exps[-1] / 2)
                basis[element][l].exps = np.array(exps)
                shell = basis[element][l]
            except IndexError:
                shell = Shell()
                shell.l = INV_AM_DICT[l]
                basis[element].append(shell)
                shell.exps = np.array([1])
            self._step = l
            self._combination = self._step - self.min_l
            uncontract_shell(shell)
            bo_logger.info(
                f"Current basis config = {''.join([str(len(shell.exps))+shell.l for shell in basis[element]])}."
            )
            return True
        else:
            self.old_basis = copy.deepcopy(basis[element])
            self.old_energy = objective
            self._possible_combinations = self.generate_combinations(basis, element)
            bo_logger.info(f"Generating new basis combinations for element {element}.")
            bo_logger.info(
                f"Combinations = {','.join([str(n+1)+INV_AM_DICT[l] for l, n in self._possible_combinations])}."
            )
            self._combinations = [()] * len(self._possible_combinations)
            self._testing = [False] * len(self._possible_combinations)
            l, n = self._possible_combinations.pop(0)
            try:
                exps = basis[element][l].exps.tolist()
                exps.append(exps[-1] / 2)
                basis[element][l].exps = np.array(exps)
                uncontract_shell(basis[element][l])
            except IndexError:
                shell = Shell()
                shell.l = INV_AM_DICT[l]
                shell.exps = np.array([1])
                uncontract_shell(shell)
            self._step = l
            self._combination = self._step - self.min_l
            return True
