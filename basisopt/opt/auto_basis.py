import copy
from typing import Any, Optional

import numpy as np
from mendeleev import element as md_element

from basisopt.basis.basis import legendre_expansion, uncontract_shell
from basisopt.containers import InternalBasis
from basisopt.data import _ATOMIC_LEGENDRE_COEFFS
from basisopt.molecule import Molecule
from basisopt.testing.rank import rank_mol_basis_cbs
from basisopt.util import bo_logger

from .preconditioners import make_positive, unit
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
    ):
        super().__init__(eval_type=eval_type, pre=unit)
        self.name = 'AutoBasisDFT'
        self.shell = []
        self.shell_done = []
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
        d["shells"] = self.shells
        d["shell_done"] = self.shell_done
        d["target"] = self.target
        d["max_n"] = self.max_n
        d["max_l"] = self.max_l
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates LegendreStrategy from MSONable dictionary"""
        strategy = Strategy.from_dict(d)
        instance = cls(
            eval_type=d.get("eval_type", 'energy'),
            target=d.get("target", 1e-5),
            max_n=d.get("max_n", 18),
            max_l=d.get("max_l", -1),
        )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.shells = d.get("shells", [])
        instance.shell_done = d.get("shell_done", [])
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
        el = md_element(element.capitalize()).atomic_number

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
        self.name = 'AutoBasisDFTLegendre'
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
        d["shells"] = self.shells
        d["shell_done"] = self.shell_done
        d["target"] = self.target
        d["max_n"] = self.max_n
        d["max_l"] = self.max_l
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates LegendreStrategy from MSONable dictionary"""
        strategy = Strategy.from_dict(d)
        instance = cls(
            eval_type=d.get("eval_type", 'energy'),
            target=d.get("target", 1e-5),
            max_n=d.get("max_n", 18),
            max_l=d.get("max_l", -1),
        )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.shells = d.get("shells", [])
        instance.shell_done = d.get("shell_done", [])
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
        el = md_element(element.capitalize()).atomic_number

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

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of object"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["shells"] = self.shells
        d["shell_done"] = self.shell_done
        d["target"] = self.target
        d["max_n"] = self.max_n
        d["max_l"] = self.max_l
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates LegendreStrategy from MSONable dictionary"""
        strategy = Strategy.from_dict(d)
        instance = cls(
            eval_type=d.get("eval_type", 'energy'),
            target=d.get("target", 1e-5),
            max_n=d.get("max_n", 18),
            max_l=d.get("max_l", -1),
        )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.shells = d.get("shells", [])
        instance.shell_done = d.get("shell_done", [])
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

        while sum(self.shells_done) != 0:
            if self.shells_done[self._step] == 0:
                self._step += 1
                if self._step == len(basis[element]):
                    self._step = 0

            errors, ranks, _, _ = rank_mol_basis_cbs(
                molecule,
                element,
                self.cbs_limit,
                self.eval_type,
                self.params,
            )

            if errors[self._step][ranks[self._step][0]] > self.target:
                self.shells_done[self._step] = 0
                self._step += 1
                if self._step == len(basis[element]):
                    self._step = 0
            else:
                new_exps = np.delete(basis[element][self._step].exps, ranks[self._step][0])
                self.set_active(new_exps, basis, element)
                uncontract_shell(basis[element][self._step])
                self._just_removed[self._step] = True
                self.n_exps_removed[self._step] += 1
                return sum(self.shells_done) != 0

        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective
        if sum(self.shells_done) == 0:
            return False

        return sum(self.shells_done) != 0


class AutoBasisPolarization(Strategy):
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
        self.name = 'AutoBasisDFT'
        self.shell = []
        self.shell_done = []
        self.target = target
        self.guess = None
        self.guess_params = {}
        self.params = {}

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of object"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["shells"] = self.shells
        d["shell_done"] = self.shell_done
        d["target"] = self.target
        d["max_n"] = self.max_n
        d["max_l"] = self.max_l
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates LegendreStrategy from MSONable dictionary"""
        strategy = Strategy.from_dict(d)
        instance = cls(
            eval_type=d.get("eval_type", 'energy'),
            target=d.get("target", 1e-5),
            max_n=d.get("max_n", 18),
            max_l=d.get("max_l", -1),
        )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        instance.shells = d.get("shells", [])
        instance.shell_done = d.get("shell_done", [])
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
        el = md_element(element.capitalize()).atomic_number

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
