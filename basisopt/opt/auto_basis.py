from typing import Any

import numpy as np
from mendeleev import element as md_element

from basisopt.basis.basis import uncontract_shell
from basisopt.containers import InternalBasis
from basisopt.data import ATOMIC_DFT_CBS
from basisopt.util import bo_logger

from .preconditioners import unit
from .strategies import Strategy


class AutoBasisDFT(Strategy):
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
        self.name = 'LegendrePairsHybrid'
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

        element_cbs_limit = ATOMIC_DFT_CBS[el]

        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective

        objective_diff = np.abs(objective - element_cbs_limit)

        if self.first_run[self._step]:
            if self._step == -1:
                self._step = 0
            else:
                self._step += 1
            self.first_run[self._step] = False
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
