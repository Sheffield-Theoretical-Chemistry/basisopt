from typing import Any

import numpy as np
from mendeleev import element as md_element

from basisopt import bo_logger

#from basisopt.basis.guesses import null_guess
from basisopt.containers import InternalBasis

from basisopt.basis.basis import contract_basis, contract_function, legendre_expansion
from .preconditioners import unit
from .strategies import Strategy


class ContractionStrategy(Strategy):
    """

    Implements a strategy for a basis set, where the contraction coefficients
    for each angular momentum are optimised to an uncontracted basis set.
    This method can be used after a set of exponents have been optimised
    or on an uncontracted basis set.

    Algorithm:
        Evaluate: energy (can change to any RMSE-compatible property)
        Loss: root-mean-square error
        Guess: null, uses _INITIAL_GUESS above
        Pre-conditioner: None

        Initialisation:
            - Find the minimum number of shells per angular momentum
            - Reorganise the basis set to the correct number of contracted functions
            -

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
        max_n (int): maximum number of primitives in shell expansion
        max_l (int): maximum angular momentum shell to do;
            if -1, does minimal configuration
    """

    def __init__(
        self,
        eval_type: str = "energy",
        target: float = 1e-5,
        max_n: int = 18,
        max_l: int = -1,
    ):
        super().__init__(eval_type=eval_type, pre=unit)
        self.name = "Contraction"
        self.shells = []
        self.shell_done = []
        self.target = target
        self.guess = None
        self.guess_params = {}
        self.max_n = max_n
        self.max_l = max_l
        self.number_of_contractions = None

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
            eval_type=d.get("eval_type", "energy"),
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

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Returns the currently active contraction coefficients"""
        coefficients = self.shells[self._step][self._n_step]
        return coefficients

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Given a series of coefficients for a shell, set the contractions for the basis"""
        coefficients = self.shells[self._step][self._n_step]
        self.shells[self._step][self._n_step] = values
        self.set_basis_shells(basis=basis, element=element, values=values)

    def set_basis_contractions(self, basis, contractions):
        """Sets the initial guess for the contraction coefficients"""
        contract_basis(basis, contractions)

    def set_basis_shells(self, basis: InternalBasis, element: str, values: np.ndarray):
        """Expands parameters into a basis set

        Arguments:
             basis (InternalBasis): the basis set to expand
             element (str): the atom type
        """
        contract_function(basis, element, self._step, self._n_step, values)

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy by determining the initial
        parameters for each angular momentum shell for the
        given element.

        Arguments:
               basis (InternalBasis): the basis set being optimized
               element (str): the atom type of interest
        """

        el = md_element(element.title())
        l_list = [l for (n, l) in el.ec.conf.keys()]
        min_l = len(set(l_list))

        self.max_l = max(min_l, self.max_l)

        self._step = 0  # Selects current shell
        self._n_step = 0  # Sets the active contraction function within the shell

        if 'initial_guess' not in self.guess_params or self.guess_params['initial_guess'] is None:
            self.shells = [
                basis[element.lower()][idx].coefs for idx in range(len(basis[element.lower()]))
            ]
        elif self.guess_params['initial_guess']:
            self.shells = self.guess_params['initial_guess']

        self.number_of_contractions = [len(shell.coefs) for shell in basis[element.lower()]]

        self.sub_shells_done = [[1] * n_func for n_func in self.number_of_contractions]

        self.last_objective = 0.0
        self.delta_objective = 0.0
        self.first_run = True

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective

        carry_on = True

        if self.first_run:
            self.first_run = False
            return carry_on
        elif self.delta_objective < self.target:
            self.sub_shells_done[self._step][self._n_step] = 0
            if self._n_step == self.number_of_contractions[self._step] - 1:
                self._n_step = 0
                self._step += 1
                if self._step == len(basis[element]):
                    return False
            else:
                self._n_step += 1
            return carry_on

        maxl = len(basis[element])
        carry_on = maxl != self._step
        return maxl != self._step
