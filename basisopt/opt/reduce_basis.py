import copy
from typing import Any, Optional

import numpy as np
from mendeleev import element as md_element

from basisopt.basis.basis import legendre_expansion, uncontract
from basisopt.containers import InternalBasis
from basisopt.data import _ATOMIC_LEGENDRE_COEFFS
from basisopt.molecule import Molecule
from basisopt.testing.rank import rank_basis, get_smallest_error_element
from basisopt.util import bo_logger

from .preconditioners import Preconditioner, make_positive, unit
from .strategies import Strategy


class ReduceStrategyAll(Strategy):
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

    def get_index_of_min(self, errors):

        def find_min_index_2d(arrays):
            # Flatten the list of arrays into a single array
            flat_array = np.concatenate(arrays)

            # Find the index of the minimum value in the flattened array
            min_index_flat = np.argmin(flat_array)

            # Compute the cumulative lengths of the arrays
            lengths = [len(arr) for arr in arrays]
            cumulative_lengths = np.cumsum([0] + lengths)

            # Determine in which array the minimum value resides
            array_index = np.searchsorted(cumulative_lengths, min_index_flat, side='right') - 1

            # Get the index within that specific array
            index_within_array = min_index_flat - cumulative_lengths[array_index]

            return array_index, index_within_array

        min_index = find_min_index_2d(errors)
        return min_index

    def initialise(self, molecule):
        """Initialises the strategy (does nothing in default)

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """
        # self.elements_dict = {}
        # self._step = 0
        # for element in basis:
        #     self.elements_dict[element] = {}
        #     self.elements_dict[element]['shells done'] = [1] * len(basis[element])
        #     self.last_objective = 0
        #     self.delta_objective = 0
        #     self.elements_dict[element]['first run'] = [True] * len(basis[element])
        #     self.init_run = True
        #     self._just_removed = [False] * len(basis[element])
        #     self.elements_dict[element]['original_shells'] = [copy.deepcopy(shell) for shell in basis[element]]
        #     self.elements_dict[element]['original_size'] = [len(shell.exps) for shell in basis[element]]
        #     self.elements_dict[element]['n_exps_removed'] = [0] * len(basis[element])
        #     self.elements_dict[element]['old_exps'] = [None] * len(basis[element])

        self.element_ranks = rank_basis(molecule, self.eval_type, self.params)
        self.current_element = get_smallest_error_element(self.element_ranks)
        self.min_l, self.exp_idx = self.get_index_of_min(
            self.element_ranks[self.current_element]['errors']
        )
        self._step = self.min_l
        self.init = True

        if self.target is None:
            raise ValueError('Target not set. This can be set with the .set_target method.')

    def set_target(self, target: float):
        """Sets the target for the strategy

        Arguments:
            target: the target for the strategy
        """
        self.target = target

    def get_active(self, basis: InternalBasis) -> np.ndarray:
        """Arguments:
             basis: internal basis dictionary
             element: symbol of the atom being optimized

        Returns:
             the set of exponents currently being optimised
        """
        elbasis = basis[self.current_element]
        x = elbasis[self.min_l].exps
        return self.pre(x, **self.pre.params)

    def set_active(self, values: np.ndarray, basis: InternalBasis):
        """Sets the currently active exponents to the given values.

        Arguments:
            values (list): list of new exponents
            basis: internal basis dictionary
            element: symbol of atom being optimized
        """
        elbasis = basis[self.current_element]
        y = np.array(values)
        elbasis[self.min_l].exps = self.pre.inverse(y, **self.pre.params)

    def next(
        self,
        molecule: Molecule,
        # backend_params: dict,
        basis: InternalBasis,
        # element: str,
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

        if self.init:
            # Remove smallest molecule and reoptimise
            self.init = False
            self.old_exps = np.copy(basis[self.current_element][self.min_l].exps)
            if len(self.old_exps) == 1:
                return False
            self.new_exps = np.delete(basis[self.current_element][self.min_l].exps, self.exp_idx)
            self.set_active(self.new_exps, basis)
            uncontract(basis)
            self.just_removed = True
            return True

        if self.just_removed:
            self.just_removed = False
            if abs(objective) < abs(self.target):
                self.set_active(self.old_exps, basis)
                uncontract(basis)
                return False
            else:
                self.element_ranks = rank_basis(molecule, self.eval_type, self.params)
                self.current_element = get_smallest_error_element(self.element_ranks)
                self.min_l, self.exp_idx = self.get_index_of_min(
                    self.element_ranks[self.current_element]['errors']
                )
                self._step = self.min_l
                self.old_exps = np.copy(basis[self.current_element][self.min_l].exps)
                self.new_exps = np.delete(
                    basis[self.current_element][self.min_l].exps, self.exp_idx
                )
                if len(self.old_exps) == 1:
                    return False
                self.set_active(self.new_exps, basis)
                uncontract(basis)
                self.just_removed = True
                return True
