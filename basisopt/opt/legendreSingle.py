from typing import Any

import numpy as np
from mendeleev import element as md_element

from basisopt.basis.basis import legendre_expansion
from basisopt.basis.guesses import null_guess
from basisopt.containers import InternalBasis

from .preconditioners import unit
from .strategies import Strategy
from .legendre import LegendreStrategy


#TODO: Implement an option that varies the number of Legendre parameters to create the basis set.

class LegendreSingleOrbitalStrategy(LegendreStrategy):
    """Implements a strategy for a basis set, where each angular
    momentum shell is determined using Petersson and co-workers' method
    based on Legendre polynomials. See J. Chem. Phys. 118, 1101 (2003).
    A tuple of parameters is required, along with the total
    number of exponents (n).
    
    This modification to the original LegendreStrategy is deisnged to only
    optimize a single angular momentum shell whilst holding all others the same.

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
        max_n (int): maximum number of primitives in shell expansion
        max_l (int): maximum angular momentum shell to do;
            if -1, does minimal configuration
    """

    def __init__(
        self, eval_type: str = 'energy', target: float = 1e-5, max_n: int = 9, max_l: int = -1, run_step=None, init_basis=None
    ):
        super().__init__(eval_type=eval_type)
        self.max_n = max_n
        self.max_l = max_l
        self.name = 'Legendre'
        self.run_step = run_step
        self.init_basis = init_basis

    def set_basis_shells(self, basis: InternalBasis, element: str, run_step: int):
        """Expands parameters into a basis set

        Arguments:
             basis (InternalBasis): the basis set to expand
             element (str): the atom type
        """
        
        basis[element] = legendre_expansion(self.shells)
        for ind, shell in enumerate(self.init_basis[element]):
            if ind != self.run_step:
                basis[element][ind] = shell

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy by determining the initial
        parameters for each angular momentum shell for the
        given element.

        Arguments:
                basis (InternalBasis): the basis set being optimized
                element (str): the atom type of interest
        """
        if 'initial_guess' in self.params:
            self._INITIAL_GUESS = self.params['initial_guess']
            self.params.pop('initial_guess')
        else:
            self._INITIAL_GUESS = ((1.5,3,1,0.5), 8)

        
        if self.max_l < 0:
            el = md_element(element.title())
            l_list = [l for (n, l) in el.ec.conf.keys()]
            min_l = len(set(l_list))
            self.max_l = max(min_l, self.max_l)
        
        if type(self.max_n)==tuple:
            self.shells = []
            for n_val in self.max_n:
                self.shells.append((self._INITIAL_GUESS[0],n_val))
        else:
            self.shells = [self._INITIAL_GUESS] * self.max_l
        self.shell_done = [1] * self.max_l
        self.set_basis_shells(basis, element, run_step=self.run_step)
          
        self.last_objective = 0.0
        self.delta_objective = 0.0
        self.first_run = True
        
    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Given the Legendre params for a shell, expands the basis"""
        (A_vals, n) = self.shells[self._step]
        self.shells[self._step] = (values, n)
        self.set_basis_shells(basis, element, self.run_step)

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        self.delta_objective = np.abs(self.last_objective - objective)
        self.last_objective = objective

        carry_on = True
        if self.first_run:
            self._step = self.run_step
            if self._step == self.run_step:
                self.first_run = False
                self._step = self.run_step
                (A_vals, n) = self.shells[self._step]
                try:
                    self.shells[self._step] = (A_vals, min(n + 1, self.max_n[self._step]))
                except:
                    self.shells[self._step] = (A_vals, min(n + 1, self.max_n))
        else:
            if self.delta_objective < self.target:
                self.shell_done[self._step] = 0
                
            (A_vals, n) = self.shells[self._step]
            if type(self.max_n)==tuple:
                if n == self.max_n[self._step]:
                    self.shell_done[self._step] = 0
                elif self.shell_done[self._step] != 0:
                    self.shells[self._step] = (A_vals, n + 1)
            else:
                if n == self.max_n:
                    self.shell_done[self._step] = 0
                elif self.shell_done[self._step] != 0:
                    self.shells[self._step] = (A_vals, n + 1)
            carry_on = self.shell_done[self._step] != 0

        return carry_on
    
class LegendreSingleOrbitalAStrategy(LegendreSingleOrbitalStrategy):
    """Implements a strategy for a basis set, where each angular
    momentum shell is determined using Petersson and co-workers' method
    based on Legendre polynomials. See J. Chem. Phys. 118, 1101 (2003).
    A tuple of parameters is required, along with the total
    number of exponents (n).
    
    This modification to the original LegendreStrategy is deisnged to only
    optimize a single angular momentum shell whilst holding all others the same.

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
        max_n (int): maximum number of primitives in shell expansion
        max_l (int): maximum angular momentum shell to do;
            if -1, does minimal configuration
    """

    def __init__(
        self, eval_type: str = 'energy', target: float = 1e-5, max_n: int = 9, max_l: int = -1, run_step=None, init_basis=None
    ):
        super().__init__(eval_type=eval_type)
    
    #def generate_initial_A_vals_guess(A_max):
        
    
    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        self.delta_objective = np.abs(self.last_objective - objective)
        self.last_objective = objective
    
        carry_on = True
        if self.first_run:
            self._step = self.run_step
            if self._step == self.run_step:
                self.first_run = False
                self._step = self.run_step
                (A_vals, n) = self.shells[self._step]
                try:
                    self.shells[self._step] = (A_vals, min(n + 1, self.max_n[self._step]))
                except:
                    self.shells[self._step] = (A_vals, min(n + 1, self.max_n))
        else:
            if self.delta_objective < self.target:
                self.shell_done[self._step] = 0
                
            (A_vals, n) = self.shells[self._step]
            if type(self.max_n)==tuple:
                if n == self.max_n[self._step]:
                    self.shell_done[self._step] = 0
                elif self.shell_done[self._step] != 0:
                    self.shells[self._step] = (A_vals, n + 1)
            else:
                if n == self.max_n:
                    self.shell_done[self._step] = 0
                elif self.shell_done[self._step] != 0:
                    self.shells[self._step] = (A_vals, n + 1)
            carry_on = np.sum(self.shell_done) != 0
    
        return carry_on
