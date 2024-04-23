#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:54:47 2024

@author: shaun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:09:55 2024

@author: shaun
"""

from typing import Any

import numpy as np
from mendeleev import element as md_element

from basisopt import data
from basisopt.basis.basis import legendre_expansion
from basisopt.basis.guesses import bse_guess, legendre_guess, load_guess, null_guess
from basisopt.containers import InternalBasis

from .preconditioners import unit
from .strategies import Strategy


class LegendrePairsHybrid(Strategy):
    """Implements a strategy for a basis set, where each angular
    momentum shell is determined using Petersson and co-workers' method
    based on Legendre polynomials. See J. Chem. Phys. 118, 1101 (2003).
    A tuple of parameters is required, along with the total
    number of exponents (n).

    This strategy aims to optimise just a single set of exponents using a
    the Legendre Expansion. The strategy should be passed an initial guess for
    the functions one wishes to optimise.

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
        self.guess = null_guess
        self.guess_params = {}
        self.params = {}
        self.max_n = max_n
        self.max_n_a = max_n_a
        self.l = l
        self.n_exp_cutoff = n_exp_cutoff
        self.npasses = None
        self.force_pass = False
        self._just_added = False
        self.initialised = None

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

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Returns the Legendre params for the current shell"""
        (A_vals, n) = self.shells[self._step][0]
        if self._just_added:
            # self._just_added = False
            if n < self.n_exp_cutoff:
                return np.array(A_vals)
            else:
                return np.array(A_vals)[-2:]
        else:
            return np.array(A_vals)

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Given the Legendre params for a shell, expands the basis"""
        (A_vals, n) = self.shells[self._step][0]
        if n < self.n_exp_cutoff:
            self.shells[self._step][0] = (values, n)
        else:
            if self._just_added:
                self.shells[self._step][0][0][-2:] = values
            else:
                self.shells[self._step][0] = (values, n)
        self.set_basis_shell(basis, element)
        # print(values)

    def set_basis_shell(self, basis: InternalBasis, element: str):
        """Expands parameters into a basis set

        Arguments:
             basis (InternalBasis): the basis set to expand
             element (str): the atom type
        """
        if self.shells[self._step][0][1] < self.n_exp_cutoff:
            basis[element][self._step].exps = self.pre.inverse(
                abs(self.shells[self._step][0][0]), **self.pre.params
            )
        else:
            basis[element][self._step] = legendre_expansion(self.shells[self._step], l=self._step)[
                0
            ]
        # print(self.shells[self._step][0][0])
        # print(basis[element][self._step].exps)

    def initialise(self, basis: InternalBasis, element: str):
        el = md_element(element.title())
        self._step = 0
        if not self.initialised:
            self.initialised = {element: False for element, _ in basis.items()}
        self.shells_done = [1] * len(basis[element])
        self.last_objective = 0.0
        self.delta_objective = 0.0
        self.first_run = True
        if not self.initialised[element]:
            # TODO: Make it so that a custom n can be set here
            try:
                self._INITIAL_Guess = self.guess_params['initial_guess']
            except:
                self._INITIAL_Guess = []
                for shell in basis[element.lower()]:
                    if len(shell.exps) < self.n_exp_cutoff:
                        self._INITIAL_Guess.append([(shell.exps, len(shell.exps))])
                    else:
                        self._INITIAL_Guess.append(
                            [
                                (
                                    np.array([2.0 if i % 2 == 0 else -2.0 for i in range(2)]),
                                    len(shell.exps),
                                )
                            ]
                        )
                # self._INITIAL_Guess = [[(np.array([.5 if i % 2 == 0 else -1. for i in range(2)]),len(shell.exps))] for shell in basis[element.lower()]]
            self.shells = self._INITIAL_Guess
            self.initialised[element] = True
        else:
            self.shells = [
                (
                    [(shell.exps, len(shell.exps))]
                    if len(shell.exps) < self.n_exp_cutoff
                    else [shell.leg_params]
                )
                for shell in basis[element]
            ]
        # self.l = self.guess_params['l']
        self.pass_number = 0
        self.set_basis_shell(basis, element)

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        self.delta_objective = np.abs(self.last_objective - objective)
        self.last_objective = objective
        carry_on = True
        (A_vals, n) = self.shells[self._step][0]
        if type(self.max_n_a) == tuple or type(self.max_n_a) == list:
            max_n_a = self.max_n_a[self._step]
        else:
            max_n_a = self.max_n_a
        if n < self.n_exp_cutoff:
            if self.first_run:
                self.first_run = False
                A_vals = basis[element][self._step].exps
                self.shells[self._step][0] = (A_vals, n)
                return carry_on
            if self.delta_objective < self.target:
                self.shells_done[self._step] = 0
            else:
                return carry_on
        else:
            if self.first_run:
                self.first_run = False
                return carry_on
            if len(A_vals) < max_n_a and len(A_vals) != n:
                if not self._just_added:
                    A_vals = np.append(A_vals, np.array([A_vals[-2:] / 10]))
                    self.shells[self._step][0] = (A_vals, n)
                    self._just_added = True
                elif self._just_added:
                    self.shells[self._step][0] = (A_vals, n)
                    self._just_added = False
            elif len(A_vals) == max_n_a and self._just_added == True:
                self.shells[self._step][0] = (A_vals, n)
                self._just_added = False
            else:
                self._just_added = False
                # self.shells_done[self._step] = 0
                if self.delta_objective < self.target:
                    self.shells_done[self._step] = 0
                else:
                    return carry_on

        carry_on = sum(self.shells_done) != 0
        self.pass_number = self.pass_number + 1
        if self.shells_done[self._step] == 0:
            self._step += 1
        return carry_on
