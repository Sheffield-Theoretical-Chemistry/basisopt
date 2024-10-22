from typing import Any

import numpy as np
from monty.json import MSONable
from mendeleev import element as md_element

from basisopt import api
from basisopt import bo_logger
from basisopt.containers import InternalBasis, basis_to_dict
from basisopt.exceptions import PropertyNotAvailable
from basisopt.util import bo_logger, dict_decode
from basisopt.containers import Shell
from basisopt.data import INV_AM_DICT
from basisopt.basis.basis import uncontract_shell

from basisopt.opt.strategies import Strategy

import copy
from .preconditioners import Preconditioner, make_positive


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
        """Sets the new basis"""
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
            if objective < self.target:
                self.first_run_just = False
                return False
        

        try:
            self._testing[self._combination] = True
            self._combinations[self._combination] = (
                self._step,
                copy.deepcopy(basis[element]),
                objective,
            )
            if all(self._testing):
                energies = np.array([test[2] for test in self._combinations])
                min_idx = np.argmin(energies)
                ang, test_basis, energy = self._combinations[min_idx]
                basis[element] = test_basis
                bo_logger.info(f'Lowest energy basis config = {" ".join([str(len(shell.exps))+shell.l for shell in basis[element]])}.')
                self.last_objective = energy
                if energy < self.target:
                    return False
        except:
            pass

        if self._possible_combinations:
            l, n = self._possible_combinations.pop(0)
            basis[element] = self.old_basis[element]
            bo_logger.info(f"Previous basis config = {''.join([str(len(shell.exps))+shell.l for shell in basis[element]])}.")
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
            bo_logger.info(f"Current basis config = {''.join([str(len(shell.exps))+shell.l for shell in basis[element]])}.")
            return True
        else:
            self.old_basis = copy.deepcopy(basis)
            self._possible_combinations = self.generate_combinations(basis, element)
            bo_logger.info(f"Generating new basis combinations for element {element}.")
            bo_logger.info(f"Combinations = {','.join([str(n+1)+INV_AM_DICT[l] for l, n in self._possible_combinations])}.")
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
