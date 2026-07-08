import numpy as np

from basisopt.basis.basis import legendre_expansion
from basisopt.containers import InternalBasis

from .tempered import TemperedStrategy


class LegendreStrategy(TemperedStrategy):
    """Implements a strategy for a basis set, where each angular
    momentum shell is determined using Petersson and co-workers' method
    based on Legendre polynomials. See J. Chem. Phys. 118, 1101 (2003).
    A tuple of parameters is required, along with the total
    number of exponents (n).

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

    See :class:`~basisopt.opt.tempered.TemperedStrategy` for the shared state
    machine and serialization.
    """

    _NAME = "Legendre"
    _INITIAL_GUESS = ((3.5, 5.0, 0.8, 0.3, 0.1, 0.1), 10)

    def set_basis_shells(self, basis: InternalBasis, element: str):
        """Expands Legendre parameters into a basis set"""
        basis[element] = legendre_expansion(self.shells)

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Returns the Legendre params for the current shell"""
        (A_vals, _) = self.shells[self._step]
        return np.array(A_vals)

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Given the Legendre params for a shell, expands the basis"""
        (A_vals, n) = self.shells[self._step]
        self.shells[self._step] = (values, n)
        self.set_basis_shells(basis, element)
