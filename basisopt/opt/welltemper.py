import numpy as np

from basisopt.basis.basis import well_temper_expansion
from basisopt.containers import InternalBasis

from .tempered import TemperedStrategy


class WellTemperedStrategy(TemperedStrategy):
    """Implements a strategy for a well tempered basis set, where each angular
    momentum shell is described by five parameters: (c, x, g, d, n)
    Each exponent in that shell is then given by
        y_k = c*(x**k)*(1 + g_l*((k+1)/n_l)**d_l) for k=0,...,n

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

    _NAME = "WellTemper"
    _INITIAL_GUESS = (0.1, 2.2, 20.0, 10.0, 8)

    def set_basis_shells(self, basis: InternalBasis, element: str):
        """Expands well tempered parameters into a basis set"""
        basis[element] = well_temper_expansion(self.shells)

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Returns the well temper params for the current shell"""
        (c, x, g, d, _) = self.shells[self._step]
        return np.array([c, x, g, d])

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Given the well temper params for a shell, expands the basis
        Checks that the smallest exponent is >= 1e-5
        that the ratio is >= 1.01, to prevent impossible exponents
        that the gamma spacing parameter is >= 1.01 to add well tempering
        and that the delta parameter is also >= 1.01 to aid in deviation
        from a rigid geometric series.
        """
        (c, x, g, d, n) = self.shells[self._step]
        c = max(values[0], 1e-5)
        x = max(values[1], 1.01)
        g = max(values[2], 1.01)
        d = max(values[3], 1.01)
        self.shells[self._step] = (c, x, g, d, n)
        self.set_basis_shells(basis, element)
