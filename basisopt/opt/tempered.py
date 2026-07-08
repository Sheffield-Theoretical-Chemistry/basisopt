from typing import Any

import numpy as np
from mendeleev import element as md_element

from basisopt.basis.guesses import null_guess
from basisopt.containers import InternalBasis

from .preconditioners import unit
from .strategies import Strategy


class TemperedStrategy(Strategy):
    """Shared base for the even-/well-tempered and Legendre strategies.

    Each angular-momentum shell is described by a parameter tuple whose **last**
    element is the primitive count ``n``. The three concrete strategies differ
    only in that tuple's shape, the expansion used to turn it into a basis, and
    which entries are optimised; everything below - the ``initialise``/``next``
    state machine (sequential first pass over shells, then growing ``n`` up to
    ``max_n`` and marking a shell done once its objective delta drops below
    ``target``), serialization, and construction - is shared.

    Subclasses provide:
        _NAME (str): strategy identifier used for ``self.name``
        _INITIAL_GUESS: starting parameter tuple applied to every shell
        set_basis_shells(basis, element): expand ``self.shells`` into the basis
        get_active/set_active: read/write the optimisable params of the current shell

    Attributes:
        shells (list): per-shell parameter tuples (last element is ``n``)
        shell_done (list): 0 if a shell is finished, 1 otherwise
        target (float): threshold on the objective delta
        max_n (int): maximum number of primitives in a shell expansion
        max_l (int): maximum angular momentum to do; if -1, minimal configuration
    """

    _NAME = "Tempered"
    _INITIAL_GUESS: Any = None

    def __init__(
        self,
        eval_type: str = "energy",
        target: float = 1e-5,
        max_n: int = 18,
        max_l: int = -1,
    ):
        super().__init__(eval_type=eval_type, pre=unit)
        self.name = self._NAME
        self.shells = []
        self.shell_done = []
        self.target = target
        self.guess = null_guess
        self.guess_params = {}
        self.max_n = max_n
        self.max_l = max_l

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
        """Creates the strategy from an MSONable dictionary"""
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

    def set_basis_shells(self, basis: InternalBasis, element: str):
        """Expands ``self.shells`` into the basis - overridden per strategy"""
        raise NotImplementedError

    def initialise(self, basis: InternalBasis, element: str):
        """Initialises the strategy by determining the initial parameters for
        each angular momentum shell for the given element.

        Arguments:
               basis (InternalBasis): the basis set being optimized
               element (str): the atom type of interest
        """
        el = md_element(element.title())
        l_list = [l for (n, l) in el.ec.conf.keys()]
        min_l = len(set(l_list))

        self.max_l = max(min_l, self.max_l)
        self.shells = [self._INITIAL_GUESS] * self.max_l
        self.shell_done = [1] * self.max_l
        self.set_basis_shells(basis, element)
        self.last_objective = 0.0
        self.delta_objective = 0.0
        self.first_run = True

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        self.delta_objective = np.abs(self.last_objective - objective)
        self.last_objective = objective

        carry_on = True
        if self.first_run:
            self._step = self._step + 1
            if self._step == self.max_l:
                self.first_run = False
                self._step = 0
                shell = self.shells[self._step]
                self.shells[self._step] = shell[:-1] + (min(shell[-1] + 1, self.max_n),)
        else:
            if self.delta_objective < self.target:
                self.shell_done[self._step] = 0

            self._step = (self._step + 1) % self.max_l
            shell = self.shells[self._step]
            n = shell[-1]
            if n == self.max_n:
                self.shell_done[self._step] = 0
            elif self.shell_done[self._step] != 0:
                self.shells[self._step] = shell[:-1] + (n + 1,)

            carry_on = np.sum(self.shell_done) != 0

        return carry_on
