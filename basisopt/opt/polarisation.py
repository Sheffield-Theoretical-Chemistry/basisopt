"""Polarisation-function growth strategy.

``AutoBasisPolarisation`` grows polarisation shells (d, f, g ...; p for H) onto an
existing sp atomic basis. It reuses the :class:`AutoBasisStrategy` growth
machinery -- raw-exponent get/set, the optional ``max_n``/``max_its``/``stall_tol``
cutoffs and MSONable serialization -- but *adds new angular-momentum shells*
rather than growing existing ones, and is driven inside the multi-molecule
``collective_polarize`` optimiser.
"""

from typing import Any, Optional

import numpy as np

from basisopt.containers import InternalBasis, Shell
from basisopt.data import INV_AM_DICT
from basisopt.util import bo_logger

from ..basis.basis import uncontract_shell
from .auto_basis import AutoBasisStrategy
from .preconditioners import Preconditioner, make_positive


class AutoBasisPolarisation(AutoBasisStrategy):
    """Grows polarisation shells (d, f, g ...; p for H) onto an existing sp basis.

    Unlike the atomic growth strategies, this one *adds* new angular-momentum
    shells (``l = min_l .. max_l``) after the existing shells and grows the
    primitives within each, leaving the lower (sp) shells untouched.

    It runs inside the multi-molecule ``collective_polarize`` optimiser, so the
    ``objective`` it sees is the chosen basis-set-incompleteness loss over the
    reference-molecule set (see ``POLARISATION_LOSSES``) -- a small, non-negative
    number that shrinks as the basis improves. Every stopping criterion is
    energy/convergence centred, and each is active only if set:

    - **target**: stop once the loss drops below ``target`` (the absolute or
      tier-relative "good enough" cutoff; ``stop_reason = "target"``);
    - **delta_e**: when a growth step lowers the loss by less than ``delta_e`` the
      added function has not earned its place -- *restore the previous step* (drop
      it) and stop (``stop_reason = "converged"``);
    - **stall_tol**: when adding a primitive lowers the loss by less than
      ``stall_tol``, treat the *current shell* as saturated and advance to the
      next ``l`` (per-shell diminishing returns);
    - bounded by ``max_n`` (primitives per shell), ``max_l`` (highest ``l`` added)
      and ``max_its`` (total growth steps). ``stop_reason`` records which fired.

    Attributes:
        min_l (int): first polarisation angular momentum (2 = d; use 1 for H)
        max_l (int): highest polarisation angular momentum to add
        seed_exponent (float): starting exponent for a freshly-added shell
        target (float): loss threshold for the absolute/relative "good enough" stop
        delta_e (float): per-step loss gain below which the last function is
            dropped and growth stops (restore-previous-step convergence)
        stall_tol (float): per-primitive within-shell diminishing-returns threshold
    """

    def __init__(
        self,
        eval_type: str = "energy",
        target: float = 1e-4,
        min_l: int = 2,
        max_l: int = 3,
        seed_exponent: float = 1.0,
        max_n: Optional[int] = None,
        max_its: Optional[int] = None,
        stall_tol: Optional[float] = 1e-5,
        delta_e: Optional[float] = None,
        pre: Preconditioner = make_positive,
    ):
        super().__init__(eval_type=eval_type, target=target, pre=pre)
        self.name = "AutoBasisPolarisation"
        self.min_l = min_l
        self.max_l = max_l
        self.seed_exponent = seed_exponent
        self.max_n = max_n
        self.max_its = max_its
        self.stall_tol = stall_tol
        self.delta_e = delta_e

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of object."""
        d = super().as_dict()
        d["min_l"] = self.min_l
        d["max_l"] = self.max_l
        d["seed_exponent"] = self.seed_exponent
        d["delta_e"] = self.delta_e
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates the strategy from an MSONable dictionary."""
        instance = super().from_dict(d)
        instance.min_l = d.get("min_l", 2)
        instance.max_l = d.get("max_l", 3)
        instance.seed_exponent = d.get("seed_exponent", 1.0)
        instance.delta_e = d.get("delta_e", None)
        return instance

    def initialise(self, basis: InternalBasis, element: str):
        """Snapshots the existing (sp) basis and resets per-run growth state.

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom whose polarisation shells are grown
        """
        if self.max_l < self.min_l:
            raise ValueError(f"max_l ({self.max_l}) must be >= min_l ({self.min_l}).")
        self._pol_ls = list(range(self.min_l, self.max_l + 1))
        self._pol_idx = 0
        # point at the last existing shell so the driver's initial
        # objective(get_active(...)) evaluates the sp-only baseline harmlessly
        self._step = max(len(basis[element]) - 1, 0)
        self.first_run = True
        self.last_objective = 0.0
        self.delta_objective = 0.0
        self._iter = 0
        self._grown_once = False
        self._shell_grown = False
        # restore-previous-step (delta_e) bookkeeping
        self._grew_last = False
        self._objective_before_grow = None
        self._last_step = None
        self.stop_reason = None

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        """Advances the growth one step; see the class docstring for the algorithm.

        Returns:
            True if there is a next step, False if the strategy is finished.
        """
        self.delta_objective = np.abs(objective - self.last_objective)
        # signed improvement delivered by the last step (>0 means the loss fell)
        improvement = (
            self._objective_before_grow - objective
            if (self._grew_last and self._objective_before_grow is not None)
            else None
        )
        self.last_objective = objective

        # (1) absolute / relative target: the loss is small enough.
        if not self.first_run and objective < self.target:
            self.stop_reason = "target"
            return False

        # first entry: add the first polarisation shell and optimise it.
        if self.first_run:
            self.first_run = False
            self._add_pol_shell(basis, element)
            return True

        # (2) delta_e: the last added function barely helped -> drop it (restore
        # the previous step) and stop. Checked after target, so a function that
        # *meets* the target is always kept.
        if self.delta_e is not None and improvement is not None and improvement < self.delta_e:
            self._restore_last_step(basis, element)
            self.stop_reason = "converged"
            bo_logger.info(
                "AutoBasisPolarisation: last function lowered the loss by %.2e "
                "< delta_e=%.2e; restored the previous step and stopped.",
                improvement,
                self.delta_e,
            )
            return False

        # (backstop) hard iteration cap.
        if self.max_its is not None and self._iter >= self.max_its:
            self.stop_reason = "max_its"
            bo_logger.warning(
                "AutoBasisPolarisation: hit max_its=%d at loss=%.2e; stopping "
                "before convergence.",
                self.max_its,
                objective,
            )
            return False

        n = len(basis[element][self._step].exps)
        # is the current shell saturated? (per-primitive diminishing returns, or max_n)
        shell_saturated = (
            self.stall_tol is not None
            and self._shell_grown
            and self.delta_objective < self.stall_tol
        ) or (self.max_n is not None and n >= self.max_n)

        if shell_saturated:
            # move up in angular momentum, unless we have run out.
            self._pol_idx += 1
            if self._pol_idx >= len(self._pol_ls):
                self.stop_reason = "max_l"
                return False
            self._add_pol_shell(basis, element)
            return True

        # grow the current polarisation shell by one primitive
        self._grow_current_shell(basis, element)
        self._iter += 1
        self._grown_once = True
        self._shell_grown = True
        return True

    def _add_pol_shell(self, basis: InternalBasis, element: str):
        """Append a new l-shell (one seed primitive) and make it current.

        Records a rollback marker so a delta_e cutoff can drop the whole shell and
        restore the previous step.
        """
        l_index = self._pol_ls[self._pol_idx]
        # snapshot for a possible delta_e rollback of this seed
        self._last_step = ("seed", len(basis[element]))
        self._objective_before_grow = self.last_objective
        self._grew_last = True
        shell = Shell()
        shell.l = INV_AM_DICT[l_index]
        shell.exps = np.array([float(self.seed_exponent)])
        uncontract_shell(shell)
        basis[element].append(shell)
        self._step = len(basis[element]) - 1
        self._shell_grown = False

    def _grow_current_shell(self, basis: InternalBasis, element: str):
        """Append one primitive to the current shell (extrapolated ratio).

        Snapshots the pre-grow exponents so a delta_e cutoff can restore this
        (previous) step if the added primitive fails to earn its place.
        """
        shell = basis[element][self._step]
        exps = shell.exps
        # snapshot for a possible delta_e rollback of this grow
        self._last_step = ("grow", self._step, np.array(exps, copy=True))
        self._objective_before_grow = self.last_objective
        self._grew_last = True
        if len(exps) >= 2:
            new_exp = exps[-1] * (exps[-1] / exps[-2])
        else:
            new_exp = exps[-1] / 3.0
        shell.exps = np.append(exps, new_exp)
        uncontract_shell(shell)

    def _restore_last_step(self, basis: InternalBasis, element: str):
        """Undo the last growth step for a delta_e rollback: drop the just-seeded
        shell, or truncate the just-added primitive, restoring the previous
        (better-value) basis."""
        if self._last_step is None:
            return
        kind = self._last_step[0]
        if kind == "seed":
            _, n_before = self._last_step
            del basis[element][n_before:]
            self._step = max(len(basis[element]) - 1, 0)
        elif kind == "grow":
            _, step, exps = self._last_step
            shell = basis[element][step]
            shell.exps = np.array(exps, copy=True)
            uncontract_shell(shell)
        self._last_step = None
        self._grew_last = False
