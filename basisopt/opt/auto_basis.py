import copy
from typing import Any, Optional

import numpy as np

from basisopt.basis.basis import legendre_expansion, uncontract_shell
from basisopt.containers import InternalBasis
from basisopt.data import _ATOMIC_LEGENDRE_COEFFS
from basisopt.testing.rank import rank_mol_basis_cbs
from basisopt.util import bo_logger

from .preconditioners import Preconditioner, make_positive, unit
from .strategies import Strategy


class AutoBasisStrategy(Strategy):
    """Shared base for the automatic basis-set optimization strategies.

    Holds the CBS-limit/target bookkeeping, the raw-exponent get/set, and the
    MSONable serialization common to :class:`AutoBasisFree` and the reduce
    strategies. Subclasses implement ``initialise`` and ``next``;
    :class:`AutoBasisLegendre` additionally overrides ``get_active``/
    ``set_active`` because it optimises Legendre expansion coefficients rather
    than raw exponents.

    Attributes:
        target (float): convergence threshold on ``|objective - cbs_limit|``
        cbs_limit (float): complete-basis-set limit for the property being
            optimized; must be set via :meth:`set_cbs_limit` before ``initialise``
    """

    def __init__(
        self,
        eval_type: str = 'energy',
        target: float = 1e-6,
        pre: Preconditioner = make_positive,
    ):
        super().__init__(eval_type=eval_type, pre=pre)
        self.target = target
        self.guess = None
        self.guess_params = {}
        self.params = {}
        self.cbs_limit = None
        # Optional growth cutoffs (None = disabled). The growth strategies stop
        # primarily on the signed CBS target; these are safety nets set by the
        # caller/config. See AutoBasisLegendre.next / AutoBasisFree.next.
        self.max_n = None  # hard cap on primitives per shell
        self.max_its = None  # cap on total growth iterations
        self.stall_tol = None  # stop a shell once adding a primitive moves the
        #                        objective by less than this (saturation)
        # Set by next() to why it stopped: 'target' | 'stall' | 'max_n' | 'max_its'
        self.stop_reason = None
        # Optional Ray parallelism for the reduce strategies' ranking trials
        # (set by the pipeline step from backend.parallel). None -> serial.
        self.parallel = False
        self.ray_params = None

    def set_cbs_limit(self, cbs_limit: float):
        """Sets the CBS limit used as the optimization target."""
        self.cbs_limit = cbs_limit

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Returns the (preconditioned) exponents of the current shell."""
        x = basis[element][self._step].exps
        return self.pre(x, **self.pre_params)

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Sets the current shell's exponents from preconditioned values."""
        y = np.array(values)
        basis[element][self._step].exps = self.pre.inverse(y, **self.pre_params)

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of object"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["cbs_limit"] = self.cbs_limit
        d["max_n"] = self.max_n
        d["max_its"] = self.max_its
        d["stall_tol"] = self.stall_tol
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates the strategy from an MSONable dictionary"""
        strategy = Strategy.from_dict(d)
        instance = cls(
            eval_type=d.get("eval_type", 'energy'),
            target=d.get("target", 1e-6),
        )
        instance.name = strategy.name
        instance.params = strategy.params
        instance.first_run = strategy.first_run
        instance._step = strategy._step
        instance.last_objective = strategy.last_objective
        instance.delta_objective = strategy.delta_objective
        # carry the base attributes as_dict wrote but the ctor does not take, so
        # an auxiliary-basis strategy does not silently revert to "orbital"
        instance.basis_type = strategy.basis_type
        instance.orbital_basis = strategy.orbital_basis
        instance.pre_params = strategy.pre_params
        instance.cbs_limit = d.get("cbs_limit", None)
        instance.max_n = d.get("max_n", None)
        instance.max_its = d.get("max_its", None)
        instance.stall_tol = d.get("stall_tol", None)
        return instance


class AutoBasisFree(AutoBasisStrategy):
    """Grows a fully free (non-parametrised) atomic basis to the CBS limit.

    Each shell is optimised in turn; after an initial sweep over the existing
    shells, exponents are appended one at a time (extrapolating the outermost
    ratio) and re-optimised until ``|objective - cbs_limit|`` drops below
    ``target``.

    Attributes:
        target (float): convergence threshold on ``|objective - cbs_limit|``
        cbs_limit (float): complete-basis-set limit for the property
    """

    def __init__(
        self,
        eval_type: str = 'energy',
        target: float = 1e-6,
        max_n: Optional[int] = None,
        max_its: Optional[int] = None,
        stall_tol: Optional[float] = None,
        pre: Preconditioner = make_positive,
    ):
        super().__init__(eval_type=eval_type, target=target, pre=pre)
        self.name = 'AutoBasisFree'
        self.max_n = max_n
        self.max_its = max_its
        self.stall_tol = stall_tol

    def initialise(self, basis: InternalBasis, element: str):
        """Resets per-run state and checks the CBS limit has been set.

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """

        self._step = -1
        self.shells_done = [1] * len(basis[element])
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = [True] * len(basis[element])
        self.init_run = True
        self.just_added = [False] * len(basis[element])
        self._iter = 0
        self._grown_once = False
        self._shell_capped = [False] * len(basis[element])
        self.stop_reason = None
        if self.cbs_limit is None:
            raise ValueError('CBS limit not set. This can be set with the .set_cbs_limit method.')

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
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
        # Signed distance to the CBS limit (see AutoBasisLegendre.next for why
        # this must not be abs()): energy approaches the limit from above, so
        # <= target -- including a negative overshoot of a too-shallow limit --
        # means converged.
        objective_diff = objective - self.cbs_limit

        if self.init_run:
            if self._step == -1:
                self._step = 0
                return True
            else:
                self._step += 1
                if self._step == len(basis[element]):
                    self.init_run = False
                    self._step = 0
                else:
                    return True

        # (1) primary stop: reached the energy target (converged from above)
        if objective_diff < self.target:
            self.stop_reason = "target"
            return False

        # (2) optional stall cutoff: the last growth barely moved the objective
        if (
            self.stall_tol is not None
            and self._grown_once
            and self.delta_objective < self.stall_tol
        ):
            self.stop_reason = "stall"
            bo_logger.warning(
                "AutoBasisFree: objective stalled (delta=%.2e < stall_tol=%.2e) at "
                "dE_CBS=%.2e Eh; stopping before the CBS target was met.",
                self.delta_objective,
                self.stall_tol,
                objective_diff,
            )
            return False

        # (3) optional global iteration cap
        if self.max_its is not None and self._iter >= self.max_its:
            self.stop_reason = "max_its"
            bo_logger.warning(
                "AutoBasisFree: hit max_its=%d at dE_CBS=%.2e Eh; stopping before "
                "the CBS target was met.",
                self.max_its,
                objective_diff,
            )
            return False

        # (4) optional per-shell cap: skip a shell already at max_n; stop if all are
        if self.max_n is not None and len(basis[element][self._step].exps) >= self.max_n:
            self._shell_capped[self._step] = True
            if all(self._shell_capped):
                self.stop_reason = "max_n"
                bo_logger.warning(
                    "AutoBasisFree: every shell reached max_n=%d at dE_CBS=%.2e Eh; "
                    "stopping before the CBS target was met.",
                    self.max_n,
                    objective_diff,
                )
                return False
            self._step = (self._step + 1) % len(basis[element])
            return True

        x = self.get_active(basis, element)
        last_func, penult_func = x[-1], x[-2]
        ratio = last_func / penult_func
        x = np.append(x, last_func * ratio)
        self.set_active(x, basis, element)
        uncontract_shell(basis[element][self._step])
        self._iter += 1
        self._grown_once = True
        self._step += 1
        if self._step == len(basis[element]):
            self._step = 0

        return True


class AutoBasisLegendre(AutoBasisStrategy):
    """Grows a Legendre-parametrised atomic basis to the CBS limit.

    Each shell's exponents are generated from a short Legendre expansion; the
    optimised quantities are the expansion coefficients (``A_vals``). After an
    initial sweep, the number of primitives per shell is increased and
    re-optimised until ``|objective - cbs_limit|`` drops below ``target``.
    ``get_active``/``set_active`` are overridden to operate on the Legendre
    coefficients rather than raw exponents.

    Attributes:
        n_prim (tuple): number of primitives per shell
        legendre_params (list): Legendre A-coefficients per shell; if None,
            ``initialise`` falls back to the built-in ``_ATOMIC_LEGENDRE_COEFFS``
    """

    def __init__(
        self,
        eval_type: str = 'energy',
        target: float = 1e-6,
        max_n: Optional[int] = None,
        max_its: Optional[int] = None,
        stall_tol: Optional[float] = None,
        n_coefs: Optional[tuple] = None,
    ):
        super().__init__(eval_type=eval_type, target=target, pre=unit)
        self.name = 'AutoBasisLegendre'
        self.n_prim = n_coefs
        self.max_n = max_n
        self.max_its = max_its
        self.stall_tol = stall_tol
        # Legendre A-coefficients per shell; if left as None, initialise() falls
        # back to the built-in _ATOMIC_LEGENDRE_COEFFS for the element.
        self.legendre_params = None

    def as_dict(self) -> dict[str, Any]:
        """Returns MSONable dictionary of object.

        Overrides the base to also persist the Legendre configuration
        (``n_prim`` and ``legendre_params``); without it a reloaded strategy
        reset both to None and crashed on the next ``initialise``.
        """
        d = super().as_dict()
        d["n_prim"] = list(self.n_prim) if self.n_prim is not None else None
        d["legendre_params"] = self.legendre_params
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates the strategy from an MSONable dictionary."""
        instance = super().from_dict(d)
        n_prim = d.get("n_prim", None)
        instance.n_prim = tuple(n_prim) if n_prim is not None else None
        instance.legendre_params = d.get("legendre_params", None)
        return instance

    def initialise(self, basis: InternalBasis, element: str):
        """Builds the Legendre-expanded starting basis and resets per-run state.

        Uses ``legendre_params`` if set, otherwise the built-in
        ``_ATOMIC_LEGENDRE_COEFFS`` for the element, pairing each shell's
        coefficients with the requested primitive count ``n_prim``.

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """
        if self.n_prim is None:
            raise ValueError(
                "AutoBasisLegendre needs the number of primitives per shell; pass "
                "n_coefs=(...) to the constructor, set .n_prim, or give the step's "
                "'n_prim' config."
            )
        if self.legendre_params:
            coeffs = self.legendre_params
        else:
            bo_logger.warning(
                'No Legendre parameters set. Using default parameters. This may '
                'result in poorly conditioned expansions.'
            )
            if element.capitalize() not in _ATOMIC_LEGENDRE_COEFFS:
                raise ValueError(
                    f"No built-in Legendre coefficients for '{element}'. Available: "
                    f"{sorted(_ATOMIC_LEGENDRE_COEFFS)}. Set them explicitly via the "
                    f"strategy's `legendre_params` attribute."
                )
            self._initial_guess = _ATOMIC_LEGENDRE_COEFFS[element.capitalize()]
            coeffs = self._initial_guess
        if len(self.n_prim) != len(coeffs):
            raise ValueError(
                f"n_coefs has {len(self.n_prim)} entries but {element} has "
                f"{len(coeffs)} Legendre shell(s)."
            )
        self._shells = [(A_vals, n) for A_vals, n in zip(coeffs, self.n_prim)]
        if not isinstance(basis, dict):
            basis = {}
        self.set_basis_shell(basis, element)

        self._step = -1
        self.shells_done = [1] * len(basis[element])
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = [True] * len(basis[element])
        self.init_run = True
        self.just_added = [False] * len(basis[element])
        self._iter = 0
        self._grown_once = False
        self._shell_capped = [False] * len(basis[element])
        self.stop_reason = None
        if self.cbs_limit is None:
            raise ValueError('CBS limit not set. This can be set with the .set_cbs_limit method.')

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """Arguments:
             basis: internal basis dictionary
             element: symbol of the atom being optimized

        Returns:
             the set of exponents currently being optimised
        """
        A_vals, n = self._shells[self._step]
        return A_vals

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Sets the currently active exponents to the given values.

        Arguments:
            values (list): list of new exponents
            basis: internal basis dictionary
            element: symbol of atom being optimized
        """
        A_vals, n = basis[element][self._step].leg_params
        self._shells[self._step] = (values, n)
        basis[element][self._step].leg_params = (values, n)

        self.set_basis_shell(basis, element)

    def set_basis_shell(self, basis: InternalBasis, element: str):
        """Expands parameters into a basis set

        Arguments:
             basis (InternalBasis): the basis set to expand
             element (str): the atom type
        """
        basis[element] = legendre_expansion(self._shells)

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        """Moves the strategy forward a step (see algorithm)

        Arguments:
            basis: internal basis dictionary
            element: symbol of atom being optimized
            objective: value of objective function from last steps

        Returns:
            True if there is a next step, False if strategy is finished
        """
        element_cbs_limit = self.cbs_limit

        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective

        # Signed distance to the CBS limit -- deliberately NOT abs(). The
        # (variational-ish) energy approaches the limit from above, so
        # objective_diff starts positive and shrinks; <= target means converged.
        # A negative value means the energy has passed a too-shallow cbs_limit,
        # which is also "done". Taking abs() here turned that overshoot into an
        # ever-growing gap and looped forever appending saturated primitives.
        objective_diff = objective - element_cbs_limit

        # Initial run: optimise each shell's A_vals once, sequentially, so they
        # suit the requested primitive count before any growth.
        if self.init_run:
            if self._step == -1:
                self._step = 0
                return True
            else:
                self._step += 1
                if self._step == len(basis[element]):
                    self.init_run = False
                    self._step = 0
                else:
                    return True

        # (1) primary stop: reached the energy target (converged from above)
        if objective_diff < self.target:
            self.stop_reason = "target"
            return False

        A_vals, n = self._shells[self._step]
        if not self.just_added[self._step]:
            # (2) optional global iteration cap (checked before growing)
            if self.max_its is not None and self._iter >= self.max_its:
                self.stop_reason = "max_its"
                bo_logger.warning(
                    "AutoBasisLegendre: hit max_its=%d at dE_CBS=%.2e Eh; stopping "
                    "before the CBS target was met.",
                    self.max_its,
                    objective_diff,
                )
                return False
            # (3) optional per-shell cap: skip a shell at max_n; stop if all are
            if self.max_n is not None and n >= self.max_n:
                self._shell_capped[self._step] = True
                if all(self._shell_capped):
                    self.stop_reason = "max_n"
                    bo_logger.warning(
                        "AutoBasisLegendre: every shell reached max_n=%d at "
                        "dE_CBS=%.2e Eh; stopping before the CBS target was met.",
                        self.max_n,
                        objective_diff,
                    )
                    return False
                self._step = (self._step + 1) % len(basis[element])
                return True
            bo_logger.info(
                f'Increasing number of {basis[element][self._step].l} functions from {n} to {n+1}'
            )
            self._shells[self._step] = (A_vals, n + 1)
            self.set_basis_shell(basis, element)
            self.just_added[self._step] = True
            self._iter += 1
            self._grown_once = True
            return True
        else:
            # The freshly-grown shell was just re-optimised.
            self.just_added[self._step] = False
            # (4) optional stall cutoff: if that growth barely moved the
            # objective the shell is saturated for this l -- stop rather than
            # keep appending dead functions (this is limit-independent, so it
            # catches a wrong cbs_limit too).
            if self.stall_tol is not None and self.delta_objective < self.stall_tol:
                self.stop_reason = "stall"
                bo_logger.warning(
                    "AutoBasisLegendre: objective stalled (delta=%.2e < "
                    "stall_tol=%.2e) at dE_CBS=%.2e Eh; stopping before the CBS "
                    "target was met.",
                    self.delta_objective,
                    self.stall_tol,
                    objective_diff,
                )
                return False
            bo_logger.info(f'Shell exponents: {list(basis[element][self._step].exps)}')
            self._step += 1
            if self._step == len(basis[element]):
                self._step = 0

        return True


class AutoBasisReduceStrategy(AutoBasisStrategy):
    """Reduces an atomic basis by removing the least important exponents.

    Ranks every exponent by its contribution (via ``rank_mol_basis_cbs``),
    removes the single least-important one, and re-optimises. A removal that
    pushes ``objective - cbs_limit`` above ``target`` is reverted and the
    reduction stops.

    Attributes:
        target (float): tolerance above the CBS limit before a removal is
            rejected
        cbs_limit (float): complete-basis-set limit for the property
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
        super().__init__(eval_type=eval_type, target=target, pre=unit)
        self.name = 'AutoBasisReduce'
        self.skip_init = False

    def initialise(self, basis: InternalBasis, element: str):
        """Resets per-run state and snapshots the starting basis.

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """

        self._step = -1
        self.shells_done = [1] * len(basis[element])
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = [True] * len(basis[element])
        self.init_run = True
        self._just_removed = False
        self.original_shells = [copy.deepcopy(shell) for shell in basis[element]]
        self.original_size = [len(shell.exps) for shell in basis[element]]
        self.n_exps_removed = [0] * len(basis[element])
        self.old_exps = [None] * len(basis[element])
        if self.cbs_limit is None:
            raise ValueError('CBS limit not set. This can be set with the .set_cbs_limit method.')

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        """Moves the strategy forward a step (see algorithm)

        The Molecule needed for ranking is read from ``self.molecule`` (set by
        the driver via ``set_context``), so this matches the standard three-
        argument ``next`` signature.

        Arguments:
            basis: internal basis dictionary
            element: symbol of atom being optimized
            objective: value of objective function from last steps

        Returns:
            True if there is a next step, False if strategy is finished
        """
        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective

        if self.init_run:
            if self._step == -1:
                self._step = 0
                return True
            else:
                self._step += 1
                if self._step == len(basis[element]):
                    self.init_run = False
                    self._step = 0
                else:
                    return True

        if not self._just_removed:
            errors, ranks, _, _ = rank_mol_basis_cbs(
                self.molecule,
                element,
                self.cbs_limit,
                self.eval_type,
                self.params,
                parallel=self.parallel,
                ray_params=self.ray_params,
            )
            # Find the (shell, exponent) with the globally smallest error.
            # The shells can have different numbers of exponents, so index the
            # jagged `errors` directly rather than unravelling a flat argmin into
            # a rectangular (n_shells, max_len) grid, which mismapped the index
            # for unequal shell sizes and could point past a shell's length.
            min_shell, min_exp = min(
                ((si, ei) for si, sub in enumerate(errors) for ei in range(len(sub))),
                key=lambda idx: errors[idx[0]][idx[1]],
            )

            self._step = min_shell
            self.old_exps[self._step] = basis[element][self._step].exps
            new_exps = np.delete(basis[element][self._step].exps, min_exp)
            self.set_active(new_exps, basis, element)
            uncontract_shell(basis[element][self._step])
            bo_logger.debug(
                "Removing exponent %d from shell %s", min_exp, basis[element][self._step].l
            )
            self._just_removed = True
            self.n_exps_removed[self._step] += 1
            return True
        elif objective - self.cbs_limit > self.target:
            bo_logger.info('Restoring previous basis set')
            self.set_active(self.old_exps[self._step], basis, element)
            uncontract_shell(basis[element][self._step])
            self.n_exps_removed[self._step] -= 1
            return False
        else:
            self._just_removed = False
            return True


class AutoBasisReduceStrategyAll(AutoBasisStrategy):
    """Reduces an atomic basis, re-optimising every shell after each removal.

    Like :class:`AutoBasisReduceStrategy`, but after removing an exponent it
    cycles through and re-optimises all shells before evaluating whether the
    removal was acceptable, marking a shell done once a removal is rejected.

    Attributes:
        target (float): tolerance above the CBS limit before a removal is
            rejected
        cbs_limit (float): complete-basis-set limit for the property
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
        super().__init__(eval_type=eval_type, target=target, pre=unit)
        self.name = 'AutoBasisReduceALl'
        self.run_all = False

    def initialise(self, basis: InternalBasis, element: str):
        """Resets per-run state and snapshots the starting basis.

        Arguments:
            basis: internal basis dictionary
            element: symbol of the atom being optimized
        """

        self._step = 0
        self.shells_done = [1] * len(basis[element])
        self.last_objective = 0
        self.delta_objective = 0
        self.first_run = [True] * len(basis[element])
        self.init_run = True
        # index of the shell whose exponent was most recently removed (-1 = none
        # yet). next() compares it to self._step; it must be a scalar int, not the
        # list it used to be (which only "worked" because list != int is always True).
        self._just_removed = -1
        self.original_shells = [copy.deepcopy(shell) for shell in basis[element]]
        self.original_size = [len(shell.exps) for shell in basis[element]]
        self.n_exps_removed = [0] * len(basis[element])
        self.old_exps = [None] * len(basis[element])
        if self.cbs_limit is None:
            raise ValueError('CBS limit not set. This can be set with the .set_cbs_limit method.')

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        """Moves the strategy forward a step (see algorithm)

        The Molecule needed for ranking is read from ``self.molecule`` (set by
        the driver via ``set_context``), so this matches the standard three-
        argument ``next`` signature.

        Arguments:
            basis: internal basis dictionary
            element: symbol of atom being optimized
            objective: value of objective function from last steps

        Returns:
            True if there is a next step, False if strategy is finished
        """
        self.delta_objective = np.abs(objective - self.last_objective)
        self.last_objective = objective

        while sum(self.shells_done) != 0:
            if self.run_all:
                self._step += 1
                if self._step == len(basis[element]):
                    self._step = 0
                if self._just_removed == self._step:
                    self.run_all = False
            if self.run_all:
                return True

            if self._just_removed != self._step:
                errors, ranks, _, _ = rank_mol_basis_cbs(
                    self.molecule,
                    element,
                    self.cbs_limit,
                    self.eval_type,
                    self.params,
                    parallel=self.parallel,
                    ray_params=self.ray_params,
                )
                removed_index = ranks[self._step][0]
                self._removed_index = removed_index
                self.old_exps[self._step] = basis[element][self._step].exps
                new_exps = np.delete(basis[element][self._step].exps, removed_index)
                self.set_active(new_exps, basis, element)
                uncontract_shell(basis[element][self._step])
                self._just_removed = self._step
                self.run_all = True
                bo_logger.info(
                    f"Removed exponent {removed_index} from shell {basis[element][self._step].l}"
                )

                return sum(self.shells_done) != 0
            else:
                if objective - self.cbs_limit > self.target:
                    self.set_active(self.old_exps[self._step], basis, element)
                    uncontract_shell(basis[element][self._step])
                    bo_logger.info(
                        f"Re-adding exponent {self._removed_index} to shell "
                        f"{basis[element][self._step].l}"
                    )
                    self.shells_done[self._step] = 0
                    self._step += 1
                    if self._step == len(basis[element]):
                        self._step = 0
                else:
                    self.n_exps_removed[self._step] += 1
                    self._step += 1
                    if self._step == len(basis[element]):
                        self._step = 0

        if sum(self.shells_done) == 0:
            return False

        return sum(self.shells_done) != 0
