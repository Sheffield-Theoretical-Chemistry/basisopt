"""Characterization tests for optimization strategy state machines.

These pin the ``initialise`` + ``next`` control flow of the strategies against
the exponent-independent Dummy backend, so the later de-duplication refactors
(shared AutoBasis base class, unified ``next()`` signature) can be shown to be
behaviour-preserving. The ``objective`` value is fed directly, so no backend
calculation is needed to drive the state machines.
"""

import numpy as np
import pytest

from basisopt.opt.auto_basis import (
    AutoBasisFree,
    AutoBasisLegendre,
    AutoBasisReduceStrategy,
)
from basisopt.opt.strategies import Strategy
from tests.data.factories import make_basis


def _two_shell_basis():
    return make_basis("h", (("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3))))


# --------------------------------------------------------------------------- #
# Base Strategy
# --------------------------------------------------------------------------- #
def test_default_strategy_progression(dummy_backend):
    """Base Strategy optimizes each shell once in increasing l, then stops."""
    strategy = Strategy()
    basis = _two_shell_basis()
    strategy.initialise(basis, "h")
    assert strategy._step == -1

    assert strategy.next(basis, "h", 1.0) is True
    assert strategy._step == 0
    assert strategy.next(basis, "h", 1.0) is True
    assert strategy._step == 1
    # step reaches len(basis['h']) == 2 -> finished
    assert strategy.next(basis, "h", 1.0) is False
    assert strategy._step == 2


def test_reduce_strategy_uses_standard_next_signature(dummy_backend):
    """Reduce strategies get the molecule via set_context, not a wide next()."""
    import inspect

    strategy = AutoBasisReduceStrategy(target=1e-6)
    # standard 3-arg next signature (self, basis, element, objective)
    params = list(inspect.signature(strategy.next).parameters)
    assert params == ["basis", "element", "objective"]
    # context stash
    assert strategy.molecule is None
    sentinel = object()
    strategy.set_context(molecule=sentinel)
    assert strategy.molecule is sentinel


def test_reduce_strategy_selects_min_error_across_jagged_shells(dummy_backend, monkeypatch):
    """The globally least-important exponent must be found even when shells have
    unequal sizes. Regression for the flat-argmin/unravel mismap that indexed
    past a shell's length (crashed on real bases like Ne = 9s4p1d)."""
    import basisopt.opt.auto_basis as ab

    # jagged basis: 3 s, 2 p, 1 d
    basis = make_basis("h", (("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3)), ("d", (0.8,))))
    strategy = AutoBasisReduceStrategy(target=1e-4)
    strategy.set_cbs_limit(-1.0)
    strategy.initialise(basis, "h")

    # step through the initial sweep over the three shells
    assert strategy.next(basis, "h", -0.5) is True  # -1 -> 0
    assert strategy.next(basis, "h", -0.5) is True  # 0 -> 1
    assert strategy.next(basis, "h", -0.5) is True  # 1 -> 2

    # global minimum error is in the LAST (d) shell; with the old flat-argmin +
    # (n_shells, max_len) unravel this mismapped onto the p shell and raised
    # IndexError.
    errors = [np.array([0.9, 0.8, 0.7]), np.array([0.6, 0.5]), np.array([0.1])]
    ranks = [np.argsort(e) for e in errors]
    monkeypatch.setattr(ab, "rank_mol_basis_cbs", lambda *a, **k: (errors, ranks, None, None))

    n_d_before = len(basis["h"][2].exps)
    assert strategy.next(basis, "h", -0.5) is True  # removal step
    assert strategy._step == 2  # removed from the d shell
    assert len(basis["h"][2].exps) == n_d_before - 1


def test_default_strategy_has_target_attribute(dummy_backend):
    """Base Strategy exposes a `target` (None) so Optimizer/Minimizer can read it."""
    strategy = Strategy()
    assert strategy.target is None
    strategy.target = 1e-5
    restored = Strategy.from_dict(strategy.as_dict())
    assert restored.target == 1e-5


# --------------------------------------------------------------------------- #
# AutoBasisFree
# --------------------------------------------------------------------------- #
def test_autobasisfree_requires_cbs_limit(dummy_backend):
    strategy = AutoBasisFree()
    basis = _two_shell_basis()
    with pytest.raises(ValueError):
        strategy.initialise(basis, "h")


def test_autobasisfree_state_machine(dummy_backend):
    """Locks: sequential init pass over shells, then one extra exponent added
    per shell while the objective is far from the CBS limit, terminating once
    the objective reaches the CBS limit."""
    strategy = AutoBasisFree(target=1e-6)
    # NOTE: AutoBasisFree.initialise uses `if not self.cbs_limit`, so a CBS
    # limit of exactly 0.0 is (buggily) treated as unset -- use a non-zero value.
    strategy.set_cbs_limit(-1.0)
    basis = _two_shell_basis()
    strategy.initialise(basis, "h")
    n_s0 = len(basis["h"][0].exps)
    n_p0 = len(basis["h"][1].exps)

    # initial pass: one step per shell
    assert strategy.next(basis, "h", 10.0) is True
    assert strategy._step == 0
    assert strategy.next(basis, "h", 10.0) is True
    assert strategy._step == 1

    # third call ends the initial pass and, since objective is far from the CBS
    # limit, appends an exponent to shell 0
    assert strategy.next(basis, "h", 10.0) is True
    assert len(basis["h"][0].exps) == n_s0 + 1

    # next call appends an exponent to shell 1
    assert strategy.next(basis, "h", 10.0) is True
    assert len(basis["h"][1].exps) == n_p0 + 1

    # objective at the CBS limit -> finished
    assert strategy.next(basis, "h", -1.0) is False


# --------------------------------------------------------------------------- #
# AutoBasisLegendre
# --------------------------------------------------------------------------- #
def test_autobasislegendre_state_machine(dummy_backend):
    """With Legendre params supplied, next() grows the primitive count of a
    shell by one, then advances to the next shell (just_added toggling)."""
    strategy = AutoBasisLegendre(target=1e-6, n_coefs=(4, 3))
    strategy.set_cbs_limit(0.0)
    # supply Legendre A-coefficients for two shells (s, p)
    strategy.legendre_params = [
        np.array([1.6, -5.1, 0.05, -0.17]),
        np.array([-0.97, 1.77, -0.27]),
    ]
    basis = {}
    strategy.initialise(basis, "o")

    n0 = strategy._shells[0][1]
    # init pass over both shells
    assert strategy.next(basis, "o", 10.0) is True
    assert strategy.next(basis, "o", 10.0) is True
    # end init pass + first real step: objective far from CBS -> grow shell 0's n
    assert strategy.next(basis, "o", 10.0) is True
    assert strategy._shells[0][1] == n0 + 1
    assert strategy.just_added[0] is True
    # re-optimised the added primitive -> toggle just_added back off, advance
    assert strategy.next(basis, "o", 10.0) is True
    assert strategy.just_added[0] is False
    # objective reaches CBS limit -> finished
    assert strategy.next(basis, "o", 0.0) is False


def test_autobasislegendre_default_initialise(dummy_backend):
    """Default construction should fall back to the built-in Legendre coeffs."""
    strategy = AutoBasisLegendre(target=1e-6, n_coefs=(4, 3))
    strategy.set_cbs_limit(0.0)
    basis = {}
    strategy.initialise(basis, "o")
    assert "o" in basis
