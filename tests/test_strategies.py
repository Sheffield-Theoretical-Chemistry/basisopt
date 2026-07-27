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
from basisopt.opt.polarisation import AutoBasisPolarisation
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


@pytest.mark.parametrize(
    "module_name, cls_name",
    [
        ("basisopt.opt.contraction", "ContractionStrategy"),
        ("basisopt.opt.legendreHybrid", "LegendrePairsHybrid"),
        ("basisopt.opt.eventemper", "EvenTemperedStrategy"),
        ("basisopt.opt.welltemper", "WellTemperedStrategy"),
        ("basisopt.opt.legendre", "LegendreStrategy"),
    ],
)
def test_strategy_family_serialization_roundtrip(dummy_backend, module_name, cls_name):
    """as_dict/from_dict must round-trip without referencing unset attributes."""
    import importlib

    cls = getattr(importlib.import_module(module_name), cls_name)
    strategy = cls()
    restored = cls.from_dict(strategy.as_dict())
    assert restored.name == strategy.name


def test_tempered_strategies_initialise_with_explicit_max_l(dummy_backend):
    """initialise must not raise UnboundLocalError when max_l >= 0.

    Regression: `el = md_element(...)` was bound only inside `if max_l < 0`,
    but `l_list` used it unconditionally, so any explicit max_l crashed (and
    took AtomicBasis.set_even/well_tempered/legendre down with it).
    """
    from basisopt.opt.eventemper import EvenTemperedStrategy
    from basisopt.opt.legendre import LegendreStrategy
    from basisopt.opt.welltemper import WellTemperedStrategy

    for cls in (EvenTemperedStrategy, WellTemperedStrategy, LegendreStrategy):
        strategy = cls(max_l=1)
        basis = {}
        strategy.initialise(basis, "he")
        assert "he" in basis


def test_tempered_strategy_reinitialise_across_elements(dummy_backend):
    """Reusing one tempered strategy (as Optimizer.run / _collective's npass loop
    do) must reset _step and not accumulate max_l onto lighter elements."""
    from basisopt.opt.eventemper import EvenTemperedStrategy

    strategy = EvenTemperedStrategy()  # default max_l = -1 (minimal config)
    basis = {}

    strategy.initialise(basis, "C")  # C -> s,p => min_l 2
    assert strategy.max_l == 2
    strategy.next(basis, "C", 1.0)
    assert strategy._step >= 0  # advanced

    strategy.initialise(basis, "H")  # H -> s => min_l 1
    assert strategy._step == -1  # reset (was left advanced before the fix)
    assert strategy.max_l == 1  # not accumulated up to 2


def test_tempered_from_dict_preserves_basis_type(dummy_backend):
    """Auxiliary-basis strategies must not silently revert to 'orbital' on reload."""
    from basisopt.opt.eventemper import EvenTemperedStrategy

    strategy = EvenTemperedStrategy()
    strategy.basis_type = "jkfit"
    restored = EvenTemperedStrategy.from_dict(strategy.as_dict())
    assert restored.basis_type == "jkfit"


def test_reduce_strategy_pads_shell_mins(dummy_backend):
    # regression: default [] made possible_changes empty -> silent no-op, and a
    # too-short list IndexError'd when next() indexed by shell.
    from basisopt.opt.reduce import ReduceStrategy

    basis = make_basis("h", (("s", (9.0, 3.0, 1.0, 0.3)), ("p", (1.5, 0.4))))

    default = ReduceStrategy(basis)
    default.initialise(basis, "h")
    assert default.shell_mins == [0, 0]  # one floor per shell

    short = ReduceStrategy(basis, shell_mins=[2])
    short.initialise(basis, "h")
    assert short.shell_mins == [2, 0]


def test_reduce_strategy_instances_do_not_share_defaults(dummy_backend):
    # regression: shell_mins=[]/params={} were shared mutable defaults.
    from basisopt.opt.reduce import ReduceStrategy

    basis = make_basis("h", (("s", (1.0,)),))
    a = ReduceStrategy(basis)
    b = ReduceStrategy(basis)
    a.shell_mins.append(3)
    a.params["x"] = 1
    assert b.shell_mins == []
    assert b.params == {}


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


def test_preconditioner_params_are_per_instance(dummy_backend):
    """Regression: pre.params was set on the shared module-level preconditioner
    function object, so every strategy aliased one dict. Each strategy must own
    its own pre_params, and the shared function must not carry a params attr."""
    from basisopt.opt.preconditioners import make_positive

    s1 = Strategy()
    s2 = Strategy()
    assert s1.pre_params == {} and s2.pre_params == {}
    s1.pre_params["k"] = 1
    assert s2.pre_params == {}
    assert s1.pre_params is not s2.pre_params
    assert not hasattr(make_positive, "params")


def test_default_strategy_has_target_attribute(dummy_backend):
    """Base Strategy exposes a `target` (None) so Optimizer/Minimizer can read it."""
    strategy = Strategy()
    assert strategy.target is None
    strategy.target = 1e-5
    restored = Strategy.from_dict(strategy.as_dict())
    assert restored.target == 1e-5


# --------------------------------------------------------------------------- #
# Tempered strategy family (even/well/legendre share one state machine)
# --------------------------------------------------------------------------- #
def _run_tempered(cls, max_n, element="c", target=1e-5, schedule=None):
    """Drive initialise + a fixed next() schedule, recording the state machine.

    Returns (strategy, basis, trace) where each trace entry is
    (return_value, _step, first_run, [n per shell], shell_done)."""
    if schedule is None:
        schedule = (10.0, 10.0, 10.0, 10.0, 1e-9, 1e-9, 1e-9, 1e-9)
    strategy = cls(target=target, max_n=max_n)
    basis = {}
    strategy.initialise(basis, element)
    trace = []
    for obj in schedule:
        ret = strategy.next(basis, element, obj)
        trace.append(
            (
                ret,
                strategy._step,
                strategy.first_run,
                [shell[-1] for shell in strategy.shells],
                list(strategy.shell_done),
            )
        )
        if not ret:
            break
    return strategy, basis, trace


# Golden trace for carbon (minimal config -> 2 shells), max_n=10. Even- and
# well-tempered share the identical control flow; the refactor to a shared
# TemperedStrategy base must reproduce this exactly.
_TEMPERED_GOLDEN = [
    (True, 0, True, [8, 8], [1, 1]),
    (True, 1, True, [8, 8], [1, 1]),
    (True, 0, False, [9, 8], [1, 1]),
    (True, 1, False, [9, 9], [0, 1]),
    (True, 0, False, [9, 9], [0, 1]),
    (True, 1, False, [9, 10], [0, 1]),
    (False, 0, False, [9, 10], [0, 0]),
]


@pytest.mark.parametrize(
    "module_name, cls_name",
    [
        ("basisopt.opt.eventemper", "EvenTemperedStrategy"),
        ("basisopt.opt.welltemper", "WellTemperedStrategy"),
    ],
)
def test_tempered_state_machine(dummy_backend, module_name, cls_name):
    import importlib

    cls = getattr(importlib.import_module(module_name), cls_name)
    strategy, basis, trace = _run_tempered(cls, max_n=10)
    assert strategy.max_l == 2
    assert len(basis["c"]) == 2
    assert trace == _TEMPERED_GOLDEN


def test_legendre_state_machine(dummy_backend):
    """Legendre uses the same machine; with max_n=12 (initial n=10) it grows
    identically to the even/well case, just offset by the larger starting n."""
    from basisopt.opt.legendre import LegendreStrategy

    strategy, basis, trace = _run_tempered(LegendreStrategy, max_n=12)
    assert strategy.max_l == 2
    assert trace == [
        (True, 0, True, [10, 10], [1, 1]),
        (True, 1, True, [10, 10], [1, 1]),
        (True, 0, False, [11, 10], [1, 1]),
        (True, 1, False, [11, 11], [0, 1]),
        (True, 0, False, [11, 11], [0, 1]),
        (True, 1, False, [11, 12], [0, 1]),
        (False, 0, False, [11, 12], [0, 0]),
    ]


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


def test_autobasislegendre_missing_element_raises(dummy_backend):
    """Default path for an untabulated element gives a clear error, not KeyError."""
    strategy = AutoBasisLegendre(target=1e-6, n_coefs=(4, 3))
    strategy.set_cbs_limit(0.0)
    with pytest.raises(ValueError, match="No built-in Legendre coefficients"):
        strategy.initialise({}, "h")


# --------------------------------------------------------------------------- #
# AutoBasisLegendre growth cutoffs (signed CBS target + max_n/max_its/stall)
# --------------------------------------------------------------------------- #
def _legendre_o(target=2.0e-3, cbs_limit=-75.0603, n_coefs=(4, 4), **cutoffs):
    """Initialised AutoBasisLegendre on O, past construction, ready to drive."""
    strategy = AutoBasisLegendre(target=target, n_coefs=n_coefs)
    strategy.set_cbs_limit(cbs_limit)
    strategy.legendre_params = [
        np.array([1.6, -5.1, 0.05, -0.17]),
        np.array([-0.97, 1.77, -0.27]),
    ]
    for key, value in cutoffs.items():
        setattr(strategy, key, value)
    basis = {}
    strategy.initialise(basis, "o")
    return strategy, basis


def test_autobasislegendre_signed_convergence_stops_on_overshoot(dummy_backend):
    """Regression for the runaway: the stop test is signed
    ``energy - cbs_limit < target``, not ``abs(...)``. When the energy shoots
    past a too-shallow cbs_limit, abs() reported an ever-growing gap and looped
    forever appending saturated primitives; the signed test terminates."""
    strategy, basis = _legendre_o(target=2.0e-3, cbs_limit=-75.0603)
    # sequential init pass over the two shells
    assert strategy.next(basis, "o", -74.5) is True
    assert strategy.next(basis, "o", -74.5) is True
    # energy is 17.7 mEh *below* the (too-shallow) limit; abs() -> "17.7 mEh
    # away, keep going"; signed -> converged, stop.
    assert strategy.next(basis, "o", -75.078) is False
    assert strategy.stop_reason == "target"


def test_autobasislegendre_far_from_limit_keeps_growing(dummy_backend):
    """Above the limit by more than target -> keep adding functions, no stop."""
    strategy, basis = _legendre_o(target=2.0e-3, cbs_limit=-75.0603)
    assert strategy.next(basis, "o", -74.0) is True
    assert strategy.next(basis, "o", -74.0) is True
    assert strategy.next(basis, "o", -74.0) is True  # still 1.06 Eh above limit
    assert strategy.stop_reason is None


def test_autobasislegendre_max_n_cutoff(dummy_backend):
    """max_n caps primitives per shell: once every shell is at the cap and the
    target is still unmet, stop with reason 'max_n'."""
    strategy, basis = _legendre_o(n_coefs=(4, 4), max_n=4)  # both shells start at cap
    assert strategy.next(basis, "o", -74.0) is True  # init: -1 -> 0
    assert strategy.next(basis, "o", -74.0) is True  # init: 0 -> 1
    assert strategy.next(basis, "o", -74.0) is True  # shell 0 capped -> skip to shell 1
    assert strategy.next(basis, "o", -74.0) is False  # shell 1 capped -> all capped, stop
    assert strategy.stop_reason == "max_n"


def test_autobasislegendre_max_its_cutoff(dummy_backend):
    """max_its bounds the total number of growth steps regardless of the target."""
    strategy, basis = _legendre_o(n_coefs=(4, 4), max_its=2)
    stopped = False
    for _ in range(50):  # far from the limit, so only max_its can stop it
        if not strategy.next(basis, "o", -74.0):
            stopped = True
            break
    assert stopped
    assert strategy.stop_reason == "max_its"
    assert strategy._iter == 2


def test_autobasislegendre_stall_cutoff(dummy_backend):
    """stall_tol stops a shell once an added primitive barely moves the objective
    (saturation) -- limit-independent, so it catches a wrong cbs_limit too."""
    strategy, basis = _legendre_o(n_coefs=(4, 4), stall_tol=1.0e-6)
    assert strategy.next(basis, "o", -74.0) is True  # init: -1 -> 0
    assert strategy.next(basis, "o", -74.0) is True  # init: 0 -> 1
    assert strategy.next(basis, "o", -74.0) is True  # grow shell 0
    # re-optimised the added primitive but it barely changed the objective
    assert strategy.next(basis, "o", -74.0 - 1.0e-9) is False
    assert strategy.stop_reason == "stall"


def test_autobasis_cutoffs_survive_serialization(dummy_backend):
    """max_n/max_its/stall_tol must round-trip through as_dict/from_dict."""
    strategy = AutoBasisLegendre(target=1e-4, n_coefs=(4, 3), max_n=15, max_its=50, stall_tol=1e-7)
    strategy.set_cbs_limit(-75.0)
    restored = AutoBasisLegendre.from_dict(strategy.as_dict())
    assert restored.max_n == 15
    assert restored.max_its == 50
    assert restored.stall_tol == 1e-7
    assert restored.cbs_limit == -75.0


# --------------------------------------------------------------------------- #
# AutoBasisPolarisation (grows d/f/g shells onto an existing sp basis)
# --------------------------------------------------------------------------- #
def _sp_basis():
    return make_basis("o", (("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3))))


def test_polarisation_adds_shells_then_stops_on_max_l(dummy_backend):
    """Add d, grow it, advance to f on stall, then stop at max_l."""
    basis = _sp_basis()
    strat = AutoBasisPolarisation(target=1e-9, min_l=2, max_l=3, stall_tol=1e-3)
    strat.initialise(basis, "o")
    assert [sh.l for sh in basis["o"]] == ["s", "p"]  # no polarisation shells yet

    assert strat.next(basis, "o", 0.05) is True  # add d
    assert basis["o"][-1].l == "d"
    assert strat.next(basis, "o", 0.03) is True  # grow d -> 2 primitives
    assert len(basis["o"][-1].exps) == 2
    assert strat.next(basis, "o", 0.0295) is True  # d stalled -> add f
    assert basis["o"][-1].l == "f"
    assert strat.next(basis, "o", 0.02) is True  # grow f
    assert strat.next(basis, "o", 0.0199) is False  # f stalled, no higher l -> stop
    assert strat.stop_reason == "max_l"


def test_polarisation_stops_on_target(dummy_backend):
    """Loss below target stops immediately (whichever trips first)."""
    basis = _sp_basis()
    strat = AutoBasisPolarisation(target=1e-3, min_l=2, max_l=4, stall_tol=1e-9)
    strat.initialise(basis, "o")
    assert strat.next(basis, "o", 0.05) is True
    assert strat.next(basis, "o", 0.02) is True
    assert strat.next(basis, "o", 5e-4) is False  # loss below target
    assert strat.stop_reason == "target"


def test_polarisation_max_n_advances_to_next_l(dummy_backend):
    """A shell at max_n is treated as saturated -> advance to the next l."""
    basis = _sp_basis()
    strat = AutoBasisPolarisation(target=1e-9, min_l=2, max_l=3, max_n=2, stall_tol=None)
    strat.initialise(basis, "o")
    strat.next(basis, "o", 0.05)  # add d (1 primitive)
    strat.next(basis, "o", 0.04)  # grow d -> 2 primitives (== max_n)
    assert len(basis["o"][-1].exps) == 2
    strat.next(basis, "o", 0.03)  # d at max_n -> add f
    assert basis["o"][-1].l == "f"


def test_polarisation_max_its_backstop(dummy_backend):
    """Constant loss (never target/stall) is bounded by max_its."""
    basis = _sp_basis()
    strat = AutoBasisPolarisation(target=1e-9, min_l=2, max_l=9, max_its=3, stall_tol=None)
    strat.initialise(basis, "o")
    stopped = False
    for _ in range(50):
        if not strat.next(basis, "o", 0.05):
            stopped = True
            break
    assert stopped
    assert strat.stop_reason == "max_its"
    assert strat._iter == 3


def test_polarisation_leaves_sp_untouched(dummy_backend):
    """The existing sp shells must not be modified while polarising."""
    basis = _sp_basis()
    s_exps = np.array(basis["o"][0].exps, copy=True)
    strat = AutoBasisPolarisation(target=1e-9, min_l=2, max_l=2, stall_tol=1e-3)
    strat.initialise(basis, "o")
    for loss in (0.05, 0.03, 0.0299):
        if not strat.next(basis, "o", loss):
            break
    assert basis["o"][0].l == "s" and basis["o"][1].l == "p"
    assert np.allclose(basis["o"][0].exps, s_exps)


def test_polarisation_max_l_below_min_l_raises(dummy_backend):
    strat = AutoBasisPolarisation(min_l=3, max_l=2)
    with pytest.raises(ValueError, match="max_l"):
        strat.initialise(_sp_basis(), "o")


def test_polarisation_serialization_roundtrip(dummy_backend):
    strat = AutoBasisPolarisation(
        target=1e-4,
        min_l=2,
        max_l=3,
        seed_exponent=0.8,
        max_n=5,
        max_its=20,
        stall_tol=1e-6,
        delta_e=2e-5,
    )
    restored = AutoBasisPolarisation.from_dict(strat.as_dict())
    assert (restored.min_l, restored.max_l, restored.seed_exponent) == (2, 3, 0.8)
    assert (restored.max_n, restored.max_its, restored.stall_tol) == (5, 20, 1e-6)
    assert restored.delta_e == 2e-5
    assert restored.name == "AutoBasisPolarisation"


def test_polarisation_delta_e_restores_last_grow(dummy_backend):
    """A grown primitive that barely helps is dropped (previous step restored)."""
    basis = _sp_basis()
    strat = AutoBasisPolarisation(target=1e-12, min_l=2, max_l=3, stall_tol=1e-12, delta_e=1e-3)
    strat.initialise(basis, "o")
    assert strat.next(basis, "o", 0.20) is True  # baseline 0.20 -> add d (1 primitive)
    assert basis["o"][-1].l == "d" and len(basis["o"][-1].exps) == 1
    assert strat.next(basis, "o", 0.05) is True  # d seed helped a lot -> grow d (2 primitives)
    assert len(basis["o"][-1].exps) == 2
    # the 2nd d primitive helps by only 1e-4 < delta_e 1e-3 -> drop it and stop
    assert strat.next(basis, "o", 0.0499) is False
    assert strat.stop_reason == "converged"
    assert basis["o"][-1].l == "d" and len(basis["o"][-1].exps) == 1  # primitive restored


def test_polarisation_delta_e_restores_seed_shell(dummy_backend):
    """A newly-seeded l-shell that barely helps is dropped entirely."""
    basis = _sp_basis()
    strat = AutoBasisPolarisation(target=1e-12, min_l=2, max_l=3, stall_tol=1e12, delta_e=1e-3)
    strat.initialise(basis, "o")
    assert strat.next(basis, "o", 0.20) is True  # baseline -> add d
    # d seed helps by only 1e-4 < delta_e -> remove the whole d shell and stop
    assert strat.next(basis, "o", 0.1999) is False
    assert strat.stop_reason == "converged"
    assert [sh.l for sh in basis["o"]] == ["s", "p"]  # d shell restored away


def test_polarisation_loss_registry():
    """The named loss aggregates behave as documented (Eh vs Eh/electron)."""
    from basisopt.opt.optimizers import POLARISATION_LOSSES

    bsies, nelec = [1.0, 3.0], [10, 20]
    assert POLARISATION_LOSSES["mean"](bsies, nelec) == 2.0
    assert POLARISATION_LOSSES["total"](bsies, nelec) == 4.0
    assert POLARISATION_LOSSES["max"](bsies, nelec) == 3.0
    assert POLARISATION_LOSSES["mean_per_electron"](bsies, nelec) == pytest.approx(
        (1.0 / 10 + 3.0 / 20) / 2
    )
    assert POLARISATION_LOSSES["max_per_electron"](bsies, nelec) == pytest.approx(
        max(1.0 / 10, 3.0 / 20)
    )


def test_polarize_contribution_floors_below_limit():
    """Below the CBS limit the objective is floored at 0, but the signed value
    is still recorded on the molecule for diagnostics."""
    from types import SimpleNamespace

    from basisopt.opt.optimizers import _polarize_contribution

    recorded = {}
    mol = SimpleNamespace(
        cbs_limit=-1.0,
        name="m",
        nelectrons=lambda: 2,
        add_result=lambda k, v: recorded.__setitem__(k, v),
    )
    strat = SimpleNamespace(eval_type="energy")
    # above the limit: raw positive BSIE returned and recorded
    assert _polarize_contribution(mol, -0.9, strat, "o") == pytest.approx(0.1)
    assert recorded["energy_O"] == pytest.approx(0.1)
    # below the limit: floored to 0, but the (negative) signed value still recorded
    assert _polarize_contribution(mol, -1.2, strat, "o") == 0.0
    assert recorded["energy_O"] == pytest.approx(-0.2)


# --------------------------------------------------------------------------- #
# Exponent ranking (reduction) -- serial correctness + parallel == serial
# --------------------------------------------------------------------------- #
def test_rank_mol_basis_cbs_ranks_by_removal_impact(dummy_backend, monkeypatch):
    """With energy = -sum(exponents), dropping exponent e_i gives err_i = e_i, so
    ranks order smallest-exponent (least important) first. Pins the refactored
    serial ranking to a known, non-degenerate answer."""
    import basisopt.api as api
    from basisopt.testing.rank import rank_mol_basis_cbs
    from tests.data.factories import make_molecule

    basis = make_basis("o", (("s", (5.0, 1.0, 0.2)),))
    mol = make_molecule(("O",), method="linear", basis=basis, name="Oatom")
    total = 5.0 + 1.0 + 0.2

    def fake_energy(m, tmp="", **p):
        return -sum(float(x) for shells in m.basis.values() for sh in shells for x in sh.exps)

    monkeypatch.setitem(api.get_backend()._methods, "energy", fake_energy)

    errors, ranks, _, dE = rank_mol_basis_cbs(mol, "o", cbs_limit=-total)
    assert np.allclose(errors[0], [5.0, 1.0, 0.2])  # err_i = e_i
    assert list(ranks[0]) == [2, 1, 0]  # 0.2 least important
    assert dE == pytest.approx(0.0)  # full-basis energy == cbs_limit here


def test_rank_mol_basis_cbs_parallel_matches_serial():
    """The parallel ranking (actor-pool fan-out) returns identical errors/ranks to
    serial -- catching any lost or misordered trial in the run_all mapping."""
    import basisopt.api as api

    if not api._PARALLEL:
        pytest.skip("Ray not available; parallel path inactive")
    import ray

    from basisopt.testing.rank import rank_mol_basis_cbs
    from tests.data.factories import make_molecule

    basis = make_basis("o", (("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3))))
    mol = make_molecule(("O",), method="linear", basis=basis, name="Oatom")
    ray_params = {"backend": "dummy", "tmp_dir": "./tmp/", "threads_per_job": 1}
    try:
        s_err, s_ranks, _, s_dE = rank_mol_basis_cbs(mol, "o", -1.0, parallel=False)
        p_err, p_ranks, _, p_dE = rank_mol_basis_cbs(
            mol, "o", -1.0, parallel=True, ray_params=ray_params
        )
        assert len(p_err) == len(s_err) == 2
        for pe, se in zip(p_err, s_err):
            assert np.allclose(pe, se)
        for pr, sr in zip(p_ranks, s_ranks):
            assert list(pr) == list(sr)
        assert p_dE == s_dE
    finally:
        api.shutdown_actor_pool()
        ray.shutdown()


# --------------------------------------------------------------------------- #
# Contraction-coefficient ranking (uncontraction + pruning) parallelism
# --------------------------------------------------------------------------- #
def _contracted_o_molecule():
    """An O 'molecule' with genuinely contracted shells (non-identity coefs), so
    the uncontraction/pruning rankings have real trials to build."""
    from basisopt.containers import Shell
    from basisopt.molecule import Molecule

    s = Shell()
    s.l = "s"
    s.exps = np.array([5.0, 1.0, 0.2])
    s.coefs = [np.array([0.6, 0.3, 0.1])]
    p = Shell()
    p.l = "p"
    p.exps = np.array([1.5, 0.3])
    p.coefs = [np.array([0.7, 0.3])]
    mol = Molecule(name="Ocontr")
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.method = "linear"
    mol.basis = {"o": [s, p]}
    return mol


def test_rank_uncontract_serial_ranks_by_contribution(dummy_backend, monkeypatch):
    """With energy = -sum(exp . coef), freeing exponent e_i adds e_i, so err_i = e_i:
    ranks order smallest-exponent first. Pins the refactored serial uncontraction
    ranking to a known, non-degenerate answer."""
    import basisopt.api as api
    from basisopt.uncontract import rank_uncontract_element_robust

    mol = _contracted_o_molecule()

    def fake_energy(m, tmp="", **p):
        return -sum(
            float(np.dot(sh.exps, coef))
            for shells in m.basis.values()
            for sh in shells
            for coef in sh.coefs
        )

    monkeypatch.setitem(api.get_backend()._methods, "energy", fake_energy)
    _, errors, ranks, _, _ = rank_uncontract_element_robust(mol, "o", {})
    assert np.allclose(errors[0], [5.0, 1.0, 0.2])  # s shell: err_i = exponent
    assert list(ranks[0]) == [2, 1, 0]
    assert np.allclose(errors[1], [1.5, 0.3])  # p shell


def test_rank_uncontract_parallel_matches_serial():
    import basisopt.api as api

    if not api._PARALLEL:
        pytest.skip("Ray not available; parallel path inactive")
    import ray

    from basisopt.uncontract import rank_uncontract_element_robust

    mol = _contracted_o_molecule()
    ray_params = {"backend": "dummy", "tmp_dir": "./tmp/", "threads_per_job": 1}
    try:
        _, s_er, _, s_ri, s_se = rank_uncontract_element_robust(mol, "o", {}, parallel=False)
        _, p_er, _, p_ri, p_se = rank_uncontract_element_robust(
            mol, "o", {}, parallel=True, ray_params=ray_params
        )
        for pe, se in zip(p_er, s_er):
            assert np.allclose(pe, se)
        assert p_ri == s_ri
        assert np.allclose(p_se, s_se)
    finally:
        api.shutdown_actor_pool()
        ray.shutdown()


def test_rank_basis_prune_parallel_matches_serial():
    import basisopt.api as api

    if not api._PARALLEL:
        pytest.skip("Ray not available; parallel path inactive")
    import ray

    from basisopt.prune import rank_basis

    mol = _contracted_o_molecule()
    ray_params = {"backend": "dummy", "tmp_dir": "./tmp/", "threads_per_job": 1}
    try:
        _, s_er, s_ri, s_se = rank_basis(mol, "o", {}, parallel=False)
        _, p_er, p_ri, p_se = rank_basis(mol, "o", {}, parallel=True, ray_params=ray_params)
        assert p_ri == s_ri  # identical jagged (shell, contraction, primitive) ranking
        assert np.allclose(p_se, s_se)
    finally:
        api.shutdown_actor_pool()
        ray.shutdown()
