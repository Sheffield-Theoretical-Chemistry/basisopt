"""Tests for AtomicBasis construction (basis/atomic.py)."""

import inspect

from basisopt.basis.atomic import AtomicBasis


def test_valid_element():
    a = AtomicBasis("C")
    assert a.element is not None
    assert a._symbol == "c"
    assert a.element.symbol == "C"


def test_invalid_element_does_not_raise():
    # unknown element -> logged and left unset (guarded later), not a raise
    a = AtomicBasis("Xx")
    assert a.element is None


def test_charged_atom_without_multiplicity():
    # regression: mult=None used to reach the setter's `None < 1` -> TypeError
    cation = AtomicBasis("Li", charge=1)
    assert cation.multiplicity is not None
    assert cation.multiplicity >= 1


def test_setup_strategy_default_is_not_a_shared_instance():
    # regression: strategy defaulted to a single Strategy() created at import,
    # so every setup() call without an explicit strategy shared (and mutated)
    # the same object. The default must be a None sentinel.
    from basisopt.basis.molecular import MolecularBasis
    from basisopt.opt.optimizers import optimize

    for func in (AtomicBasis.setup, MolecularBasis.setup, optimize):
        assert inspect.signature(func).parameters["strategy"].default is None
