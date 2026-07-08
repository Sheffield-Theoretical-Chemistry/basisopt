"""Tests for AtomicBasis construction (basis/atomic.py)."""

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
