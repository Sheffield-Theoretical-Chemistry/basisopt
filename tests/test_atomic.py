"""Tests for AtomicBasis construction (basis/atomic.py)."""

import inspect

import numpy as np
import pytest

import basisopt.data as data
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


@pytest.mark.parametrize(
    "method_name, lookup_name, attr, fake_params, n_exps",
    [
        ("set_even_tempered", "get_even_temper_params", "et_params", [(0.5, 2.0, 4)], 4),
        (
            "set_well_tempered",
            "get_well_temper_params",
            "wt_params",
            [(0.1, 2.2, 20.0, 10.0, 4)],
            4,
        ),
        (
            "set_legendre",
            "get_legendre_params",
            "leg_params",
            [((3.5, 5.0, 0.8, 0.3, 0.1, 0.1), 6)],
            6,
        ),
    ],
)
def test_set_tempered_expands_looked_up_params(
    dummy_backend, monkeypatch, method_name, lookup_name, attr, fake_params, n_exps
):
    # When pre-optimized params are tabulated, the shared _set_tempered helper
    # takes the expansion branch (no optimization) and stores them on the
    # right attribute. Guards the set_even/well_tempered/set_legendre dedup.
    monkeypatch.setattr(data, lookup_name, lambda atom, accuracy: fake_params)
    atom = AtomicBasis("He")
    getattr(atom, method_name)()

    assert getattr(atom, attr) == fake_params
    shells = atom._molecule.basis[atom._symbol]
    assert len(shells) == 1
    assert len(shells[0].exps) == n_exps


def test_set_legendre_expands_shipped_data():
    # Regression: shipped _LEGENDRE_DATA entries are bare coefficient lists;
    # get_legendre_params must pair each with a primitive count n so that
    # legendre_expansion (which unpacks (A_vals, n)) does not crash on the only
    # populated tempered table.
    params = data.get_legendre_params("H")
    assert params, "H should be tabulated"
    a_vals, n = params[0]  # must unpack cleanly as (A_vals, n)
    assert len(a_vals) == 6 and isinstance(n, int)

    atom = AtomicBasis("H")
    atom.set_legendre()  # else-branch: pure expansion, no backend needed
    shells = atom._molecule.basis[atom._symbol]
    assert len(shells) == len(params)
    assert shells[0].l == "s"
    assert len(shells[0].exps) == n
    assert (shells[0].exps > 0).all()
    assert np.isfinite(shells[0].exps).all()


def test_setup_strategy_default_is_not_a_shared_instance():
    # regression: strategy defaulted to a single Strategy() created at import,
    # so every setup() call without an explicit strategy shared (and mutated)
    # the same object. The default must be a None sentinel.
    from basisopt.basis.molecular import MolecularBasis
    from basisopt.opt.optimizers import optimize

    for func in (AtomicBasis.setup, MolecularBasis.setup, optimize):
        assert inspect.signature(func).parameters["strategy"].default is None
