"""Tests for JKFitBasis (basis/jkfit.py)."""

from basisopt.basis.jkfit import JKFitBasis
from basisopt.opt.strategies import Strategy
from tests.data.factories import make_molecule


def test_jkfit_importable_and_serializes(dummy_backend, tmp_path):
    # JKFitBasis used to fail at import (from basisopt.opt import Strategy)
    mol = make_molecule(("H", "H"), method="linear", name="h2")
    jk = JKFitBasis(name="H", mol=mol)
    jk.strategy = Strategy(eval_type="energy")
    jk._done_setup = True

    # as_dict must store the strategy as a dict, not the raw object
    d = jk.as_dict()
    assert isinstance(d["strategy"], dict)
    assert d["strategy"]["@class"] == "Strategy"

    # JSON round-trip must rebuild the strategy as a Strategy object
    path = str(tmp_path / "jk.json")
    jk.save(path)
    loaded = JKFitBasis(mol=mol).load(path)
    assert isinstance(loaded.strategy, Strategy)
    assert loaded.basis_type == "jkfit"
