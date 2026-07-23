"""Characterization tests for the optimization drivers and collective functions.

Run end-to-end against the exponent-independent Dummy backend so the result-dict
structure and control flow are locked before the driver/collective functions are
collapsed. Tests that document not-yet-fixed bugs are marked xfail and reference
the commit that will fix them.
"""

import pytest

from basisopt.opt.auto_basis import (
    AutoBasisFree,
    AutoBasisReduceStrategy,
    AutoBasisReduceStrategyAll,
)
from basisopt.opt.contraction import ContractionStrategy
from basisopt.opt.optimizers import (
    Minimizer,
    Optimizer,
    atom_auto,
    atom_auto_reduce,
    collective_minimize,
    collective_optimize,
    collective_polarize,
    contraction_optimize,
    optimize,
)
from basisopt.opt.strategies import Strategy
from tests.data.factories import make_basis, make_molecule
from tests.data.shells import get_vdz_internal


# --------------------------------------------------------------------------- #
# _atomic_opt via optimize()
# --------------------------------------------------------------------------- #
def test_optimize_default_strategy(dummy_backend):
    mol = make_molecule(("H", "H"), method="linear")
    results = optimize(mol, element="H", algorithm="l-bfgs-b", strategy=Strategy())
    # base Strategy optimizes each of the two shells once
    assert set(results) == {"atomicopt1", "atomicopt2"}


# --------------------------------------------------------------------------- #
# _atomic_opt_auto via atom_auto()
# --------------------------------------------------------------------------- #
def test_atom_auto_autobasisfree(dummy_backend):
    mol = make_molecule(("H", "H"), method="linear")
    strategy = AutoBasisFree(target=1e-6)
    strategy.set_cbs_limit(-2.0)  # == dummy energy, so the initial pass finishes it
    results = atom_auto(mol, element="H", strategy=strategy)
    assert len(results) >= 1
    # auto driver records the distance to the CBS limit on each step
    for res in results.values():
        assert "dE_CBS" in res


# --------------------------------------------------------------------------- #
# _atomic_opt_auto_reduce via atom_auto_reduce()
# --------------------------------------------------------------------------- #
def test_atom_auto_reduce_removes_and_restores(dummy_backend):
    basis = make_basis("h", (("s", (9.0, 3.0, 1.0, 0.3)), ("p", (1.5, 0.4))))
    mol = make_molecule(("H", "H"), method="linear", basis=basis)
    strategy = AutoBasisReduceStrategy(target=1e-6)
    # CBS limit below the (constant) energy so the first removal is rejected and
    # the strategy restores the basis and stops
    strategy.set_cbs_limit(-3.0)
    results = atom_auto_reduce(mol, element="H", strategy=strategy)
    assert isinstance(results, dict)
    # driver runs a final calculation and records it on the molecule
    assert mol.get_result(strategy.eval_type) != 0.0
    # no net exponents removed (removal was rejected)
    assert strategy.n_exps_removed == [0, 0]


def test_atom_auto_reduce_all(dummy_backend):
    basis = make_basis("h", (("s", (9.0, 3.0, 1.0, 0.3)), ("p", (1.5, 0.4))))
    mol = make_molecule(("H", "H"), method="linear", basis=basis)
    strategy = AutoBasisReduceStrategyAll(target=1e-6)
    strategy.set_cbs_limit(-3.0)
    results = atom_auto_reduce(mol, element="H", strategy=strategy)
    assert isinstance(results, dict)


# --------------------------------------------------------------------------- #
# _atomic_contract via contraction_optimize()
# --------------------------------------------------------------------------- #
def test_contraction_optimize_structure(dummy_backend):
    mol = make_molecule(("H", "H"), method="linear", basis=get_vdz_internal())
    # contraction_optimize requires an uncontracted-energy reference
    mol.add_reference("uncontracted_energy", -1.0)
    results = contraction_optimize(
        mol, ContractionStrategy(target=1e-5), element="H", opt_params={"options": {"maxiter": 2}}
    )
    assert isinstance(results, dict)
    # at least one contraction function was optimized
    assert len(results) >= 1


# --------------------------------------------------------------------------- #
# collective_* (sequential path)
# --------------------------------------------------------------------------- #
def test_collective_optimize_sequential(dummy_backend):
    mol = make_molecule(("H", "H"), method="linear")
    basis = mol.basis
    mol.add_reference("energy", -2.0)
    opt_data = [("h", "l-bfgs-b", Strategy(), lambda x: 0, {})]
    results = collective_optimize([mol], basis, opt_data=opt_data, npass=1, parallel=False)
    assert results


def test_collective_minimize_sequential(dummy_backend):
    mol = make_molecule(("H", "H"), method="linear")
    basis = mol.basis
    opt_data = [("h", "l-bfgs-b", Strategy(), lambda x: 0, {})]
    results = collective_minimize([mol], basis, opt_data=opt_data, npass=1, parallel=False)
    assert results


def test_collective_polarize_sequential(dummy_backend):
    mol = make_molecule(("H", "H"), method="linear", cbs_limit=-2.5)
    basis = mol.basis
    opt_data = [("h", "l-bfgs-b", Strategy(), lambda x: 0, {})]
    results = collective_polarize([mol], basis, opt_data=opt_data, npass=1, parallel=False)
    assert results
    # polarize stores abs(value - cbs_limit) as the per-molecule result
    assert mol.get_result("energy_H") == pytest.approx(abs(-2.0 - (-2.5)))


# --------------------------------------------------------------------------- #
# Optimizer / Minimizer classes
# --------------------------------------------------------------------------- #
def test_optimizer_class_runs(dummy_backend):
    basis = make_basis("h", (("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3))))
    mol = make_molecule(("H", "H"), method="linear", basis=basis)
    opt = Optimizer(
        strategy=Strategy(), params={}, basis=basis, elements=["h"],
        opt_params={"options": {"maxiter": 2}},
    )
    opt.run(molecules=[mol], algorithm="l-bfgs-b")
    assert set(opt.get_results()) == {"h_opt1", "h_opt2"}


def test_optimizer_multi_element_keeps_all_results(dummy_backend):
    # regression: results were keyed f"opt{ctr}" with ctr reset per element into
    # one shared dict, so a multi-element run silently kept only the last
    # element's steps (and which one survived was nondeterministic via a set).
    basis = {
        **make_basis("h", (("s", (5.0, 1.0, 0.2)),)),
        **make_basis("o", (("s", (9.0, 3.0, 1.0)), ("p", (1.5, 0.4)))),
    }
    mol = make_molecule(("H", "O"), method="linear", basis=basis)
    opt = Optimizer(
        strategy=Strategy(), params={}, basis=basis, elements=["h", "o"],
        opt_params={"options": {"maxiter": 2}},
    )
    opt.run(molecules=[mol], algorithm="l-bfgs-b")
    keys = set(opt.get_results())
    assert "h_opt1" in keys  # H's step survives instead of being clobbered
    assert {"o_opt1", "o_opt2"} <= keys


def test_optimizer_instances_do_not_share_mutable_defaults(dummy_backend):
    """Regression: basis/elements used to be mutable defaults shared across all
    Optimizer instances, and _initialize mutates elements in place."""
    a = Optimizer(strategy=Strategy(), params={})
    b = Optimizer(strategy=Strategy(), params={})
    a.elements.append("h")
    a.basis["h"] = []
    assert b.elements == []
    assert b.basis == {}
    assert a.elements is not b.elements
    assert a.basis is not b.basis


def test_minimizer_class_runs(dummy_backend):
    basis = make_basis("h", (("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3))))
    mol = make_molecule(("H", "H"), method="linear", basis=basis)
    mn = Minimizer(
        strategy=Strategy(), params={}, basis=basis, elements=["h"],
        opt_params={"options": {"maxiter": 2}},
    )
    mn.run(molecules=[mol], algorithm="l-bfgs-b")
    assert set(mn.get_results()) == {"h_opt1", "h_opt2"}


# --------------------------------------------------------------------------- #
# Serialization round-trips (broken today -> xfail until C2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cls", [AutoBasisFree, AutoBasisReduceStrategy, AutoBasisReduceStrategyAll]
)
def test_autobasis_serialization_roundtrip(dummy_backend, cls):
    strategy = cls(target=1e-5)
    strategy.set_cbs_limit(-2.0)
    d = strategy.as_dict()
    restored = cls.from_dict(d)
    assert restored.target == strategy.target
    assert restored.cbs_limit == strategy.cbs_limit


def test_autobasis_legendre_initialise_requires_n_prim(dummy_backend):
    # regression: n_coefs defaulted to None, so initialise did zip(..., None)
    # -> TypeError. It must now raise a clear, actionable error.
    from basisopt.opt.auto_basis import AutoBasisLegendre

    strategy = AutoBasisLegendre()  # no n_coefs
    strategy.set_cbs_limit(-75.0)
    with pytest.raises(ValueError, match="n_coefs"):
        strategy.initialise({}, "O")


def test_autobasis_legendre_serialization_roundtrip(dummy_backend):
    # regression: n_prim/legendre_params were dropped by as_dict/from_dict, so a
    # reloaded strategy reset both to None and crashed on the next initialise.
    from basisopt.opt.auto_basis import AutoBasisLegendre

    strategy = AutoBasisLegendre(n_coefs=(9, 3))
    strategy.legendre_params = [[1.0] * 6, [2.0] * 6]
    strategy.set_cbs_limit(-75.0)

    restored = AutoBasisLegendre.from_dict(strategy.as_dict())
    assert restored.n_prim == (9, 3)
    assert restored.legendre_params == [[1.0] * 6, [2.0] * 6]
    assert restored.cbs_limit == -75.0
    assert restored.target == strategy.target
