import numpy as np

import basisopt.basis.basis as basis
from tests.data.shells import get_vdz_internal
from tests.data.utils import almost_equal


def test_uncontract_shell():
    shell = basis.Shell()
    shell.exps = np.array([0.1, 1.0, 2.0, 4.0, 8.0])
    shell.coefs = [
        np.array([0.5, -0.5, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, -0.5, -0.5]),
    ]
    basis.uncontract_shell(shell)
    assert len(shell.coefs) == len(shell.exps)
    for c in shell.coefs:
        assert np.sum(c) == 1.0


def test_uncontract():
    vdz = get_vdz_internal()

    new_vdz = basis.uncontract(vdz, elements=["o"])
    assert "o" not in new_vdz

    new_vdz = basis.uncontract(vdz)
    for s in new_vdz["h"]:
        assert len(s.exps) == len(s.coefs)


def test_uncontract_does_not_mutate_original():
    vdz = get_vdz_internal()
    n_coefs_before = len(vdz["h"][0].coefs)  # contracted (< n_exps)
    new = basis.uncontract(vdz)
    # the returned basis is uncontracted...
    assert len(new["h"][0].coefs) == len(new["h"][0].exps)
    # ...but the caller's basis is untouched (was mutated in place before)
    assert len(vdz["h"][0].coefs) == n_coefs_before


def test_legendre_expansion_explicit_l():
    leg = [
        ((3.0, 4.5, 0.75, 0.25, 0.1, 0.1), 5),
        ((2.2, 4.5, 0.44, 0.29, 0.07, 0.02), 4),
    ]
    # explicit l=0 now forces s for every shell (was treated as "unset")
    assert [s.l for s in basis.legendre_expansion(leg, l=0)] == ["s", "s"]
    # default (l unset) derives each shell's l from its position
    assert [s.l for s in basis.legendre_expansion(leg)] == ["s", "p"]


def test_even_temper_expansion():
    et_params = [(1.5, 1.9, 15), (2.7, 1.6, 12)]
    et_basis = basis.even_temper_expansion(et_params)
    assert len(et_basis) == 2

    s_shell = et_basis[0]
    p_shell = et_basis[1]
    assert len(s_shell.exps) == 15
    assert almost_equal(s_shell.exps[0], 1.5, thresh=1e-6)
    assert almost_equal(s_shell.exps[6], 70.568822, thresh=1e-6)
    assert len(p_shell.exps) == 12
    assert almost_equal(p_shell.exps[11], 474.989023, thresh=1e-6)


def test_legendre_expansion():
    leg_params = [
        ((3.0, 4.5, 0.75, 0.25, 0.1, 0.1), 13),
        ((2.2, 4.5, 0.44, 0.29, 0.07, 0.02), 12),
    ]
    leg_basis = basis.legendre_expansion(leg_params)
    assert len(leg_basis) == 2

    s_shell = leg_basis[0]
    p_shell = leg_basis[1]
    assert len(s_shell.exps) == 13
    # Values on the corrected Petersson node grid x_j = 2j/(n-1) - 1
    assert almost_equal(s_shell.exps[1], 0.689883, thresh=1e-6)
    assert almost_equal(s_shell.exps[7], 30.005022, thresh=1e-6)
    assert len(p_shell.exps) == 12
    assert almost_equal(p_shell.exps[10], 457.218308, thresh=1e-6)


def test_legendre_expansion_uses_petersson_grid():
    # The Legendre nodes must be Petersson's x_j = 2j/(n-1) - 1, spanning [-1, +1]
    # symmetrically. Selecting only A_1 (P_1(x) = x) makes ln(exp) == x_j, so the
    # log-exponents recover the node grid directly.
    n = 6
    a_vals = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    shell = basis.legendre_expansion([(a_vals, n)])[0]
    nodes = np.log(shell.exps)
    expected = (2 * np.arange(n)) / (n - 1) - 1
    assert np.allclose(nodes, expected)
    assert almost_equal(nodes[0], -1.0, thresh=1e-9)
    assert almost_equal(nodes[-1], 1.0, thresh=1e-9)


def test_well_temper_expansion():
    wt_params = [(0.1, 2.3, 12.2, 8.6, 13), (0.2, 2.1, 32.8, 9.9, 10)]
    wt_basis = basis.well_temper_expansion(wt_params)
    assert len(wt_basis) == 2

    s_shell = wt_basis[0]
    p_shell = wt_basis[1]
    assert len(s_shell.exps) == 13
    assert almost_equal(s_shell.exps[0], 0.1, thresh=1e-6)
    assert almost_equal(s_shell.exps[10], 1615.706126, thresh=1e-6)
    assert len(p_shell.exps) == 10
    assert almost_equal(p_shell.exps[6], 33.623095, thresh=1e-6)


def test_fix_ratio():
    exps = np.array([0.1, 0.2, 0.58, 1.3, 3.0, 8.2])
    new_exps = basis.fix_ratio(exps)
    assert almost_equal(np.sum(exps - new_exps), 0.0)

    new_exps = basis.fix_ratio(exps, 2.4)
    expected = np.array([0.1, 0.24, 0.58, 1.392, 3.3408, 8.2])
    assert almost_equal(np.sum(expected - new_exps), 0.0)


def test_basis_init():
    b = basis.Basis()
    assert type(b.results).__name__ == "Result"
    assert b.opt_results is None
    assert b._tests == []
    assert b._molecule is None


def test_basis_load(tmp_path):
    # JSON (MSONable) save/load round-trip (Basis.as_dict needs a molecule)
    from basisopt.molecule import Molecule

    b = basis.Basis()
    b._molecule = Molecule(name="test")
    path = str(tmp_path / "basis.json")
    b.save(path)

    loaded = basis.Basis().load(path)
    assert isinstance(loaded, basis.Basis)
    assert loaded._molecule.name == "test"
    assert loaded._tests == []


def test_basis_tests():
    from basisopt.testing.test import Test

    b = basis.Basis()
    assert b.get_test("missing") is None
    t = Test(name="my_test")
    b.register_test(t)
    assert b.get_test("my_test") is t
    assert b.get_test("still_missing") is None
