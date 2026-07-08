"""Tests for the exponent-ranking helpers (testing/rank.py).

These pin the shared _drop_each_exponent loop used by rank_primitives and
rank_mol_basis_cbs. The Dummy 'linear' backend gives energy = -natoms,
independent of the exponents, so dropping any single exponent leaves the energy
unchanged - which makes the returned shapes and difference values predictable.
"""

from basisopt.testing.rank import rank_mol_basis_cbs
from tests.data.factories import make_basis, make_molecule


def test_rank_mol_basis_cbs_shapes_and_values(dummy_backend):
    basis = make_basis("h", (("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3))))
    mol = make_molecule(("H",), method="linear", basis=basis)

    errors, ranks, energies, dE_initial = rank_mol_basis_cbs(mol, "H", cbs_limit=-2.0)

    # one entry per shell; one value per exponent within each
    assert [len(e) for e in errors] == [3, 2]
    assert [len(r) for r in ranks] == [3, 2]
    assert [len(en) for en in energies] == [3, 2]

    # dummy 'linear' energy = -natoms = -1, so every drop reports -1 ...
    assert all(en == -1.0 for row in energies for en in row)
    # ... and each error is |-1 - (-2)| = 1
    assert all(abs(e - 1.0) < 1e-12 for arr in errors for e in arr)
    # initial difference to the CBS limit: -1 - (-2) = 1
    assert abs(dE_initial - 1.0) < 1e-12


def test_rank_mol_basis_cbs_restores_the_basis(dummy_backend):
    # the loop mutates a deepcopy, so the caller's molecule is untouched
    basis = make_basis("h", (("s", (5.0, 1.0, 0.2)),))
    mol = make_molecule(("H",), method="linear", basis=basis)
    before = mol.basis["h"][0].exps.copy()

    rank_mol_basis_cbs(mol, "H", cbs_limit=-1.0)

    assert list(mol.basis["h"][0].exps) == list(before)
