"""Tests for MolecularBasis and MoleculeLoader (basis/molecular.py)."""

from basisopt.basis.molecular import MolecularBasis, MoleculeLoader
from basisopt.molecule import Molecule
from tests.data.factories import make_molecule


def test_molecular_basis_json_roundtrip(dummy_backend, tmp_path):
    mol = make_molecule(("H", "H"), method="linear")
    mb = MolecularBasis(name="testmb")
    mb.add_molecule(mol)

    path = str(tmp_path / "mb.json")
    mb.save(path)
    loaded = MolecularBasis().load(path)

    assert isinstance(loaded, MolecularBasis)
    # molecules must come back as Molecule objects, not raw dicts
    restored = loaded.get_molecule(mol.name)
    assert isinstance(restored, Molecule)
    assert set(loaded.unique_atoms()) == {"h"}


def test_molecule_loader_run_calculations(dummy_backend):
    m1 = make_molecule(("H", "H"), method="linear", name="m1")
    m2 = make_molecule(("H",), method="linear", name="m2")
    loader = MoleculeLoader(molecules=[m1, m2])
    loader.run_calculations(params={})
    # DummyWrapper energy == -natoms; both molecules should have a recorded result
    assert m1.get_result("energy") == -2.0
    assert m2.get_result("energy") == -1.0
    assert set(loader.unique_atoms()) == {"H"}
    assert len(loader) == 2
