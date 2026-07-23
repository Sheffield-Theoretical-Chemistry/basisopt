"""Tests for the testing subsystem (testing/test.py, testing/dunham.py).

Focus on JSON round-trips, which previously left the molecule as a raw dict
(Test/PropertyTest) or crashed on construction without a molecule (DunhamTest).
"""

from basisopt.molecule import Molecule
from basisopt.testing.dunham import DunhamTest
from basisopt.testing.test import PropertyTest
from tests.data.factories import make_molecule


def test_property_test_roundtrip_rebuilds_molecule(dummy_backend):
    mol = make_molecule(("H", "H"), method="linear")
    pt = PropertyTest("h2-energy", prop="energy", mol=mol)
    pt.reference = -1.0

    restored = PropertyTest.from_dict(pt.as_dict())
    # regression: molecule used to come back as a raw dict
    assert isinstance(restored.molecule, Molecule)
    assert restored.name == "h2-energy"
    assert restored.reference == -1.0
    assert restored.eval_type == "energy"
    assert restored.molecule.natoms() == 2


def test_dunham_construct_without_molecule_does_not_crash():
    # regression: __init__ called reduced_mass() unconditionally, dereferencing
    # a None molecule.
    dt = DunhamTest("bare")
    assert dt.molecule is None


def test_dunham_roundtrip_rebuilds_molecule():
    dt = DunhamTest("hf", mol_str="HF,0.9", poly_order=4, step=0.02, Emax=0.1)
    assert dt.molecule.natoms() == 2

    restored = DunhamTest.from_dict(dt.as_dict())
    assert isinstance(restored.molecule, Molecule)
    assert restored.molecule.natoms() == 2
    assert restored.poly_order == 4
    assert restored.step == 0.02
    assert restored.Emax == 0.1
    # with a real molecule restored, reduced_mass() works
    assert restored.reduced_mass() > 0.0
