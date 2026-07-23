# Wrappers for testing functionality
import numpy as np

from basisopt.data import AM_DICT
from basisopt.molecule import Molecule
from basisopt.wrappers.wrapper import Wrapper, available


def _linear(x, a=1.0):
    return a * x


def _exp(x, a=1.0):
    return a * np.exp(a * x)


def _quadratic(x, a=1.0):
    return 1 + a * x * (1 + a * x)


def _uniform(x, a=1.0):
    return a


"""Available dummy methods"""
_method_lookup = {
    "linear": _linear,
    "exp": _exp,
    "quadratic": _quadratic,
    "uniform": _uniform,
}


class DummyWrapper(Wrapper):
    """A Wrapper that can't actually be used to do computations.
    It has two purposes: to be a default Wrapper so that the library
    can be used even when calculations aren't needed (e.g. if looking
    at and analysing previously computed results); and to make testing
    a lot easier, as we don't have to have e.g. Psi4 installed. As such
    this has minimal functionality.
    """

    def __init__(self):
        Wrapper.__init__(self, name="Dummy")
        self._method_strings = {
            "linear": ["energy", "dipole", "quadrupole", "polarizability"],
            "exp": ["energy", "dipole"],
            "quadratic": ["energy"],
            "uniform": ["energy", "dipole", "quadrupole"],
        }
        self._memory_set = False
        self._value = 0
        self._basis_value = 0

    def convert_molecule(self, m: Molecule) -> int:
        """Dummy molecule converter"""
        return m.natoms()

    def initialise(self, m: Molecule, name: str = "", tmp: str = ""):
        """Initialises calc by converting molecule,
        setting globals and self._basis_value
        """
        # deal with molecule conversion
        self._value = self.convert_molecule(m)

        # logic to check global options
        if "memory" in self._globals:
            self._memory_set = True

        # set basis
        self._basis_value = len(m.basis)

    @available
    def energy(self, mol, tmp="", **params):
        self.initialise(mol, name="energy", tmp=tmp)
        return _method_lookup[mol.method](self._value, a=-1.0)

    @available
    def dipole(self, mol, tmp="", **params):
        self.initialise(mol, name="dipole", tmp=tmp)
        return _method_lookup[mol.method](self._value, a=0.5)

    @available
    def quadrupole(self, mol, tmp="", **params):
        self.initialise(mol, name="quadrupole", tmp=tmp)
        return _method_lookup[mol.method](self._value, a=0.1)

    @available
    def polarizability(self, mol, tmp="", **params):
        self.initialise(mol, name="polarizability", tmp=tmp)
        return _method_lookup[mol.method](self._value, a=self._basis_value)

    def natural_orbitals(self, molecule, params=None):
        """Deterministic stand-in for the pipeline's NAO contraction path.

        Each primitive is treated as its own natural orbital (identity
        coefficients) with occupations descending, so the auto-basis contraction
        step can be exercised without a real quantum-chemistry backend.
        """
        result = {}
        for element in molecule.unique_atoms():
            for shell in molecule.basis[element.lower()]:
                n = len(shell.exps)
                result[AM_DICT[shell.l]] = (np.arange(n, 0, -1, dtype=float), np.eye(n))
        return result
