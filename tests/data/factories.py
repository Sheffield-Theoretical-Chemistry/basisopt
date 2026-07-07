"""Factory helpers for building shells, bases, and molecules in tests.

These support characterization tests for the optimization machinery using the
exponent-independent ``DummyWrapper`` backend.
"""

import numpy as np

from basisopt.basis.basis import uncontract_shell
from basisopt.containers import Shell
from basisopt.molecule import Molecule


def make_shell(l: str, exps) -> Shell:
    """Build a single uncontracted Shell with the given angular momentum/exponents."""
    shell = Shell()
    shell.l = l
    shell.exps = np.array(exps, dtype=float)
    uncontract_shell(shell)
    return shell


def make_basis(element: str = "h", config=(("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3)))):
    """Build an internal basis dict for a single element.

    ``config`` is an iterable of ``(angular_momentum, exponents)`` tuples.
    """
    return {element.lower(): [make_shell(l, exps) for l, exps in config]}


def make_molecule(
    atoms=("H", "H"),
    method: str = "linear",
    basis=None,
    name: str = "test_mol",
    cbs_limit=None,
) -> Molecule:
    """Build a Molecule wired for the Dummy backend.

    ``method`` must be a DummyWrapper method ("linear", "exp", "quadratic",
    "uniform"); "linear" gives energy = -natoms (constant in the exponents).
    """
    mol = Molecule(name=name)
    for i, atom in enumerate(atoms):
        mol.add_atom(element=atom, coord=[0.0, 0.0, 0.7 * i])
    mol.method = method
    if basis is None:
        basis = make_basis(atoms[0])
    mol.basis = basis
    if cbs_limit is not None:
        mol.cbs_limit = cbs_limit
    return mol
