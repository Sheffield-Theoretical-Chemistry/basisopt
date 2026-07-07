"""Shared pytest fixtures for the optimization test suite.

These support characterization/regression tests for the optimization machinery
(strategies, drivers, the collective functions, and the Optimizer/Minimizer
classes) using the exponent-independent ``DummyWrapper`` backend, so the tests
lock control-flow/state-machine behaviour without needing a real QC backend.

Factory helpers (make_shell/make_basis/make_molecule) live in
``tests.data.factories`` so they can be imported directly by test modules.
"""

import pytest

from basisopt import api
from basisopt.wrappers.dummy import DummyWrapper
from tests.data.factories import make_basis, make_molecule, make_shell


@pytest.fixture
def dummy_backend():
    """Reset the global backend to a fresh DummyWrapper for the test.

    DummyWrapper energy is a function of atom count only (independent of the
    exponents), which makes optimization control flow deterministic.
    """
    previous = api.get_backend()
    api._CURRENT_BACKEND = DummyWrapper()
    yield api._CURRENT_BACKEND
    api._CURRENT_BACKEND = previous


@pytest.fixture
def shell_factory():
    return make_shell


@pytest.fixture
def basis_factory():
    return make_basis


@pytest.fixture
def molecule_factory():
    return make_molecule
