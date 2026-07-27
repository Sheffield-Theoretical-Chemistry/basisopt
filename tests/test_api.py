import logging
import os
import subprocess
import sys

import pytest

from basisopt import api
from basisopt.wrappers import Wrapper


def test_run_all_parallel_requires_ray_params(dummy_backend):
    # regression: parallel=True with no ray_params silently dropped every result
    # (None['backend'] TypeError swallowed in the worker). Now it fails fast.
    from tests.data.factories import make_molecule

    if not api._PARALLEL:
        pytest.skip("Ray not available; parallel path inactive")
    mol = make_molecule(("H", "H"), method="linear")
    with pytest.raises(ValueError, match="ray_params"):
        api.run_all(evaluate="energy", mols=[mol], parallel=True)


def test_run_all_serial_applies_shared_basis(dummy_backend):
    """The shared basis is assigned to every molecule in the serial path."""
    from tests.data.factories import make_basis, make_molecule

    mols = [make_molecule(("H", "H"), method="linear", name=f"s{i}") for i in range(2)]
    shared = make_basis("h", (("s", (2.0,)),))
    results = api.run_all(
        evaluate="energy", mols=mols, params={}, parallel=False, shared_basis=shared
    )
    assert set(results) == {"s0", "s1"}
    for m in mols:
        assert m.basis is shared  # serial path assigns the shared basis directly
        assert results[m.name] == -m.natoms()  # dummy 'linear' energy = -natoms


def test_run_all_parallel_actor_pool_dummy():
    """The warm actor pool fans a molecule set out in parallel (dummy backend),
    returns correct per-molecule energies, applies shared_basis in-actor, and
    restores the callers' molecule bases afterwards."""
    if not api._PARALLEL:
        pytest.skip("Ray not available; parallel path inactive")
    import ray

    from tests.data.factories import make_basis, make_molecule

    mols = [make_molecule(("H", "H"), method="linear", name=f"m{i}") for i in range(3)]
    originals = [m.basis for m in mols]
    shared = make_basis("h", (("s", (1.0, 0.3)),))
    ray_params = {"backend": "dummy", "tmp_dir": "./tmp/", "threads_per_job": 1}
    try:
        results = api.run_all(
            evaluate="energy",
            mols=mols,
            params={},
            parallel=True,
            ray_params=ray_params,
            shared_basis=shared,
        )
        assert set(results) == {m.name for m in mols}
        for m in mols:
            assert results[m.name] == -m.natoms()
        # local molecule bases are restored (the shared basis travelled via Ray)
        for m, original in zip(mols, originals):
            assert m.basis is original
    finally:
        api.shutdown_actor_pool()
        ray.shutdown()


def test_backend_registration():
    assert len(api._BACKENDS.keys()) > 0
    assert "dummy" in api._BACKENDS.keys()


def test_set_backend():
    api.set_backend("DuMmy")
    assert api.which_backend() == "Dummy"
    api.set_backend("NotABackendType")
    assert api.which_backend() == "Dummy"


def test_get_backend():
    assert isinstance(api.get_backend(), Wrapper)
    assert api.get_backend()._name == "Dummy"


def test_which_backend():
    assert api.which_backend() == "Dummy"


def test_get_set_tmp_dir():
    assert api.get_tmp_dir() == "."

    NEW_TMP = "_tmp/"
    api.set_tmp_dir(NEW_TMP)
    assert api.get_tmp_dir() == NEW_TMP
    assert os.path.isdir(NEW_TMP)

    api._TMP_DIR = ""
    api.set_tmp_dir(NEW_TMP)
    assert api.get_tmp_dir() == NEW_TMP

    if os.path.isdir(NEW_TMP):
        os.rmdir(NEW_TMP)


def test_import_without_ray():
    # regression: `@ray.remote` used to decorate `_run_one_job` unconditionally at
    # import, so if the Ray import failed `ray` was unbound -> NameError at import.
    # Mask Ray in a fresh interpreter and confirm the package still imports.
    script = (
        "import sys; sys.modules['ray'] = None\n"
        "import basisopt.api as api\n"
        "assert api._PARALLEL is False\n"
        "assert callable(api._run_one_job)\n"
        "assert not hasattr(api._run_one_job, 'remote')\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_set_logger():
    logger = logging.getLogger("basisopt")
    assert logger.getEffectiveLevel() == logging.INFO

    api.set_logger(level=logging.WARNING)
    assert logger.getEffectiveLevel() == logging.WARNING
