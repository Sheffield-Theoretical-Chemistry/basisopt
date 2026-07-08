import logging
import os
import subprocess
import sys

from basisopt import api
from basisopt.wrappers import Wrapper


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
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_set_logger():
    logger = logging.getLogger("basisopt")
    assert logger.getEffectiveLevel() == logging.INFO

    api.set_logger(level=logging.WARNING)
    assert logger.getEffectiveLevel() == logging.WARNING
