import logging
import os
from datetime import datetime
from typing import Any, Callable

import colorlog

from basisopt.exceptions import FailedCalculation
from basisopt.molecule import Molecule
from basisopt.wrappers.dummy import DummyWrapper
from basisopt.wrappers.wrapper import Wrapper

bo_logger = logging.getLogger('basisopt')

try:
    _PARALLEL = True
    num_cores = 2
    import ray

    # from basisopt.parallelise import distribute


except ImportError:
    bo_logger.error('DASK Import Error')
    _PARALLEL = False

_BACKENDS = {}
_CURRENT_BACKEND = DummyWrapper()
_TMP_DIR = "."


def set_parallel(value: bool = True, number_cores: int = 2):
    """Turns parallelism on/off using Ray."""
    global _PARALLEL
    global num_cores
    num_cores = number_cores
    if value:
        if not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True, num_cpus=num_cores)
                _PARALLEL = True
            except Exception as e:
                _PARALLEL = False
                bo_logger.warning(f"Could not initialize Ray: {str(e)}")
        else:
            _PARALLEL = True  # Ray is already initialized
            ray.shutdown()  # Optional: restart Ray to configure with new number of cores
            ray.init(ignore_reinit_error=True, num_cpus=num_cores)
    else:
        if ray.is_initialized():
            ray.shutdown()
        _PARALLEL = False


# def set_parallel(value: bool = True, number_cores: int = 2):
#     """Turns parallelism on/off"""
#     global _PARALLEL
#     global num_cores
#     num_cores = number_cores
#     if value:
#         try:
#             import dask

#             _PARALLEL = True
#         except ImportError:
#             _PARALLEL = False
#             bo_logger.warning("Could not import dask, parallelism turned off")
#     else:
#         _PARALLEL = False


def register_backend(func: Callable[[str, str], None]) -> Callable[[str, str], None]:
    """Registers a function to set the backend for basisopt"""
    _BACKENDS[func.__name__] = func
    return func


def set_backend(name: str, path: str = "", verbose=True):
    """Sets the global backend for basisopt calculations

    Arguments:
         name (str): the name of the program to use
         path (str): absolute path to the program executable
    """
    try:
        if verbose:
            func = _BACKENDS[name.lower()]
            if _CURRENT_BACKEND._name != "Dummy":
                bo_logger.warning("Overwriting previous backend")
            func(path)
            bo_logger.info("Backend set to %s", _CURRENT_BACKEND._name)
        else:
            func = _BACKENDS[name.lower()]
            if _CURRENT_BACKEND._name != "Dummy":
                if verbose:
                    bo_logger.warning("Overwriting previous backend")
            func(path)
    except KeyError:
        bo_logger.error("%s is not a registered backend for basisopt", name)


def get_backend() -> Wrapper:
    """Returns:
    backend (Wrapper): the Wrapper object for the current backend
    """
    return _CURRENT_BACKEND


def set_tmp_dir(path: str, verbose=True):
    """Sets the working directory for all backend calculations,
    creating the directory if it doesn't already exist.

    Arguments:
         path (str): path to the scratch directory
    """
    global _TMP_DIR
    # check if dir exists, and create if not
    if not os.path.isdir(path):
        bo_logger.info("Created directory at %s", path)
        os.makedirs(path, exist_ok=True)
    _TMP_DIR = path
    if verbose:
        bo_logger.info("Scratch directory set to %s", _TMP_DIR)


def get_tmp_dir() -> str:
    """Returns:
    Path to the current scratch/temp directory
    """
    return _TMP_DIR


def which_backend() -> str:
    """Returns:
    str: The name of the currently registered backend
    """
    return _CURRENT_BACKEND._name


def set_logger(level: int = logging.INFO, filename: str = None):
    """Initialises Python logging, formatting it nicely,
    and optionally printing to a file.
    """
    log_format = '%(asctime)s - ' '%(funcName)s - ' '%(levelname)s - ' '%(message)s'
    bold_seq = '\033[1m'
    colorlog_format = f'{bold_seq} ' '%(log_color)s ' f'{log_format}'
    colorlog.basicConfig(format=colorlog_format)
    bo_logger.setLevel(level)

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        bo_logger.addHandler(fh)


@register_backend
def dummy(path: str):
    """Sets backend to the DummyWrapper for testing and
    for when calculations aren't needed.
    """
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = DummyWrapper()


@register_backend
def psi4(path: str):
    """Tests Psi4 import and prepares to be used as calculation backend"""
    try:
        global _CURRENT_BACKEND
        from basisopt.wrappers.psi4 import Psi4Wrapper

        _CURRENT_BACKEND = Psi4Wrapper()
    except ImportError:
        bo_logger.error("Psi4 backend not found!")


@register_backend
def orca(path: str):
    """Tests orca import and prepares to be used as calculation backend"""
    global _CURRENT_BACKEND
    from basisopt.wrappers.orca import OrcaWrapper

    _CURRENT_BACKEND = OrcaWrapper(path)
    bo_logger.info("ORCA install dir at: %s", path)


@register_backend
def molpro(path: str):
    """Tests pymolpro import and prepares to be used as calculation backend"""
    try:
        global _CURRENT_BACKEND
        from basisopt.wrappers.molpro import MolproWrapper

        _CURRENT_BACKEND = MolproWrapper()
    except ImportError:
        bo_logger.error("Molpro backend (using pymolpro) not found!")


def run_calculation(
    evaluate: str = 'energy', mol: Molecule = None, params: dict[Any, Any] = {}
) -> int:
    """Interface to the wrapper used to run a calculation.

    Arguments:
        evaluate (str): The function to be called for the computation
        mol (Molecule): molecule to run the calculation on
        params (dict): A dictionary of parameters needed for the computation

    Returns:
        int: 0 on success, non-zero on failure
    """
    result = _CURRENT_BACKEND.run(evaluate, mol, params, tmp=_TMP_DIR)
    _CURRENT_BACKEND.clean()
    return result


def _one_job(
    mol: Molecule, evaluate: str = 'energy', params: dict[Any, Any] = {}
) -> tuple[str, Any]:
    """Internal helper to run a single job in a distributed array"""
    success = _CURRENT_BACKEND.run(evaluate, mol, params, tmp=_TMP_DIR)
    if success != 0:
        raise FailedCalculation
    value = _CURRENT_BACKEND.get_value(evaluate)
    _CURRENT_BACKEND.clean()
    return mol.name, value


# def run_all(
#     evaluate: str = 'energy',
#     mols: list[Molecule] = [],
#     params: dict[Any, Any] = {},
#     parallel: bool = False,
#     count=None,
# ) -> dict[str, Any]:
#     """Runs calculations over a set of molecules, optionally in parallel

#     Arguments:
#          evaluate (str): the property to evaluate
#          mols (list): a list of Molecule objects to run
#          params (dict): parameters for backend
#          parallel (bool): if True, will try to run distributed

#     Returns:
#          a dictionary  of the form {molecule name: value}
#     """
#     results = {}
#     if parallel and _PARALLEL:
#         kwargs = {"evaluate": evaluate, "params": params}
#         with dask.config.set({"multiprocessing.context": "fork"}):
#             tmp_results = distribute(num_cores, _one_job, mols, **kwargs)
#         for n, v in tmp_results:
#             results[n] = v
#     else:
#         for m in mols:
#             name, value = _one_job(m, evaluate=evaluate, params=params)
#             results[name] = value
#     return results


@ray.remote
def run_one_job(molecule, evaluate, params):
    """Remote function to process each molecule using the backend."""
    set_backend('psi4', verbose=False)  # Set the backend for each remote task
    set_tmp_dir('./temp_directory', verbose=False)  # Set temporary directory if needed
    try:
        name, value = _one_job(molecule, evaluate=evaluate, params=params)
        return name, value
    except FailedCalculation:
        bo_logger.error(f"Calculation failed for molecule: {molecule.name}")
        return molecule.name, None


def run_all(
    evaluate: str = 'energy',
    mols: list = [],
    params: dict = {},
    parallel: bool = False,
    count=None,
) -> dict:
    """Runs calculations over a set of molecules, optionally in parallel

    Arguments:
        evaluate (str): the property to evaluate
        mols (list): a list of Molecule objects to run
        params (dict): parameters for backend
        parallel (bool): if True, will try to run distributed

    Returns:
        a dictionary of the form {molecule name: value}
    """
    results = {}

    if parallel and _PARALLEL:
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=num_cores)

        # Submit jobs to Ray
        futures = [run_one_job.remote(m, evaluate, params) for m in mols]
        tmp_results = ray.get(futures)

        # Collect results
        for name, value in tmp_results:
            if value is not None:
                results[name] = value
    else:
        # Sequential processing
        for m in mols:
            set_backend('psi4', verbose=False)  # Set the backend for each job in sequential mode
            set_tmp_dir('./temp_directory', verbose=False)  # Set temporary directory if needed
            try:
                name, value = _one_job(m, evaluate=evaluate, params=params)
                results[name] = value
            except FailedCalculation:
                bo_logger.error(f"Calculation failed for molecule: {m.name}")
                results[m.name] = None

    return results
