import logging
import os
from typing import Any, Callable

import colorlog

from basisopt.exceptions import BackendNotFound, FailedCalculation
from basisopt.molecule import Molecule
from basisopt.parallelise import chunk
from basisopt.wrappers.dummy import DummyWrapper
from basisopt.wrappers.wrapper import Wrapper

bo_logger = logging.getLogger('basisopt')

try:
    _PARALLEL = True
    num_cores = 2
    import ray
except ImportError:
    bo_logger.debug("Ray not installed; parallelism disabled")
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
                bo_logger.warning("Could not initialize Ray: %s", e)
        else:
            _PARALLEL = True
            ray.shutdown()  # Restart Ray to configure with new number of cores
            ray.init(ignore_reinit_error=True, num_cpus=num_cores)
    else:
        if ray.is_initialized():
            ray.shutdown()
        _PARALLEL = False


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
    # Check for a trailing slash and add if missing
    if not path.endswith('/'):
        path += '/'
    # check if dir exists, and create if not
    if not os.path.isdir(path):
        bo_logger.info("Created directory at %s", path)
        os.makedirs(path, exist_ok=True)
    # Check the path is valid
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


def set_logger(
    level: int = logging.INFO,
    filename: str = None,
    *,
    rich: bool = False,
    console=None,
):
    """(Re)configure the ``basisopt`` logger.

    Owns a single console handler on ``bo_logger`` itself (not the root logger), so
    repeat calls actually change the level/handler instead of being a no-op, and
    handlers never stack. ``bo_logger.propagate`` is left at its default ``True`` and
    the root logger is never configured, so there is no double output and pytest's
    ``caplog`` (which captures at the root) still sees records.

    Arguments:
        level: logging level for the console (and file) handler.
        filename: if given, also tee to this file (plain, uncoloured format).
        rich: render the console handler through ``rich.logging.RichHandler`` so log
            lines share the auto-basis CLI's visual system.
        console: an optional ``rich.console.Console`` for the RichHandler to write to
            (implies ``rich=True``); defaults to Rich's own stderr console.
    """
    log_format = '%(asctime)s - ' '%(funcName)s - ' '%(levelname)s - ' '%(message)s'

    # Drop any handler we installed on a previous call (idempotent reconfigure).
    for handler in list(bo_logger.handlers):
        if getattr(handler, "_basisopt_managed", False):
            bo_logger.removeHandler(handler)

    if rich or console is not None:
        from rich.logging import RichHandler

        console_handler = RichHandler(
            console=console, rich_tracebacks=True, show_path=False, markup=False
        )
        console_handler.setFormatter(logging.Formatter('%(funcName)s: %(message)s'))
    else:
        bold_seq = '\033[1m'
        colorlog_format = f'{bold_seq} ' '%(log_color)s ' f'{log_format}'
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(colorlog_format))

    console_handler.setLevel(level)
    console_handler._basisopt_managed = True
    bo_logger.addHandler(console_handler)
    bo_logger.setLevel(level)

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(log_format))
        fh._basisopt_managed = True
        bo_logger.addHandler(fh)


class _AutoBasisAdapter(logging.LoggerAdapter):
    """Tags every message with ``[auto-basis]`` so pipeline lines stand out in a run."""

    def process(self, msg, kwargs):
        return f"[auto-basis] {msg}", kwargs


# Child logger for the auto-basis pipeline: records propagate up to ``bo_logger``'s handlers.
ab_logger = _AutoBasisAdapter(logging.getLogger("basisopt.autobasis"), {})


def _apply_worker_log_level(ray_params) -> None:
    """Inside a Ray worker, match the driver's log level when it was threaded through
    ``ray_params`` (each worker otherwise re-initialises at INFO on import)."""
    if ray_params and ray_params.get("log_level") is not None:
        bo_logger.setLevel(ray_params["log_level"])


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
    except ImportError as exc:
        raise BackendNotFound(
            "Psi4 backend not found. Install psi4 (e.g. `conda install -c psi4 psi4`) "
            "to use it, or choose another backend."
        ) from exc


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
    except ImportError as exc:
        raise BackendNotFound(
            "Molpro backend not found. Install pymolpro to use it, or choose another backend."
        ) from exc


def run_calculation(
    evaluate: str = 'energy', mol: Molecule = None, params: dict[Any, Any] = None
) -> int:
    """Interface to the wrapper used to run a calculation.

    Arguments:
        evaluate (str): The function to be called for the computation
        mol (Molecule): molecule to run the calculation on
        params (dict): A dictionary of parameters needed for the computation

    Returns:
        int: 0 on success, non-zero on failure
    """
    params = {} if params is None else params
    result = _CURRENT_BACKEND.run(evaluate, mol, params, tmp=_TMP_DIR)
    _CURRENT_BACKEND.clean()
    return result


def _one_job(
    mol: Molecule, evaluate: str = 'energy', params: dict[Any, Any] = None
) -> tuple[str, Any]:
    """Internal helper to run a single job in a distributed array"""
    params = {} if params is None else params
    success = _CURRENT_BACKEND.run(evaluate, mol, params, tmp=_TMP_DIR)
    if success != 0:
        raise FailedCalculation
    value = _CURRENT_BACKEND.get_value(evaluate)
    _CURRENT_BACKEND.clean()
    return mol.name, value


def _apply_additional_params(ray_params):
    """Apply additional parameters based on the backend."""
    if ray_params and ray_params.get('backend') == 'psi4':
        num_threads = ray_params.get('threads_per_job')
        import psi4

        if num_threads:
            psi4.core.set_num_threads(num_threads)
        default_path = ray_params.get('default_path')
        if default_path:
            psi4_io = psi4.core.IOManager.shared_object()
            psi4_io.set_default_path(default_path)
            # bo_logger.info(f"Setting psi4 path to {default_path}")


def _run_one_job(molecule, evaluate, params, ray_params=None):
    """Remote function to process each molecule using the backend."""
    try:
        set_backend(ray_params['backend'], verbose=False)
        _apply_additional_params(ray_params)
        _apply_worker_log_level(ray_params)
    except TypeError:
        bo_logger.error(
            'No backend set for Ray. Please pass a dictionary with the "backend" key assigned to a valid backend. The ray parameters should be passed into the optimization through the ray_params argument.'
        )
    if ray_params:
        set_tmp_dir(ray_params['tmp_dir'], verbose=False)
    else:
        set_tmp_dir('./tmp/', verbose=False)
    try:
        name, value = _one_job(molecule, evaluate=evaluate, params=params)
        return name, value
    except FailedCalculation:
        bo_logger.warning("Calculation failed for molecule '%s'", molecule.name)
        return molecule.name, None


# Only wrap as a Ray remote when Ray actually imported; decorating at import
# time unconditionally would raise NameError on a machine without Ray. When Ray
# is unavailable `_run_one_job` stays a plain function and is never called via
# `.remote` (run_all only uses the remote path under `parallel and _PARALLEL`).
if _PARALLEL:
    _run_one_job = ray.remote(_run_one_job)


# --------------------------------------------------------------------------- #
# Warm backend actor pool
# --------------------------------------------------------------------------- #
class _BackendActorImpl:
    """A persistent Ray actor that sets up the backend ONCE (keeping e.g. psi4
    warm) and then processes many molecules across successive ``run_all`` calls,
    avoiding the per-task ``set_backend`` of the stateless path. Never decorated
    at import (so a Ray-less import is fine); wrapped with ``ray.remote`` in
    :func:`_get_actor_pool`, which only runs under the parallel branch."""

    def __init__(self, ray_params):
        set_backend(ray_params["backend"], verbose=False)
        set_tmp_dir(ray_params.get("tmp_dir", "./tmp/"), verbose=False)
        _apply_additional_params(ray_params)
        _apply_worker_log_level(ray_params)

    def run_batch(self, molecules, evaluate, params, shared_basis=None, robust=False):
        """Run a chunk of molecules on the warm backend. ``shared_basis`` (if
        given, resolved by Ray from a single object-store entry) is applied to
        each molecule so the basis travels once per call, not once per molecule.
        ``robust`` makes any per-molecule failure (e.g. a linear dependency that
        raises rather than returning a failure code) yield ``None`` instead of
        killing the whole batch -- needed by the ranking passes."""
        out = []
        for mol in molecules:
            if shared_basis is not None:
                mol.basis = shared_basis
            try:
                success = _CURRENT_BACKEND.run(evaluate, mol, params, tmp=_TMP_DIR)
                value = _CURRENT_BACKEND.get_value(evaluate) if success == 0 else None
            except FailedCalculation:
                bo_logger.warning("Calculation failed for molecule '%s'", mol.name)
                value = None
            except Exception:
                if not robust:
                    raise
                bo_logger.exception("Unexpected error for molecule '%s'", mol.name)
                value = None
            _CURRENT_BACKEND.clean()
            out.append((mol.name, value))
        return out


_ACTOR_POOL = None
_ACTOR_POOL_KEY = None


def _get_actor_pool(ray_params: dict, pool_size: int) -> list:
    """Return a cached pool of ``pool_size`` warm backend actors, (re)building it
    only when the backend / scratch / threads / size change. Each actor reserves
    ``threads_per_job`` CPUs so Ray does not oversubscribe the cores (previously
    every task took 1 CPU while psi4 span up ``threads_per_job`` threads)."""
    global _ACTOR_POOL, _ACTOR_POOL_KEY
    threads = ray_params.get("threads_per_job") or 1
    key = (ray_params.get("backend"), ray_params.get("tmp_dir"), threads, pool_size)
    if _ACTOR_POOL is not None and _ACTOR_POOL_KEY == key:
        return _ACTOR_POOL
    shutdown_actor_pool()
    actor_cls = ray.remote(num_cpus=threads)(_BackendActorImpl)
    _ACTOR_POOL = [actor_cls.remote(ray_params) for _ in range(pool_size)]
    _ACTOR_POOL_KEY = key
    return _ACTOR_POOL


def shutdown_actor_pool():
    """Tear down the cached actor pool (between differently-configured runs, or
    at shutdown). Best-effort: ignores actors already gone."""
    global _ACTOR_POOL, _ACTOR_POOL_KEY
    if _ACTOR_POOL:
        for actor in _ACTOR_POOL:
            try:
                ray.kill(actor)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
    _ACTOR_POOL = None
    _ACTOR_POOL_KEY = None


def run_all(
    evaluate: str = 'energy',
    mols: list = None,
    params: dict = None,
    parallel: bool = False,
    count=None,
    ray_params=None,
    shared_basis=None,
    robust=False,
) -> dict:
    """Runs calculations over a set of molecules, optionally in parallel

    Arguments:
        evaluate (str): the property to evaluate
        mols (list): a list of Molecule objects to run
        params (dict): parameters for backend
        parallel (bool): if True, will try to run distributed
        shared_basis: a basis dict applied to every molecule (shipped once via
            the object store in the parallel path)
        robust (bool): if True, any per-molecule failure -- including a backend
            error that raises rather than returning a failure code (e.g. a linear
            dependency from freeing a function) -- yields ``None`` for that
            molecule instead of propagating. Used by the ranking passes.

    Returns:
        a dictionary of the form {molecule name: value}
    """
    mols = [] if mols is None else mols
    params = {} if params is None else params
    results = {}
    if not mols:
        return results

    if parallel and _PARALLEL:
        # Ray workers start fresh and must be told which backend to use; without
        # ray_params, _run_one_job used to hit None['backend'] (a swallowed
        # TypeError) and silently drop every result. Fail fast instead.
        if not ray_params or 'backend' not in ray_params:
            raise ValueError(
                "run_all(parallel=True) requires ray_params with at least a "
                "'backend' key so the Ray workers can set the backend."
            )
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=num_cores)

        # Size the pool by cores/threads (stable across iterations so the actors
        # stay warm); use a subset when there are fewer molecules than workers.
        threads = ray_params.get('threads_per_job') or 1
        pool_size = ray_params.get('n_workers') or max(1, num_cores // threads)
        pool = _get_actor_pool(ray_params, pool_size)
        active = max(1, min(pool_size, len(mols)))

        # Put the (large, shared) basis and params in the object store ONCE; Ray
        # resolves the refs to values in each actor call, so they are serialized
        # once per run_all rather than once per molecule.
        params_ref = ray.put(params)
        basis_ref = ray.put(shared_basis) if shared_basis is not None else None
        # Drop the shared basis from each molecule while submitting (it travels
        # via basis_ref), then restore, so molecule serialization stays light.
        stashed = None
        if shared_basis is not None:
            stashed = [m.basis for m in mols]
            for m in mols:
                m.basis = {}
        try:
            batches = chunk(mols, active)
            futures = [
                pool[i].run_batch.remote(batches[i], evaluate, params_ref, basis_ref, robust)
                for i in range(active)
            ]
            gathered = ray.get(futures)
        finally:
            if stashed is not None:
                for m, saved in zip(mols, stashed):
                    m.basis = saved

        for batch in gathered:
            for name, value in batch:
                if value is not None:
                    results[name] = value
    else:
        # Sequential processing: use the already-configured backend and scratch
        # directory unless ray_params explicitly overrides them (previously this
        # dereferenced ray_params['backend'] unconditionally and raised TypeError
        # whenever ray_params was None, e.g. every non-parallel collective_* run).
        if ray_params:
            set_backend(ray_params['backend'], verbose=False)
            set_tmp_dir(ray_params['tmp_dir'], verbose=False)
        for m in mols:
            if shared_basis is not None:
                m.basis = shared_basis
            try:
                name, value = _one_job(m, evaluate=evaluate, params=params)
                results[name] = value
            except FailedCalculation:
                bo_logger.warning("Calculation failed for molecule '%s'", m.name)
                results[m.name] = None
            except Exception:
                if not robust:
                    raise
                bo_logger.exception("Unexpected error for molecule '%s'", m.name)
                results[m.name] = None

    return results
