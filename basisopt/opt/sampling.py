"""Parallel multi-start sampling (Level B).

Run *N* independent optimisations from perturbed seeds and keep the best, in
parallel across Ray -- each start runs its *inner* calcs serially, so there is no
nested Ray and no core oversubscription (the starts get the workers). Toggled per
step via a ``sampling:`` config block. Seedable stages:

- **primitives** -- perturb the Legendre expansion coefficients;
- **polarisation** -- perturb the shell seed exponent.

Start 0 is always the unperturbed default, so multi-start is never worse than the
single default start. Selection keeps the lowest objective (energy / BSIE loss).

Latin-hypercube seeding and population methods (CMA-ES/DE) are future swaps for
``perturb`` / ``_dispatch``; see docs/improvements/sampling-methods.md.
"""

import copy

import numpy as np

from basisopt import api
from basisopt.exceptions import FailedCalculation
from basisopt.util import bo_logger


def _zero_reg(x):
    """Module-level (picklable) no-op regulariser for the polarisation opt_data."""
    return 0.0


def perturb(base, n_starts, spread, rng_seed=None):
    """Return ``n_starts`` seed variants of ``base``.

    Start 0 is the unperturbed base; the rest are multiplicative Gaussian
    perturbations ``x * (1 + spread * N(0, 1))``. ``base`` is either a scalar
    (a polarisation seed exponent) or a list of arrays (Legendre coefficients
    per shell).
    """
    rng = np.random.default_rng(rng_seed)
    seeds = [copy.deepcopy(base)]
    for _ in range(max(0, int(n_starts) - 1)):
        if np.isscalar(base):
            seeds.append(float(base) * (1.0 + spread * float(rng.standard_normal())))
        else:
            seeds.append(
                [
                    np.asarray(a, dtype=float) * (1.0 + spread * rng.standard_normal(np.shape(a)))
                    for a in base
                ]
            )
    return seeds


def _activate(ray_params):
    """Set up the backend inside a worker (parallel path); no-op when serial."""
    if ray_params:
        api.set_backend(ray_params["backend"], verbose=False)
        api.set_tmp_dir(ray_params.get("tmp_dir", "./tmp/"), verbose=False)
        api._apply_additional_params(ray_params)
        api._apply_worker_log_level(ray_params)


def _primitives_start(mol, strategy, algorithm, opt_params, ray_params):
    """One primitives optimisation (serial inside). Returns
    ``(objective, stop_reason, basis)`` or ``None`` if this seed failed."""
    from basisopt import opt

    _activate(ray_params)
    try:
        opt.atom_auto(molecule=mol, strategy=strategy, algorithm=algorithm, opt_params=opt_params)
    except Exception as exc:  # a bad seed must not kill the whole sweep
        bo_logger.warning("multistart: a primitives start failed (%s)", exc)
        return None
    return strategy.last_objective, getattr(strategy, "stop_reason", None), mol.basis


def _polarisation_start(molecules, combined, opt_data, npass, loss, ray_params):
    """One polarisation optimisation (serial over molecules inside). Returns
    ``(objective, stop_reason, basis)`` or ``None`` if this seed failed."""
    from basisopt.opt import collective_polarize

    _activate(ray_params)
    strategy = opt_data[0][2]
    try:
        collective_polarize(
            molecules, combined, opt_data=opt_data, npass=npass, parallel=False, loss=loss
        )
    except Exception as exc:
        bo_logger.warning("multistart: a polarisation start failed (%s)", exc)
        return None
    return strategy.last_objective, getattr(strategy, "stop_reason", None), combined


def _dispatch(worker, jobs, sampling_cfg, ray_params):
    """Run ``worker(*job, ray_params)`` for each job -- across Ray when available
    (each start serial inside), else serially in-process."""
    if api._PARALLEL and len(jobs) > 1:
        import ray

        api.set_parallel(True, int(sampling_cfg.get("n_cores", len(jobs))))
        remote = ray.remote(num_cpus=int(sampling_cfg.get("threads_per_job", 1)))(worker)
        return ray.get([remote.remote(*job, ray_params) for job in jobs])
    return [worker(*job, None) for job in jobs]


def _best(results, n_jobs):
    """Keep the start with the lowest objective (lowest energy / loss = best)."""
    valid = [r for r in results if r is not None]
    if not valid:
        raise FailedCalculation("all multistart runs failed")
    obj, reason, basis = min(valid, key=lambda r: r[0])
    bo_logger.info(
        "multistart: best of %d starts (%d succeeded) -> objective %.8g",
        n_jobs,
        len(valid),
        obj,
    )
    return obj, reason, basis, n_jobs


def multistart_primitives(
    mol, strategy, base_legendre, algorithm, opt_params, sampling_cfg, ray_params
):
    """Multi-start the primitives (Legendre) optimisation. Returns
    ``(objective, stop_reason, basis, n_starts)``."""
    seeds = perturb(
        base_legendre,
        sampling_cfg.get("n_starts", 4),
        sampling_cfg.get("seed_spread", 0.25),
        sampling_cfg.get("rng_seed"),
    )
    jobs = []
    for seed in seeds:
        strat = copy.deepcopy(strategy)
        strat.legendre_params = seed
        jobs.append((copy.deepcopy(mol), strat, algorithm, opt_params))
    results = _dispatch(_primitives_start, jobs, sampling_cfg, ray_params)
    return _best(results, len(jobs))


def multistart_polarisation(
    molecules,
    combined,
    strategy,
    element,
    algorithm,
    opt_params,
    npass,
    sampling_cfg,
    ray_params,
    loss="mean_per_electron",
):
    """Multi-start the polarisation optimisation (perturbing the seed exponent).
    Returns ``(objective, stop_reason, basis, n_starts)``; ``basis`` is the winning
    combined basis dict (its element entry carries the grown polarisation shells)."""
    seeds = perturb(
        strategy.seed_exponent,
        sampling_cfg.get("n_starts", 4),
        sampling_cfg.get("seed_spread", 0.25),
        sampling_cfg.get("rng_seed"),
    )
    jobs = []
    for seed in seeds:
        strat = copy.deepcopy(strategy)
        strat.seed_exponent = float(seed)
        opt_data = [(element, algorithm, strat, _zero_reg, opt_params)]
        jobs.append((copy.deepcopy(molecules), copy.deepcopy(combined), opt_data, npass, loss))
    results = _dispatch(_polarisation_start, jobs, sampling_cfg, ray_params)
    return _best(results, len(jobs))
