"""Tests for the parallel multi-start sampler (Level B)."""

import numpy as np
import pytest

from basisopt.opt import sampling


def test_perturb_start0_unperturbed_and_shapes():
    base = [np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5])]
    seeds = sampling.perturb(base, n_starts=4, spread=0.3, rng_seed=0)
    assert len(seeds) == 4
    for a, b in zip(seeds[0], base):  # start 0 is the exact base
        assert np.allclose(a, b)
    for s in seeds[1:]:  # perturbations keep shapes but differ
        assert [x.shape for x in s] == [x.shape for x in base]
    assert not np.allclose(seeds[1][0], base[0])


def test_perturb_scalar_and_deterministic():
    s1 = sampling.perturb(1.0, 3, 0.2, rng_seed=7)
    s2 = sampling.perturb(1.0, 3, 0.2, rng_seed=7)
    assert s1[0] == 1.0  # start 0 unperturbed
    assert s1 == s2  # reproducible with a fixed seed


def test_best_picks_min_objective_and_raises_when_all_fail():
    from basisopt.exceptions import FailedCalculation

    results = [(-1.0, "target", {"o": "A"}), None, (-2.0, "stall", {"o": "B"})]
    obj, reason, basis, n = sampling._best(results, n_jobs=3)
    assert obj == -2.0 and reason == "stall" and basis == {"o": "B"} and n == 3
    with pytest.raises(FailedCalculation):
        sampling._best([None, None], n_jobs=2)


def test_multistart_polarisation_over_dummy(dummy_backend):
    """End-to-end multi-start polarisation over the dummy backend: n_starts seed
    exponents are perturbed, each start grows shells serially, and the best combined
    basis (with the grown polarisation shells on the target element) is returned."""
    import basisopt.api as api
    from basisopt.opt.polarisation import AutoBasisPolarisation
    from tests.data.factories import make_basis, make_molecule

    combined = make_basis("o", config=(("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3))))
    mol = make_molecule(("O", "O"), method="linear", basis=combined, name="O2", cbs_limit=-2.0)
    strat = AutoBasisPolarisation(target=1e-6, min_l=2, max_l=2, seed_exponent=1.0, max_n=2)
    strat.params = {}
    sampling_cfg = {"n_starts": 2, "seed_spread": 0.2, "rng_seed": 0, "n_cores": 2}
    ray_params = {"backend": "dummy", "tmp_dir": "./tmp/", "threads_per_job": 1}
    try:
        obj, reason, basis, n_starts = sampling.multistart_polarisation(
            [mol],
            combined,
            strat,
            "o",
            "Nelder-Mead",
            {},
            1,
            sampling_cfg,
            ray_params,
            loss="mean_per_valence_electron",
        )
        assert n_starts == 2
        # a d polarisation shell was grown onto the sp base
        assert any(sh.l == "d" for sh in basis["o"])
    finally:
        if api._PARALLEL:
            import ray

            if ray.is_initialized():
                ray.shutdown()


def test_multistart_primitives_over_dummy(dummy_backend):
    """End-to-end multi-start over the dummy backend: n_starts Legendre
    optimisations run (via Ray if available, else serial) and the best basis is
    returned. Dummy energy of a 1-atom molecule is -1, so cbs_limit=-1 converges
    each start immediately."""
    import basisopt.api as api
    from basisopt.opt.auto_basis import AutoBasisLegendre
    from tests.data.factories import make_molecule

    mol = make_molecule(("O",), method="linear", name="O")
    strat = AutoBasisLegendre(target=1e-6, n_coefs=(4, 3))
    strat.set_cbs_limit(-1.0)
    strat.legendre_params = [
        np.array([1.6, -5.1, 0.05, -0.17]),
        np.array([-0.97, 1.77, -0.27]),
    ]
    sampling_cfg = {"n_starts": 2, "seed_spread": 0.2, "rng_seed": 0, "n_cores": 2}
    ray_params = {"backend": "dummy", "tmp_dir": "./tmp/", "threads_per_job": 1}
    try:
        obj, reason, basis, n_starts = sampling.multistart_primitives(
            mol, strat, strat.legendre_params, "Nelder-Mead", {}, sampling_cfg, ray_params
        )
        assert n_starts == 2
        assert "o" in basis and len(basis["o"]) == 2  # s + p shells
    finally:
        if api._PARALLEL:
            import ray

            if ray.is_initialized():
                ray.shutdown()
