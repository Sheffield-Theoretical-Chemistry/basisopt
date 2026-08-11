"""Competing-configuration polarisation search.

An alternative to the greedy :class:`~basisopt.opt.polarisation.AutoBasisPolarisation`
growth. Instead of saturating one angular-momentum shell before advancing to the
next, this *compares competing angular-momentum configurations at equal function
budget* -- the historical method described in ``docs/improvements/SCIENCE_TODO.md``
("2d vs 1d1f, then 3d1f vs 2d2f vs 2d1f1g ...; keep the lowest loss"). It is the
mode intended for the second row, where the optimal split of functions across d/f/g
matters more than for the first row and greedy misallocates.

Algorithm (greedy best-first over configurations):

1. seed the configuration ``{min_l: 1}`` (1d), optimise all its exponents, score it;
2. stop if the loss is below ``target``;
3. enumerate every single-function-addition *neighbour* (all at the same total
   count, so the comparison is fair): +1 to an existing ``l``, or seed the next
   ``l`` up to ``max_l`` -- subject to ``max_n`` per shell and an optional
   non-increasing pattern (``n_d >= n_f >= n_g``);
4. optimise every neighbour's exponents (independent -> fanned across Ray exactly
   like the sampler: candidates get the workers, molecules serial inside each);
5. commit the lowest-loss neighbour if it improves the best by more than
   ``min_improvement``; else stop (converged);
6. repeat, bounded by ``max_l`` / ``max_n`` / ``max_total``.

Each candidate is scored through the same :func:`collective_polarize` path as the
greedy strategy, so the floored-BSIE contribution and the ``POLARISATION_LOSSES``
aggregate (total- or valence-normalised) are identical.
"""

import copy

import numpy as np

from basisopt import api
from basisopt.containers import InternalBasis, Shell
from basisopt.data import AM_DICT, INV_AM_DICT
from basisopt.exceptions import FailedCalculation
from basisopt.util import bo_logger

from ..basis.basis import uncontract_shell
from .optimizers import _pol_loss_mean_per_electron, _pol_loss_mean_per_valence_electron
from .polarisation import AutoBasisPolarisation
from .preconditioners import make_positive
from .sampling import _activate, _dispatch, _zero_reg


def _config_str(config: dict) -> str:
    """Render a ``{l: n}`` configuration as e.g. ``2d1f`` (occupied shells only)."""
    return "".join(f"{config[l]}{INV_AM_DICT[l]}" for l in sorted(config) if config[l] > 0)


def _seed_exponents(seed: float, n: int, ratio: float = 3.0) -> np.ndarray:
    """``n`` starting exponents in a geometric progression centred on ``seed``.

    A single primitive is just ``[seed]``; scipy reoptimises them, so this only
    needs to be a sensible starting spread (ratio ~3 between neighbours).
    """
    n = int(n)
    if n <= 1:
        return np.array([float(seed)])
    ks = np.arange(n) - (n - 1) / 2.0
    return float(seed) * (ratio**ks)


class FixedConfigPolarisation(AutoBasisPolarisation):
    """Non-growing polarisation strategy: builds a FIXED ``{l: n}`` configuration of
    polarisation shells onto the sp base and optimises *all* their exponents together
    in a single minimisation. Used to score one candidate configuration; it never
    grows or advances shells."""

    def __init__(self, config: dict, seed_exponent: float = 1.0, eval_type: str = "energy"):
        super().__init__(eval_type=eval_type, seed_exponent=seed_exponent, pre=make_positive)
        self.name = "FixedConfigPolarisation"
        self.config = {int(l): int(n) for l, n in config.items() if int(n) > 0}
        self._base_len = None
        self._pol_indices = []

    def initialise(self, basis: InternalBasis, element: str):
        """Append the fixed configuration's shells onto the sp base (idempotent)."""
        # Drop shells we appended on a previous pass so npass>1 does not duplicate.
        if self._base_len is not None:
            del basis[element][self._base_len :]
        self._base_len = len(basis[element])
        for l in sorted(self.config):
            shell = Shell()
            shell.l = INV_AM_DICT[l]
            shell.exps = _seed_exponents(self.seed_exponent, self.config[l])
            uncontract_shell(shell)
            basis[element].append(shell)
        self._pol_indices = list(range(self._base_len, len(basis[element])))
        self._step = max(self._base_len - 1, 0)
        self.first_run = True
        self.last_objective = 0.0
        self.delta_objective = 0.0
        self.stop_reason = "fixed"
        self._done = False

    def get_active(self, basis: InternalBasis, element: str) -> np.ndarray:
        """All polarisation-shell exponents concatenated (preconditioned)."""
        parts = [basis[element][i].exps for i in self._pol_indices]
        x = np.concatenate(parts) if parts else np.array([])
        return self.pre(x, **self.pre_params)

    def set_active(self, values: np.ndarray, basis: InternalBasis, element: str):
        """Split the flat vector back across the polarisation shells."""
        y = self.pre.inverse(np.array(values), **self.pre_params)
        pos = 0
        for i in self._pol_indices:
            shell = basis[element][i]
            n = len(shell.exps)
            shell.exps = y[pos : pos + n]
            uncontract_shell(shell)
            pos += n

    def next(self, basis: InternalBasis, element: str, objective: float) -> bool:
        """Optimise the fixed exponents exactly once, then stop."""
        self.last_objective = objective
        if self._done:
            return False
        self._done = True
        return True


def _is_sensible(config: dict, min_l: int) -> bool:
    """A configuration is a contiguous block from ``min_l`` with non-increasing
    counts (``n_d >= n_f >= n_g >= ...``)."""
    ls = sorted(l for l, n in config.items() if n > 0)
    if not ls:
        return True
    if ls != list(range(min_l, min_l + len(ls))):
        return False
    counts = [config[l] for l in ls]
    return all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))


def _neighbours(config, min_l, max_l, max_n, max_total, non_increasing):
    """Every single-function-addition neighbour of ``config`` (same +1 total count)."""
    if max_total is not None and sum(config.values()) >= max_total:
        return []
    out = []
    for l in range(min_l, max_l + 1):
        cand = dict(config)
        cand[l] = cand.get(l, 0) + 1
        if max_n is not None and cand[l] > max_n:
            continue
        # no gaps: a freshly-seeded shell must sit directly above an occupied one
        if cand[l] == 1 and l != min_l and config.get(l - 1, 0) == 0:
            continue
        if non_increasing and not _is_sensible(cand, min_l):
            continue
        out.append(cand)
    return out


def _evaluate_config(
    molecules,
    combined_sp,
    config,
    element,
    algorithm,
    opt_params,
    params,
    seed_exponent,
    loss,
    npass,
    ray_params,
):
    """Optimise one candidate configuration's exponents and return its score.

    Runs serially over the molecule set (the caller parallelises *candidates*).
    Returns ``(loss, basis, config)`` or ``None`` if the candidate failed.
    """
    from basisopt.opt import collective_polarize

    _activate(ray_params)
    basis = copy.deepcopy(combined_sp)
    strategy = FixedConfigPolarisation(config, seed_exponent=seed_exponent)
    strategy.params = params
    opt_data = [(element, algorithm, strategy, _zero_reg, opt_params)]
    try:
        collective_polarize(
            molecules, basis, opt_data=opt_data, npass=npass, parallel=False, loss=loss
        )
    except Exception as exc:  # a bad candidate must not kill the level
        bo_logger.warning("config_search: candidate %s failed (%s)", _config_str(config), exc)
        return None
    return float(strategy.last_objective), basis, dict(config)


def _trace_entry(config, basis, molecules, params):
    """Per-committed-step diagnostics: per-molecule BSIE, total vs valence electron
    counts, and the aggregate loss under BOTH the total- and valence-normalisations
    (so a single run shows the core-dilution effect side by side)."""
    energies = api.run_all(
        evaluate="energy", mols=molecules, params=params, parallel=False, shared_basis=basis
    )
    per_mol, bsies, nelec, nval = {}, [], [], []
    for m in molecules:
        bsie = max(0.0, float(energies[m.name]) - m.cbs_limit)
        ne, nv = m.nelectrons(), m.nvalence_electrons()
        bsies.append(bsie)
        nelec.append(ne)
        nval.append(nv)
        per_mol[m.name] = {
            "bsie": bsie,
            "nelectrons": ne,
            "nvalence": nv,
            "bsie_per_electron": bsie / ne,
            "bsie_per_valence_electron": bsie / nv,
        }
    return {
        "config": _config_str(config),
        "n_functions": int(sum(config.values())),
        "loss_mean_per_electron": _pol_loss_mean_per_electron(bsies, nelec, nval),
        "loss_mean_per_valence_electron": _pol_loss_mean_per_valence_electron(bsies, nelec, nval),
        "per_molecule": per_mol,
    }


def config_search_polarisation(
    molecules,
    combined_sp,
    element,
    algorithm,
    opt_params,
    params,
    *,
    min_l=2,
    max_l=4,
    seed_exponent=1.0,
    target=1e-4,
    loss="mean_per_valence_electron",
    max_n=None,
    max_total=None,
    min_improvement=0.0,
    non_increasing=True,
    npass=1,
    parallel=False,
    ray_params=None,
    dispatch_cfg=None,
):
    """Greedy best-first search over polarisation configurations (see module docstring).

    Returns ``(final_loss, stop_reason, best_basis, best_config_str, trace)`` where
    ``best_basis`` is the winning combined basis (its element entry carries the grown
    polarisation shells) and ``trace`` is the per-committed-step diagnostic list.
    """
    if max_l < min_l:
        raise ValueError(f"max_l ({max_l}) must be >= min_l ({min_l}).")
    dispatch_cfg = dict(dispatch_cfg or {})

    def score(config):
        return _evaluate_config(
            copy.deepcopy(molecules),
            combined_sp,
            config,
            element,
            algorithm,
            opt_params,
            params,
            seed_exponent,
            loss,
            npass,
            ray_params if parallel else None,
        )

    best = _evaluate_config(
        copy.deepcopy(molecules),
        combined_sp,
        {min_l: 1},
        element,
        algorithm,
        opt_params,
        params,
        seed_exponent,
        loss,
        npass,
        None,
    )
    if best is None:
        raise FailedCalculation("config_search: the seed configuration failed to optimise")
    best_loss, best_basis, best_config = best
    trace = [_trace_entry(best_config, best_basis, molecules, params)]
    bo_logger.info("config_search: seed %s -> loss %.6e", _config_str(best_config), best_loss)

    stop_reason = "exhausted"
    while True:
        if best_loss < target:
            stop_reason = "target"
            break
        candidates = _neighbours(best_config, min_l, max_l, max_n, max_total, non_increasing)
        if not candidates:
            stop_reason = "max_total" if max_total is not None else "max_l"
            break

        if parallel and api._PARALLEL and len(candidates) > 1:
            jobs = [
                (
                    copy.deepcopy(molecules),
                    combined_sp,
                    c,
                    element,
                    algorithm,
                    opt_params,
                    params,
                    seed_exponent,
                    loss,
                    npass,
                )
                for c in candidates
            ]
            results = _dispatch(_evaluate_config, jobs, dispatch_cfg, ray_params)
        else:
            results = [score(c) for c in candidates]

        valid = [r for r in results if r is not None]
        if not valid:
            stop_reason = "failed"
            break
        cand_loss, cand_basis, cand_config = min(valid, key=lambda r: r[0])
        improvement = best_loss - cand_loss
        bo_logger.info(
            "config_search: level %d best candidate %s -> loss %.6e (improvement %.2e)",
            sum(best_config.values()) + 1,
            _config_str(cand_config),
            cand_loss,
            improvement,
        )
        if improvement <= min_improvement:
            stop_reason = "converged"
            break
        best_loss, best_basis, best_config = cand_loss, cand_basis, cand_config
        trace.append(_trace_entry(best_config, best_basis, molecules, params))

    bo_logger.info(
        "config_search: stopped (%s) at %s, loss %.6e",
        stop_reason,
        _config_str(best_config),
        best_loss,
    )
    return best_loss, stop_reason, best_basis, _config_str(best_config), trace
