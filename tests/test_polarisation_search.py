"""Tests for the competing-configuration polarisation search (opt/polarisation_search.py).

Covers the pure neighbour-generation logic, the fixed-config evaluator, and the
end-to-end greedy best-first search over a config-dependent dummy backend.
"""

import numpy as np
import pytest

from basisopt import api
from basisopt.data import AM_DICT
from basisopt.opt import polarisation_search as ps
from basisopt.wrappers.dummy import DummyWrapper
from basisopt.wrappers.wrapper import available
from tests.data.factories import make_basis, make_molecule


# --------------------------------------------------------------------------- #
# Pure logic: neighbour generation + config rendering + seed exponents
# --------------------------------------------------------------------------- #
def test_neighbours_reproduce_science_todo_examples():
    """Single-function-addition neighbours at equal budget match the documented
    competing configurations (1d -> {2d, 1d1f}; 2d1f -> {3d1f, 2d2f, 2d1f1g})."""
    step1 = [ps._config_str(c) for c in ps._neighbours({2: 1}, 2, 4, None, None, True)]
    assert step1 == ["2d", "1d1f"]
    step2 = [ps._config_str(c) for c in ps._neighbours({2: 2, 3: 1}, 2, 4, None, None, True)]
    assert step2 == ["3d1f", "2d2f", "2d1f1g"]


def test_neighbours_honour_caps_and_shape_rules():
    # max_n caps a shell: from 2d with max_n=2 only the f seed survives
    assert [ps._config_str(c) for c in ps._neighbours({2: 2}, 2, 4, 2, None, True)] == ["2d1f"]
    # max_total already reached -> no neighbours
    assert ps._neighbours({2: 2}, 2, 4, None, 2, True) == []
    # non_increasing forbids n_f > n_d (would give 1d2f)
    cfgs = [ps._config_str(c) for c in ps._neighbours({2: 1, 3: 1}, 2, 4, None, None, True)]
    assert "1d2f" not in cfgs and "2d1f" in cfgs
    # a gap is never seeded (no f without d, no g without f)
    assert not ps._is_sensible({3: 1}, 2)
    assert not ps._is_sensible({2: 1, 4: 1}, 2)


def test_seed_exponents_geometric_and_single():
    assert ps._seed_exponents(0.5, 1).tolist() == [0.5]
    three = ps._seed_exponents(0.5, 3, ratio=3.0)
    assert len(three) == 3
    assert three[1] == pytest.approx(0.5)  # centred on the seed
    assert three[2] / three[1] == pytest.approx(3.0)  # geometric ratio


# --------------------------------------------------------------------------- #
# A config-dependent dummy backend so the search can actually choose configs
# --------------------------------------------------------------------------- #
class _ConfigDummy(DummyWrapper):
    """Energy lowers as polarisation functions are added, weighted so lower l and
    the first primitive of a shell matter most (diminishing returns within a
    shell). This makes different equal-budget configs score differently, so the
    greedy search has a well-defined path: 1d -> 1d1f -> 2d1f -> 3d1f ..."""

    _W = {2: 1.0, 3: 0.6, 4: 0.3}

    @available
    def energy(self, mol, tmp="", **params):
        self.initialise(mol, name="energy", tmp=tmp)
        s = 0.0
        for shells in mol.basis.values():
            for sh in shells:
                l = AM_DICT[sh.l]
                if l >= 2:
                    s += sum(self._W.get(l, 0.1) / k for k in range(1, len(sh.exps) + 1))
        return -float(mol.natoms()) - 0.01 * s


@pytest.fixture
def config_dummy_backend():
    previous = api.get_backend()
    api._CURRENT_BACKEND = _ConfigDummy()
    yield api._CURRENT_BACKEND
    api._CURRENT_BACKEND = previous


def _s2_setup():
    """A homonuclear S2 (no spectators) with an sp base to grow d/f/g onto."""
    combined = make_basis("s", config=(("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3))))
    mol = make_molecule(("S", "S"), method="linear", basis=combined, name="S2", cbs_limit=-1000.0)
    return [mol], combined


def test_fixed_config_builds_and_scores(config_dummy_backend):
    """FixedConfigPolarisation appends exactly the requested shells and its loss
    reflects the config (2d1f scores lower than 1d under the diminishing dummy)."""
    molecules, combined = _s2_setup()
    loss_1d, basis_1d, _ = ps._evaluate_config(
        molecules,
        combined,
        {2: 1},
        "s",
        "Nelder-Mead",
        {},
        {},
        1.0,
        "mean_per_valence_electron",
        1,
        None,
    )
    loss_2d1f, basis_2d1f, _ = ps._evaluate_config(
        molecules,
        combined,
        {2: 2, 3: 1},
        "s",
        "Nelder-Mead",
        {},
        {},
        1.0,
        "mean_per_valence_electron",
        1,
        None,
    )
    # 1d appended one d shell; 2d1f appended a d(2) + f(1)
    pol_1d = [sh.l for sh in basis_1d["s"] if AM_DICT[sh.l] >= 2]
    pol_2d1f = [(sh.l, len(sh.exps)) for sh in basis_2d1f["s"] if AM_DICT[sh.l] >= 2]
    assert pol_1d == ["d"]
    assert pol_2d1f == [("d", 2), ("f", 1)]
    assert loss_2d1f < loss_1d  # more/better functions -> lower BSIE -> lower loss
    # the sp base is untouched by the evaluator (works on a deep copy)
    assert [sh.l for sh in combined["s"]] == ["s", "p"]


def test_config_search_progression_and_max_total(config_dummy_backend):
    """The greedy best-first search follows the expected path and stops at max_total."""
    molecules, combined = _s2_setup()
    final_loss, stop_reason, best_basis, final_config, trace = ps.config_search_polarisation(
        molecules,
        combined,
        "s",
        "Nelder-Mead",
        {},
        {},
        min_l=2,
        max_l=4,
        seed_exponent=1.0,
        target=0.0,  # target unreachable -> caps stop it
        loss="mean_per_valence_electron",
        max_total=4,
    )
    assert stop_reason == "max_total"
    assert final_config == "3d1f"
    assert [t["config"] for t in trace] == ["1d", "1d1f", "2d1f", "3d1f"]
    # loss decreases monotonically as the search commits better configs
    losses = [t["loss_mean_per_valence_electron"] for t in trace]
    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1))
    # both normalisations are recorded, and valence > total (core dilution removed)
    entry = trace[-1]
    assert entry["loss_mean_per_valence_electron"] > entry["loss_mean_per_electron"]
    # the winning basis actually carries the grown shells
    assert [(sh.l, len(sh.exps)) for sh in best_basis["s"] if AM_DICT[sh.l] >= 2] == [
        ("d", 3),
        ("f", 1),
    ]


def test_config_search_target_stops_at_seed(config_dummy_backend):
    """When the seed already meets the target, the search stops immediately."""
    molecules, combined = _s2_setup()
    # seed 1d loss is ~ small positive; a large target is met at once
    _, stop_reason, _, final_config, trace = ps.config_search_polarisation(
        molecules,
        combined,
        "s",
        "Nelder-Mead",
        {},
        {},
        min_l=2,
        max_l=4,
        seed_exponent=1.0,
        target=1e9,
        loss="mean_per_valence_electron",
    )
    assert stop_reason == "target"
    assert final_config == "1d"
    assert len(trace) == 1
