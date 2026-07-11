"""Tests for the optimization objective/loss functions (opt/objectives.py).

All objectives measure the distance to the CBS limit, ``E - E_CBS``. ``mean_bsie_per_electron``
and ``mean_per_mol`` normalise that per electron so the loss is comparable
across atoms/molecules of different size (the confirmed intended behaviour).
"""

import numpy as np

from basisopt.opt.objectives import mae, mean_bsie_per_electron, mean_per_mol, rmse
from tests.data.factories import make_molecule


def _mol(atoms, cbs_limit, energy):
    mol = make_molecule(atoms, method="linear", cbs_limit=cbs_limit)
    mol.add_result("energy", energy)
    return mol


def test_mean_bsie_per_electron_is_mean_abs_distance_to_cbs():
    # H: dE = 0.5 over 1 electron -> 0.5;  H2: dE = 0.4 over 2 electrons -> 0.2
    m1 = _mol(("H",), -1.5, -1.0)
    m2 = _mol(("H", "H"), -2.0, -1.6)
    assert abs(mean_bsie_per_electron([m1, m2]) - np.mean([0.5, 0.2])) < 1e-12


def test_bsie_normalisation_stops_large_systems_dominating():
    # same per-electron error, very different sizes -> equal contribution
    small = _mol(("H",), -1.0, -0.9)          # dE 0.1 / 1 electron = 0.1
    big = _mol(("H", "H"), -2.0, -1.8)        # dE 0.2 / 2 electrons = 0.1
    assert abs(mean_bsie_per_electron([small, big]) - 0.1) < 1e-12


def test_bsie_uses_abs_where_mean_per_mol_is_signed():
    # a system dipping just below its (numerical) CBS limit -> negative distance
    m1 = _mol(("H",), -1.0, -1.2)             # dE = -0.2 / 1
    m2 = _mol(("H", "H"), -2.0, -1.6)         # dE = +0.4 / 2 = +0.2
    assert abs(mean_bsie_per_electron([m1, m2]) - np.mean([0.2, 0.2])) < 1e-12          # abs -> 0.2
    assert abs(mean_per_mol([m1, m2]) - np.mean([-0.2, 0.2])) < 1e-12  # signed -> 0.0


def test_mae_and_rmse_use_raw_cbs_distance_not_per_electron():
    m1 = _mol(("H",), -1.5, -1.0)             # dE = 0.5
    m2 = _mol(("H", "H"), -2.0, -1.6)         # dE = 0.4
    assert abs(mae([m1, m2]) - np.mean([0.5, 0.4])) < 1e-12
    assert abs(rmse([m1, m2]) - np.sqrt(np.mean([0.25, 0.16]))) < 1e-12
