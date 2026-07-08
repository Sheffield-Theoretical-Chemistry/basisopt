import numpy as np
import pandas as pd

from basisopt import api
from basisopt.basis.basis import uncontract_shell
from basisopt.containers import Shell
from basisopt.data import get_even_temper_params, get_legendre_params
from basisopt.util import (
    fit_poly,
    format_with_prefix,
    get_composition,
    rank_shell_contractions,
    read_json,
)
from basisopt.wrappers.dummy import DummyWrapper
from tests.data.factories import make_molecule


def test_read_json():
    obj = read_json("tests/data/neon.json")
    assert type(obj).__name__ == "AtomicBasis"
    assert "ne" in obj.get_basis()
    shell = obj.get_basis()["ne"][0]
    assert hasattr(shell, "exps")
    assert type(shell.exps).__name__ == "ndarray"


def test_fit_poly():
    data = pd.read_csv("tests/data/cl2.csv")
    _, xref, re, pt = fit_poly(data["R"], data["ECC"], n=6)
    assert abs(xref - 2.00749686) < 1e-8
    assert abs(re - 1.98792829) < 1e-8
    assert len(pt) == 7
    assert abs(pt[0] + 919.45844231) < 1e-8


def test_get_even_temper():
    # even_tempered_data is currently empty
    result = get_even_temper_params()
    assert len(result) == 0


def test_get_legendre_params_accepts_accuracy():
    # regression: set_legendre passes accuracy=, which used to be an unexpected kwarg
    result = get_legendre_params(atom="O", accuracy=1e-5)
    assert len(result) > 0
    assert get_legendre_params(atom="Zzz", accuracy=1e-5) == []


def _shell(l, n_exps, n_coefs):
    shell = Shell()
    shell.l = l
    shell.exps = np.arange(1.0, n_exps + 1.0)
    if n_coefs == n_exps:
        uncontract_shell(shell)
    else:
        shell.coefs = [np.ones(n_exps) for _ in range(n_coefs)]
    return shell


def test_format_with_prefix_scales_to_prefix():
    assert format_with_prefix(1500, "Hz") == "1.500 kHz"
    assert format_with_prefix(2.5e-3, "s") == "2.500 ms"


def test_format_with_prefix_zero_has_no_prefix():
    # regression: zero fell through every prefix test and returned None
    assert format_with_prefix(0, "Ha") == "0.000 Ha"
    # a magnitude below the smallest prefix also gets no prefix
    assert format_with_prefix(1e-30, "Ha") == "0.000 Ha"


def test_get_composition_uncontracted():
    # one coefficient per exponent -> plain primitive string
    basis = {"h": [_shell("s", 3, 3), _shell("p", 2, 2)]}
    assert get_composition(basis, "H") == "3s2p"


def test_get_composition_contracted():
    # fewer coefficient vectors than exponents -> arrow notation
    basis = {"h": [_shell("s", 4, 2), _shell("p", 2, 1)]}
    assert get_composition(basis, "H") == "(4s2p) -> [2s1p]"


def test_rank_shell_contractions_failed_calc_ranks_last(monkeypatch):
    api._CURRENT_BACKEND = DummyWrapper()
    shell = Shell()
    shell.l = "s"
    shell.exps = np.array([1.0, 2.0])
    shell.coefs = [np.array([0.5, 0.5])]
    mol = make_molecule(("H",), method="linear")

    state = {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return 0  # initial reference calculation succeeds
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "run_calculation", flaky)
    _, errors, _, _ = rank_shell_contractions(mol, shell, {})

    flat = [e for row in errors for e in row]
    # a failed removal must be inf (rank last), never 0.0 (which would rank it
    # as the best candidate to prune)
    assert flat and all(np.isinf(e) for e in flat)
