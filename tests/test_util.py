import numpy as np
import pandas as pd

from basisopt import api
from basisopt.basis.basis import uncontract_shell
from basisopt.containers import Shell
from basisopt.data import get_even_temper_params, get_legendre_params
from basisopt.util import (
    _canonicalise_degenerate_naos,
    fit_poly,
    format_with_prefix,
    get_composition,
    natural_orbitals_from_density_block,
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


def test_natural_orbitals_orthonormal_reconstruct_and_ordered():
    # a non-orthogonal overlap and a symmetric density block
    S = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.25], [0.1, 0.25, 1.0]])
    D = np.array([[1.5, 0.2, 0.05], [0.2, 0.8, 0.1], [0.05, 0.1, 0.3]])

    occ, C = natural_orbitals_from_density_block(D, S)

    # occupations sorted descending
    assert np.all(np.diff(occ) <= 1e-12)
    # natural orbitals are orthonormal in the S metric
    assert np.allclose(C.T @ S @ C, np.eye(3), atol=1e-10)
    # the density is exactly rebuilt from its natural orbitals: D = C diag(occ) C^T
    assert np.allclose(C @ np.diag(occ) @ C.T, D, atol=1e-10)
    # trace(D S) (the electron count) is preserved as the sum of occupations
    assert abs(occ.sum() - np.trace(D @ S)) < 1e-10


def test_natural_orbitals_diagonal_case():
    # S = I, D = diag(occupations) -> natural orbitals are the axes, occ recovered
    D = np.diag([2.0, 1.0, 0.0])
    occ, C = natural_orbitals_from_density_block(D, np.eye(3))
    assert np.allclose(occ, [2.0, 1.0, 0.0])
    assert np.allclose(np.abs(C), np.eye(3), atol=1e-10)  # up to column sign


def test_canonicalise_degenerate_naos_unmixes_by_fock():
    # Two occupation-degenerate NAOs (occ 2, 2) handed in as an arbitrary 45-degree
    # mixture of the true core/valence axes; a third, distinct NAO (occ 0.5). The
    # Fock block marks axis 0 as core (lowest orbital energy, F=1) and axis 1 as
    # valence (F=5), so canonicalisation puts the core first.
    r = 1.0 / np.sqrt(2.0)
    coefficients = np.array([[r, r, 0.0], [r, -r, 0.0], [0.0, 0.0, 1.0]])
    occupations = np.array([2.0, 2.0, 0.5])
    fock = np.diag([1.0, 5.0, 3.0])

    occ, C = _canonicalise_degenerate_naos(occupations, coefficients, fock, 1e-3)

    # the mixture is resolved back to the clean axes, core (lowest energy) first
    assert np.allclose(np.abs(C[:, 0]), [1.0, 0.0, 0.0], atol=1e-10)
    assert np.allclose(np.abs(C[:, 1]), [0.0, 1.0, 0.0], atol=1e-10)
    assert np.allclose(C[:, 2], [0.0, 0.0, 1.0], atol=1e-10)
    assert np.allclose(occ, [2.0, 2.0, 0.5])
    # sign convention: largest-magnitude coefficient of each column is positive
    for k in range(C.shape[1]):
        assert C[np.argmax(np.abs(C[:, k])), k] > 0


def test_natural_orbitals_fock_orders_degenerate_by_energy_and_preserves_span():
    # A degenerate density block (occ 2, 2) with a distinct third orbital (occ 0.5).
    # Fock puts the lower orbital energy ("core") on axis 1, so canonicalisation must
    # reorder the degenerate pair to put axis 1 first -- while span, occupations and
    # the S-orthonormality all survive unchanged.
    D = np.diag([2.0, 2.0, 0.5])
    S = np.eye(3)
    F = np.diag([5.0, 1.0, 3.0])

    occ, C = natural_orbitals_from_density_block(D, S, F)

    assert np.allclose(occ, [2.0, 2.0, 0.5])
    assert np.allclose(np.abs(C[:, 0]), [0.0, 1.0, 0.0], atol=1e-10)  # core = lowest energy
    assert np.allclose(np.abs(C[:, 1]), [1.0, 0.0, 0.0], atol=1e-10)
    assert np.allclose(C.T @ S @ C, np.eye(3), atol=1e-10)  # still S-orthonormal
    assert np.allclose(C @ np.diag(occ) @ C.T, D, atol=1e-10)  # span/energy preserved


def test_natural_orbitals_resolver_noop_when_nondegenerate():
    # distinct occupations -> no degenerate group -> passing a resolver (Fock) block
    # must not change the result versus omitting it.
    S = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.25], [0.1, 0.25, 1.0]])
    D = np.array([[1.5, 0.2, 0.05], [0.2, 0.8, 0.1], [0.05, 0.1, 0.3]])
    F = np.diag([4.0, 2.0, 1.0])

    occ0, C0 = natural_orbitals_from_density_block(D, S)
    occ1, C1 = natural_orbitals_from_density_block(D, S, F)

    assert np.allclose(occ0, occ1)
    assert np.allclose(C0, C1)
