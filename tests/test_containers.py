import pytest

import basisopt.containers as boc
from basisopt.exceptions import DataNotFound, InvalidResult
from tests.data import shells as shell_data
from tests.data.utils import almost_equal


def test_default_shell():
    new_shell = boc.Shell()
    assert new_shell.l == "s"
    assert len(new_shell.coefs) == 0
    assert new_shell.exps.size == 0


def test_shell_dict_roundtrip_preserves_leg_params():
    import numpy as np

    shell = boc.Shell()
    shell.l = "s"
    shell.exps = np.array([5.0, 1.0, 0.2])
    shell.coefs = [np.array([1.0, 0.0, 0.0])]
    shell.leg_params = (np.array([1.6, -5.1, 0.05]), 3)

    restored = boc.Shell.from_dict(shell.as_dict())
    assert restored.l == "s"
    assert restored.exps.size == 3
    # leg_params must survive the round-trip (previously dropped by from_dict)
    assert len(restored.leg_params) == 2
    assert almost_equal(np.sum(np.abs(np.asarray(restored.leg_params[0]) - shell.leg_params[0])), 0.0)
    assert restored.leg_params[1] == 3


_COMPUTE_POINTS = [
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
    (-1.0, 0.0, 0.5),
    (2.0, 0.2, -2.0),
    (5.0, 1.0, -0.5),
    (0.3, -0.7, 1.1),
]


def _s_closed(shell, x, y, z, i):
    """Closed-form unnormalised s (l=0) GTO: (sum c_k e^{-a_k r^2}) * Y_0^0."""
    import numpy as np

    r2 = x * x + y * y + z * z
    radial = np.sum(shell.coefs[i] * np.exp(-shell.exps * r2))
    return radial * (1.0 / (2.0 * np.sqrt(np.pi)))


def _p_closed(shell, x, y, z, m):
    """Closed-form unnormalised p (l=1) GTO for the m used by Shell.compute.

    radial = r * (sum c_k e^{-a_k r^2}); the real spherical harmonics give
    m=0 -> sqrt(3/4pi) cos(theta) (i.e. proportional to z), and
    m=+/-1 -> -/+ sqrt(3/8pi) sin(theta) cos(phi) (proportional to x).
    """
    import numpy as np

    r2 = x * x + y * y + z * z
    radial = np.sum(shell.coefs[0] * np.exp(-shell.exps * r2))  # r cancels the 1/r
    if m == 0:
        return radial * np.sqrt(3.0 / (4.0 * np.pi)) * z
    sign = -1.0 if m > 0 else 1.0
    return radial * sign * np.sqrt(3.0 / (8.0 * np.pi)) * x


def test_shell_compute_matches_closed_form():
    """Shell.compute must reproduce analytic spherical GTO values.

    Regression: the polar angle was built from the squared cylindrical radius
    with the arctan2 arguments swapped, and the azimuthal/polar angles were
    handed to sph_harm in the wrong order - so e.g. a p(m=0) orbital picked up
    x-dependence instead of z-dependence. These check against the closed forms.
    """
    s_shell, p_shell = shell_data.get_vdz_internal()["h"]

    for x, y, z in _COMPUTE_POINTS:
        for i in range(len(s_shell.coefs)):
            assert almost_equal(s_shell.compute(x, y, z, i=i), _s_closed(s_shell, x, y, z, i))
        for m in (-1, 0, 1):
            assert almost_equal(
                p_shell.compute(x, y, z, m=m), _p_closed(p_shell, x, y, z, m)
            )


def test_shell_compute_s_is_rotationally_invariant():
    """An s (l=0) GTO depends only on |r|, so points at equal radius match."""
    import numpy as np

    s_shell = shell_data.get_vdz_internal()["h"][0]
    r = 0.5
    on_axis = [s_shell.compute(*p) for p in ((r, 0, 0), (0, r, 0), (0, 0, r))]
    diagonal = s_shell.compute(*(r / np.sqrt(3),) * 3)
    for v in on_axis[1:] + [diagonal]:
        assert almost_equal(v, on_axis[0])


def test_shell_compute_pz_vanishes_in_z_plane():
    """A p(m=0) orbital (~ z) is identically zero anywhere in the z=0 plane."""
    p_shell = shell_data.get_vdz_internal()["h"][1]
    for x, y in ((0.5, 0.0), (0.0, 0.5), (1.0, 1.0), (-2.0, 3.0)):
        assert almost_equal(p_shell.compute(x, y, 0.0, m=0), 0.0)


def _d_solid_harmonic(m, x, y, z):
    """r^2 * Re(Y_2^m), the tabulated solid-harmonic polynomials (Condon-Shortley)."""
    import numpy as np

    r2 = x * x + y * y + z * z
    if m == 0:
        return 0.25 * np.sqrt(5.0 / np.pi) * (3.0 * z * z - r2)
    if abs(m) == 1:
        return (-1.0 if m > 0 else 1.0) * 0.5 * np.sqrt(15.0 / (2.0 * np.pi)) * x * z
    return 0.25 * np.sqrt(15.0 / (2.0 * np.pi)) * (x * x - y * y)  # |m| == 2


def test_shell_compute_d_matches_solid_harmonics():
    """l=2 must reproduce the tabulated Y_2^m shapes (3z^2-r^2, xz, x^2-y^2),
    checking angle handling and sph_harm argument order beyond the p case."""
    import numpy as np

    d_shell = boc.Shell()
    d_shell.l = "d"
    d_shell.exps = np.array([0.5])
    d_shell.coefs = [np.array([1.0])]

    for x, y, z in _COMPUTE_POINTS:
        radial = float(np.exp(-0.5 * (x * x + y * y + z * z)))  # r^2 absorbed into the harmonic
        for m in (-2, -1, 0, 1, 2):
            assert almost_equal(
                d_shell.compute(x, y, z, m=m), radial * _d_solid_harmonic(m, x, y, z)
            )


def test_basis_dict():
    hbas = shell_data.get_vdz_internal()
    d = boc.basis_to_dict(hbas)
    assert "h" in d
    b = boc.dict_to_basis(d)
    assert "h" in b
    for s, s_ in zip(hbas["h"], b["h"]):
        assert s.exps.size == s_.exps.size


def test_default_result():
    r = boc.Result()
    assert r.name == "Empty"
    assert r._depth == 1
    assert len(r._children) == 0


def test_add_get_data():
    r = boc.Result(name="Test")
    assert r.name == "Test"
    r.add_data("Is_Banana", True)
    assert r.get_data("Is_Banana")
    r.add_data("Is_Banana", False)
    assert not r.get_data("Is_Banana")
    assert r.get_data("Is_Banana", step_back=1)
    assert r.get_data("Is_Banana", step_back=4)

    with pytest.raises(DataNotFound):
        r.get_data("Is_Apple")


def build_frame():
    r1 = boc.Result()
    r1.add_data("Is_Banana", True)
    r1.add_data("Is_Banana", False)
    r2 = boc.Result(name="Child1")
    r2.add_data("Is_Banana", False)
    r2.add_data("Size", 10.1)
    r3 = boc.Result(name="Child2")
    r3.add_data("Surname", "Flump")
    r4 = boc.Result(name="Grandchild")
    r4.add_data("Size", 4.3)
    r4.add_data("Is_Banana", True)
    r1.add_child(r2)
    r1.add_child(r3)
    r2.add_child(r4)
    return r1, r2, r3, r4


def test_add_get_child():
    r1, r2, r3, r4 = build_frame()

    assert r1.depth == 1
    assert r2.depth == 2
    assert r3.depth == 2
    assert r4.depth == 3
    assert len(r1._children) == 2
    assert len(r2._children) == 1
    assert len(r3._children) == 0

    assert r1.get_child("Child1").depth == 2
    assert r1.get_child("Child2").depth == 2
    assert r2.get_child("Grandchild").depth == 3

    with pytest.raises(DataNotFound):
        r1.get_child("Grandchild")

    with pytest.raises(InvalidResult):
        shell = boc.Shell()
        r3.add_child(shell)


def test_result_summary():
    r1, r2, r3, r4 = build_frame()
    # statistics() must not raise, and summary() should include data + children
    stats = r1.statistics()
    assert "Is_Banana" in stats
    summary = r1.summary()
    # titles are upper-cased by _summary; data keys/values are not
    assert "CHILD1" in summary
    assert "GRANDCHILD" in summary
    assert "Surname" in summary
    assert "Flump" in summary


def test_search_result():
    r1, r2, r3, r4 = build_frame()

    results = r1.search("Is_Banana")
    assert len(results) == 4
    assert not results["Child1_Is_Banana1"]
    assert results["Grandchild_Is_Banana1"]

    results = r3.search("Is_Banana")
    assert len(results) == 0

    results = r2.search("Size")
    assert len(results) == 2
    assert 4.3 in results.values()

    results = r1.search("Surname")
    assert "Flump" in results.values()

    results = r2.search("Surname")
    assert "Flump" not in results.values()


def test_save_load_result_json(tmp_path):
    # round-trip a Result tree through the JSON (MSONable) save/load
    r1, r2, r3, r4 = build_frame()
    path = str(tmp_path / "result.json")
    r1.save(path)

    loaded = boc.Result().load(path)
    assert type(loaded).__name__ == "Result"
    # data survives (latest value and a step back)
    assert loaded.get_data("Is_Banana") == r1.get_data("Is_Banana")
    assert loaded.get_data("Is_Banana", step_back=1) == r1.get_data("Is_Banana", step_back=1)
    # children survive recursively
    assert len(loaded._children) == 2
    child1 = loaded.get_child("Child1")
    assert child1.get_data("Size") == 10.1
    grandchild = child1.get_child("Grandchild")
    assert grandchild.get_data("Is_Banana")
