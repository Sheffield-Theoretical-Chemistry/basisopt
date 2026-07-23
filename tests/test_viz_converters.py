"""Tests for visualization helpers and the internal->text basis converter."""

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import pytest

from basisopt.basis_set_converters import convert_internal_to_basis_str
from basisopt.viz.basis import create_exponent_plot, extract_steps
from tests.data.factories import make_basis


def test_extract_steps_parses_varied_keys():
    # keys vary across drivers: atomicopt1, opt1, element-namespaced, pass-prefixed
    results = {"h_opt1": {"fun": 1.0}, "h_opt2": {"fun": 0.5}, "pass0_opt3": {"fun": 0.2}}
    steps, values = extract_steps(results, key="fun")
    assert steps == [1, 2, 3]
    assert list(values) == [1.0, 0.5, 0.2]


def test_convert_internal_to_basis_str_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        convert_internal_to_basis_str(make_basis("h"), "gaussian94")


def test_create_exponent_plot_default_labels_do_not_crash():
    # regression: basis_labels defaulted to None but was indexed/measured
    basis = make_basis("c", (("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3))))
    fig, ax = create_exponent_plot([basis], "c")  # no basis_labels
    assert fig is not None
    # a linear plot must span the data rather than the fixed [-2, 2]
    fig2, ax2 = create_exponent_plot([basis], "c", log=False)
    bottom, top = ax2.get_ylim()
    assert top > 2  # exponents up to 5.0 are visible
    import matplotlib.pyplot as plt

    plt.close(fig)
    plt.close(fig2)
