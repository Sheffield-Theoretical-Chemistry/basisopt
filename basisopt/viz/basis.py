# correlation consistent plots
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from basisopt.basis.basis import Basis
from basisopt.containers import InternalBasis, OptResult
from scipy.optimize import minimize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

def extract_steps(opt_results: OptResult, key: str = "fun"):
    """Get the given key value for each step
    in an opt_results dictionary
    """
    steps, values = [], []
    for k, d in opt_results.items():
        steps.append(int(k[9:]))
        values.append(d.get(key, 0.0))
    return steps, np.array(values)


Transform = Callable[[np.ndarray], np.ndarray]


def plot_objective(
    basis: Basis,
    figsize: tuple[float, float] = (9, 9),
    x_transform: Transform = lambda x: x,
    y_transform: Transform = lambda y: y,
) -> tuple[object, object]:
    """Create a matplotlib figure of the objective function value
    at each step of an optimization, separated by atom type if
    multiple atoms given.

    Arguments:
        basis (Basis): basis object with opt_results attribute
        figsize (tuple): (width, height) of figure in inches
        x_transform, y_transform (callable): functions that take
                a numpy array of values and return an array of the
                same size

    Returns:
        matplotlib (figure, axis) tuple
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Objective value")

    if hasattr(basis, "opt_results"):
        steps = {}
        values = {}
        results = basis.opt_results
        for k, v in results.items():
            if "atomicopt" in k:
                key = basis._symbol
                steps[key], values[key] = extract_steps(results, key="fun")
                break
            steps[k], values[k] = extract_steps(v, key="fun")

        for k, v in steps.items():
            ax.plot(x_transform(v), y_transform(values[k]), "x", ms=8, label=k)
        if (len(steps)) > 1:
            ax.legend()
    else:
        raise TypeError("Not a suitable Basis object")
    return fig, ax


def create_exponent_plot(
    basis_sets: list,
    element: str,
    log: bool = True,
    fit: bool = False,
    polynomial_order: int = 3,
    title: Optional[str] = None,
    max_l: Optional[int] = None,
    min_l: Optional[int] = 0,
    filepath: Optional[str] = None,
    basis_labels: Optional[list[str]] = None,
    basis_labels_heading: Optional[str] = "Basis Set",
):
    """
    Generates and optionally saves a plot of exponents for a given element across different basis sets.

    Parameters:
    - basis_sets (list): A list of basis sets to be plotted.
    - element (str): The chemical element for which the exponents are plotted.
    - log (bool, optional): If True, the y-axis will be logarithmic. Defaults to True.
    - fit (bool, optional): If True, fits a polynomial of order `polynomial_order` to the exponents. Defaults to False.
    - polynomial_order (int, optional): The order of the polynomial to fit if `fit` is True. Defaults to 3.
    - title (Optional[str], optional): The title of the plot. If None, no title is set. Defaults to None.
    - max_l (Optional[int], optional): The maximum angular momentum quantum number (l) to include. If None, includes all. Defaults to None.
    - min_l (Optional[int], optional): The minimum angular momentum quantum number (l) to include. Defaults to 0.
    - filepath (Optional[str], optional): The path to save the plot to. If None, the plot is not saved. Defaults to None.
    - basis_labels (Optional[list[str]], optional): Labels for the basis sets. If None, basis sets are not labeled. Defaults to None.
    - basis_labels_heading (Optional[str], optional): The heading for the basis set labels. Defaults to "Basis Set".

    Returns:
    None: The function generates and displays a plot. If `filepath` is provided, the plot is saved to the specified location.
    """
    element = element.lower()
    def set_ax_format(ax):

        ax.tick_params(axis="both", which="major", labelsize=20)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(4)

        ax.grid(True, axis="both", which="both")

        ax.tick_params(
            axis="x",
            which="both",
            direction="out",
            length=6,
            width=4,
            colors="black",
            grid_color="k",
            grid_alpha=0.1,
        )
        ax.tick_params(axis="x", which="minor", length=3, width=2, colors="gray")

        ax.set_xlim(left=0)

        ax.set_xticks([i for i in range(0, max_x + 2, 5)])
        ax.set_xticks(range(0, max_x + 2), minor=True)

        ax.set_ylim(bottom=min_y - 2, top=max_y + 2)
        ax.set_xlim(left=-0.8)

        ax.set_yticks(
            [
                i
                for i in range(
                    min_y - 2,
                    max_y + 2,
                    2,
                )
            ]
        )
        ax.set_yticks(range(min_y - 2, max_y + 2, 1), minor=True)

        ax.tick_params(
            axis="x",
            which="both",
            direction="out",
            length=12,
            width=4,
            colors="black",
            grid_color="k",
            grid_alpha=0.0,
        )

        ax.tick_params(axis="x", which="minor", length=6, width=4, colors="black")

        ax.tick_params(
            axis="y",
            which="both",
            direction="out",
            length=12,
            width=4,
            colors="black",
            grid_color="k",
            grid_alpha=0.2,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            length=6,
            width=4,
            colors="black",
            grid_alpha=0.2,
        )

    def polynomial(j: np.array, coefficients: list[float]):
        return sum([coefficients[k] * j**k for k in range(len(coefficients))])

    def fit_polynomial(exponents):

        atom_opt_s_exps = exponents

        # Define the objective function
        def objective(coefficients):
            return np.sum(
                (
                    polynomial(np.arange(0, len(atom_opt_s_exps)), coefficients)
                    - atom_opt_s_exps
                )
                ** 2
            )

        # Initial guess for the coefficients
        # initial_guess = [1, -1, 0.1, -0.1]
        initial_guess = [
            1 / 10**i if i % 2 == 0 else -1 / 10**i for i in range(polynomial_order)
        ]

        # Perform the optimization
        result = minimize(objective, initial_guess, method="Nelder-Mead")

        # Get the optimized coefficients
        optimized_coefficients = result.x
        return optimized_coefficients

    # Init plot
    if not isinstance(basis_sets, list):
        basis_sets = [basis_sets]

    if max_l is None:
        max_l = 0
        for b in basis_sets:
            if len(b[element]) > max_l:
                max_l = len(b[element])
                
    max_x = 0
    for b in basis_sets:
        for shell in b[element][min_l : max_l + 1]:
            if len(shell.exps) > max_x:
                max_x = len(shell.exps)
    max_y = 0
    min_y = 0
    for b in basis_sets:
        for shell in b[element][min_l : max_l + 1]:
            if log:
                if max(np.log(shell.exps)) > max_y:
                    max_y = int(max(np.log(shell.exps)))
                if min(np.log(shell.exps)) < min_y:
                    min_y = int(min(np.log(shell.exps)))

    # Setup the plot

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    fig, ax = plt.subplots(
        1, 1, figsize=(6, 8), sharex=True, sharey=True, constrained_layout=True, dpi=150
    )

    set_ax_format(ax)
    # Plotting the data

    l_string = "spdfghijkl"

    legend_elements = []
    basis_set_label_elements = []
    line_styles = ["-", "--", "-.", ":"]
    marker_styles = ["s", "^", "D", "P", "X", "v", "<", ">", "1", "2", "3", "4"]
    max_angular_momentum = 0

    def plot_single(shell, z, ax):
        if log:
            y = np.log(shell.exps)
        else:
            y = shell.exps
        ax.scatter(
            [i for i in range(0, len(shell.exps))],
            y,
            s=100,
            marker=marker_styles[b_id],
            edgecolor="black",
            linewidth=2,
            alpha=1,
            zorder=z,
            color=colors[l],
        )

        z += 1
        return z

    def plot_single_fit_line(shell, z, ax, fit):
        if len(shell.exps) > 1:
            fit_coefs = fit_polynomial(np.log(shell.exps))
            y = polynomial(np.arange(-0.5, len(shell.exps), 0.1), fit_coefs)
            ax.plot(
                np.arange(-0.5, len(shell.exps), 0.1),
                y,
                lw=2,
                ls=line_styles[b_id],
                zorder=z,
                color=colors[l],
            )
        z += 1
        return z

    z = 1
    if fit:
        for b_id, basis in enumerate(basis_sets):
            for l, shell in enumerate(basis[element]):
                if l < min_l:
                    continue
                else:
                    z = plot_single_fit_line(shell, z, ax=ax, fit=fit)
                    if l == max_l:
                        break
    for b_id, basis in enumerate(basis_sets):
        for l, shell in enumerate(basis[element]):
            if l < min_l:
                continue
            else:
                z = plot_single(shell, z, ax=ax)
                if l == max_l:
                    break
        if fit:
            basis_set_label_elements.append(
                Line2D(
                    [],
                    [],
                    color="k",
                    lw=2,
                    ls=line_styles[b_id],
                    label=basis_labels[b_id],
                    marker=marker_styles[b_id],
                )
            )
        else:
            basis_set_label_elements.append(
                Line2D(
                    [],
                    [],
                    color="k",
                    lw=0,
                    ls=line_styles[b_id],
                    label=basis_labels[b_id],
                    marker=marker_styles[b_id],
                )
            )

    if len(basis[element]) > max_angular_momentum:
        max_angular_momentum = len(basis[element])

    if max_l:
        for i in range(min_l, max_l + 1):
            legend_elements.append(Patch(color=colors[i], label=f"{l_string[i]}"))
    else:
        for i in range(min_l, max_angular_momentum):
            legend_elements.append(Patch(color=colors[i], label=f"{l_string[i]}"))
    #if fit:
    second_legend = ax.legend(
        handles=basis_set_label_elements,
        bbox_to_anchor=(1.0, 1.0),
        title=basis_labels_heading,
        frameon=True,
        framealpha=1,
        title_fontproperties={"weight": "bold"},
    )
    second_legend.get_frame().set_edgecolor("white")
    ax.add_artist(second_legend)

    legend = ax.legend(
        handles=legend_elements[:max_angular_momentum],
        bbox_to_anchor=(1.0, 0.964 - 0.03 * len(basis_labels)),
        title="Angular Momentum",
        frameon=True,
        framealpha=1,
        title_fontproperties={"weight": "bold"},
    )
    legend.get_frame().set_edgecolor("white")

    ax.set_xlabel("Exponent Index (j)", fontsize=20)
    ax.set_ylabel(r"ln$\alpha_j$", fontsize=20)
    if title:
        ax.set_title(title, fontsize=20)

    if filepath:
        fig.savefig(filepath)
        
    return fig, ax


def plot_exponents(
    basis: InternalBasis,
    atoms: list[str] = [],
    titles: list[str] = [],
    split_by_shell: bool = True,
    log_scale: bool = True,
    figsize: tuple[float, float] = (9, 9),
) -> tuple[object, list[object]]:
    """Creates event plots to visualize exponents in a basis set.

    Arguments:
            basis (dict): internal basis object
            atoms (list): list of atoms to plot for
            split_by_shell (bool): if True, the event plots will be
               split by shell, with a different plot for each atom
            log_scale (bool): if True, exponents will be in log_10
            figsize (tuple): (width, heigh) in inches of the figure

    Returns:
            matplotlib figure, [list of matplotlib axes]
    """
    natoms = len(atoms)
    if natoms > 1 and split_by_shell:
        fig, axes = plt.subplots(ncols=natoms, sharey=True)
        to_build = [{k: basis[k.lower()]} for k in atoms]
    else:
        fig, ax = plt.subplots()
        axes = [ax]
        to_build = [{k: basis[k.lower()] for k in atoms}]
    fig.set_size_inches(figsize)

    def _single_plot(bas, ax, title=None):
        flat_bases = []
        for k, v in bas.items():
            flat_basis = [s.exps for s in v]
            if log_scale:
                flat_basis = [np.log10(x) for x in flat_basis]
            if not split_by_shell:
                flat_basis = np.concatenate(flat_basis)
                flat_bases.append(flat_basis)
            else:
                flat_bases = flat_basis
        colors = [f"C{i}" for i in range(len(flat_bases))]
        ax.eventplot(flat_bases, orientation="vertical", linelengths=0.5, colors=colors)

        if split_by_shell:
            ax.set_xticks(list(range(len(flat_bases))))
            ax.set_xticklabels([s.l for v in bas.values() for s in v])
        else:
            ax.set_xticks(list(range(len(bas))))
            ax.set_xticklabels(list(bas.keys()))

        if log_scale:
            ax.set_ylabel(r"$\log_{10}$ (exponent)")
        else:
            ax.set_ylabel("Exponent")
        ax.set_title(title)
    if titles:
        for bas, ax, title in zip(to_build, axes, titles):
            _single_plot(bas, ax, title=title)
    else:
        for bas, ax in zip(to_build, axes):
            _single_plot(bas, ax)

    return fig, axes


def compare_exponents(
    basis_sets: list,
    atoms: list[str] = [],
    split_by_shell: bool = True,
    log_scale: bool = True,
    figsize: tuple[float, float] = (9, 9),
) -> tuple[object, list[object]]:
    """Creates event plots to visualize exponents in a basis set.

    Arguments:
            basis1 (dict): internal basis object
            basis2 (dict): internal basis object
            atoms (list): list of atoms to plot for
            split_by_shell (bool): if True, the event plots will be
               split by shell, with a different plot for each atom
            log_scale (bool): if True, exponents will be in log_10
            figsize (tuple): (width, heigh) in inches of the figure

    Returns:
            matplotlib figure, [list of matplotlib axes]
    """
    natoms = len(atoms)
