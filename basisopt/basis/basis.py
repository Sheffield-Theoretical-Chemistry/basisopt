import copy
import pickle
from typing import Any, Optional, Union

import numpy as np
from monty.json import MSONable
from scipy.special import legendre

from basisopt import data
from basisopt.containers import InternalBasis, Result, Shell
from basisopt.data import ETParams, LegParams, WTParams
from basisopt.testing import Test
from basisopt.util import bo_logger, dict_decode


def uncontract_shell(shell: Shell):
    """Converts a Shell into an uncontracted Shell
    (overwrites any existing contraction coefs)
    """
    shell.coefs = []
    n = shell.exps.size
    for ix in range(n):
        c = np.zeros(n)
        c[ix] = 1.0
        shell.coefs.append(c)


def uncontract(basis: InternalBasis, elements: Optional[list[str]] = None) -> InternalBasis:
    """Uncontracts all shells in a basis for the elements specified
    (does not overwrite the old basis).

    Arguments:
         basis (dict): the basis dictionary to be uncontracted
         elements (list): list of atomic symbols

    Returns:
         a new basis dictionary with uncontracted shells
    """
    if elements is None:
        elements = basis.keys()  # do all
    new_basis = copy.copy(basis)
    for el in elements:
        if el in new_basis:
            el_basis = new_basis[el]
            for s in el_basis:
                uncontract_shell(s)
    return new_basis


def contract_shell(shell: Shell, contractions: list[np.ndarray]):
    """Converts a Shell into a contracted Shell
    (overwrites any existing contraction coefs)
    """
    shell.coefs = contractions


def contract_basis(basis: InternalBasis, contractions: dict):
    """Contracts all shells in a basis for the elements specified
    (Overwrites the old basis).

    Arguments:
         basis (dict): the basis dictionary to be contracted
         contractions (dict): dictionary of contractions for each element
                            contractions should be a list of numpy arrays
                            where each array is a set of coefficients
    """
    elements = contractions.keys()
    for el in elements:
        for idx, shell in enumerate(basis[el.lower()]):
            contract_shell(shell, contractions[el][idx])


def contract_function(
    basis: InternalBasis, element: str, l: int, func: int, contractions: np.ndarray
):
    """Contracts a single function in a shell for the element specified
    (Overwrites the existing function contraction coefs)
    """
    shell = basis[element.lower()][l]
    shell.coefs[func] = contractions


def contract_element(basis: InternalBasis, element: str, l: int, contractions: list[np.ndarray]):
    """Contracts all shells in a basis for the elements specified
    (Overwrites the existing elements contraction coefs)
    """
    shell = basis[element.lower()][l]
    contract_shell(shell, contractions)


def even_temper_expansion(params: ETParams) -> list[Shell]:
    """Forms a basis for an element from even tempered expansion parameters

    Arguments:
         params (list): list of tuples corresponding to shells
         e.g. [(c_s, x_s, n_s), (c_p, x_p, n_p), ...] where each shell
         is expanded as c_l * (x_l**k) for k=0,...,n_l

    Returns:
         list of Shell objects for the expansion
    """
    el_basis = []
    for ix, (c, x, n) in enumerate(params):
        new_shell = Shell()
        new_shell.l = data.INV_AM_DICT[ix]
        new_shell.exps = np.array([c * (x**p) for p in range(n)])
        uncontract_shell(new_shell)
        el_basis.append(new_shell)
    return el_basis


def legendre_expansion(params: LegParams, l=0, contractions=None) -> list[Shell]:
    """Forms a basis for an element from Petersson's Legendre expansion

    Arguments:
        params (list): list of tuples corresponding to shells
        e.g. [(c_s, x_s, n_s), (c_p, x_p, n_p), ...] where each shell
        is expanded as ln c_l = Sum_{k=0}^{k_{max}} a_k P_k
        * (\frac{2c_l - 2}{n_l - 1} - 1)
        Where a_k is a parameter and P_k is the k'th Legendre polynomial.
        k_max is the total number of parameters (6).

    Returns:
        list of Shell objects for the expansion
    """
    el_basis = []
    for ix, (A_vals, n) in enumerate(params):
        new_shell = Shell()
        if l:
            new_shell.l = data.INV_AM_DICT[l]
        else:
            new_shell.l = data.INV_AM_DICT[ix]
        exponents = []

        for j in range(n):
            ln_a = 0e1
            for k in range(len(A_vals)):
                ln_a += A_vals[k] * legendre(k)((((2 * j) - 2) / (n)) - 1)
            exponents.append(np.exp(ln_a))
        new_shell.exps = np.array(exponents)
        new_shell.leg_params = (np.array(A_vals), n)
        if contractions:
            new_shell.coefs = contractions
        else:
            uncontract_shell(new_shell)
        el_basis.append(new_shell)
    return el_basis


def well_temper_expansion(params: WTParams) -> list[Shell]:
    """Forms a basis for an element from well tempered expansion parameters

    Arguments:
         params (list): list of tuples corresponding to shells
         e.g. [(c_s, x_s, g_s, d_s, n_s), (c_p, x_p, g_p, d_p, n_p), ...] where each shell
         is expanded as c_l * (x_l**k)*(1 + g_l*((k+1)/n_l)**d_l) for k=0,...,n_l

    Returns:
         list of Shell objects for the well tempered expansion
    """
    el_basis = []
    for ix, (c, x, g, d, n) in enumerate(params):
        new_shell = Shell()
        new_shell.l = data.INV_AM_DICT[ix]
        new_shell.exps = np.array([c * (x**p) * (1.0 + g * ((p + 1) / n) ** d) for p in range(n)])
        uncontract_shell(new_shell)
        el_basis.append(new_shell)
    return el_basis


def fix_ratio(exps: np.ndarray, ratio: float = 1.4) -> np.ndarray:
    """Returns a sorted numpy array of exponents
    where x_{i+1}/x_i >= ratio
    """
    new_exps = np.sort(exps)
    for i in range(exps.size - 1):
        if new_exps[i + 1] / new_exps[i] < ratio:
            new_exps[i + 1] = new_exps[i] * ratio
    return new_exps


class Basis(MSONable):
    """Abstract parent class object representing a basis type
    All basis types must inherit from here to work, see e.g. AtomicBasis, MolecularBasis

    Attributes:
         results: a Result object where any results (e.g. calculations, optimizations, ...)
         can be archived

    Private attributes:
         _tests (list): a list of Test objects that can be run and collated together, with results
         going into the results attribute
    """

    def __init__(self):
        self.results = Result()
        self.opt_results = None
        self._tests = []
        self._molecule = None

    def save(self, filename: str):
        """Pickles the Basis object into a binary file"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
            f.close()
        bo_logger.info("Dumped object of type %s to %s", type(self), filename)

    def load(self, filename: str) -> object:
        """Loads and returns a Basis object from a binary file pickle"""
        with open(filename, "rb") as f:
            pkl_data = pickle.load(f)
            f.close()
        bo_logger.info("Loaded object of type %s from %s", type(pkl_data), filename)
        return pkl_data

    def get_basis(self) -> InternalBasis:
        return self._molecule.basis

    def register_test(self, test: Test):
        """Add a Test object to the set of tests"""
        self._tests.append(test)

    def get_test(self, name: str) -> Union[Test, None]:
        """Retrieve a Test with a given name if it exists"""
        for t in self._tests:
            if t.name == name:
                return t
        return None

    def run_test(self, name: str, params: dict[str, Any] = {}):
        """Runs a test with the given name, printing result"""
        t = self.get_test(name)
        if t is None:
            bo_logger.warning("No test with name %s", name)
        else:
            t.result = t.calculate(self._molecule.method, self._molecule.basis, params=params)
            bo_logger.info("Test %s: %s", name, t.result)

    def run_all_tests(self, params: dict[str, Any] = {}):
        """Runs all the tests in basis, printing results"""
        for t in self._tests:
            t.result = t.calculate(self._molecule.method, self._molecule.basis, params=params)
            bo_logger.info("Test %s: %s", t.name, t.result)

    def optimize(self, algorithm: str = "Nelder-Mead", params: dict[str, Any] = {}) -> dict:
        """All basis objects should implement an optimize method with this signature"""
        raise NotImplementedError

    def copy(self) -> object:
        """Returns a deepcopy of self"""
        return copy.deepcopy(self)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "results": self.results.as_dict(),
            "opt_results": self.opt_results,
            "tests": [t.as_dict() for t in self._tests],
            "molecule": self._molecule.as_dict(),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        d = dict_decode(d)
        instance = cls()
        instance.results = d.get("results", Result())
        instance.opt_results = d.get("opt_results", None)
        instance._molecule = d.get("molecule", None)
        instance._tests = d.get("tests", [])
        return instance
