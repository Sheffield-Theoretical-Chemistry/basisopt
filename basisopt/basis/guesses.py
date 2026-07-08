# ways of generating guesses for exponents
# NEEDS GREATLY EXPANDING
import basis_set_exchange as bse
import numpy as np

from basisopt import data
from basisopt.bse_wrapper import bse_to_internal, fetch_basis
from basisopt.containers import Shell

from .basis import (
    even_temper_expansion,
    fix_ratio,
    legendre_expansion,
    uncontract_shell,
    well_temper_expansion,
)

# All guess functions need this signature
# func(atomic, params=None), where atomic is an AtomicBasis object
# and params is a dictionary of parameters. atomic must have attribute
# atomic.config set.
# Return an array of Shell objects (i.e. an internal basis for a single atom)


def null_guess(atomic, params=None):
    """Default guess type for testing, returns empty array"""
    return []


def log_normal_guess(atomic, params=None):
    """Generates exponents randomly from a log-normal distribution

    Params:
         mean: centre of the log-normal distribution
         sigma: standard deviation of log-normal distribution
    """
    params = {'mean': 0.0, 'sigma': 1.0} if params is None else params
    config = atomic.config
    basis = []
    for k, v in config.items():
        shell = Shell()
        shell.l = k
        shell.exps = np.random.lognormal(mean=params['mean'], sigma=params['sigma'], size=v)
        shell.exps = fix_ratio(shell.exps)
        uncontract_shell(shell)
        basis.append(shell)
    return basis


_LEGENDRE_INITIAL_GUESS = ((3.5, 5.0, 0.8, 0.3, 0.1, 0.1), 6)


def legendre_guess(atomic, params=None):
    """Generates a Legendre-expansion guess for an atom.

    With no params (or unrecognised params) each angular momentum in the atom's
    minimal configuration gets a built-in initial guess. Recognised keys:

    - 'exponents': a list of primitive counts per shell, paired with the
      database Legendre coefficients;
    - 'name': take the number of primitives per shell from a named BSE basis;
    - 'initial_guess': an explicit list of ``(A_vals, n)`` tuples.
    """

    def _default_shells():
        l_list = [l for (n, l) in atomic.element.ec.conf.keys()]
        max_l = len(set(l_list))
        return [_LEGENDRE_INITIAL_GUESS] * max_l

    if not params:
        return legendre_expansion(_default_shells())
    if 'exponents' in params:
        a_vals = data.get_legendre_params(atom=atomic._symbol.title())
        shells = list(zip(a_vals, params['exponents']))
        return legendre_expansion(shells)
    if 'name' in params:
        ref_basis = fetch_basis(params['name'], [atomic._symbol])
        lengths = [len(shell.exps) for shell in ref_basis[atomic._symbol]]
        a_vals = data.get_legendre_params(atom=atomic._symbol.title())
        shells = [(tuple(a), n) for a, n in zip(a_vals, lengths)]
        return legendre_expansion(shells)
    if 'initial_guess' in params:
        return legendre_expansion(params['initial_guess'])
    return legendre_expansion(_default_shells())


def load_guess(atomic, params):
    """
    Loads a basis set from a file
    """
    basis = bse_to_internal(
        bse.read_formatted_basis_file(params['filepath'][0], basis_fmt=params['filepath'][1])
    )
    return basis[atomic._symbol]


def bse_guess(atomic, params={'name': 'cc-pvdz'}):
    """Takes guess from an existing basis on the BSE

    Params:
         name (str): name of desired basis set
    """
    basis = fetch_basis(params['name'], [atomic._symbol])
    return basis[atomic._symbol]


def even_tempered_guess(atomic, params=None):
    """Takes guess from an even-tempered expansion

    Params:
         see signature for AtomicBasis.set_even_tempered
    """
    params = {} if params is None else params
    if atomic.et_params is None:
        atomic.set_even_tempered(**params)
    return even_temper_expansion(atomic.et_params)


def well_tempered_guess(atomic, params=None):
    """Takes guess from a well-tempered expansion

    Params:
         see signature for AtomicBasis.set_well_tempered
    """
    params = {} if params is None else params
    if atomic.wt_params is None:
        atomic.set_well_tempered(**params)
    return well_temper_expansion(atomic.wt_params)
