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
# func(atomic, params={}), where atomic is an AtomicBasis object
# and params is a dictionary of parameters. atomic must have attribute
# atomic.config set.
# Return an array of Shell objects (i.e. an internal basis for a single atom)


def null_guess(atomic, params={}):
    """Default guess type for testing, returns empty array"""
    return []


def log_normal_guess(atomic, params={'mean': 0.0, 'sigma': 1.0}):
    """Generates exponents randomly from a log-normal distribution

    Params:
         mean: centre of the log-normal distribution
         sigma: standard deviation of log-normal distribution
    """
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


def legendre_guess(atomic, params):
    if not params:
        leg_params = data.get_legendre_params(atom=atomic._symbol.title())
        if leg_params:
            _INITIAL_GUESS = leg_params
            shells = leg_params
        else:
            _INITIAL_GUESS = ((3.5, 5.0, 0.8, 0.3, 0.1, 0.1), 6)
            l_list = [l for (n, l) in atomic.element.ec.conf.keys()]
            max_l = len(set(l_list))
            shells = [_INITIAL_GUESS] * max_l
        return legendre_expansion(shells)
    elif 'initial_guess' in params:
        return legendre_expansion(params['initial_guess'])
    elif 'name' in params.keys():
        l_list = [l for (n, l) in atomic.element.ec.conf.keys()]
        ref_basis = fetch_basis(params['name'], atomic._symbol)
        max_l = len(set(l_list))
        lengths = [len(shell.exps) for shell in ref_basis[atomic._symbol]]
        try:
            database_values = data.get_legendre_params(atom=atomic._symbol.title())
            for i, shell in enumerate(database_values):
                shell = list(shell)
                if len(shell[0]) >= lengths[i]:
                    shell[0] = tuple(list(shell[0])[: lengths[i]])
                shell[1] = lengths[i]
                database_values[i] = tuple(shell)

        except:
            _INITIAL_GUESS = ((3.5, 5.0, 0.8, 0.3, 0.1, 0.1), 6)
            shells = [_INITIAL_GUESS] * max_l
            for i, shell in enumerate(_INITIAL_GUESS):
                shell = list(shell)
                if len(shell[0]) >= lengths[i]:
                    shell[0] = tuple(list(shell[0])[: lengths[i]])
                shell[1] = lengths[i]
                _INITIAL_GUESS[i] = tuple(shell)
        return legendre_expansion(database_values)
    else:
        _INITIAL_GUESS = params
        shells = [_INITIAL_GUESS]
        return legendre_expansion(shells)


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


def even_tempered_guess(atomic, params):
    if not params:
        leg_params = data.get_legendre_params(atom=atomic._symbol.title())
        if leg_params:
            _INITIAL_GUESS = leg_params
            shells = leg_params
        else:
            _INITIAL_GUESS = ((3.5, 5.0, 0.8, 0.3, 0.1, 0.1), 6)
            l_list = [l for (n, l) in atomic.element.ec.conf.keys()]
            max_l = len(set(l_list))
            shells = [_INITIAL_GUESS] * max_l
        return legendre_expansion(shells)
    elif 'initial_guess' in params:
        return legendre_expansion(params['initial_guess'])
    elif 'name' in params.keys():
        l_list = [l for (n, l) in atomic.element.ec.conf.keys()]
        ref_basis = fetch_basis(params['name'], atomic._symbol)
        max_l = len(set(l_list))
        lengths = [len(shell.exps) for shell in ref_basis[atomic._symbol]]
        try:
            database_values = data.get_legendre_params(atom=atomic._symbol.title())
            for i, shell in enumerate(database_values):
                shell = list(shell)
                if len(shell[0]) >= lengths[i]:
                    shell[0] = tuple(list(shell[0])[: lengths[i]])
                shell[1] = lengths[i]
                database_values[i] = tuple(shell)

        except:
            _INITIAL_GUESS = ((3.5, 5.0, 0.8, 0.3, 0.1, 0.1), 6)
            shells = [_INITIAL_GUESS] * max_l
            for i, shell in enumerate(_INITIAL_GUESS):
                shell = list(shell)
                if len(shell[0]) >= lengths[i]:
                    shell[0] = tuple(list(shell[0])[: lengths[i]])
                shell[1] = lengths[i]
                _INITIAL_GUESS[i] = tuple(shell)
        return legendre_expansion(database_values)
    else:
        _INITIAL_GUESS = params
        shells = [_INITIAL_GUESS]
        return legendre_expansion(shells)


def even_tempered_guess(atomic, params={}):
    """Takes guess from an even-tempered expansion

    Params:
         see signature for AtomicBasis.set_even_tempered
    """
    if atomic.et_params is None:
        atomic.set_even_tempered(params)
    return even_temper_expansion(params)


def well_tempered_guess(atomic, params={}):
    """Takes guess from a well-tempered expansion

    Params:
         see signature for AtomicBasis.set_well_tempered
    """
    if atomic.wt_params is None:
        atomic.set_well_tempered(**params)
    return well_temper_expansion(atomic.wt_params)
