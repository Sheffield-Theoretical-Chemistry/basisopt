"""Objective / loss functions for optimizing over a set of atoms/molecules.

All of these measure the **distance to the CBS limit**, ``E - E_CBS`` (the basis
set incompleteness error), for each system. The mean-per-electron variants
divide by the electron count so the loss is comparable across atoms and
molecules of different size - otherwise larger systems would dominate the loss
simply by having more electrons.
"""

import numpy as np


class ObjectiveRegistry:
    """Registry for objective functions."""

    _registry = {}

    @classmethod
    def register(cls, name, func):
        cls._registry[name] = func

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

    @classmethod
    def all_objectives(cls):
        return cls._registry.keys()


def registered_objective_decorator():
    """Decorator to register objective functions in a global registry."""

    def decorator(func):
        ObjectiveRegistry.register(func.__name__, func)
        return func

    return decorator


@registered_objective_decorator()
def rmse(molecules):
    """Root Mean Squared Error"""
    objective = np.sqrt(
        np.mean([(mol.get_result('energy') - mol.cbs_limit) ** 2 for mol in molecules])
    )
    return objective


@registered_objective_decorator()
def mae(molecules):
    """Mean Absolute Error"""
    objective = np.mean(np.abs([mol.get_result('energy') - mol.cbs_limit for mol in molecules]))
    return objective


@registered_objective_decorator()
def mape(molecules):
    """Mean absolute distance-to-CBS per electron.

    The absolute CBS distance ``|E - E_CBS|`` of each system, normalised by its
    electron count and averaged over the set. This normalisation makes the loss
    comparable across atoms/molecules of different size. (Named ``mape`` for
    historical reasons; it is a size-normalised CBS-distance loss, not a
    statistical mean-absolute-percentage-error against a reference.)
    """
    objective = np.mean(
        np.abs([(mol.get_result('energy') - mol.cbs_limit) / mol.nelectrons() for mol in molecules])
    )
    return objective


@registered_objective_decorator()
def mean_per_mol(molecules):
    """Mean (signed) distance-to-CBS per electron.

    As :func:`mape` but without the absolute value. For variational energies
    ``E - E_CBS >= 0``, so the two coincide; the signed form is used where the
    sign of the incompleteness error is meaningful (e.g. polarisation).
    """
    objective = np.mean(
        [(mol.get_result('energy') - mol.cbs_limit) / mol.nelectrons() for mol in molecules]
    )
    return objective


@registered_objective_decorator()
def default_opt_loss(molecules):
    """Default loss function, the norm of the difference between the result and the reference"""
    deltas = np.array([mol.get_result('energy') - mol.get_reference('energy') for mol in molecules])
    return np.linalg.norm(deltas)


@registered_objective_decorator()
def default_min_loss(molecules):
    """Default loss function, the mean of energy per electron"""
    deltas = np.mean([mol.get_result('energy') / mol.nelectrons() for mol in molecules])
    return deltas


@registered_objective_decorator()
def sp_polarisation_energy(molecules):
    objective = np.mean(
        [
            ((mol.get_result('energy') - mol.cbs_limit) - mol.sp_polarisation) / mol.nelectrons()
            for mol in molecules
        ]
    )
    return objective
