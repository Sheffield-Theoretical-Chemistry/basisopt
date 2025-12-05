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
    """Mean Absolute Percentage Error"""
    objective = np.mean(
        np.abs([(mol.get_result('energy') - mol.cbs_limit) / mol.nelectrons() for mol in molecules])
    )
    return objective


@registered_objective_decorator()
def mean_per_mol(molecules):
    """Mean per molecule"""
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
