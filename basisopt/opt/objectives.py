import numpy as np


def rmse(molecules):
    """Root Mean Squared Error"""
    objective = np.sqrt(
        np.mean([(mol.get_result('energy') - mol.cbs_limit) ** 2 for mol in molecules])
    )
    return objective


def mae(molecules):
    """Mean Absolute Error"""
    objective = np.mean(np.abs([mol.get_result('energy') - mol.cbs_limit for mol in molecules]))
    return objective


def mape(molecules):
    """Mean Absolute Percentage Error"""
    objective = np.mean(
        np.abs([(mol.get_result('energy') - mol.cbs_limit) / mol.nelectrons() for mol in molecules])
    )
    return objective


def mean_per_mol(molecules):
    """Mean per molecule"""
    objective = np.mean(
        [(mol.get_result('energy') - mol.cbs_limit) / mol.nelectrons() for mol in molecules]
    )
    return objective


def default_opt_loss(molecules):
    """Default loss function, the norm of the difference between the result and the reference"""
    deltas = np.array([mol.get_result('energy') - mol.get_reference('energy') for mol in molecules])
    return np.linalg.norm(deltas)
