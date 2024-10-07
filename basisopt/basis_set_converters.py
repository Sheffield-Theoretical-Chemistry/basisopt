# Converter for changing basis sets into different formats

import numpy as np

from .containers import InternalBasis

class BasisShell:
    def __init__(self, l, exps, coefs):
        self.l = l
        self.exps = exps
        self.coefs = coefs


class BasisElement:
    def __init__(self, label):
        self.label = label
        self.shells = []

    def add_shell(self, shell):
        self.shells.append(shell)


class BasisSet:
    def __init__(self):
        self.elements = {}

    def add_element(self, element):
        self.elements[element.label] = element.shells


def internal_to_basis_set(basis: InternalBasis):
    basis_set = BasisSet()
    for element, shells in basis.items():
        b_element = BasisElement(element)
        for shell in shells:
            b_shell = BasisShell(shell.l, shell.exps, shell.coefs)
            b_element.add_shell(b_shell)
        basis_set.add_element(b_element)
    return basis_set


def convert_to_psi4(basis: BasisSet):
    basis_str = "spherical\n\n****\n"
    for label, shells in basis.elements.items():
        element_str = f"{label.upper()}\t0\n"
        for shell in shells:
            exps = shell.exps
            coefs = np.array(shell.coefs)
            for c in coefs:
                non_zero = c != 0
                contraction_exps = [format(exp, ".6E").replace('E', 'D') for exp in exps[non_zero]]
                contraction_coefs = [format(coef, ".6E").replace('E', 'D') for coef in c[non_zero]]
                contraction_title = f"{shell.l.upper()}\t{len(contraction_exps)}\t1.00\n"
                element_str += contraction_title
                for e, c in zip(contraction_exps, contraction_coefs):
                    element_str += f"\t{e}\t\t{c}\n"
        basis_str += element_str
        basis_str += "****\n"
    return basis_str


def convert_internal_to_basis_str(basis: InternalBasis, fmt: str) -> str:
    if fmt == "psi4":
        return convert_to_psi4(internal_to_basis_set(basis))
