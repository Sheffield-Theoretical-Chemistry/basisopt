from typing import Any, Callable, Optional, Union

import numpy as np

from basisopt import api
from basisopt.bse_wrapper import fetch_basis
from basisopt.containers import (
    InternalBasis,
    OptCollection,
    Result,
    basis_to_dict,
    dict_to_basis,
)
from basisopt.exceptions import DataNotFound, EmptyBasis
from basisopt.molecule import Molecule
from basisopt.opt import collective_minimize, collective_optimize, collective_polarize
from basisopt.opt.strategies import Strategy
from basisopt.util import bo_logger, write_json

from .atomic import AtomicBasis
from .basis import Basis


class MolecularBasis(Basis):
    """Object for preparation and optimization of a basis set for
    multiple atoms across one or more Molecules.

    Attributes:
         basis (dict): internal basis used for all molecules

    Private Attributes:
         _molecules (dict): dictionary of Molecule objects
         _atoms (set): unique atoms across all molecules
         _atomic_bases (dict): dictionary of AtomicBasis objects
             for each atom in _atoms
         _done_setup (bool): if True, setup has been called
    """

    def __init__(self, name: str = "Empty", molecules: list[Molecule] = None):
        super().__init__()
        self.name = name
        self.basis = {}
        self._molecules = {}
        self._atoms = set()
        self._atomic_bases = {}
        self._done_setup = False
        for m in molecules or []:
            self.add_molecule(m)

    def save(self, filename: str):
        """Saves the MolecularBasis to a JSON file (MSONable)"""
        write_json(filename, self)

    def as_dict(self) -> dict[str, Any]:
        """Returns as MSONable dictionary"""
        d = super().as_dict()
        d["@module"] = type(self).__module__
        d["@class"] = type(self).__name__
        d["name"] = self.name
        d["basis"] = basis_to_dict(self.basis)
        d["atoms"] = list(self._atoms)  # set is not JSON-serializable
        d["atomic_bases"] = {k: ab.as_dict() for k, ab in self._atomic_bases.items()}
        d["done_setup"] = self._done_setup
        d["molecules"] = {k: m.as_dict() for k, m in self._molecules.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> object:
        """Creates a MolecularBasis from an MSONable dictionary"""
        basis = Basis.from_dict(d)
        instance = cls(name=d.get("name", "Empty"))
        instance.results = basis.results
        instance.opt_results = basis.opt_results
        instance._tests = basis._tests
        instance.basis = dict_to_basis(d.get("basis", {}))
        instance._atoms = set(d.get("atoms", set()))
        instance._done_setup = d.get("done_setup", False)
        # decode nested objects (they are stored as as_dict() output); tolerate
        # either raw dicts or objects already decoded by the MSON decoder
        instance._atomic_bases = {
            k: v if isinstance(v, AtomicBasis) else AtomicBasis.from_dict(v)
            for k, v in d.get("atomic_bases", {}).items()
        }
        instance._molecules = {
            k: v if isinstance(v, Molecule) else Molecule.from_dict(v)
            for k, v in d.get("molecules", {}).items()
        }
        return instance

    def add_molecule(self, molecule: Molecule):
        """Adds a Molecule object to the optimization pool"""
        if molecule.name in self._molecules:
            bo_logger.warning("Molecule with name %s being overwritten", molecule.name)
        self._molecules[molecule.name] = molecule
        for atom in molecule.unique_atoms():
            self._atoms.add(atom.lower())

    def get_molecule(self, name: str) -> Molecule:
        """Returns a Molecule with the given name, if it exists, otherwise None"""
        if name in self._molecules:
            return self._molecules[name]
        bo_logger.warning("No molecule with name %s", name)
        return None

    def get_basis(self) -> InternalBasis:
        """Returns the basis set used for all molecules"""
        return self.basis

    def get_atomic_basis(self, atom: str) -> AtomicBasis:
        """Returns the AtomicBasis object for a given atom, if it exists,
        otherwise None
        """
        if atom in self._atomic_bases:
            return self._atomic_bases[atom]
        return None

    def unique_atoms(self) -> list[str]:
        """Returns list of unique atoms across all molecules"""
        return list(self._atoms)

    def molecules(self) -> list[Molecule]:
        """Returns a list of all the Molecule objects"""
        return list(self._molecules.values())

    def run_test(
        self,
        name: str,
        params: dict[str, Any] = None,
        reference_basis: Optional[Union[str, InternalBasis]] = None,
        do_print: bool = True,
    ) -> dict[str, Any]:
        """Runs a single test with a given name across all molecules

        Arguments:
             name (str): name of the test
             params (dict): parameters for backend
             reference_basis (str or dict): either string name for basis to fetch
                 from the BSE, or an internal basis dictionary, or None
             do_print (bool): if True, test results will be printed to Logger

        Returns:
             Dicionary of results for each test, indexed by molecule name
        """
        params = {} if params is None else params
        t = self.get_test(name)
        results = {}
        if t is None:
            bo_logger.warning("No test with name %s", name)
        else:
            try:
                child = self.results.get_child(name)
            except DataNotFound:
                new_result = Result(name=name)
                self.results.add_child(new_result)
                child = new_result

                # calculate reference values
                bo_logger.info("Calculating reference values for test %s", name)
                str_basis = isinstance(reference_basis, str)
                for m in self.molecules():
                    t.molecule = m
                    if str_basis:
                        t.calculate_reference(m.method, basis_name=reference_basis, params=params)
                    else:
                        t.calculate_reference(m.method, basis=reference_basis, params=params)
                    child.add_data(f"{m.name}_ref", t.reference)

            for m in self.molecules():
                t.result = t.calculate(m.method, self.basis, params=params)
                child.add_data(m.name, t.result)
                results[m.name] = t.result
                if do_print:
                    bo_logger.info("%s: %s", m.name, str(t.result))
        return results

    def run_all_tests(
        self,
        params: dict[str, Any] = None,
        reference_basis: Optional[Union[str, InternalBasis]] = None,
    ) -> None:
        """Runs all of the tests across all molecules, and prints the results to logger

        Arguments:
             params (dict): paramerters to pass to the backend
             reference_basis (str or dict): either string name for basis to fetch
                 from the BSE, or an internal basis dictionary, or None
        """
        params = {} if params is None else params
        results = {}
        for t in self._tests:
            results[t.name] = self.run_test(
                t.name, params=params, reference_basis=reference_basis, do_print=False
            )
        # print results
        header = "Molecule"
        for t in self._tests:
            header += f"\t{t.name}"
        bo_logger.info(header)
        for m in self.molecules():
            res_string = f"{m.name}"
            for v in results.values():
                res_string += f"\t{v[m.name]}"
            bo_logger.info(res_string)

    def setup(
        self,
        method: str = "ccsd(t)",
        quality: str = "dz",
        strategy: Strategy = None,
        reference: str = "cc-pvqz",
        params: dict[str, Any] = None,
    ):
        """Sets up the basis ready for optimization by creating AtomicBasis objects for each unique
        atom in the set, and calling setup for those - see the signature of AtomicBasis.setup for
        explanation.
        """
        params = {} if params is None else params
        strategy = Strategy() if strategy is None else strategy
        if len(self._atoms) == 0:
            raise EmptyBasis

        for m in self.molecules():
            m.method = method
        if not self.basis:
            self._atomic_bases = {}
            for atom in self._atoms:
                self._atomic_bases[atom] = AtomicBasis(atom)
                bo_logger.info("Doing setup for atom %s", atom)
                self._atomic_bases[atom].setup(
                    method=method,
                    quality=quality,
                    strategy=strategy,
                    reference=("dummy", 0.0),
                    params=params,
                )
            self.basis = {k: v.get_basis()[k] for k, v in self._atomic_bases.items()}
            if reference is not None:
                if api.which_backend() in ("Dummy", "Empty"):
                    bo_logger.warning("No computational backend set, can't compute reference value")
                else:
                    ref_basis = fetch_basis(reference, self.unique_atoms())
                    for m in self.molecules():
                        bo_logger.info(
                            "Calculating reference value for molecule %s using %s and %s/%s",
                            m.name,
                            api.which_backend(),
                            method,
                            reference,
                        )
                        m.basis = ref_basis
                        success = api.run_calculation(
                            evaluate=strategy.eval_type, mol=m, params=params
                        )
                        if success != 0:
                            bo_logger.warning("Reference calculation failed")
                            value = 0.0
                        else:
                            value = api.get_backend().get_value(strategy.eval_type)
                        m.add_reference(strategy.eval_type, value)
                        bo_logger.info("Reference value set to %f", value)
        else:
            for m in self.molecules():
                bo_logger.info(
                    "Adding basis for molecule %s using %s and %s/%s",
                    m.name,
                    api.which_backend(),
                    method,
                    reference,
                )
                m.basis = self.basis
                success = api.run_calculation(evaluate=strategy.eval_type, mol=m, params=params)
                if success != 0:
                    bo_logger.warning("Reference calculation failed")
                    value = 0.0
                else:
                    value = api.get_backend().get_value(strategy.eval_type)
                m.add_reference(strategy.eval_type, value)
                bo_logger.info("Reference value set to %f", value)
        self.strategy = strategy

        self._done_setup = True
        bo_logger.info("Molecular basis setup complete")

    def optimize(
        self,
        algorithm: str = "Nelder-Mead",
        params: dict[str, Any] = None,
        reg: Callable[[np.ndarray], float] = lambda x: 0,
        npass: int = 1,
        parallel: bool = False,
        ray_params: dict = None,
    ) -> OptCollection:
        """Calls collective optimize to optimize all the atomic basis sets in this basis

        Arguments:
             algorithm (str): name of scipy.optimize algorithm to use
             params (dict): parameters to pass to scipy.optimize
             reg (callable): regularization to use
             npass (int): number of optimization passes to do
             parallel (bool): if True, molecular calculations will be distributed in parallel

         Returns:
             dictionary of scipy.optimize result objects, indexed by atom
        """
        params = {} if params is None else params
        if self._done_setup:
            opt_data = [
                (k, algorithm, v.strategy, reg, params) for k, v in self._atomic_bases.items()
            ]
            self.opt_results = collective_optimize(
                self._molecules.values(),
                self.basis,
                opt_data=opt_data,
                npass=npass,
                parallel=parallel,
                ray_params=ray_params,
            )
        else:
            bo_logger.error("Please call setup first")
            self.opt_results = None
        return self.opt_results

    def minimization(
        self,
        algorithm: str = "Nelder-Mead",
        params: dict[str, Any] = None,
        reg: Callable[[np.ndarray], float] = lambda x: 0,
        npass: int = 1,
        parallel: bool = False,
        ray_params: dict = None,
    ) -> OptCollection:
        """Calls collective optimize to optimize all the atomic basis sets in this basis

        Arguments:
             algorithm (str): name of scipy.optimize algorithm to use
             params (dict): parameters to pass to scipy.optimize
             reg (callable): regularization to use
             npass (int): number of optimization passes to do
             parallel (bool): if True, molecular calculations will be distributed in parallel

         Returns:
             dictionary of scipy.optimize result objects, indexed by atom
        """
        params = {} if params is None else params
        if self._done_setup:
            opt_data = [(k, algorithm, self.strategy, reg, params) for k, _ in self.basis.items()]
            self.opt_results = collective_minimize(
                self._molecules.values(),
                self.basis,
                opt_data=opt_data,
                npass=npass,
                parallel=parallel,
                ray_params=ray_params,
            )
        else:
            bo_logger.error("Please call setup first")
            self.opt_results = None
        return self.opt_results

    def polarization(
        self,
        element: str,
        algorithm: str = "Nelder-Mead",
        params: dict[str, Any] = None,
        reg: Callable[[np.ndarray], float] = lambda x: 0,
        npass: int = 1,
        parallel: bool = False,
        ray_params: dict = None,
    ) -> OptCollection:
        """Calls collective optimize to optimize all the atomic basis sets in this basis

        Arguments:
             algorithm (str): name of scipy.optimize algorithm to use
             params (dict): parameters to pass to scipy.optimize
             reg (callable): regularization to use
             npass (int): number of optimization passes to do
             parallel (bool): if True, molecular calculations will be distributed in parallel

         Returns:
             dictionary of scipy.optimize result objects, indexed by atom
        """
        params = {} if params is None else params
        if self._done_setup:
            opt_data = [(element.lower(), algorithm, self.strategy, reg, params)]
            self.opt_results = collective_polarize(
                self._molecules.values(),
                self.basis,
                opt_data=opt_data,
                npass=npass,
                parallel=parallel,
                ray_params=ray_params,
            )
        else:
            bo_logger.error("Please call setup first")
            self.opt_results = None
        return self.opt_results


class MoleculeLoader:
    """A dataloader class load and store Molecule objects for use in the Minimizer and Optimizer classes"""

    def __init__(self, molecules: list[Molecule] = None):
        if molecules:
            self._molecules = {mol.name: mol for mol in molecules}
            self._atoms = set()
            for mol in molecules:
                for atom in mol.unique_atoms():
                    self._atoms.add(atom)
        else:
            self._molecules = {}
            self._atoms = set()
        self._loaded = False
        self.params = {}

    def set_basis(self, basis: InternalBasis):
        """Set the basis for all molecules in the loader"""
        self.basis = basis
        for mol in self._molecules.values():
            mol.basis = self.basis

    def _add_molecule(self, molecule: Molecule):
        """Add a molecule to the loader"""
        if molecule.name in self._molecules:
            bo_logger.warning(f"Molecule with name {molecule.name} already exists. Overwriting.")
        self._molecules[molecule.name] = molecule
        for atom in molecule.unique_atoms():
            self._atoms.add(atom)

    def unique_atoms(self) -> list[str]:
        """Returns a list of all the Molecule objects"""
        return list(self._atoms)

    def add_molecules_from_xyz(self, geoms: list[str], elements: list[str] = None, **kwargs):
        """
        Add multiple molecules to the loader from XYZ files with dynamic attributes.

        Args:
            geoms (list[str]): List of XYZ file paths.
            **kwargs: Additional attributes to be assigned to the molecules.
                      Keys are attribute names, and values are dictionaries
                      mapping molecule names to their corresponding values.
        """
        for xyz_file in geoms:
            # Create the molecule from the XYZ file
            mol = Molecule.from_xyz(xyz_file)
            mol.name = xyz_file.split('/')[-1].split('.')[0]
            if elements:
                if not set(elements).intersection(
                    set(mol.unique_atoms())
                ):  # Check if the molecule contains any of the filter atoms
                    # Extract the molecule name (assuming Molecule has a `name` attribute)
                    continue

            molecule_name = mol.name

            # Dynamically assign attributes from kwargs
            for attr_name, attr_values in kwargs.items():
                if molecule_name in attr_values:
                    setattr(mol, attr_name, attr_values[molecule_name])

            # Add the molecule to the loader
            self._add_molecule(mol)

    def add_molecule_from_xyz(self, xyz_file: str, **kwargs):
        """Add a molecule to the loader from an xyz file"""
        mol = Molecule.from_xyz(xyz_file)
        for key, value in kwargs.items():
            setattr(mol, key, value)
        self._add_molecule(mol)

    def set_method(self, method: str):
        """Set the method for all molecules in the loader"""
        for mol in self._molecules.values():
            mol.method = method

    def load_molecules(self, molecules: list[str]):
        """Load molecules into the loader from a list of Molecule objects"""
        if not isinstance(molecules, list):
            molecules = [molecules]
        for mol in molecules:
            self._add_molecule(mol)

    def __getitem__(self, index):
        """Get a molecule from the loader by index"""
        return list(self._molecules.values())[index]

    def __len__(self):
        """Get the number of molecules in the loader"""
        return len(self._molecules.values())

    def __iter__(self):
        """Iterate over the molecules in the loader"""
        return iter(self._molecules.values())

    def run_calculations(self, params: dict, objective=None, clean: bool = True):
        """
        Run calculations on all molecules in the loader.
        If an objective function is provided, return the objective value from all molecules.

        Arguments:
            params (dict): backend parameters.
            objective (callable, optional): if given, called with all molecules
                and its return value is returned.
            clean (bool): if True (default), clear backend scratch/state after
                each molecule (Psi4Wrapper.clean() clears psi4 timers/state; a
                safe no-op for other backends). Set False to preserve backend
                scratch files/state for inspection.
        """
        wrapper = api.get_backend()
        n = len(self)
        for ix, mol in enumerate(self._molecules.values()):
            try:
                bo_logger.info(f"Running calculation for molecule {mol.name} (#{ix} of {n})")
                api.run_calculation(mol=mol, params=params)
                mol.add_result('energy', wrapper.get_value('energy'))
            except Exception as e:
                bo_logger.error(f"Calculation failed for molecule {mol.name}: {e}")
            finally:
                if clean:
                    wrapper.clean()
        if objective:
            return objective(self._molecules.values())
