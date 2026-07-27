"""Run state and step result types for the auto-basis pipeline.

A step is a callable ``(state, step_cfg) -> StepResult``. The driver resolves and
loads each step's input basis onto the state, runs the step, then persists the
result — steps never touch the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from basisopt.containers import InternalBasis
from basisopt.molecule import Molecule

from .config import PipelineConfig


@dataclass
class StepResult:
    """What a step returns: the output basis, a structured record, and optional
    extra exports (``{format_label: text}``) to write alongside the canonical basis."""

    basis: InternalBasis
    record: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, str] = field(default_factory=dict)


@dataclass
class RunState:
    """State handed to a step: the resolved config, the element, and the input
    basis the driver loaded for this step (``None`` for generative step 1)."""

    config: PipelineConfig
    element: str
    input_basis: Optional[InternalBasis] = None

    @property
    def reference(self):
        return self.config.reference

    def require_input(self, step_name: str) -> InternalBasis:
        """Return the input basis or raise a clear error if the step needs one."""
        if self.input_basis is None:
            raise ValueError(
                f"Step '{step_name}' needs an input basis but none was resolved. "
                f"Run an earlier step first, or set '{step_name}.input' to a file."
            )
        return self.input_basis

    def build_atom(
        self, method: str, multiplicity: Optional[int] = None, charge: Optional[int] = None
    ) -> Molecule:
        """Build the single-atom Molecule for atomic steps (primitives, reduction,
        contraction, purification, pruning). ``multiplicity``/``charge`` override the
        reference defaults for this stage -- e.g. the H atom is a doublet even when
        the molecular stage uses a singlet."""
        mol = Molecule(self.element)
        mol.add_atom(self.element, [0.0, 0.0, 0.0])
        mol.name = self.element
        mol.method = method
        charge = charge if charge is not None else self.reference.charge
        if charge:
            mol.charge = charge
        mult = multiplicity if multiplicity is not None else self.reference.multiplicity
        if mult is not None:
            mol.multiplicity = mult
        return mol

    def build_geometry_molecule(
        self, method: str, multiplicity: Optional[int] = None, charge: Optional[int] = None
    ) -> Molecule:
        """Build the (di)atomic Molecule for molecular steps (uncontraction) from the
        reference geometry. ``multiplicity``/``charge`` override the reference
        defaults for this stage -- set them per step when the atomic and molecular
        ground states differ (e.g. the H atom is a doublet but H2 is a singlet)."""
        if not self.reference.geometry:
            raise ValueError(
                f"Step needs reference.geometry for element {self.element}, none given"
            )
        mol = Molecule.from_xyz(self.reference.geometry)
        mol.name = self.element
        mol.method = method
        charge = charge if charge is not None else self.reference.charge
        if charge:
            mol.charge = charge
        mult = multiplicity if multiplicity is not None else self.reference.multiplicity
        if mult is not None:
            mol.multiplicity = mult
        return mol
