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

    def build_atom(self, method: str) -> Molecule:
        """Build the single-atom Molecule for atomic steps (primitives, reduction,
        purification, pruning), applying the config's charge/multiplicity."""
        mol = Molecule(self.element)
        mol.add_atom(self.element, [0.0, 0.0, 0.0])
        mol.name = self.element
        mol.method = method
        if self.reference.charge:
            mol.charge = self.reference.charge
        if self.reference.multiplicity is not None:
            mol.multiplicity = self.reference.multiplicity
        return mol

    def build_geometry_molecule(self, method: str) -> Molecule:
        """Build the (di)atomic Molecule for molecular steps (uncontraction) from
        the reference geometry."""
        if not self.reference.geometry:
            raise ValueError(
                f"Step needs reference.geometry for element {self.element}, none given"
            )
        mol = Molecule.from_xyz(self.reference.geometry)
        mol.name = self.element
        mol.method = method
        if self.reference.charge:
            mol.charge = self.reference.charge
        # NOTE: a single reference.multiplicity is applied to both the atom
        # (build_atom) and this (di)atomic; for species whose atomic and
        # molecular ground states differ (e.g. N vs N2) set it per run/step.
        if self.reference.multiplicity is not None:
            mol.multiplicity = self.reference.multiplicity
        return mol
