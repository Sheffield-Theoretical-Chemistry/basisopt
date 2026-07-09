"""Run manifest for the auto-basis pipeline.

The manifest is a small JSON file in the run workdir recording which steps have
completed and where their artifacts live. It is what makes runs resumable and
step-selectable: a later invocation can run only a subset of steps and resolve
each step's ``input: auto`` to the most recent completed prior step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

MANIFEST_NAME = "manifest.json"


class Manifest:
    """Tracks completed steps + artifact paths for one run workdir."""

    def __init__(self, workdir: str | Path, element: str, steps: Optional[dict] = None):
        self.workdir = Path(workdir)
        self.element = element
        self.steps: dict[str, dict] = steps or {}

    @property
    def path(self) -> Path:
        return self.workdir / MANIFEST_NAME

    @classmethod
    def load_or_create(cls, workdir: str | Path, element: str) -> "Manifest":
        """Load an existing manifest from ``workdir`` or create a fresh one."""
        workdir = Path(workdir)
        path = workdir / MANIFEST_NAME
        if path.exists():
            with open(path) as handle:
                data = json.load(handle)
            return cls(workdir, data.get("element", element), data.get("steps", {}))
        return cls(workdir, element)

    def has(self, name: str) -> bool:
        """True if ``name`` has a recorded, completed artifact."""
        return name in self.steps

    def step(self, name: str) -> dict:
        return self.steps[name]

    def basis_path(self, name: str) -> Path:
        """Absolute path to a completed step's canonical basis file."""
        return self.workdir / self.steps[name]["basis"]

    def latest_completed_before(self, name: str, order: tuple[str, ...]) -> Optional[str]:
        """Name of the last step (in canonical ``order``) before ``name`` that has
        a completed artifact, or None if there is none."""
        if name not in order:
            return None
        prior = order[: order.index(name)]
        for candidate in reversed(prior):
            if self.has(candidate):
                return candidate
        return None

    def record_step(
        self,
        name: str,
        basis_relpath: str,
        record: dict[str, Any],
        backend: str,
        timestamp: str,
        exports: Optional[dict[str, str]] = None,
    ) -> None:
        """Register a completed step and persist the manifest."""
        self.steps[name] = {
            "basis": basis_relpath,
            "record": record,
            "backend": backend,
            "timestamp": timestamp,
            "exports": exports or {},
        }
        self.save()

    def save(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump({"element": self.element, "steps": self.steps}, handle, indent=2)
