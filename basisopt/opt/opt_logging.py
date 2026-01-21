import csv
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from basisopt import bo_logger


class BasisOptimizationLogger:
    """Logger for basis set optimization that efficiently tracks energies and exponents"""

    # Class-level storage for session timestamps
    _session_timestamps = {}

    def __init__(
        self,
        basis: dict,
        element: str,
        strategy_name: str,
        basis_type: str = "orbital",
        eval_type: str = "scf",
        log_dir: Optional[str] = None,
        flush_interval: int = 50,
        enabled: bool = True,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the logger

        Arguments:
            basis: basis dictionary for the element
            element: atomic symbol
            strategy_name: name of optimization strategy
            basis_type: "orbital", "jfit", or "jkfit"
            eval_type: type of evaluation (e.g., "scf")
            log_dir: directory for log files
            flush_interval: flush buffer to disk every N evaluations
            enabled: if False, logger does nothing (for easy enable/disable)
            session_id: unique identifier for this optimization session; if None, creates new timestamp
                       Use the same session_id across multiple logger instances to share files
        """
        self.enabled = enabled
        if not enabled:
            return

        self.basis = basis
        self.element = element
        self.strategy_name = strategy_name
        self.basis_type = basis_type
        self.eval_type = eval_type
        self.flush_interval = flush_interval

        # Setup base directory
        if log_dir is None:
            log_dir = "."
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Use session-based timestamp
        if session_id is None:
            session_id = f"{element}_{basis_type}_{eval_type}"

        if session_id not in self._session_timestamps:
            self._session_timestamps[session_id] = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.timestamp = self._session_timestamps[session_id]
        self.session_id = session_id

        # Track current composition and file
        self.current_composition = None
        self.current_npy_path = None
        self.current_csv_path = None
        self.column_names = None
        self.log_buffer = []
        self.total_eval_counter = 0
        self.file_eval_counter = 0  # Counter for current file

        bo_logger.info(f"Logger initialized for {element.capitalize()} (session: {session_id})")
        bo_logger.info(f"Flush interval: {flush_interval} evaluations")

    def _get_composition(self, basis: dict, element: str) -> str:
        """Get basis composition string like '6s4p2d'"""
        composition = []
        for shell in basis[element]:
            n_exps = len(shell.exps)
            composition.append(f"{n_exps}{shell.l}")
        return ''.join(composition)

    def _create_column_names(self, basis: dict, element: str) -> List[str]:
        """Create column names for current basis composition"""
        columns = ['eval_num', 'strategy', 'energy', 'dE_CBS']
        for shell in basis[element]:
            for i in range(len(shell.exps)):
                columns.append(f'{shell.l}{i+1}')
        return columns

    def _get_file_paths(self, composition: str):
        """Get file paths for a given composition"""
        base_name = (
            f"{self.element}_{self.basis_type}_{self.eval_type}_{composition}_{self.timestamp}"
        )
        npy_path = os.path.join(self.log_dir, f"{base_name}.npy")
        csv_path = os.path.join(self.log_dir, f"{base_name}.csv")
        return npy_path, csv_path

    def _initialize_composition(self, composition: str):
        """Initialize or resume logging for a composition"""
        self.current_npy_path, self.current_csv_path = self._get_file_paths(composition)

        # If file exists, load it to get the last eval_num
        if os.path.exists(self.current_npy_path):
            existing = np.load(self.current_npy_path, allow_pickle=True)
            self.file_eval_counter = int(existing[-1, 0])  # Last eval_num
            bo_logger.info(f"Resuming composition {composition} at eval {self.file_eval_counter}")
        else:
            self.file_eval_counter = 0
            bo_logger.info(f"New composition detected: {composition}")

        bo_logger.info(f"Logging to: {self.current_npy_path}")

    def log(self, energy: float, basis: dict, element: str, cbs_limit: Optional[float] = None):
        """Log a single evaluation

        Arguments:
            energy: computed energy value
            basis: basis dictionary
            element: atomic symbol
            cbs_limit: CBS limit for computing dE_CBS (if None, uses 0.0)
        """
        if not self.enabled:
            return

        # Check if basis composition has changed
        composition = self._get_composition(basis, element)

        if composition != self.current_composition:
            # Flush previous buffer if exists
            if self.current_composition is not None:
                self._flush_to_disk()
                self._export_to_csv()

            # Initialize or resume composition
            self.current_composition = composition
            self.column_names = self._create_column_names(basis, element)
            self._initialize_composition(composition)

        self.total_eval_counter += 1
        self.file_eval_counter += 1

        # Calculate dE_CBS
        dE_CBS = energy - cbs_limit if cbs_limit is not None else 0.0

        row = [self.file_eval_counter, self.strategy_name, energy, dE_CBS]

        # Append all exponents from all shells
        for shell in basis[element]:
            row.extend(shell.exps.tolist())

        self.log_buffer.append(row)

        # Periodic flush
        if len(self.log_buffer) >= self.flush_interval:
            self._flush_to_disk()

    def _flush_to_disk(self):
        """Write buffer to disk"""
        if not self.log_buffer or self.current_npy_path is None:
            return

        buffer_array = np.array(self.log_buffer, dtype=object)  # Use object dtype for mixed types

        # Append to existing file or create new one
        if os.path.exists(self.current_npy_path):
            existing = np.load(self.current_npy_path, allow_pickle=True)
            combined = np.vstack([existing, buffer_array])
            np.save(self.current_npy_path, combined)
        else:
            np.save(self.current_npy_path, buffer_array)

        self.log_buffer.clear()

    def _export_to_csv(self):
        """Export current npy file to CSV"""
        if self.current_npy_path is None or not os.path.exists(self.current_npy_path):
            return

        data = np.load(self.current_npy_path, allow_pickle=True)
        with open(self.current_csv_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(self.column_names)
            csv_writer.writerows(data.tolist())

        bo_logger.info(
            f"Composition {self.current_composition}: {self.file_eval_counter} evaluations"
        )
        bo_logger.info(f"  CSV: {self.current_csv_path}")

    def finalize(self):
        """Final flush and CSV conversion"""
        if not self.enabled:
            return

        # Flush remaining buffer and export final file
        self._flush_to_disk()
        self._export_to_csv()

        bo_logger.info(f"Total evaluations logged: {self.total_eval_counter}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures finalize is called"""
        self.finalize()
        return False  # Don't suppress exceptions
