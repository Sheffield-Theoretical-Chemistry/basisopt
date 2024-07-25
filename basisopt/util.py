# utility functions
import json
import logging
from typing import Any

import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable


bo_logger = logging.getLogger("basisopt")  # internal logging object


def read_json(filename: str) -> MSONable:
    """Reads an MSONable object from file

    Arguments:
         filename (str): path to JSON file

    Returns:
         object
    """
    with open(filename, "r", encoding="utf-8") as f:
        obj = json.load(f, cls=MontyDecoder)
    bo_logger.info("Read %s from %s", type(obj).__name__, filename)
    return obj


def write_json(filename: str, obj: MSONable):
    """Writes an MSONable object to file

    Arguments:
         filename (str): path to JSON file
         obj (MSONable): object to be written
    """
    obj_type = type(obj).__name__
    if isinstance(obj, MSONable):
        bo_logger.info(f"Writing {obj_type} to {filename}")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(obj, f, cls=MontyEncoder)
    else:
        bo_logger.error("%s cannot be converted to JSON format", obj_type)


def dict_decode(d: dict[str, Any]) -> dict[str, Any]:
    decoder = MontyDecoder()
    return {k: decoder.process_decoded(v) for k, v in d.items()}


def fit_poly(
    x: np.ndarray, y: np.ndarray, n: int = 6
) -> tuple[np.poly1d, float, float, list[float]]:
    """Fits a polynomial of order n to the set of (x [Bohr], y [Hartree]) coordinates given,
    and calculates data necessary for a Dunham analysis.

    Arguments:
         x (numpy array): atomic separations in Bohr
         y (numpy array): energies at each point in Hartree
         n (int): order of polynomial to fit

    Returns:
         poly1d object, reference separation (Bohr), equilibrium separation (Bohr),
         first (n+1) Taylor series coefficients at eq. sep.
    """
    # Find best guess at minimum and shift coordinates
    xref = x[np.argmin(y)]
    xshift = x - xref

    # Fit polynomial to shifted system
    z = np.polyfit(xshift, y, n)
    p = np.poly1d(z)

    # Find the true minimum by interpolation, if possible
    xmin = min(xshift) - 0.1
    xmax = max(xshift) + 0.1
    crit_points = [x.real for x in p.deriv().r if np.abs(x.imag) < 1e-8 and xmin < x.real < xmax]
    if len(crit_points) == 0:
        bo_logger.warning("Minimum not found in polynomial fit")
        # Set outputs to default values
        re = xref
        pt = [0.0] * (n + 1)
    else:
        dx = crit_points[0]
        re = xref + dx  # Equilibrium geometry
        # Calculate 0th - nth Taylor series coefficients at true minimum
        pt = [p.deriv(i)(dx) / np.math.factorial(i) for i in range(n + 1)]

    # Return fitted polynomial, x-shift, equilibrium bond length,
    # and Taylor series coefficients
    return p, xref, re, pt


def format_with_prefix(value: float, unit: str, dp: int = 3) -> str:
    """ Utility function for converting a float to scientific notation with units"""
    prefixes = [
        (1e24, 'Y'), (1e21, 'Z'), (1e18, 'E'), (1e15, 'P'), (1e12, 'T'),
        (1e9, 'G'), (1e6, 'M'), (1e3, 'k'), (1, ''),
        (1e-3, 'm'), (1e-6, 'µ'), (1e-9, 'n'), (1e-12, 'p'),
        (1e-15, 'f'), (1e-18, 'a'), (1e-21, 'z'), (1e-24, 'y')
    ]
    
    # Create the format string dynamically based on the number of decimal places
    format_string = f"{{:.{dp}f}}"

    for factor, prefix in prefixes:
        if abs(value) >= factor:
            formatted_value = value / factor
            return format_string.format(formatted_value) + f" {prefix}{unit}"
    
    # Handle very small numbers that do not fit any prefix
    return format_string.format(value) + f" {unit}"
