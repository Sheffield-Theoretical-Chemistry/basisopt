# BasisOpt

BasisOpt is a python library for the optimization of molecular Gaussian basis sets as used in most quantum chemistry packages. This development version has been forked from the [original version](https://github.com/robashaw/basisopt). 

## Installation

The library is pip installable:

	pip install basisopt

You can alternatively create a fresh conda env or pyenv and in the top directory run

	 pip install -e .


## Contributing

Contributions are welcomed, either in the form of raising issues or pull requests on this repo. Please take a look at the Code of Conduct before interacting, which includes instructions for reporting any violations.

## Differences from Robert Shaw's repo

The following major changes have been made relative to Robert's original version (corresponding to release 1.0.0) of the repo.

- Support for a Molpro backend using pymolpro.
- Adds the ability to use/optimise well-tempered expansions of exponents.
- Fixes issue with multiplicities not setting correctly for ground state atoms.
- Added multiplcities upto Z=30
- Replaced DASK with Ray for distributed computing tasks
- Added the LegendreHybrid strategy for optimising basis sets with a legendre expansion
- Added a contraction strategy for contracting basis sets
- Added a native basis set converter
- Support for atom specific basis sets for Psi4
- 

## Documentation

For dependencies, detailed installation instructions, and a guide to getting started, please refer to the main documentation (currently Robert's original version) [here](https://basisopt.readthedocs.io/en/latest/index.html).

You can build the development version docs locally using sphinx from the ``doc`` directory:

	sphinx-build -b html src build

then open ``index.html`` in the resulting build directory.

## Examples

There are working examples in the examples folder, and these are (or will be) documented in the documentation. 

## Acknowledging usage

If you use this library in your program and find it helpful, that's great! Any feedback would be much appreciated. If you publish results using this library, please cite the following paper detailing version 1.0.0:

[J. Chem. Phys. 159, 044802 (2023)](https://doi.org/10.1063/5.0157878)
