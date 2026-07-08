
.. _`sec:auto_basis`:

============================
Automatic basis optimization
============================

The *auto-basis* strategies grow (or reduce) an atomic basis until it reaches a
target accuracy relative to a complete-basis-set (CBS) reference, without you
having to fix the number of functions per shell in advance. This tutorial
optimizes an SCF basis for neon; a runnable version of the ideas here can be
adapted from ``examples/`` and the test suite.

CBS reference
-------------

As in the even-tempered tutorial, we use the numerical Hartree--Fock limit for
neon, which ships with the library:

.. code-block:: python

	from basisopt import data
	cbs_limit = data._ATOMIC_HF_ENERGIES[data.atomic_number("ne")]

Setting up the atom
-------------------

The auto-basis drivers operate on a :class:`~basisopt.molecule.Molecule` (here a
single atom) with a starting basis. ``AutoBasisFree`` grows that starting basis;
the reduce strategies shrink it.

.. code-block:: python

	import basisopt as bo
	from basisopt.molecule import Molecule
	from basisopt.bse_wrapper import fetch_basis

	bo.set_backend("psi4")
	bo.set_tmp_dir("tmp/")

	mol = Molecule(name="ne")
	mol.add_atom("Ne")            # multiplicity is set automatically
	mol.method = "hf"
	mol.basis = fetch_basis("cc-pvdz", ["Ne"])   # starting point

Growing a free basis
--------------------

``AutoBasisFree`` optimizes each shell, then adds exponents one at a time and
re-optimizes until ``|energy - cbs_limit|`` drops below ``target``. The CBS
limit is supplied with :meth:`set_cbs_limit`.

.. code-block:: python

	from basisopt.opt.auto_basis import AutoBasisFree
	from basisopt.opt.optimizers import atom_auto

	strategy = AutoBasisFree(target=1e-4)
	strategy.set_cbs_limit(cbs_limit)

	results = atom_auto(mol, element="Ne", strategy=strategy)
	optimized_basis = mol.basis   # the grown, optimized internal basis

``results`` is a dictionary of the SciPy optimization result for each step; each
entry also carries a ``dE_CBS`` field recording how far that step sat from the
CBS limit.

Legendre-parametrized growth
----------------------------

``AutoBasisLegendre`` is used the same way but represents each shell with a short
Legendre expansion, optimizing the expansion coefficients rather than the raw
exponents. Supply the per-shell primitive counts with ``n_coefs`` (and,
optionally, your own starting coefficients via the ``legendre_params``
attribute; otherwise a built-in guess is used where one is tabulated):

.. code-block:: python

	from basisopt.opt.auto_basis import AutoBasisLegendre

	strategy = AutoBasisLegendre(target=1e-4, n_coefs=(8, 4))
	strategy.set_cbs_limit(cbs_limit)
	results = atom_auto(mol, element="Ne", strategy=strategy)

Reducing an existing basis
--------------------------

To go the other way -- trim an over-complete basis down to the smallest set that
stays within ``target`` of the CBS limit -- use ``AutoBasisReduceStrategy`` with
``atom_auto_reduce``. It ranks the exponents by importance, removes the least
important, re-optimizes, and stops (reverting the last removal) once a removal
would push the energy too far from the limit.

.. code-block:: python

	from basisopt.opt.auto_basis import AutoBasisReduceStrategy
	from basisopt.opt.optimizers import atom_auto_reduce

	strategy = AutoBasisReduceStrategy(target=1e-4)
	strategy.set_cbs_limit(cbs_limit)
	results = atom_auto_reduce(mol, element="Ne", strategy=strategy)

Logging the run
---------------

Pass ``log_minimisation=True`` (and optionally ``log_dir``) to ``atom_auto`` /
``atom_auto_reduce`` to record every evaluation -- energy, distance to the CBS
limit, and the exponents -- to per-composition ``.csv``/``.npy`` files via
:class:`~basisopt.opt.opt_logging.BasisOptimizationLogger`.

Several molecules at once
-------------------------

For a strategy driven across a set of molecules rather than a single atom, load
them with :class:`~basisopt.basis.molecular.MoleculeLoader` and drive them with
the :class:`~basisopt.opt.optimizers.Optimizer` (or ``Minimizer``) class, which
share the same strategy objects shown above.
