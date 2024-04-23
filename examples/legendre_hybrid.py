import basisopt as bo
from basisopt.basis.molecular import MolecularBasis
from basisopt.bse_wrapper import fetch_basis
from basisopt.opt.legendreHybrid import LegendrePairsHybrid

bo.set_backend("psi4")
bo.set_tmp_dir("./tmp/")

# %% Setup molecule

molecule = bo.Molecule(name='Oxygen')
molecule.add_atom('O')
molecule.method = 'hf'
molecule.basis = fetch_basis('cc-pvdz', ['o'])

# %% Initialise Strategy

strat_params = {'reference': 'rohf'}

strategy = LegendrePairsHybrid(max_n_a=6)
strategy.params = strat_params

# %% Run minimsation using defined opt_params

opt_params = {'options': {'xatol': 1e-6}}

bo.opt.minimizer(molecule, strategy=strategy, algorithm='Nelder-Mead', opt_params=opt_params)

# %% Collect energy and legendre parameters

wrapper = bo.get_backend()
optimised_energy = wrapper.get_value('energy')
leg_params = molecule.get_legendre_params()

# %% Save basis set as a file

f = open('./minimised_basis_set.txt', 'w')
f.write(bo.bse_wrapper.internal_basis_converter(molecule.basis, fmt='molpro'))
f.close()
