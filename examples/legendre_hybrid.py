import basisopt as bo
from basisopt.basis.molecular import MolecularBasis
from basisopt.opt.legendreHybrid import LegendrePairsHybrid

bo.set_backend("psi4")

#%% Setup molecule

molecule = bo.Molecule(name='Oxygen')
molecule.add_atom('O')
molecule.method = 'hf'

strategy = LegendrePairsHybrid(max_n_a=[6,6,6])
bo.opt.minimizer(molecule, strategy = strategy, algorithm='Nelder-Mead', opt_params=opt_params)