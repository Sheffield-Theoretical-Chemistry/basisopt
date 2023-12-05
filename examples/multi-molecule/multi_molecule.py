import basisopt as bo
from basisopt.basis import uncontract
from basisopt.basis.molecular import MolecularBasis
from basisopt.opt.strategies import Strategy
from basisopt.util import bo_logger


def test_attempt():
    mb = MolecularBasis(name="double")
    list_of_mols = ['water', 'methane', 'methanol', 'formaldehyde', 'oxygen']
    mol_objs = [bo.molecule.Molecule.from_xyz(mol + '.xyz', name=mol) for mol in list_of_mols]
    mb = MolecularBasis(name="double", molecules=mol_objs)
    
    params = {
        'scf_type': "pk",
    }
    
    strategy = Strategy()
    strategy.params = params
    strategy.guess_params = {'name': 'cc-pvdz'}
    mb.setup(method='hf', strategy=strategy, params=params, reference='cc-pvdz')
    def reduce_to_vdz_uncontract(basis):
        basis['h'] = basis['h'][0:1]
        basis['c'] = basis['c'][0:2]
        basis['o'] = basis['o'][0:2]

        basis = uncontract(basis, basis.keys())
        return basis
    basis = reduce_to_vdz_uncontract(mb.get_basis())
    
    mb.optimize(parallel=True)
    e_opt = []
    e_opt.append(strategy.last_objective)
    e_diff = e_opt[0]
    conv_crit = 1.0e-6
    counter = 0
    
    while e_diff > conv_crit:
        bo_logger.info("Starting consistency iteration %d", counter + 1)
        mb.optimize()
        e_opt.append(strategy.last_objective)
        e_diff = abs(strategy.last_objective - e_opt[counter])
        bo_logger.info("Objective function difference from previous iteration: %f\n", e_diff)
        counter += 1
    
    filename = "opt_basis_2.txt"
    bo_logger.info("Writing optimized basis to %s", filename)
    f = open(filename, "x")
    f.write(bo.bse_wrapper.internal_basis_converter(mb.get_basis(), fmt='molpro'))
    f.close()


if __name__=='__main__':
    bo.set_backend('psi4')
    bo.set_tmp_dir('./scr/')
    test_attempt()
