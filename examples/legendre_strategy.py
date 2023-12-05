import basisopt as bo

bo.set_backend('psi4')
bo.set_tmp_dir('tmp/')

from basisopt.basis.atomic import AtomicBasis
params = {'reference':'rohf','scf_type': "pk"}
ne = AtomicBasis('O')
ne.set_legendre(method='scf', accuracy=1e-4,target_ref='Partridge Uncontracted 3', strat_params=params, max_n=(18,13))
print(ne.leg_params)
for i, p in enumerate(ne.get_basis()['o']):
    print(f"l = {i}:", p.exps)
