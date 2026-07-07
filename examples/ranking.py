import basisopt as bo

bo.set_backend("psi4")
bo.set_tmp_dir("tmp/")

from basisopt.basis.atomic import AtomicBasis
from basisopt.testing.rank import *

ne = AtomicBasis().load("tmp/neon-def.obj")
errors, ranks = rank_primitives(ne)
for e, r in zip(errors, ranks):
    print(e)
    print(r)
