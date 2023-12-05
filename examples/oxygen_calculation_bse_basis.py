from basisopt.basis import uncontract
import basisopt as bo
import basis_set_exchange as bse
from basisopt.bse_wrapper import bse_to_internal


bo.set_backend('psi4') # Set backend to Psi4
bo.set_tmp_dir('./tmp/') # Set temporary directory
m = bo.Molecule(name="Oxygen",charge=0) # Define the molecule object, in this case named Oxygen
m.add_atom(element='O', coord=[0., 0., 0.]) # Add the oxygen atom to the Molecule Object

basis = bo.fetch_basis('cc-pvdz','O') # Fetch a cc-pvdz basis set for Oxygen from the Basis Set Exchange
basis['o'] = basis['o'][:2] # This reduces the VDZ basis set down to just the s and p orbitals
basis = uncontract(basis) # Uncontracts the basis set as basisopt only produces uncontract basis sets currently

m.basis = basis # Set the basis set for the molecule to the one from BSE.

m.method = 'hf' # Set the molecular method to Hartree-Fock

success = bo.run_calculation(evaluate='energy', mol=m, params = {'reference':'rohf'}) #Run calcuation using the molecule in m and the parameters passed to Psi4 that instruct it to use the reference functon for Restricted Open Hartree Fock
print(bo.get_backend().get_value('energy')) # Print the energy from the calculation.
