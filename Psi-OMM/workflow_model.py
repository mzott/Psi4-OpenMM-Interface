import sys
import operator
import time
import numpy as np
import psi4
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import simtk.openmm as mm

import molecule
import psi_omm as po
    
# Declare geometries
phenol = """ H      1.7344      0.1731      2.1387
  C      1.1780      0.0989      1.1986
  C      1.8626      0.0162     -0.0086
  H      2.9570      0.0254     -0.0180
  C      1.1586     -0.0783     -1.2063
  H      1.7026     -0.1432     -2.1544
  C     -0.2296     -0.0912     -1.2096
  H     -0.7855     -0.1655     -2.1508
  C     -0.9117     -0.0073      0.0125
  C     -0.2116      0.0881      1.2224
  H     -0.7442      0.1538      2.1778
  O     -2.2783     -0.0249     -0.0584
  H -2.6165 0.0372 0.8262
"""

wat_1 = """O    0.1747051   1.1050002  -0.7244430
H   -0.5650842   1.3134964  -1.2949455
H    0.9282185   1.0652990  -1.3134026"""


# First, we want to make Psi-OMM Molecules
solute = molecule.Molecule.from_xyz_string(phenol)
s_solvent = molecule.Molecule.from_xyz_string(wat_1)

# Assign bonds, atom types, charges (using Psi4)
solute.generate_bonds()
solute.generate_atom_types('temp_solu_ats')
solute.generate_charges()

# The s_ prefix should be read as single
s_solvent.generate_bonds()
s_solvent.generate_atom_types('temp_solv_ats')
s_solvent.generate_charges()

# Using the s_solvent info, add 50 solvent molecules to the solute and create a new system
solv_z_vals, solv_xyz = po.add_solvent(solute, s_solvent, num_solvent=50, stacked=True)

# Create an entire solute-solvent system with the standard the solute always goes before solvent
# Stack the solute and solvent z_vals and xyz
sys_z_vals = np.hstack((solute.z_vals, solv_z_vals))
sys_xyz = np.vstack((solute.xyz, solv_xyz))

# Use Numpy's tile function to copy the s_solvent atom_types and charges so they don't
# need to be recalculated. The tile method will not work for bonds; for bonds, use the method
# tile_bonds from psi_omm.
solv_atom_types = np.tile(s_solvent.atom_types, 50)
solv_charges = np.tile(s_solvent.charges, 50)
solv_bonds = po.tile_bonds(s_solvent.bonds, 50, 3)

# Stack the solute and solvent atom_types, charges, and bonds
sys_atom_types = np.hstack((solute.atom_types, solv_atom_types))
sys_charges = np.hstack((solute.charges, solv_charges))

# As with the tile_bonds, need to call a method to add solute and solvent bonds.
sys_bonds = po.add_bonds(solute.bonds, solv_bonds)

# Instantiate the new system as a new Psi-OMM Molecule.
sys = molecule.Molecule(sys_z_vals, sys_xyz)
sys.bonds = sys_bonds
sys.atom_types = sys_atom_types
sys.charges = sys_charges

# Start to make the OpenMM objects
# Create an OpenMM topology
topology = po.make_topology(sys)

# Declare the force field to be used
forcefield = ForceField('gaff2.xml')

# Create the OpenMM System
omm_sys = forcefield.createSystem(topology, nonbondedMethod=NoCutoff, atomTypes=sys.atom_types, atomCharges=sys.charges)

# Declare the integrator to be used
integrator = LangevinIntegrator(175*kelvin, 1/picosecond, 0.002*picoseconds)

# Add an external potential 
# Spherical potential; 0 if inside 1 nanometers; large if outside
external_force = CustomExternalForce('999*max(0, r-1.0)^2; r=sqrt(x*x+y*y+z*z)')
omm_sys.addForce(external_force)
for i in range(omm_sys.getNumParticles()):
    external_force.addParticle(i, ())

# Create the OpenMM Simulation
simulation = Simulation(topology, omm_sys, integrator, platform=Platform.getPlatformByName('Reference'))

# Create positions as desired by OpenMM; multiplication by .1 is to transform units from A to nm
coords = []
for i in range(len(sys.xyz)):
    coords.append(Vec3(sys.xyz[i][0]*.1, sys.xyz[i][1]*.1, sys.xyz[i][2]*.1))
positions = coords * nanometer

# Set the positions in the simulation so it knows where atoms are located
simulation.context.setPositions(positions)

# Equilibrate the geometry
simulation.step(500)

# Minimize the energy
simulation.minimizeEnergy(tolerance=2*kilojoule/mole)

# Get the updated geometry
new_z_vals, new_xyz = po.get_atom_positions(topology, simulation)

# Write the new geometry to an xyz file
po.write_traj('phenol-water_opt_geo_mm.xyz', new_z_vals, new_xyz, comment="Optimized Geometry")

