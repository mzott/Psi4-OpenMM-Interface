import sys
import operator
import time
import numpy as np
import psi4
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

from psiomm import molecule
from psiomm import psi_omm as po
"""
This is a workflow model for using Psi4 and OpenMM together. It is meant to
represent some of the useful features in the interface.
"""
   
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

# First, we want to make Psi-OMM Molecules
solute = molecule.Molecule.from_xyz_string(phenol)

# Assign bonds, atom types, charges (using Psi4)
solute.generate_bonds()
solute.generate_atom_types()
solute.generate_charges()

# Start to make the OpenMM objects
# Create an OpenMM topology
topology = po.make_topology(solute)

# Declare the force fields to be used
forcefield = ForceField('gaff2.xml', 'tip3p.xml')

# Generate template for the force field
po.generateTemplate(forcefield, topology, solute)

# Create positions as desired by OpenMM
coords = []
for i in range(len(solute.xyz)):
    coords.append(Vec3(solute.xyz[i][0], solute.xyz[i][1], solute.xyz[i][2]))
positions = coords*angstroms

# Add solvent
modeller = Modeller(topology, positions)
modeller.addSolvent(forcefield, numAdded=50)

# Create the OpenMM System
omm_sys = forcefield.createSystem(modeller.topology)

# Declare the integrator to be used
integrator = LangevinIntegrator(175*kelvin, 1/picosecond, 0.002*picoseconds)

# Create the OpenMM Simulation
# If you have GPUs, you should change 'Reference' to the appropriate platform e.g. 'CUDA'
simulation = Simulation(modeller.topology, omm_sys, integrator, platform=Platform.getPlatformByName('Reference'))

# Set the positions in the simulation so it knows where atoms are located
simulation.context.setPositions(modeller.positions)

# Equilibrate the geometry
simulation.step(500)

# Minimize the energy
simulation.minimizeEnergy(tolerance=2*kilojoule/mole)

# Get the updated geometry
new_z_vals, new_xyz = po.get_atom_positions(simulation.topology, simulation)

# Write the new geometry to an xyz file
po.write_traj('phenol-water_opt_geo_mm.xyz', new_z_vals, new_xyz, comment="Optimized Geometry")

# Calculate the B3LYP single point energy of this new geometry
new_mol = molecule.Molecule(new_z_vals, new_xyz)
# In between Psi4 calculations need to call clean to prevent issues
psi4.core.clean()
psi4.geometry(new_mol.to_xyz_string())
# Uncomment energy call if you want to actually calculate the energy; perhaps change adding 50 waters above to 5 to do the DFT
#psi4.energy('b3lyp-d3/sto-3g')
