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
This is the same as workflow_model, but instead of using OpenMM to add
solvent it uses this interface. This may be useful if OpenMM can't add
the solvent you want i.e. arbitrary molecules.
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

acac = """  H      1.9933     -0.0696      0.7114
  C      1.4621     -0.1360     -0.2471
  H      1.7972      0.7009     -0.8739
  H      1.7611     -1.0706     -0.7384
  C     -0.0193     -0.0845     -0.0386
  O     -0.8655     -0.9092     -0.3334
  O     -0.4776      1.0398      0.5632
H -1.4239 0.9943 0.6582
"""

# First, we want to make Psi-OMM Molecules
solute = molecule.Molecule.from_xyz_string(phenol)
s_solvent = molecule.Molecule.from_xyz_string(acac)

# Assign bonds, atom types, charges (using Psi4)
solute.generate_bonds()
solute.generate_atom_types()
solute.generate_charges()

# The s_ prefix should be read as single
s_solvent.generate_bonds()
s_solvent.generate_atom_types()
s_solvent.generate_charges()

# Using the s_solvent info, add 50 solvent molecules to the solute and create a new system
solv_z_vals, solv_xyz = po.find_solvent_pos(solute, s_solvent, num_solvent=50, stacked=True)

# Start to make the OpenMM objects
# Create an OpenMM topology
topology = po.make_topology(solute)

# Declare the force field to be used
forcefield = ForceField('gaff2.xml')

# Generate template for the force field
po.generateTemplate(forcefield, topology, solute)

# Add solvent (adds templates as well)
topology, forcefield = po.add_solvent_to_topology(topology, forcefield, s_solvent,
                                                  50, solv_name="AceticAcid")

# Create the OpenMM System
omm_sys = forcefield.createSystem(topology, nonbondedMethod=NoCutoff)

# Declare the integrator to be used
integrator = LangevinIntegrator(5*kelvin, 1/picosecond, 0.002*picoseconds)

# Add an external potential 
# Spherical potential; 0 if inside 5 nanometers; large if outside
external_force = CustomExternalForce('999*max(0, r-5.0)^2; r=sqrt(x*x+y*y+z*z)')
omm_sys.addForce(external_force)
for i in range(omm_sys.getNumParticles()):
    external_force.addParticle(i, ())

# Create the OpenMM Simulation
# If you have GPUs, you should change 'Reference' to the appropriate platform e.g. 'CUDA'
simulation = Simulation(topology, omm_sys, integrator, platform=Platform.getPlatformByName('Reference'))

# Create positions as desired by OpenMM; multiplication by .1 is to transform units from A to nm
coords = []
whole_xyz = np.vstack((solute.xyz, solv_xyz))
for i in range(len(whole_xyz)):
    coords.append(Vec3(whole_xyz[i][0]*.1, whole_xyz[i][1]*.1, whole_xyz[i][2]*.1))
positions = coords * nanometer

# Set the positions in the simulation so it knows where atoms are located
simulation.context.setPositions(positions)

# Equilibrate the geometry
simulation.step(5000)

# Get the updated geometry
new_z_vals, new_xyz = po.get_atom_positions(topology, simulation)

# Write the new geometry to an xyz file
po.write_traj('phenol-acac_opt_geo_mm.xyz', new_z_vals, new_xyz, comment="Equilibrated Geometry")

# Minimize the energy
simulation.minimizeEnergy(tolerance=2*kilojoule/mole)

# Get the updated geometry
new_z_vals, new_xyz = po.get_atom_positions(topology, simulation)

# Write the new geometry to an xyz file
po.write_traj('phenol-acac_opt_geo_mm.xyz', new_z_vals, new_xyz, comment="Optimized Geometry")

