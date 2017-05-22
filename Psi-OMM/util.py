from __future__ import absolute_import
from __future__ import print_function
import sys
import operator
import time
import numpy as np
sys.path.append("/theoryfs2/ds/zott/Gits/Atom_typing")
sys.path.insert(1, "/theoryfs2/ds/zott/openmm/wrappers/python")
sys.path.insert(1, "/theoryfs2/ds/cdsgroup/psi4-compile/hrw-labfork/install-psi4/lib")
#sys.path.insert(1, "/theoryfs2/ds/zott/Gits/openmm/wrappers/python")
import psi4
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import simtk.openmm as mm
import GAFF_Typer as gt
import scipy.spatial

def calc_mm_E(xyz_arr, z_vals, atom_types, atom_charges):
    gt_sys = gt.Molecule(z_vals, xyz_arr)
    topology = gt.psi_omm.make_top(gt_sys.z_vals, gt_sys.bonds, atom_types)
    forcefield = ForceField('gaff2.xml')
    omm_sys = forcefield.createSystem(topology, nonbondedMethod=NoCutoff, atomTypes=atom_types, atomCharges=atom_charges)

    integrator = LangevinIntegrator(100*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(topology, omm_sys, integrator, platform=Platform.getPlatformByName('CUDA'))

    # Create positions as desired by OpenMM; multiplication by .1 is to transform units from A to nm
    coords = []
    for i in range(len(xyz_arr)):
        coords.append(Vec3(xyz_arr[i][0]*.1, xyz_arr[i][1]*.1, xyz_arr[i][2]*.1))
    positions = coords * nanometer
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True, getForces=True)
    forces = state.getForces(asNumpy=True)
    energy = state.getPotentialEnergy()
    print('MM ENERGY: ', energy/kilocalories_per_mole, 'KCAL/MOL ', energy)
    return energy/kilocalories_per_mole



def conformation_search(sys_geo, num_conformations, filename, num_solvent=None, solvent_geo=None): 
    log_str = str(time.asctime( time.localtime(time.time()) )) + '\n'

    # First, we want to assign atom types and charges to the solute
    psi_solute = psi4.geometry(sys_geo)
    psi4.set_global_option("BASIS", "sto-3g")
    psi_solute.update_geometry()
    gt_solute = gt.Molecule(*gt.psi_omm.psi_mol_to_omm(psi_solute))
    solute_atom_types = gt.atom_typer.atom_type_system(gt_solute.z_vals, gt_solute.xyz)
    #e_solute, wfn_solute = psi4.property("b3lyp", properties=["MULLIKEN_CHARGES"], return_wfn=True)
    #solute_charges = list(np.asarray(wfn_solute.atomic_point_charges()))
    # for 1-methylhydroxyurea
    solute_charges = [-0.27403 ,-0.29012 ,-0.57139 ,-0.15400, 0.30893,-0.27921, 0.27911 , 0.27599, 0.28528, 0.13994
      , 0.15394, 0.12557]*10


    print solute_atom_types

    log_str += filename +'\n\n'
    log_str += 'SOLUTE GEOMETRY\n\n'
    for z, xyz in zip(gt_solute.z_vals, gt_solute.xyz):
        log_str += str(gt.periodictable._temp_symbol[int(z)]) + ' ' + str(xyz[0]) + ' ' + str(xyz[1]) + ' ' + str(xyz[2]) +'\n'
    log_str += '\nSOLUTE ATOM TYPES\n\n'
    for at in solute_atom_types:
        log_str += str(at) + ' ' + str(solute_atom_types[at]) + '\n'
    log_str += '\nSOLUTE CHARGES\n'
    for c in range(len(solute_charges)):
        log_str += str(c) + ' ' + str(solute_charges[c]) + '\n'

    # Next, assign atom types and charges to the solvent if present
    if solvent_geo is not None:
        psi_solvent = psi4.geometry(solvent_geo)
        psi_solvent.update_geometry()
        gt_solvent = gt.Molecule(*gt.psi_omm.psi_mol_to_omm(psi_solvent))
        solvent_atom_types = gt.atom_typer.atom_type_system(gt_solvent.z_vals, gt_solvent.xyz)
        psi4.set_global_option("BASIS", "sto-3g")
        e_solvent, wfn_solvent = psi4.property("b3lyp", properties=["MULLIKEN_CHARGES"], return_wfn=True)
        solvent_charges = list(np.asarray(wfn_solvent.atomic_point_charges()))
        ###solvent_charges = [-0.3345, 0.3115, 0.0076, 0.0076, 0.0076]
        # old - from mulliken [-0.00834, 0.16596, -0.05252, -0.05254, -0.05254]

        if num_solvent is None:
            num_solvent = 5
        # Add desired number of solvent molecules around solute
        solvent_array = add_solvent(sys_geo, solvent_geo, num_solvent)

        log_str += '\nNUMBER OF SOLVENT: ' + str(num_solvent) +'\n'
        log_str += '\nSOLVENT GEOMETRY\n'
        for z, xyz in zip(gt_solvent.z_vals, gt_solvent.xyz):
            log_str += str(gt.periodictable._temp_symbol[int(z)]) + ' ' + str(xyz[0]) + ' ' + str(xyz[1]) + ' ' + str(xyz[2]) + '\n'
        log_str += '\nSOLVENT ATOM TYPES\n'
        for at in solvent_atom_types:
            log_str += str(at) + ' ' + str(solvent_atom_types[at]) + '\n'
        log_str += '\nSOLVENT CHARGES\n'
        for c in range(len(solvent_charges)):
            log_str += str(c) + ' ' + str(solvent_charges[c]) + '\n'
    else:
        solvent_atom_types = []
    # Create an OpenMM topology
    # First, we need to find the bonds; this is carried out by GAFF Typer
    if num_solvent is None:
        num_solvent = 0
    solvent_z_vals = []
    solvent_xyz = []
    
    if solvent_geo is not None:
        for atom in solvent_array:
            solvent_z_vals.append(atom[0])
            solvent_xyz.append(atom[1:])
    solvent_xyz = np.asarray(solvent_xyz)
    sys_z_vals = gt_solute.z_vals + solvent_z_vals
    
    if num_solvent > 0:
        sys_xyz = np.vstack((gt_solute.xyz, solvent_xyz))      
    else:
        sys_xyz = gt_solute.xyz
    
    gt_sys = gt.Molecule(sys_z_vals, sys_xyz)
    # Combine the atom type dictionaries for the solute and the solvent
    sys_atom_types = solute_atom_types
    for i in range( len(solute_charges) , len(solute_charges) + num_solvent * len(solvent_atom_types) ):
        sys_atom_types[i] = solvent_atom_types[(i - len(solute_charges)) % len(solvent_atom_types)]
    # Combine the charges for the solute and the solvent
    sys_charges = solute_charges
    for i in range(num_solvent):
        sys_charges += solvent_charges
    
    topology = gt.psi_omm.make_top(gt_sys.z_vals, gt_sys.bonds, sys_atom_types)
    
    # Make the OpenMM objects
    forcefield = ForceField('gaff2.xml')
    omm_sys = forcefield.createSystem(topology, nonbondedMethod=NoCutoff, atomTypes=sys_atom_types, atomCharges=sys_charges)

    # spherical potential; 0 if inside 1.8 nanometers
    external_force = CustomExternalForce('999*max(0, r-3.5)^2; r=sqrt(x*x+y*y+z*z)')
    omm_sys.addForce(external_force)
    for i in range(omm_sys.getNumParticles()):
        external_force.addParticle(i, ())

    integrator = LangevinIntegrator(175*kelvin, 1/picosecond, 0.002*picoseconds)
    #simulation = Simulation(topology, omm_sys, integrator, platform=Platform.getPlatformByName('CUDA'), platformProperties={'CudaPrecision':'double'})
    simulation = Simulation(topology, omm_sys, integrator, platform=Platform.getPlatformByName('CUDA'))

    # Create positions as desired by OpenMM; multiplication by .1 is to transform units from A to nm
    coords = []
    for i in range(len(sys_xyz)):
        coords.append(Vec3(sys_xyz[i][0]*.1, sys_xyz[i][1]*.1, sys_xyz[i][2]*.1))
    positions = coords * nanometer
    simulation.context.setPositions(positions)

    pocket_list = []
    ix = 0
    num_chcl3 = 5
    unique_geometries = []
    geo_dict = {}
    E_dict = {}
    #simulation.step(35000)
    for k in range(1000):
        integrator.setTemperature(50*kelvin)
        if k==10:
            external_force = CustomExternalForce('999*max(0, r-2.5)^2; r=sqrt(x*x+y*y+z*z)')
            omm_sys.addForce(external_force)
            for i in range(omm_sys.getNumParticles()):
                external_force.addParticle(i, ())
        integrator.setTemperature(250*kelvin)

        for i in range(1):
            t = time.time()
            gt.psi_omm.write_traj(filename+'.xyz', topology, simulation, gt_sys) 
            simulation.step(10000)
            gt.psi_omm.write_traj(filename+'.xyz', topology, simulation, gt_sys) 
            print "Time for step iteration " + str(ix) + ": ", time.time() - t
            





    #simulation.minimizeEnergy(tolerance=0.5*kilojoule/mole)
    
    f = open(filename+'.info', 'w')
    f.write(log_str)
    f.close()


def add_solvent(solute_geo, solvent_geo, num_solvent=None):
    """Add solvent around a solute. This method is very rudimentary and adds solvent
       around a rectangular box which surrounds the solute. Thus, for accurate structures
       a energy minimization must be performed after use of this method. The solvent is 
       added on a grid whose size is determined by the size of the solvent which is also
       modeled as a rectangle. 

       solute_geo : string
           String of z values and Cartesian coordinates
       solvent_geo : string
           String of z values and Cartesian coordinates
       num_solvent : int
           number of solvent molecules to add around the solute
       box_size : tuple
           tuple of box dimensions to fill with solvent
       return_array : boolean
           If True, returns the solute and solvent as a Nx4 array
           of z values and Cartesian coordinates
       returns : string by default
           String of z values and Cartesian coordinates of solute
           and solvent
    """
    t = time.time()
    solute_array = xyz_to_array(solute_geo)
    solvent_geo_array = xyz_to_array(solvent_geo)

    solute_xyz = np.array([x[1:]  for x in solute_array])
    solvent_xyz = np.array([x[1:]  for x in solvent_geo_array])

    # Center solute
    solute_xyz -= np.mean(solute_xyz, axis=0)
    solute_box = 2.0 * np.max(solute_xyz, axis=0) 
    solute_box += 2.0

    solvent_xyz -= np.mean(solvent_xyz, axis=0)
    solvent_box = 2.0 * np.max(solvent_xyz, axis=0) 
    solvent_box += 1.25


    sphere_dist = np.linalg.norm(solvent_box)
    
    solvent_placed = 0

    if num_solvent is None:
        num_solvent = 5

    output_xyz = solute_xyz.copy()
    output_z_vals = np.array([[x[0]] for x in solute_array])
    for nshell in range(1, 30):
        if solvent_placed >= num_solvent:
            break

        #print('Starting shell %d, waters %d' % (nshell, solvent_placed))
        batch = 500 * nshell ** 2
        phi_x = np.random.rand(batch) * np.pi
        theta_x = np.random.rand(batch) * 2.0 * np.pi

        random_xyz = np.vstack((nshell * sphere_dist * np.cos(theta_x) * np.sin(phi_x),
                                nshell * sphere_dist * np.sin(theta_x) * np.sin(phi_x),
                                nshell * sphere_dist * np.cos(phi_x))).T

        for b in range(batch):
            displacement = random_xyz[b]     

            distances = np.sum((output_xyz - displacement)**2, axis=1) ** 0.5
            if np.any(distances < sphere_dist):
#                print('Skipping distance!')
                continue


            output_xyz = np.vstack((output_xyz, solvent_xyz + displacement))
            output_z_vals = np.vstack((output_z_vals, np.array([[x[0]] for x in solvent_geo_array])))

            solvent_placed += 1            
            if solvent_placed >= num_solvent:
                break
        
    # Convert output_z_vals from symbols to integers
    for x in output_z_vals:
        x[0] = gt.periodictable.el2z[x[0].upper()]
    output_z_vals = output_z_vals.astype(float)
    output_system = np.hstack((output_z_vals, output_xyz))

    ret_solute = output_system[:len(solute_xyz)]
    ret_solvent = output_system[len(solute_xyz):]

    print "Time to add solvent: ", time.time()-t
    return ret_solvent

def xyz_to_array(xyz_string):
    array = []
    for line in xyz_string.splitlines():
        Zxyz_list = line.strip().split()
        if Zxyz_list == []:
            continue
        if len(Zxyz_list) != 4:
            raise KeyError("Line should have exactly 4 elements, Z, (x, y, z).")
        Zxyz_list[1:] = [float(Zxyz_list[x]) for x in range(1,4)]
        array.append(Zxyz_list)
    return array


def calc_dihedral(mol, indices):
    """
    Calculate the dihedral angle between four points.
    
    Parameters
    ----------
    mol : GAFF_Typer Molecule
    
    indices : list
        indices of the four atoms to calculate a dihedral angle from
    
    Returns
    -------
    float
    """
    if len(indices) != 4:
        raise Exception("Should be four points to calculate a dihedral angle.")

    n1 = gt.atom_type_helper.plane_perp_vec(mol, indices[0], indices[1], indices[2])
    n2 = gt.atom_type_helper.plane_perp_vec(mol, indices[1], indices[2], indices[3])
    
    dihedral = - n1.dot(n2) / np.linalg.norm(n1) / np.linalg.norm(n2)
    dihedral = np.arccos([dihedral])

    return dihedral

hydroxyurea = ''
with open('1-methylhydroxyurea_10.xyz', 'r') as myfile:
    temp_1mhu=myfile.read()
    #temp_1mhu = temp_1mhu.strip().split('\n')
    hydroxyurea = ''
    for x in temp_1mhu.strip().split('\n')[2:]:
        hydroxyurea += x + '\n'
print hydroxyurea

ethane = """  F      1.1851     -0.0039      0.9875
  C      0.7516     -0.0225     -0.0209
  H      1.1669      0.8330     -0.5693
  H      1.1155     -0.9329     -0.5145
  C     -0.7516      0.0225      0.0209
  F     -1.1669     -0.8334      0.5687
  H     -1.1157      0.9326      0.5151
  H     -1.1850      0.0044     -0.9875
"""

#conformation_search(bal_3a_inverted, 10, 'bal_3a_inverted_20_chloroform', num_solvent=20, solvent_geo=chloroform)
#conformation_search(bal_0a1, 10, 'dihedral_test.xyz', num_solvent=1000, solvent_geo=chloroform)
#conformation_search(ethane, 10, 'dihedral_test.xyz')
#conformation_search(bal_3a, 10, 'bal_3_0_chloroform_open_min50', num_solvent=500, solvent_geo=chloroform)
conformation_search(hydroxyurea, 10, 'hydroxyurea_10_water_1000', num_solvent=1000, solvent_geo=wat_1)
 
