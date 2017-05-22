import sys
import operator
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
#from .physconst import *
#from .cov_radii import *
import GAFF_Typer as gt
import time
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

#sys_z = [6, 6, 6, 1, 6, 6, 6, 1, 6, 6, 6, 6, 6, 1, 6, 1, 6, 1, 6, 1, 6, 6, 6, 1, 6, 1, 6, 1, 6, 1, 6, 6, 6, 6, 1, 6, 1, 6, 1, 6, 6, 1, 1, 1, 7, 8, 8, 8, 8, 1, 1, 1, 6.0, 1.0, 17.0, 17.0, 17.0]
#sys_at = ['c2', 'c3', 'c3', 'hc', 'c', 'c', 'c3', 'hc', 'c3', 'ca', 'ca', 'ca', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha', 'ca', 'ca', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha', 'ca', 'ca', 'ca', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha', 'ca', 'c3', 'h1', 'h1', 'h1', 'n', 'o', 'o', 'o', 'os', 'hc', 'hc', 'ha', 'c3', 'h3', 'cl', 'cl', 'cl']
#sys_c = [0.19094, -0.08665, -0.08206, 0.09203, 0.23761, 0.24057, -0.08161, 0.092, -0.08657, -0.01207, -0.01321, -0.00985, -0.07706, 0.08575, -0.0788, 0.08478, -0.0801, 0.08242, -0.07908, 0.08107, -0.0055, -0.00292, -0.07782, 0.08071, -0.07926, 0.08227, -0.07843, 0.08416, -0.07603, 0.08487, -0.00756, 0.07281, -0.07373, -0.08976, 0.08151, -0.07418, 0.08179, -0.10939, 0.07875, 0.10471, -0.11578, 0.08987, 0.10249, 0.09258, -0.2431, -0.17662, -0.21333, -0.21276, -0.17899, 0.09159, 0.09136, 0.09561, -0.3345, 0.3115, 0.0076, 0.0076, 0.0076]

#e_dat = open('open_1_MM_E.dat', 'a')
#geo_file = open('bal_3_1_chloroform_open_sample2_unique_trunc.xyz', 'r')
#geos = []
#geo = []
#count = 0
#for line in geo_file.readlines():
#    num_atoms = 57
#    Zxyz_list = line.strip().split()
#    if len(Zxyz_list) != 4: continue
#    geo.append([float(Zxyz_list[x]) for x in range(1,4)]) 
#    count += 1
#    if count == 57:
#        geos.append(geo)
#        count = 0
#        geo = []
#
#
#for geo in geos:
#    gtt = gt.Molecule(sys_z, geo, no_properties=True)
#    e_dat.write(str(calc_mm_E(gtt.xyz, gtt.z_vals, sys_at, sys_c))+'\n')


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


    
    #### for 3
    ###solute_charges = [ 0.19094, -0.08665, -0.08206, 0.09203, 0.23761, 0.24057,-0.08161, 0.09200,
    ###                  -0.08657,-0.01207,-0.01321,-0.00985,-0.07706, 0.08575,-0.07880, 0.08478,
    ###                  -0.08010, 0.08242,-0.07908, 0.08107,-0.00550,-0.00292,-0.07782, 0.08071,
    ###                  -0.07926, 0.08227,-0.07843, 0.08416,-0.07603, 0.08487,-0.00756, 0.07281,
    ###                  -0.07373,-0.08976, 0.08151,-0.07418, 0.08179,-0.10939, 0.07875, 0.10471,
    ###                  -0.11578, 0.08987, 0.10249, 0.09258,-0.24310,-0.17662,-0.21333,-0.21276,
    ###                  -0.17899, 0.09159, 0.09136, 0.09561]
    


    #print solute_atom_types    
    #### for balance 3
    ###if solute_atom_types[44] == 'na':
    ###    solute_atom_types[44] = 'n'

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

    #### for balance 3
    ###temp_xyz = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    ###temp_xyz /= angstrom
    ###temp_mol = gt.Molecule(sys_z_vals, temp_xyz, no_properties=True)
    ###init_dihedral = calc_dihedral(temp_mol, [44,31,39,48])
    
    #gt.psi_omm.write_traj(filename+'.xyz', topology, simulation, gt_sys) 
    # equilibrate
    #simulation.step(50000)
    #gt.psi_omm.write_traj(filename, topology, simulation, gt_sys) 

    ###dif_list = []
    ###dihedral_list = []
    ###dihedral_list.append(init_dihedral)
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
            #for j in range(10):
            #    integrator.setTemperature((200-20*j)*kelvin) 
            #    simulation.step(400)
            #integrator.setTemperature((375-10*i)*kelvin) 
            #if i>10
            #    integrator.setTemperature((175+10*i)*kelvin)
            gt.psi_omm.write_traj(filename+'.xyz', topology, simulation, gt_sys) 
            simulation.step(10000)
            gt.psi_omm.write_traj(filename+'.xyz', topology, simulation, gt_sys) 
            #gt.psi_omm.write_psi_in(filename+'_'+str(ix)+'.in', topology, simulation, gt_sys) 
            print "Time for step iteration " + str(ix) + ": ", time.time() - t
            
            ###temp_xyz = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
            ###temp_xyz /= nanometer
            ###temp_xyz *= 10
            ###temp_mol = gt.Molecule(sys_z_vals, temp_xyz, no_properties=True)
            #### for balance 3
            ###new_dihedral = calc_dihedral(temp_mol, [44,31,39,48])
            ###dihedral = new_dihedral-init_dihedral
            ###dihedral %= 2 * np.pi    
            ###if (dihedral > (.5 * np.pi)) and (dihedral < (1.5 * np.pi)):
            ###    dif_list.append(ix)
            ###dihedral_list.append(dihedral)
       
            #    see if chloroform in pocket
            #rrpos = np.asarray([temp_mol.xyz[22], temp_mol.xyz[26], temp_mol.xyz[30]])
            #rrpos = np.mean(rrpos, axis=0) 
            #lrpos = np.asarray([temp_mol.xyz[14], temp_mol.xyz[11], temp_mol.xyz[18]]) 
            #lrpos = np.mean(lrpos, axis=0) 
            ###mrpos = np.asarray([temp_mol.xyz[9], temp_mol.xyz[11], temp_mol.xyz[21]]) 
            ###mrpos = np.mean(mrpos, axis=0) 

            ###indices = range(52)
            ###C_indices = [idx for idx in range(52, len(temp_mol.z_vals)) if temp_mol.z_vals[idx] == 6]
            ###flag = True
            ###num_added_chcl3 = 0
            ###added_chcl3 = False
            ###if num_chcl3 == 0:
            ###    flag = False
            ###    added_chcl3 = True
            ###for idx in C_indices:
            ###    if flag is False:
            ###        break
            ###    for pos in temp_mol.xyz[:52]:
            ###        if flag is False:
            ###            break
            ###        dist = 0
            ###        for i in range(3):
            ###           dist += pow((pos[i] - temp_mol.xyz[idx][i]), 2)
            ###        dist = pow(dist, .5)
            ###        if dist < 5:
            ###            indices += range(idx, idx+5)
            ###            print "bad"
            ###            num_added_chcl3 += 1
            ###            if num_added_chcl3 == num_chcl3:
            ###                flag = False
            ###                added_chcl3 = True
            ###            break
            ###if flag is False:
            ###    ix +=1
            ###if added_chcl3 is False:
            ###   continue 

            """
            indices = range(52)
            C_distances = {}
            C_distances2 = {}
            C_indices = [idx for idx in range(52, len(temp_mol.z_vals)) if temp_mol.z_vals[idx] == 6]
            for idx in C_indices:
                mr_dist = 0
                N_dist = 0
                for i in range(3):
                   mr_dist += pow((mrpos[i] - temp_mol.xyz[idx][i]), 2)
                   N_dist += pow((temp_mol.xyz[44][i] - temp_mol.xyz[idx][i]), 2)
                mr_dist = pow(mr_dist, .5)
                N_dist = pow(N_dist, .5)
                C_distances[idx] = N_dist
                C_distances2[idx] = mr_dist
                if mr_dist < 4.5 and N_dist < 7.5:
                    print "Chloroform possibly in pocket at geometry " + str(ix+1)                
                    pocket_list.append(ix+1)

            # find num_chcl3 closest chloroforms
            max_chcl3_dist = sorted(C_distances.values())[num_chcl3-1]
            for C_idx, d in C_distances.items():
                if d <= max_chcl3_dist:
                    # chloroform solvent molecules have 5 atoms total and the first is carbon
                    indices += range(C_idx, C_idx+5)
            # find close chloroforms; pick out 1 within max_chcl3_dist
            max_chcl3_dist = 9
            max_chcl3_dist2 = 5
            chcl3_indices = []
            for C_idx, d in C_distances.items():
                if C_distances[C_idx] <= max_chcl3_dist or C_distances2[C_idx] <= max_chcl3_dist2:
                    # chloroform solvent molecules have 5 atoms total and the first is carbon
                    chcl3_indices += range(C_idx, C_idx+5)
            indices += chcl3_indices[:5]
            """



            #### make truncated molecule (data in Angstroms)
            ###trunc_z_vals = []
            ###trunc_xyz = []
            ###for idx in indices:
            ###    trunc_z_vals.append(temp_mol.z_vals[idx])
            ###    trunc_xyz.append(temp_mol.xyz[idx])
            ###trunc_gt = gt.Molecule(trunc_z_vals, trunc_xyz, no_properties=True)
            ###trunc_gt.to_xyz_file(filename+'_trunc.xyz')

    
            ###mm_E = calc_mm_E(trunc_gt.xyz, trunc_gt.z_vals, [sys_atom_types[ac] for ac in range(len(trunc_gt.xyz))], sys_charges[:len(trunc_gt.xyz)])
            ###E_dict[ix] = mm_E
            ###geo_dict[ix] = trunc_gt.xyz            

            """
            if ix == 0:
                unique_geometries.append(trunc_gt.xyz)            
            # make input files for unique geometries
            is_unique = True
            trunc_pdist = scipy.spatial.distance.pdist(trunc_gt.xyz)
            for geo in unique_geometries:
                is_unique = True
                geo_pdist = scipy.spatial.distance.pdist(geo)
                #print np.isclose(trunc_pdist, geo_pdist, rtol=.002)
                #print np.allclose(trunc_pdist, geo_pdist, rtol=.08)
                if np.allclose(trunc_pdist, geo_pdist, rtol=.0005):
                    is_unique = False
                    break
            if is_unique or ix == 0:
                unique_geometries.append(trunc_gt.xyz)
                trunc_gt.to_xyz_file(filename+'_unique_trunc.xyz')
                mm_E = calc_mm_E(trunc_gt.xyz, trunc_gt.z_vals, [sys_atom_types[ac] for ac in range(len(trunc_gt.xyz))], sys_charges[:len(trunc_gt.xyz)])
                trunc_gt.write_psi_in(filename+'_'+str(ix)+'.in', comment='MM Energy: '+str(mm_E))
            """
        #simulation.context.setPositions(positions)
    
    ###sorted_E = sorted(E_dict.items(), key=operator.itemgetter(1))
    ###lowest_300_keys = [sorted_E[x][0] for x in range(len(sorted_E)) if x < 300]
    ###for key in lowest_300_keys:
    ###    trunc_gt.xyz = geo_dict[key]
    ###    trunc_gt.write_psi_in(filename+'_'+str(key)+'.in', comment='MM Energy: '+str(E_dict[key]))


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
benzene = """ C   -0.695750000000    0.000000000000   -1.205074349366
 C    0.695750000000    0.000000000000   -1.205074349366
 C    1.391500000000    0.000000000000    0.000000000000
 C    0.695750000000    0.000000000000    1.205074349366
 C   -0.695750000000    0.000000000000    1.205074349366
 C   -1.391500000000    0.000000000000    0.000000000000
 H   -1.235750000000    0.000000000000   -2.140381785453
 H    1.235750000000    0.000000000000   -2.140381785453
 H    2.471500000000    0.000000000000    0.000000000000
 H    1.235750000000    0.000000000000    2.140381785453
 H   -1.235750000000    0.000000000000    2.140381785453
 H   -2.471500000000    0.000000000000    0.000000000000"""

wat_1 = """O    -2.0449536949999998    -1.6898322539999999     0.0354707364500000
H    -2.3427132454308994    -2.1474611062791298     0.8216939386571565
H    -1.1344686596866658    -1.9649570182333860    -0.0720244010028244
"""

chloroform = """C           -0.000060389024    -0.476370509963     0.000000000000
H           -0.000091977725    -1.566304923479     0.000000000000
CL          -1.703916804497     0.069488839774     0.000000000000
CL           0.851970089302     0.069562797391     1.475551447929
CL           0.851970089302     0.069562797391    -1.475551447929
"""

hf = """H 0.0 0.0 0.0
F 0.917 0.0 0.0"""

hcl = """H 0.0 0.0 0.0
Cl 1.274 0.0 0.0"""

bal_3a = """ C        -1.681221     3.231251     0.853686
 C        -1.195036     2.093407     1.784019
 C         0.348959     2.272164     1.542686
 H         0.765821     3.011346     2.232044
 C         1.096327     0.953943     1.659982
 C         1.239192     1.537676    -0.611204
 C         0.453450     2.667349     0.038727
 H         0.944150     3.628327    -0.138289
 C        -1.041861     2.702980    -0.453450
 C        -1.587255     1.294234    -0.383100
 C        -1.669784     0.930294     0.940591
 C        -2.047026    -0.388300     1.340608
 C        -2.050778    -0.780874     2.699146
 H        -1.744475    -0.062522     3.453303
 C        -2.406887    -2.066626     3.062279
 H        -2.398154    -2.361615     4.108006
 C        -2.768596    -2.995877     2.066091
 H        -3.049742    -4.008170     2.344899
 C        -2.757799    -2.633550     0.728243
 H        -3.030646    -3.379501    -0.010036
 C        -2.392904    -1.328463     0.317344
 C        -2.324874    -0.934543    -1.089099
 C        -2.641877    -1.820651    -2.146740
 H        -2.980929    -2.827478    -1.929880
 C        -2.526018    -1.435826    -3.473042
 H        -2.775165    -2.142905    -4.260071
 C        -2.077866    -0.141875    -3.806347
 H        -1.971358     0.148189    -4.848122
 C        -1.765953     0.752920    -2.799554
 H        -1.400631     1.746160    -3.041704
 C        -1.893936     0.384167    -1.440730
 C         2.130188    -0.671236     0.091250
 C         1.370227    -1.835831     0.132249
 C         1.964184    -3.066609    -0.160399
 H         1.373389    -3.977167    -0.128419
 C         3.317890    -3.107810    -0.500161
 H         3.790799    -4.057422    -0.737461
 C         4.087743    -1.939369    -0.539912
 H         5.138111    -1.995282    -0.802321
 C         3.497480    -0.706248    -0.233132
 C         5.509547     0.507773    -0.634157
 H         6.136477    -0.104921     0.028544
 H         5.821117     1.552900    -0.578005
 H         5.599552     0.144962    -1.667099
 N         1.510088     0.580552     0.376070
 O        -2.325216     4.221063     1.079335
 O         1.270342     0.306035     2.670277
 O         1.540620     1.441577    -1.780831
 O         4.150643     0.485639    -0.210725
 H        -1.506572     2.134123     2.826492
 H        -1.215411     3.278631    -1.360966
 H         0.317758    -1.766222     0.391409
"""

bal_3a_inverted = """ C        -0.564409        1.587232       -0.478793
 C        -0.560421        1.286801        1.101938
 C         0.543648        2.243454        1.468059
 H         0.289457        3.314206        1.601034
 C         1.453420        1.855986        2.668132
 C         2.748095        1.321092        0.913446
 C         1.539133        2.008386        0.309205
 H         1.944906        2.907770       -0.187173
 C         0.823544        1.029819       -0.704379
 C         0.524194       -0.385910       -0.181137
 C        -0.158177       -0.176471        0.944983
 C        -0.724411       -1.285375        1.545135
 C        -1.569699       -1.068634        2.670542
 H        -1.616098       -0.186996        3.137461
 C        -2.040529       -2.200673        3.400829
 H        -2.441387       -1.987819        4.398660
 C        -1.985053       -3.490445        2.865076
 H        -2.356972       -4.260386        3.386696
 C        -1.199603       -3.678919        1.690157
 H        -0.899187       -4.618902        1.400922
 C        -0.703058       -2.546775        0.950655
 C         0.043926       -2.723712       -0.215802
 C         0.117822       -4.009480       -0.867193
 H        -0.486961       -4.920807       -0.512894
 C         0.925203       -4.151224       -1.973807
 H         0.967300       -5.124635       -2.535337
 C         1.454040       -3.036311       -2.618509
 H         2.052297       -3.057764       -3.549839
 C         1.368155       -1.761902       -1.976289
 H         2.028186       -0.936157       -2.308063
 C         0.750011       -1.617650       -0.795085
 C         3.667380        0.820638        3.016837
 C         4.817836        1.507791        3.293197
 C         5.824763        0.942602        4.031733
 H         6.609240        1.551475        4.329109
 C         5.826043       -0.530847        4.267753
 H         6.660714       -1.096856        4.777113
 C         4.724409       -1.194412        3.687601
 H         4.742298       -2.276672        3.928057
 C         3.640852       -0.551871        3.137113
 C         2.343757       -2.753978        2.555937
 H         2.301436       -3.038457        1.481982
 H         1.378576       -2.884162        2.887641
 H         3.113539       -3.372892        3.022274
 N         2.646018        1.424187        2.237894
 O        -0.987959        2.582959       -1.065034
 O         1.123323        2.025997        3.922335
 O         3.683920        0.903644        0.187655
 O         2.659018       -1.302055        2.494632
 H        -1.525594        1.557269        1.383435
 H         1.251580        1.095848       -1.695695
 H         4.990026        2.522531        3.020096
 """

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
 
