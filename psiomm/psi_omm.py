from sys import stdout
import itertools
import time
import copy
import numpy as np

import simtk.openmm.app 
import simtk.openmm 
from simtk.unit import *

from psi4.driver.qcdb import periodictable
from psi4.driver.qcdb import physconst 

from . import BFS_bonding
from . import molecule
from . import analysis_methods as am

# variables prefixed with 'po_' indicate objects for use linking Psi (the p) and OpenMM (the o)
# the po essentially means that these OpenMM objects have not been implemented to fully utilize
# all OpenMM functionality
def generateTemplate(forcefield, topology, molecule):
    """
    Method to make OpenMM Templates for small molecules that do not
    have a template yet. This is useful for manually assigning charges
    and atom types.
    forcefield : OpenMM Forcefield
    topology : OpenMM Topology
    molecule : Psi4-OpenMM Molecule 

    Returns
    -------
    OpenMM Forcefield object with new template registered and the topology
    with the atom types and charges added.
    """
    [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(topology)
    if len(templates) > 1:
        raise Exception("Adding more than one new molecule class at a time. Not supported yet!!!")
    for t in templates:
        if t is None:
            continue 
        for index, atom in enumerate(t.atoms):
            atom.type = molecule.atom_types[index]
            atom.parameters = {'charge':molecule.charges[index]}
        forcefield.registerResidueTemplate(t)

def make_topology(mol, top=None, chain_name=None, residue_name=None, res_id=None, unit_cell=(1,1,1)):
    """
    Method to make an OpenMM topology from a Molecule instance.
    For the current iteration of the interface, this method will
    be called when you want to add any new residue/chain; residues
    and chains are treated as being one and the same here since
    we don't typically use proteins.

    Currently this breaks some OpenMM features. Needs revision!

    mol : Molecule object
        Takes in a Molecule object which has z_vals, bonds, and atom_types.
    top : OpenMM Topology
        If an OpenMM topology is passed in, instead of making a new
        topology, this topology will be updated with mol.
    chain_name : string
        Name of chain to add to the Topology.
    residue_name : string
        Name of residue to add to the Topology.
    res_id : string
        id of residue to add to the Topology.
    unit_cell : length 3 Container 
        Dimensions of unit cell for the Topology. Pass in dimensions in nanometers as 
        normal float/integer values. Not necessary to pass in OpenMM Unit objects.

    Returns
    -------
    OpenMM Topology object 
    """
    # Check that bonds, atom types exist
    if mol.bonds is None:
        mol.generate_bonds()
    if mol.atom_types is None:
        mol.generate_atom_types()

    # Get the number of bonds; bonds[0] is a list of length 3 lists that contain
    # the atom indices of the two atoms in a bond as well as the bond order
    nbonds = len(mol.bonds[0])
    # Instantiate an empty Topology object
    po_top = simtk.openmm.app.topology.Topology() if top is None else top
    
    po_chain = po_top.addChain("po_chain" if chain_name is None else chain_name)
    po_residue = po_top.addResidue("po_res" if residue_name is None else residue_name, po_chain, id=("po_res" if res_id is None else res_id))

    for a_ix in range(mol.natoms()):
        # add atoms to topology; here, the name we give each atom is its atom type
        # OpenMM labels all atoms with a unique id, thus it is fine to have equivalent names
        po_top.addAtom(mol.atom_types[a_ix]+"%d" %a_ix, simtk.openmm.app.Element.getByAtomicNumber(mol.z_vals[a_ix]), po_residue)  

    # place the OpenMM Atom objects into a dictionary for simple access
    atoms_dict = {}
    for ix, at in enumerate(po_top.atoms()):
        atoms_dict[ix] = at

    # with the Atoms ordered above, we can now easily add the bonds 
    for b_ix in range(nbonds):
        # recall the mol.bonds[0] is ordered as [bond index][atom index 1, atom index 2, bond order]
        po_top.addBond(atoms_dict[mol.bonds[0][b_ix][0]], atoms_dict[mol.bonds[0][b_ix][1]])     

    po_top.setUnitCellDimensions( simtk.openmm.Vec3(unit_cell[0]*nanometer, unit_cell[1]*nanometer, unit_cell[2]*nanometer) )
    
    return po_top   

def offset_bonds(bonds, offset):
    """
    Offset all of the numbers in the bonds array by value 
    offset; useful for adding molecules to a system that's
    already established because the bond arrays start at 0.
    For a system with N atoms, all indices in the new molecule
    to be added's bonds array need to be increased by N.

    bonds : Psi-OMM Bonds Array

    offset : Integer
        number to increase indices in bonds by

    Returns
    -------
    List of lists of lists in the form of the bonds array.
    """
    bc = copy.deepcopy(bonds)
    for b0 in bc[0]:
        # Increment the indices at b0[0] and b0[1] by the offset.
        b0[0] += offset
        b0[1] += offset

    for b1 in bc[1]:
        # Increment every value in the list by offset.
        for ix in range(len(b1)):
            b1[ix] += offset
    return bc

def add_bonds(bonds1, bonds2):
    """
    Add the two bonds arrays together. For solute-solvent
    systems, the standard is that solute is added before
    solvent, thus bonds1 should be from solute and bonds2
    from solvent.

    bonds1 : Psi-OMM Bonds Array
    
    bonds2 : Psi-OMM Bonds Array
    
    Returns
    -------
    List of lists of lists in the form of the bonds array.
    """
    # Offset is equal to the number of atoms in the bonds1 array 
    offset = len(bonds1[1])
    b2o = offset_bonds(bonds2, offset)

    return (bonds1[0] + b2o[0], bonds1[1] + b2o[1])

def tile_bonds(bonds, reps, num_solvent_atoms):
    """
    Like the Numpy tile function, tile the bonds array.
    The Numpy tile function cannot be used because the 
    values in the bonds array need to be incremented by
    the number of atoms in the solvent molecules.

    bonds : Psi-OMM Bonds array
        The bonds array for the solvent molecule.
    reps : int
        Number of repeats of the bonds array. For a simple
        array, arr=[1,2], tile(arr, 3) would result in
        [1,2,1,2,1,2].
    num_solvent_atoms : int
        Number of atoms in the solvent molecule. This value
        is used to increment every atom index in the bonds array
        (this is every value except bond orders).

    Returns
    -------
    List of lists of lists in the form of the bonds array.
    """
    ret_bonds0 = bonds[0].copy()
    ret_bonds1 = bonds[1].copy()

    for tile_ix in range(1, reps):
        for b0 in bonds[0]:
            working_bond = b0.copy()
            # Increment the indices at working_bond[0] and [1] by
            # the num_solvent_atoms. working_bond[2] is the bond
            # order and should be left alone.
            working_bond[0] += num_solvent_atoms * tile_ix
            working_bond[1] += num_solvent_atoms * tile_ix

            ret_bonds0.append(working_bond)

        for b1 in bonds[1]:
            working_bond = b1.copy()
            # Increment every value in the list by num_solvent_atoms.
            for ix in range(len(working_bond)):
                working_bond[ix] += num_solvent_atoms * tile_ix

            ret_bonds1.append(working_bond)

    # Return the new tiled bond array in the form of a typical bonds
    # array, (bonds0, bonds1)
    return (ret_bonds0, ret_bonds1)

def add_solvent_to_topology(topology, forcefield, solv_mol, num_solvent, solv_name="user_solvent"):
    """
    Method to add solvent molecules to a topology as 'residues'
    individually. More in line with OpenMM usage than calling
    the solute and solvent one system altogether.

    topology : OpenMM topology
    forcefield : OpenMM forcefield
    solv_mol : Psi-OMM Molecule
        Psi-OMM Molecule of the solvent molecule that is to be added
    num_solvent : Integer
        Number of solvent molecules to add
    solv_name : String
        The name that the solvent molecules will be prepended with.
        For example, adding 5 water molecules with name 'water' would
        yield 'water1', ... , 'water5' as residue names

    Returns
    -------
    OpenMM Topology and Forcefield as tuple
    (topology, forcefield)
    """
    # Save original bonds list for future use
    orig_bonds = copy.deepcopy(solv_mol.bonds)
    # Add solvent molecules to topology
    for i in range(num_solvent):
        # Get number of atoms in topology already to offset solv_mol's bonds array by
        n_atoms = topology.getNumAtoms()
        solv_mol.bonds = offset_bonds(orig_bonds, n_atoms)
        topology = make_topology(solv_mol, top=topology,
                                 chain_name='solvent', residue_name=solv_name,
                                res_id=solv_name+"%d" % i)
       
        # Generate template for this solvent molecule
        generateTemplate(forcefield, topology, solv_mol)
    return (topology, forcefield)

def get_atom_positions(topology, simulation):
    """
    Method to get atom positions from an OpenMM simulation
    by accessing the Simulation's Context. Converts atom positions
    to Angstroms.

    topology : OpenMM Topology object
        Topology object that contains the atoms we want to include in the trajectory.
    simulation : OpenMM Simulation object
        Simulation object that contains the Context that contains up to date atom 
        positions.
    
    Returns
    -------
    Tuple of (xyz, z_vals) where xyz is a Numpy array of length number of atoms (N)
    and z_vals is a Numpy array of length number of atoms of lists of length 3 (N, 3).
    """
    unit = None
    # retrieve atomic number and coordinates in Nanometers 
    z_vals, xyz = [], []
    for atom, xyz_tri in zip(topology.atoms(), simulation.context.getState(getPositions=True).getPositions()):
        z_vals.append( atom.__dict__['element'].atomic_number )
        xyz.append( [ xyz_tri[i].__dict__['_value'] for i in range(3) ] )
        unit = xyz_tri[0].__dict__['unit'].get_name().lower().strip()
    
    # Convert from nanometers to Angstroms if unit is nanometers
    unit_scaling = 10 if unit=='nanometer' else 1

    return (np.asarray(z_vals), np.asarray(xyz)*unit_scaling)

def write_traj(filename, z_vals, xyz, comment=" generated by Psi-OMM"):
    """
    Method to write trajectories for a simulation. 
    
    filename : string
        Name of file to write trajectory to.
    z_vals : container 
        Container of length number of atoms with atomic numbers of atoms.
    xyz : container
        Container of length number of atoms of containers of length 3.
    comment : string
        Comment to add in comment field of XYZ files that make up a trajectory.
        A nice standard would be to include the index of this snapshot in the comment.

    Yields
    ------
    New text file with name filename.
    """
    # create a Molecule instance and write the trajectory
    mol = molecule.Molecule(z_vals, xyz, unit='Angstrom')
    mol.to_xyz_file(filename, comment, append_mode='a')

def calc_mm_E(mol, forcefield=simtk.openmm.app.forcefield.ForceField('gaff2.xml')):
    """
    Method to calculate an OpenMM energy for an arbitrary geometry.
    Useful for post-analysis if the simulation is over but you still
    have the trajectory. Only returns potential energy as kinetic 
    energy is not able to be calculated from an XYZ file alone.

    mol : Molecule object
        Psi-OMM Molecule object that has z_vals and xyz and is able to 
        calculate atom_types or charges if it does not already have them. 
    forcefield : OpenMM ForceField
        Force field to use. Currently only GAFF atom types are able
        to be generated automatically, thus this is the default. If a force field
        other than GAFF is used, be sure that the atom_types in the Molecule
        object match. This is primarily included such that GAFF1 versus
        GAFF2 can be chosen.

    Returns
    -------
    float : OpenMM Potential Energy in kcal/mol
    """
    # Check that bonds, atom types, charges exist
    if mol.bonds is None:
        mol.generate_bonds()
    if mol.atom_types is None:
        mol.generate_atom_types()
    if mol.charges is None:
        mol.generate_charges()

    po_top = make_topology(mol)        
    omm_sys = simtk.openmm.app.forcefield.createSystem(top, nonbondedMethod=NoCutoff, atomTypes=mol.atom_types, atomCharges=mol.atom_charges)

    integrator = LangevinIntegrator(100*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(po_top, omm_sys, integrator, platform=Platform.getPlatformByName('Reference'))

    # Create positions as desired by OpenMM; multiplication by .1 is to transform units from A to nm
    coords = []
    for i in range(len(mol.xyz)):
        coords.append(Vec3(mol.xyz[i][0]*.1, mol.xyz[i][1]*.1, mol.xyz[i][2]*.1))
    positions = coords * nanometer
    simulation.context.setPositions(positions)

    state = simulation.context.getState(getEnergy=True, getForces=False)
    #forces = state.getForces(asNumpy=True)
    energy = state.getPotentialEnergy()
    #print('MM ENERGY: ', energy/kilocalories_per_mole, 'KCAL/MOL ', energy)

    return energy/kilocalories_per_mole

def find_solvent_pos(solute_mol, solvent_mol, num_solvent=None, box_size=(10,10,10), stacked=True):
    """
    Add solvent around a solute. This method is very rudimentary and adds solvent
    around a rectangular box which surrounds the solute. Thus, for accurate structures
    a energy minimization must be performed after use of this method. The solvent is 
    added on a grid whose size is determined by the size of the solvent which is also
    modeled as a rectangle.
    
    TODO: Change from rectangle with padding around solvent to a Van Der Waals radii based
    approach.

    The solvent is added radially around the solute. Thus, with enough solvent molecules added, 
    the entire solute/solvent system will approximate a sphere. 

    In OpenMM, there exists a method to addSolvent, but it only adds water molecules. In order to 
    use other solvents (and not use PDB files), this method exists. When OpenMM adds solvent molecules,
    the standard appears to be to create a new Chain for the solvent system. Within this Chain, every
    individual solvent molecule becomes a Residue. In order to keep track of individual solvent molecules,
    set stacked=False.

    solute_mol : Psi-OMM Molecule
        Psi-OMM Molecule representing the solute system. Only needs to contain
        xyz - this is required of all Psi-OMM Molecules, thus any Psi-OMM 
        Molecule will work.
    solvent_mol : Psi-OMM Molecule
        Psi-OMM Molecule representing a single solvent molecule. See solute_mol.
    num_solvent : int
        Number of solvent molecules to add around the solute.
    box_size : tuple
        Tuple of box dimensions to fill with solvent. Dimensions should be
        in Angstroms.
    stacked : Boolean
        If True, returns the added solvent's z_vals and xyz arrays as single arrays;
        arrays would have the form [N] and [N,[3]].
        If False, z_vals and xyz will be lists of length num_solvent of z_vals and
        xyz arrays; lists would have form [N[1]] and [N[n,[3]]] where n is the 
        number of atoms in the solvent molecule.

    Returns
    -------
    Tuple of Numpy arrays of form described in stacked. Note that the returned
    arrays only contain information about the SOLVENT that has been added. In
    order to create a simluation with both the solute and solvent, the solute and
    solvent arrays will need to be combined.
    """
    z_arr_list = []
    xyz_arr_list = []

    t = time.time()
    solute_xyz = solute_mol.xyz
    solvent_xyz = solvent_mol.xyz

    # Pad the region around the solute
    solute_box = 2.0 * np.max(solute_xyz, axis=0)
    solute_box += 2.0

    # Center solvent
    solvent_xyz -= np.mean(solvent_xyz, axis=0)
    # Pad the region around the solvent
    solvent_box = 2.0 * np.max(solvent_xyz, axis=0)
    solvent_box += 1.25

    # Find the maximum allowable extent of the solvent molecules, constrained
    # by the box_size
    sphere_dist = np.linalg.norm(solvent_box)

    solvent_placed = 0

    # If the number of solvent molecules desired is not specified, add a default value of 5
    if num_solvent is None:
        num_solvent = 5

    ref_stacked_xyz = solute_xyz.copy()
    # Break the addition of solvent into radial shells; default number of 
    # shells is up to 30 which seems to typically work well.
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

            distances = np.sum((ref_stacked_xyz - displacement)**2, axis=1) ** 0.5
            if np.any(distances < sphere_dist):
                #print('Skipping distance!')
                continue

            z_arr_list.append(solvent_mol.z_vals)
            xyz_arr_list.append(solvent_xyz + displacement)

            ref_stacked_xyz = np.vstack((ref_stacked_xyz, solvent_xyz + displacement))
            #output_xyz = np.vstack((output_xyz, solvent_xyz + displacement))
            #output_z_vals = np.vstack((output_z_vals, np.array([[x[0]] for x in solvent_geo_array])))

            solvent_placed += 1
            if solvent_placed >= num_solvent:
                break

    print( "Time to add solvent: ", time.time()-t)

    if stacked is False:
        return (z_arr_list, xyz_arr_list)

    else:
        # Need to stack the z_vals and xyz arrays
        stacked_z = np.hstack(tuple(z_arr_list))
        stacked_xyz = np.vstack(tuple(xyz_arr_list))
        return (stacked_z, stacked_xyz)    


def mm_setup(mol, forcefield=simtk.openmm.app.forcefield.ForceField('gaff2.xml')):
    """
    Method to set up an OpenMM simulation and associated objects. Useful as a precursor
    to other methods. Advanced users should probably perform setup themselves.

    mol : Psi-OMM Molecule object
        Psi-OMM Molecule object that has z_vals and xyz and is able to 
        calculate atom_types or charges if it does not already have them. 
    forcefield : OpenMM ForceField
        Force field to use. Currently only GAFF atom types are able
        to be generated automatically, thus this is the default. If a force field
        other than GAFF is used, be sure that the atom_types in the Molecule
        object match. This is primarily included such that GAFF1 versus
        GAFF2 can be chosen.

    Returns
    -------
    A tuple of the OpenMM topology, forcefield, system, integrator, and simulation
    objects initialized for this molecule and with suitable defaults.
    """
    # Check that bonds, atom types, charges exist
    if mol.bonds is None:
        mol.generate_bonds()
    if mol.atom_types is None:
        mol.generate_atom_types()
    if mol.charges is None:
        mol.generate_charges()
    
    # Create an OpenMM topology
    topology = make_topology(mol)
    
    # Declare the force field to be used
    forcefield = simtk.openmm.app.forcefield.ForceField('gaff2.xml')

   # Generate template for the force field
    forcefield = generateTemplate(forcefield, topology, mol)

    # Create the OpenMM System
    omm_sys = forcefield.createSystem(topology)
    
    # Declare the integrator to be used
    integrator = simtk.openmm.openmm.LangevinIntegrator(25*kelvin, 1/picosecond, 0.002*picoseconds)
    
    # Create the OpenMM Simulation
    simulation = simtk.openmm.app.simulation.Simulation(topology, omm_sys, integrator)
    
    # Create positions as desired by OpenMM; multiplication by .1 is to transform units from A to nm
    coords = []
    for i in range(len(mol.xyz)):
        coords.append(simtk.openmm.vec3.Vec3(mol.xyz[i][0]*.1, mol.xyz[i][1]*.1, mol.xyz[i][2]*.1))
    positions = coords * nanometer
    
    # Set the positions in the simulation so it knows where atoms are located
    simulation.context.setPositions(positions)
    
    return (topology, forcefield, omm_sys, integrator, simulation)

def find_mm_min(mol, E_tolerance=0.5, step_block=5000):
    """
    Method to calculate an OpenMM minimum energy geometry.

    mol : Psi-OMM Molecule object
        Psi-OMM Molecule object that has z_vals and xyz and is able to 
        calculate atom_types or charges if it does not already have them. 
    E_tolerance : float
        Energy tolerance in kcal/mol to converge the energy minimization.
        Default is 0.5 kcal/mol.
    step_block : int
        Number of steps to take per iteration in the annealing process. Default
        is 5000 - raising this value will increase the amount of time it takes
        to complete this method, but it may result in a more accurate geometry.

    Returns
    -------
    Psi-OMM Molecule with updated geometry
    """
    top, ff, omm_sys, integrator, sim = mm_setup(mol)

    # Anneal the system to try to get into the global minimum
    sim.step(step_block)
    for i in range(1, 11):
        integrator.setTemperature(200-20*i)
        sim.step(step_block)

    # Minimize the energy
    sim.minimizeEnergy(tolerance=E_tolerance*kilocalorie/mole)
 
    # Get the updated geometry
    new_z_vals, new_xyz = get_atom_positions(top, sim)
    
    mol.z_vals = new_z_vals
    mol.xyz = new_xyz
   
    return mol 

def find_conformations(mol, N, T, MM_sort=True, unique_geometries=False, RMSD_threshold=.1, has_solvent=False, return_E=False):
    """
    Method to find N different low energy conformations of the molecule.
    This method works by first finding N*10 geometries annealed down to
    temperature T and then sorting out the N lowest energy (MM energy) 
    geometries if the MM_sort option is set to True. If the MM_sort
    option is set to False, only N geometries are found and they are 
    returned.

    If you trust the forcefield, it is probably a good idea to leave
    MM_sort set to True because you will likely get better geometries.
    However, if you do not think the forcefield represents the system
    well (and thus the potential energy surface), you might set
    MM_sort to False in order to not miss false positive "high energy"
    structures that are actually more stable than the forcefield indicates.

    MM_sort and unique_geometries are mutually exclusive.

    mol : Psi-OMM Molecule object
        Psi-OMM Molecule object that has z_vals and xyz and is able to 
        calculate atom_types or charges if it does not already have them. 
    N : int
        Number of conformations to find.
    T : float
        Temperature in Kelvin to find geometries at.
    MM_sort : boolean
        If True, finds 10*N MM geometries and returns the N lowest energy
        structures. If False, only finds N geometries and returns them.
    unique_geometries : boolean
        If True, only returns unique geometries. These may be difficult to find;
        thus, an unknown number of geometries must be computed in order to find 
        N unique conformations. This will affect the running time. Unique geometries
        are assessed using RMSD as a metric. A maximum of 200*N geometries are searched.
    RMSD_threshold : float
        Value that the RMSD must be greater than for two conformations to be deemed unique.
        Value should be in Angstroms.
    has_solvent : boolean
        Set True if there is solvent in the simulation. This is important
        to know if unique geometries are desired.
    return_E : boolean  
        If True, return the energies of the conformations as well. Energies are
        in kcal/mol.

    Returns
    -------
    List of length N of Numpy (3,L) arrays where L is the number of atoms in the system.
    If return_E is True, returns a tuple of the above list and a list of energies, ((N,3), (N))
    """
    # If MM_sort and unique_geometries are set, unique_geometries takes precedence
    if MM_sort and unique_geometries:
        MM_sort = False

    return_list = []
    E_list = []

    # Set max temperature to 30% greater than annealing temperature
    max_T = T * 1.3

    # Find the number of geometries that are initially desired
    num_geos = N * 10 if MM_sort else N

    # Maximum number of geometries if unique_geometries is True
    max_num_geos = N * 200

    top, ff, omm_sys, integrator, sim = mm_setup(mol)

    # Let the system settle before annealing
    sim.step(2500)

    geos_seen = 0
    while len(return_list) < num_geos:
        # If unique geometries are desired, check that we haven't went over max_num_geos
        if unique_geometries and geos_seen > max_num_geos:
            print("""We have searched through %d geometries. We have only found %d
                     unique geometries. We are stopping the search at this point
                     in order to avoid excessive computational time. The %d unique
                     geometries have been returned.""" % (max_num_geos, len(return_list), len(return_list)))
            if return_E:
                return (return_list, E_list)
            return return_list

        # Anneal the system to try to get into a local minimum
        for i in range(1, 11):
            integrator.setTemperature(max_T- (max_T-T)/10 * i )
            sim.step(500)
    
        # Get the updated geometry
        new_z_vals, new_xyz = get_atom_positions(top, sim)

        if unique_geometries:    
            # Check that the geometries are unique
            is_unique = True

            #TODO: has_solvent is unusued. It will be used here in RMSD method. Need
            # a way to account for solvent molecules swapping places and inflating RMSD
            for known_geo in return_list:
                RMSD = am.rmsd(known_geo, new_xyz) 
                print("RMSD", RMSD)
                # If the RMSD is too large, do not return the geometry
                if RMSD < RMSD_threshold:
                    is_unique = False
                    break
                
            if is_unique:
                return_list.append(new_xyz)
                state = sim.context.getState(getEnergy=True, getForces=False)
                energy = state.getPotentialEnergy()
                energy /= kilocalories_per_mole
                E_list.append(energy)
                print("Found a geometry")
        else:
            return_list.append(new_xyz)
            state = sim.context.getState(getEnergy=True, getForces=False)
            energy = state.getPotentialEnergy()
            energy /= kilocalories_per_mole
            E_list.append(energy)
            print("Found a geometry")
            
        print("Saw new geometry")
        geos_seen += 1            

    # If MM_sort is True, we need to return only the lowest 10% of geometries by energy
    if MM_sort:
        low_E_geos = []
        low_E = []
        # Find the maximum energy allowable
        max_E = sorted(E_list)[N-1]
        for ix, E in enumerate(E_list):   
            if E < max_E:
                low_E_geos.append(return_list[ix])
                low_E.append(E)
        return_list = low_E_geos
        E_list = low_E

    # We have found the number of geometries desired now. Return the list of the geometries.
    if return_E:
        return (return_list, E_list)
    return return_list


