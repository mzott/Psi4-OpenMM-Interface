import itertools
import numpy as np
from sys import stdout
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import simtk.openmm as mm
from psi4.driver.qcdb import periodictable
import BFS_bonding


# use this if just running python: from psi4.python.qcdb import parkerZOTT

# pdb.positions is list of tuples(Vec3 objects) w units [(x,y,z), (x,y,z)] nm
"""CONSTANTS COPIED FROM PSI"""
bohr2angstroms = .52917720859
angstroms2nanometers = .1

""" zvals is a list of atomic numbers, xyz is a list of lists [ [x,y,z] , [x,y,z] ], bonds is a list of two items: index 0 is as list of lists [ [atom1, atom2, bondorder] ], and index 1 is a list of lists where index 0 is atom 0 and the list at index 0 is the list of indices bonded to """

def make_top(zvals, bonds, atomTypes):
    natom = len(zvals)
    nbonds = len(bonds[0])
    top = Topology()
    smallChain = top.addChain("smallCh")
    smallResidue = top.addResidue("smallRes", smallChain, "1")
    for a in range(natom):
        # add atoms to topology
        top.addAtom(atomTypes[a], Element.getByAtomicNumber(zvals[a]), smallResidue, str(a))  
    atoms_dict = {}
    i = 0
    for a in top.atoms():
        atoms_dict[i] = a
        i += 1
    for b in range(nbonds):
        # add bond to topology
        #top.addBond(bonds[0][b][0], bonds[0][b][1])     
        top.addBond(atoms_dict[bonds[0][b][0]], atoms_dict[bonds[0][b][1]])     
        # determine what type of atom type the atoms should have
    top.setUnitCellDimensions(Vec3(1*nanometer,1*nanometer,1*nanometer))
    
    return top   

def build_omm_system(top, bondinfo, psi_charges):
    forcefield = ForceField('gaff.xml', 'tip3p.xml')
    system = mm.System()
    
    # add atoms to system
    for atom in top.atoms():
        mass = forcefield._atomTypes[atom.name].mass
        system.addParticle(mass)

    """ COPIED from openmm forcefield.py and modified """
    """ PRIMARY MODIFICATIONS:
            bond.atom1 -> bond[0] because using bonds from topology
    """
    # build bondedToAtom from bondinfo[1] keys
    bondedToAtom = []
    for indexList in range(len(bondinfo[1])):
        bondedToAtom.append(bondinfo[1][indexList])
    
    uniqueAngles = set()
    for bond in top.bonds():
        for atom in bondedToAtom[bond[0]]:
            if atom != bond[1]:
                if atom < bond[1]:
                    uniqueAngles.add((atom, bond[0], bond[1]))
                else:
                    uniqueAngles.add((bond[1], bond[0], atom))
        for atom in bondedToAtom[bond[1]]:
            if atom != bond[0]:
                if atom > bond[0]:
                    uniqueAngles.add((bond[0], bond[1], atom))
                else:
                    uniqueAngles.add((atom, bond[1], bond[0]))
    uniqueAngles = sorted(list(uniqueAngles))
    
    uniquePropers = set()
    for angle in uniqueAngles:
        for atom in bondedToAtom[angle[0]]:
            if atom not in angle:
                if atom < angle[2]:
                    uniquePropers.add((atom, angle[0], angle[1], angle[2]))
                else:
                    uniquePropers.add((angle[2], angle[1], angle[0], atom))
        for atom in bondedToAtom[angle[2]]:
            if atom not in angle:
                if atom > angle[0]:
                    uniquePropers.add((angle[0], angle[1], angle[2], atom))
                else:
                    uniquePropers.add((atom, angle[2], angle[1], angle[0]))
    uniquePropers = sorted(list(uniquePropers))
    
    # Make a list of all unique improper torsions
    impropers = []
    
    for atom in range(len(bondedToAtom)):
        bondedTo = bondedToAtom[atom]
        if len(bondedTo) > 2:
            for subset in itertools.combinations(bondedTo, 3):
                impropers.append((atom, subset[0], subset[1], subset[2]))
    
    
    """ END copy """

    data = forcefield._SystemData()
    
    for atom in top.atoms():
        data.atomType[atom] = atom.name
        data.atomParameters[atom] = {}
        data.atoms.append(atom)
        data.excludeAtomWith.append([])
        data.isAngleConstrained.append(False)
        data.atomBonds.append([])
    
    for bond in top.bonds():
        data.bonds.append(ForceField._BondData(bond[0], bond[1]))
    
    data.angles = uniqueAngles
    data.propers = uniquePropers
    data.impropers = impropers
    
    for i in range(len(data.bonds)):
        bond = data.bonds[i]
        data.atomBonds[bond.atom1].append(i)
        data.atomBonds[bond.atom2].append(i)


    """ copied from openmm forcefield """
    
    # add forces
    args = ''
    #None
    nonbondedMethod =  NoCutoff
    nonbondedCutoff = 100*nanometer 
    removeCMMotion = True


    forcefield._forces[3].params.paramsForType['c3']['charge']=0.06217
    forcefield._forces[3].params.paramsForType['c2']['charge']=0.06217
    forcefield._forces[3].params.paramsForType['hc']['charge']=0.06217
    forcefield._forces[3].params.paramsForType['oh']['charge']=0.06217
    forcefield._forces[3].params.paramsForType['h1']['charge']=0.06217
    forcefield._forces[3].params.paramsForType['hw']['charge']=0.06217
    forcefield._forces[3].params.paramsForType['ho']['charge']=0.06217
    forcefield._forces[3].params.paramsForType['ow']['charge']=0.06217

    #for force in system.getForces():

    for force in forcefield._forces:
        force.createForce(system, data, nonbondedMethod, nonbondedCutoff, args)
   
    for force in system.getForces():
        if isinstance(force, NonbondedForce): 
            for i in range(system.getNumParticles()):
                charge, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, psi_charges[i], sigma, epsilon)
                charge, sigma, epsilon = force.getParticleParameters(i)
                #print (data.atoms[i].name)
                #print "charge", charge, "sigma", sigma, "epsilon", epsilon



    if removeCMMotion:
        system.addForce(mm.CMMotionRemover())


    for force in forcefield._forces:
        if 'postprocessSystem' in dir(force):
            force.postprocessSystem(system, data, args)
    
    """ COPIED FROM PEASTMAN AT https://github.com/pandegroup/openmm/issues/1473 """
    
    """ IMPORTANT!!!! THIS IS HOW TO INDIVIDUALLY SET CHARGES FOR ATOMS THAT YOU WANT TO HAVE CHARGES DIFFERENT FROM THE CHARGES SET BY THE FORCEFIELD!"""
     
    
    """ end copy"""
    return system


def write_traj(filename, topology, simulation, mol):
    f = open(filename, 'a')
    xyz_string = str(mol.natoms()) + "\n"
    xyz_string += " generated by Psi4\n"
    for atom, xyz in zip(topology.atoms(), simulation.context.getState(getPositions=True).getPositions()):
        sym = atom.__dict__['element'].symbol
        xyz_string += " " + str(sym) + "  "
        for i in range(3):
            #xyz_string += str(xyz[i].__dict__['_value'])
            xyz_string += str(xyz[i].__dict__['_value']*10)
            xyz_string += "   "
        xyz_string += "\n"
    f.write(xyz_string)
    f.close()

def write_psi_in(filename, topology, simulation, mol):
    f = open(filename, 'a')
    xyz_string = 'molecule{'
    for atom, xyz in zip(topology.atoms(), simulation.context.getState(getPositions=True).getPositions()):
        sym = atom.__dict__['element'].symbol
        xyz_string += " " + str(sym) + "  "
        for i in range(3):
            #xyz_string += str(xyz[i].__dict__['_value'])
            xyz_string += str(xyz[i].__dict__['_value']*10)
            xyz_string += "   "
        xyz_string += "\n"
    xyz_string += "}\n"
    f.write(xyz_string)
    f.write('memory 60000 mb\n')
    f.write("energy('b3lyp-d3/aug-cc-pvdz')")
    f.close()

def make_xyz_matrix(natom, simulation):
    # Make an XYZ matrix from openmm nanometer positions; return bohr positions
    psi_mat = np.zeros(shape=(natom,3))
    index = 0
    for xyz in simulation.context.getState(getPositions=True).getPositions():
        for i in range(3):
            psi_mat[index][i] = xyz[i].__dict__['_value']*10 / bohr2angstroms 
        index += 1
    return psi_mat       

def b_weight(e_list, temp):
    # hartree/K
    k_b = 43.6e-19 * 1.381e-23
    Q = 0
    for e in e_list:
        Q += np.exp(-e / k_b / temp)
    p_list = []
    for e in e_list:
        p_list.append(e / Q)
    return p_list

def psi_mol_to_omm(psi_mol):
    z_vals = []
    xyz = []
    for atom_index in range(psi_mol.natom()):
        xyz.append([0]*3)
        xyz[atom_index][0] = psi_mol.x(atom_index) * bohr2angstroms
        xyz[atom_index][1] = psi_mol.y(atom_index) * bohr2angstroms
        xyz[atom_index][2] = psi_mol.z(atom_index) * bohr2angstroms
        z_vals.append(periodictable.el2z[psi_mol.symbol(atom_index)])
        #z_vals.append(GAFF_Typer.periodictable.el2z[psi_mol.symbol(atom_index)])
    return (z_vals, xyz)

