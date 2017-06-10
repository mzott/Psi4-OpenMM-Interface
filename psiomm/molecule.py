from __future__ import absolute_import
from __future__ import print_function
import os
import subprocess
import itertools
import numpy as np

import psi4
from psi4.driver.qcdb import periodictable
from psi4.driver.qcdb import physconst
from psi4.driver.qcdb import cov_radii

from . import BFS_bonding
from . import helper_methods as hm
from . import analysis_methods as am

class Molecule(object):
    def __init__(self, z_vals, xyz, unit='Angstrom'):
        """
        Molecule objects are used for passing information between Psi4 and OpenMM.
        Psi4 and OpenMM have different unit conventions. Psi4 does not label units,
        but typically takes units as Angstrom or Bohr. Internally, units are typically
        in Bohr. OpenMM typically uses nanometers, but units in OpenMM are treated as
        objects where every number has its value as well as its unit in a Unit object.

        In order to preserve clarity between programs, it is important to keep track of units. 
        The Molecule class here keeps all units in Angstroms. Because the Cartesian coordinates
        stored in Molecule.xyz are kept in a Numpy array, conversion between units is facile.

        If units other than Angstrom are passed in, the keyword argument unit should be set
        to the unit in place. Supported units (not case sensitive): 'Angstrom', 'Bohr', 
        'Nanometer'.

        z_vals : container of integers
            Container of z values (atomic numbers) of length N.
            Ex: [1, 1, 6, 6]
        xyz : container of container of doubles
            Container of Cartesian coordinates of N atoms. Dimensions (N, 3).
        """
        
        self.z_vals = np.array(z_vals)
        self.xyz = np.array(xyz)      

        unit = unit.lower()
        # if the units are not in Angstrom, convert Nanometers OR Bohr to Angstrom
        if unit != 'angstrom':
            self.xyz = self.xyz * 0.1 if unit == 'nanometer' else self.xyz / physconst.psi_bohr2angstroms 

        if self.xyz.shape[1] != 3:
            raise ValueError("XYZ should only have three coordinates")

        atom_indices = range(self.natoms())
        
        self.bonds = None
        self.charges = None
        self.atom_types = None

    @staticmethod
    def from_xyz_string(xyz_string):
        """
        Create a molecule from a string.
        Zxyz_string : string
            Example and format:

            Atomic number   X    Y    Z
            _____________________________
            
            Zxyz_str = 'C               0.0  0.0  0.0
                        O               1.2  0.0  0.0'
        
        """
        z_vals = []
        xyz = []
        for line in xyz_string.splitlines():
            Zxyz_list = line.strip().split()
            if Zxyz_list == []:
                continue
            if len(Zxyz_list) != 4:
                raise KeyError("Line should have exactly 4 elements, Z, x, y, z.")

            z_vals.append(periodictable.el2z[Zxyz_list[0].upper()])
            xyz.append([float(x) for x in Zxyz_list[1:]])

        return Molecule(z_vals, xyz)

    @staticmethod
    def from_xyz_file(filename):
        """
        Create a molecule from an xyz file.
        First line should be number of atoms, second line a comment,
        and subsequent lines (of length the number of atoms) atomic symbol
        and Cartesian coordinates.

        filename : string
            String of the name of the file containing the molecular input.
        """
        f = open(filename)
        data = f.read().splitlines()
        f.close()
        data_string = ""

        num_atoms = int( data[0].strip() )
        comment = data[1]

        for line in data[2:]:
            if len(line.split()) != 4: continue
            data_string += line + '\n'

        num_atom_lines = data_string.count('\n')
        if num_atom_lines != num_atoms:
            raise Exception("The XYZ file should contain %d atoms. It only contains %d lines (one atom per line)." % (num_atoms, num_atom_lines))

        return Molecule.from_xyz_string(data_string)
    
    def generate_bonds(self):
        """
        Generate bonding information for this molecule.

        Yields
        -------
        List of lists of lists

        self.bonds[0] will yield a list of length number of bonds. This list is a list of 
        lists of length 3; [atom1, atom2, bond_order] where atom1 and atom2 are atom indices
        for the Molecule's xyz array and bond_order is an integer 1, 2, or 3 for single, double, 
        or triple bond.

        self.bonds[1] will yield a list of length number of atoms. This list is a list of lists
        where each index in the list corresponds to the index in the Molecule's xyz array. The list
        at this location contains the indices of every atom that is bound to this atom.

        Example:
        O=C=O carbon dioxide
        self.bonds[0] = [ [0,1,2], [1,2,2] ]
        self.bonds[1] = [ [1], [0,2], [1] ]

        """
        self.bonds = BFS_bonding.bond_profile(self)
    
    def to_xyz_string(self):
        """
        Create an XYZ string with this molecule. Useful for passing to
        Psi4 via psi4.geometry or for writing XYZ files.

        Returns
        -------
        string

        Example: String looks like this (with newlines inserted instead of shown as backslash n. Note the one space.
         O 0.0 0.0 0.0           
         C 1.2 0.0 0.0
         O 2.4 0.0 0.0 

        """
        output = '' 
        for atom in range(self.natoms()):
            output += '\n' if atom != 0 else ''
            output += "%3s  " % self.symbol(atom)
            output += '   '.join("% 10.6f" % x for x in self.xyz[atom])
        return output

    def to_xyz_file(self, filename, comment=None, append_mode='w'):
        """
        Create an XYZ file with this molecule.
        
        filename : string
            Name of the new XYZ file.
        comment : string
            String to be placed on second line of XYZ file. 
        append_mode : string
            Mode for open() to use. Default is 'w' for write - overwrite anything
            in the file with name filename. For writing XYZ files, this should 
            typically be in append_mode='w'. Alternative mode 'a' for append is 
            useful for writing trajectories.

        Example:
            An XYZ file looks like the following
            3                        # Number of atoms
             Carbon Dioxide          # Comment
             O 0.0 0.0 0.0           # Atoms; atomic symbol followed by coordinates
             C 1.2 0.0 0.0
             O 2.4 0.0 0.0 
        """
        # if append_mode is 'a', we need to check if there is text in the file
        # if there is text, we add a newline such that the new XYZ data is readable for trajectories
        file_exists = os.path.exists(filename)
        output = '\n' if (file_exists and append_mode=='a') else '' 
        output += str(self.natoms())
        output += ('\n '+comment if comment is not None else '\n #Generated by GT')
        output += '\n' + self.to_xyz_string()
        
        f = open(filename, append_mode)
        f.write(output)
        f.close()

    """TODO add mol2 file writer. Probably exists in qcdb of Psi4."""

    """TODO add method to write psi4 input files. Probably exists in qcdb of Psi4."""

    def generate_charges(self, qc_method=None, basis=None, charge_method=None):
        """
        Generate charges for the molecule using Psi4. Probably need to support semi-empirical
        charges in the future i.e. MOPAC support for use in generating am1-bcc charges via
        Antechamber. For now, Mulliken and Lowdin charges can be generated by Psi4. Default is 
        Mulliken with SCF and sto-3g.

        qc_method : string
            Quantum chemistry method to generate charges. Any method supported by Psi4 can be 
            passed i.e. SCF, HF, B3LYP, etc.
        basis : string
            Basis set for generation of charges. Any basis set supported by Psi4 can be passed.
        charge_method : string
            Method to generate atomic charges from one-electron density. Psi4 supported methods
            are Mulliken and Lowdin; case insensitive.

        Yields
        -------
        Numpy array of length number of atoms of floats; accessible at self.charges
        """
        qc_method = 'scf' if qc_method is None else qc_method
        basis = 'sto-3g' if basis is None else basis
        charge_method = 'MULLIKEN' if charge_method is None else charge_method.upper()
        charge_method += '_CHARGES'

        # make geometry into a string for passing to psi4.geometry
        xyz_string = self.to_xyz_string()
        # First, we want to assign atom types and charges to the solute
        mol = psi4.geometry(xyz_string)
        # Tell Psi4 that we have updated the geometry
        mol.update_geometry()
        
        psi4.core.set_global_option("BASIS", basis)

        E, wfn = psi4.property(qc_method, properties=[charge_method], return_wfn=True)
        self.charges = np.asarray(wfn.atomic_point_charges())

    def generate_atom_types(self, temp_filename="TEMP_FILENAME_PSI-OMM", babel_path=None, antechamber_path=None):
        """
        Generate General Amber Force Field atom types for the molecule using Antechamber.
        Hopefully in the future there will be a Python module that will be able to do this...

        In order to generate the atom types, first an XYZ file is written. Then, this XYZ file
        is translated into a pseudo PDB file by babel. This PDB file is then passed to Antechamber
        which produces an output file with the atom types. This file is then parsed in order to return
        a list of the atom types.

        If babel_path and antechamber_path are not provided, they are searched for in the system path.

        temp_filename : string
            Filename for XYZ file, PDB file, and Antechamber output file to be written to. All of these
            will be cleaned up (deleted) upon success of this method.
        babel_path : string
            Full path including executable name for Babel i.e. '/usr/bin/babel'
        antechamber_path : string
            Full path including executable name for Antechamber i.e. '/usr/bin/antechamber'

        Yields
        ------
        Numpy array of length number of atoms of strings; accessible at self.atom_types
        """
        # find paths to the programs we will need to call later
        babel_path = hm.which('babel') if babel_path is None else babel_path
        antechamber_path = hm.which('antechamber') if antechamber_path is None else antechamber_path

        # make an XYZ file 
        self.to_xyz_file(temp_filename+'.xyz')

        # use Babel to convert the XYZ file to a PDB file for use with Antechamber
        subprocess.call([babel_path, temp_filename+'.xyz', temp_filename+'.pdb']) 

        # use Antechamber to assign atom types
        subprocess.call([antechamber_path, '-i', temp_filename+'.pdb', '-fi', 
                         'pdb', '-o', temp_filename+'.TEMP_ATS', '-fo', 'prepc',
                         '-pf', 'y'])

        # parse the atom types from the Antechamber output file; not simple, thus a helper method is called
        self.atom_types = hm.align_antechamber(temp_filename+'.xyz', temp_filename+'.TEMP_ATS')

        # remove files generated during the process
        remove_list = [temp_filename+'.xyz', temp_filename+'.pdb', temp_filename+'.TEMP_ATS']
        for rm in remove_list:
            subprocess.call(['rm', rm])

    def natoms(self):
        """
        Return the number of atoms from the length of the xyz array.
        
        Returns
        -------
        int - the number of atoms
        """
        return self.xyz.shape[0]

    def distance(self, atom1, atom2):
        """
        Return the distance between two atoms in Angstroms.

        atom1, atom2 : int
            The index of atom1 and atom2 in the xyz array.
        
        Returns
        -------
        float - the distance in Angstroms
        """
        return np.linalg.norm(self.xyz[atom1]- self.xyz[atom2])

    def neighbors(self, atom1):
        """
        Return the list of atoms that are neighbors to atom1. Requires bonding
        information. If none exists, it will be generated.
        
        atom1 : int
            Index of the atom in Molecule that neighbors are desired for.

        Returns : list
            List of atomic indices in Molecule of atoms that are bonded to atom1.
        """
        # Check that bonding information exists; if it does not, generate it
        if self.bonds is None:
            self.generate_bonds()
        
        return self.bonds[1][atom1] 

    def angle(self, atom1, atom2, atom3):
        """
        Compute the angle between the points at indices atom1, atom2, and atom3
        in the xyz array.

        atom1, atom2, atom3 : int
            The indicies of the atoms in the XYZ array.
        
        Returns
        -------
        float - the angle in radians.
        """
        point1 = np.asarray(mol.xyz[atom1])
        point2 = np.asarray(mol.xyz[atom2])
        point3 = np.asarray(mol.xyz[atom3])

        v1 = point2 - point1
        v2 = point3 - point1

        return hm.vector_angle(v1, v2)

    def plane_perp_vec(self, atom1, atom2, atom3):
        """
        Compute the unit length vector perpendicular to the plane defined by points at indices atom1, atom2, and atom3
        in the xyz array.

        atom1, atom2, atom3 : int
            The indicies of the atoms in the XYZ array.

        Returns
        -------
        Length 3 Numpy array of floats
        """
        point1 = np.asarray(mol.xyz[atom1])
        point2 = np.asarray(mol.xyz[atom2])
        point3 = np.asarray(mol.xyz[atom3])
    
        v1 = point2 - point1
        v2 = point3 - point1
    
        perp_vec = np.cross(v1, v2)
    
        return perp_vec / np.linalg.norm(perp_vec)

    def dihedral(self, atom1, atom2, atom3, atom4):
        """
        Calculate the dihedral angle between four points.
        
        atom1, atom2, atom3, atom4 : int
            Indices of the four atoms in the XYZ array to calculate a dihedral angle from
        
        Returns
        -------
        float - the dihedral angle in radians
        """
        n1 = self.plane_perp_vec(atom1, atom2, atom3)
        n2 = self.plane_perp_vec(atom2, atom3, atom4)

        dihedral = - n1.dot(n2) / np.linalg.norm(n1) / np.linalg.norm(n2)
        dihedral = np.arccos([dihedral])

        return dihedral

    def symbol(self, atom1):
        """
        Return the chemical symbol of an atom. For example, 6 would yield 'C'.
        atom1 : int
            The index of atom1 in the z_vals array.
        
        Returns
        -------
        string - the chemical symbol
        """
        return periodictable._temp_symbol[int(self.z_vals[atom1])]

#TODO: Add methods to return all bonds, angles, dihedrals that exist between BONDED ATOMS only

    def is_overlapped(self, atom1, atom2):
        """
        Using covalent radii, check if atom1 and atom2 are overlapping. 
    
        atom1, atom2 : Indices of atom1 and atom2 in self.xyz
    
        Returns
        -------
        Boolean - True if the distance between atom1 and atom2 is less
        than the sum of their covalent radii 
        """
        # Find covalent radii
        r1 = cov_radii.psi_cov_radii[self.symbol(self.z_vals[atom1])]
        r2 = cov_radii.psi_cov_radii[self.symbol(self.z_vals[atom2])]

        # Find distance
        d = self.distance(atom1, atom2)
        
        return d < (r1 + r2)*1.2

    def substitute(self, atom1, group, group_ix=0):
        """
        Substitute the atom at index atom1 with the group defined by the Psi-OMM Molecule
        group by attaching the group by the atom at index group_ix. Place the group (attached
        by the atom at index group_ix) at the position of atom1. This group is aligned such that
        the mean atomic position of group_ix is aligned with the vector that bisects all angles containing
        atom1 and atoms it is attached to. Essentially, it places group in a VSEPR-like configuration.

        atom1 : int
            The index of atom1, the atom to be substituted, in this Molecule.
        group : Psi-OMM Molecule
            A Psi-OMM Molecule that represents the group to be added at the position of atom1.
            Only needs to contain the atomic positions.
        group_ix : int
            The index of the atom in group which will take the place of atom1. The default is 0
            i.e. the first atom in group should be the atom that will take the place of atom1. For
            example, hydroxylating a hydrogen site would require the oxygen to be at index group_ix.

        Updates
        -------
        Updates the self.xyz array with a new array with the substituted group.
        """
        # Make a copy of the old geometry
        new_geo = self.xyz.copy()      
        new_z_vals = self.z_vals.copy()        

        # Find the bisecting vector between atom1 and its neighbors
        # First, identify the list of atoms that are neighbors to atom1
        neighbors = self.neighbors(atom1)

        # Find the unit vector pointing from atom1 to where group's mean atomic position will lie
        # This is the unit vector of the sum of all vectors atom1-n where n is a neighboring atom                
        uvs = []
        for n in neighbors:
            v = self.xyz[atom1] - self.xyz[n]
            v /= np.linalg.norm(v)
            uvs.append(v)

        uv = np.sum(np.asarray(uvs), axis=0)
        uv /= np.linalg.norm(uv)
        uv.shape = (1,3)
        #uv = np.append(uv, np.array([[0,0,0]]), axis=0)

        # Translate the group's atom at group_ix to (0,0,0) and the rest of the group
        # to prepare for rotation (needs to be at origin for rotation)
        group.xyz -= group.xyz[group_ix]

        # Find the centroid of the orientation and group
        centroid = np.mean(uv, axis=0)
        g_centroid = np.mean(group.xyz, axis=0)

        # Find the group unit vector pointing from atom at group_ix to g_centroid
        g_uv = g_centroid - group.xyz[group_ix]
        g_uv /= np.linalg.norm(g_uv)
        g_uv.shape = (1,3)
        
        g_uv_orig = g_uv.copy()
        g_uv_orig.shape = (3,)

        #g_uv = np.append(g_uv, np.array([[0,0,0]]), axis=0)

        # Align group by rotating g_uv to uv in order to find the rotation matrix
        R = am.find_rotation(g_uv, uv)

        group.xyz = np.dot(R, group.xyz.T)
        group.xyz = group.xyz.T 

        # Translate the group geometry such that the atom at group_ix lies at the position of atom1
        group.xyz += self.xyz[atom1] 

        # Translate the group such that the bond distance is more accurate if there is only one neighbor
        # excluding the group itself. Else, too complicated and it should stay at its current position.
        if len(neighbors) == 1:
            bond_dist = self.distance(atom1, neighbors[0])
            debond_v = uv * bond_dist
            group.xyz -= debond_v
            bond_v_dist = cov_radii.psi_cov_radii[self.symbol(self.z_vals[atom1])] + cov_radii.psi_cov_radii[self.symbol(self.z_vals[neighbors[0]])]
            bond_v = uv * bond_v_dist
            group.xyz += bond_v

        # Add the group to the molecule
        # First, remove atom1
        new_geo = np.delete(new_geo, atom1, axis=0)
        new_z_vals = np.delete(new_z_vals, atom1)

        # Add group
        new_geo = np.append(new_geo, group.xyz, axis=0)
        new_z_vals = np.append(new_z_vals, group.z_vals)    

        # Update geometry of molecule with new_geo
        self.xyz = new_geo
        self.z_vals = new_z_vals

        # Check that the rotated, translated group does not overlap with any atoms in the original
        # molecule. If so, rotate it in 30 degree increments to try to resolve any overlaps.
        n = group.xyz.shape[0]
        N = self.xyz.shape[0]

        # 12 possible rotations by 30 degrees
        no_overlaps = False
        for x in range(12):
            if no_overlaps is True:
                print("Took %d rotations to arrive at new geometry." % (x))
                break

            found_overlap = False

            # Don't want atom at group_ix because it IS bonded to the original molecule and WILL overlap
            group_indices = list(range(N-n, N))
            group_indices.remove(group_ix + N - n)
            for at, g_at in itertools.product(range(N-n), group_indices):
                # If atoms are overlapped, rotate
                if self.is_overlapped(at, g_at):
                    found_overlap = True
                    # Find rotation matrix
                    rot_axis = uv
                    R = am.rotation_matrix(uv, np.pi/6) 
                    
                    # Rotate
                    group.xyz = self.xyz[N-n:]
                    old_pos = group.xyz[group_ix].copy()
                    group.xyz -= old_pos
                    group.xyz = np.dot(R, group.xyz.T)
                    group.xyz = group.xyz.T 
                    group.xyz += old_pos

                    # Replace old group with rotated group
                    self.xyz[N-n:] = group.xyz
                    
                    break

            no_overlaps = True if found_overlap is False else False

        # Finally, set self.bonds to None again because the old self.bonds is now wrong (doesn't account for new atoms)
        self.bonds = None





