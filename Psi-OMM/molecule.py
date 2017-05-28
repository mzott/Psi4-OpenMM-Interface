from __future__ import absolute_import
from __future__ import print_function
import os
import subprocess
import numpy as np
import psi4
from psi4.driver.qcdb import periodictable
from psi4.driver.qcdb import physconst
import BFS_bonding
import helper_methods as hm

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

        # if the units are not in angstrom, convert nanometers OR Bohr to Angstrom
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

            z_vals.append(periodictable.el2z[Zxyz_list[0]])
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
        print(self.bonds)
    
    def to_xyz_string(self):
        """
        Create an XYZ string with this molecule. Useful for passing to
        Psi4 via psi4.geometry or for writing XYZ files.

        Returns
        -------
        string
        """
        output = '' 
        for atom in range(self.natoms()):
            output += '\n' if atom != 0 else ''
            output += "%3s  " % self.symbol(atom)
            output += '   '.join("% 10.6f" % x for x in self.xyz[atom])
        return output

    def to_xyz_file(self, filename, comment=None):
        """
        Create an XYZ file with this molecule.
        
        filename : string
            Name of the new XYZ file.
        comment : string
            String to be placed on second line of XYZ file. 
        """
        output = str(self.natoms())
        output += ('\n'+comment if comment is not None else '\n #Generated by GT')
        output += '\n' + self.to_xyz_string()
        
        f = open(filename, 'w')
        f.write(output)
        f.close()

    """TODO add mol2 file writer. Probably exists in qcdb of Psi4."""

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

    def generate_atom_types(self, temp_filename, babel_path=None, antechamber_path=None):
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

    def angle(self, atom1, atom2):
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

        return np.arccos( np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2) )

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


