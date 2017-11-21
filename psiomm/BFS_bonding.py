from __future__ import absolute_import
from __future__ import print_function
import math
from psi4.driver.qcdb.physconst import *
from psi4.driver.qcdb.cov_radii import *

"""
This code is primarily written by Trent M. Parker
"""

#TODO: implement penalty scores in bond algorithm
#TODO: enable radicals/charged species to have bonds determined correctly
#TODO: eliminate duplication of code; align code to rest of code base

BOND_FACTOR = 1.2  # fudge factor for bond length threshold

_expected_bonds = {
    'H': 1,
    'B': 3,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'AL':3,
    'SI':4,
    'P': 3,
    'S': 2,
    'CL':1
    }

def dist(self, atom1, atom2):
    x = self.xyz[atom1][0] - self.xyz[atom2][0]
    y = self.xyz[atom1][1] - self.xyz[atom2][1]
    z = self.xyz[atom1][2] - self.xyz[atom2][2]
    return math.sqrt(x*x + y*y + z*z)


def missing_bonds(bonds, bond_tree, at_types):
    """Determine number of bonds missing for each atom"""
    n_missing = []
    for i in range(len(at_types)):
        n_bonds_i = 0
        for p in range(len(bonds)):
            at1 = bonds[p][0]
            at2 = bonds[p][1]
            if (at1 == i or at2 == i):
                bond_order = bonds[p][2]
                n_bonds_i += bond_order
        n_expect_i = _expected_bonds[at_types[i]]
        n_missing.append(n_expect_i - n_bonds_i)

    return n_missing


def missing_neighbors(bond_tree, n_missing):
    """Determine number of neighboring atoms missing bonds for each atom"""
    missing_neighbors = []
    for i in range(len(bond_tree)):
        N_neighbors = len(bond_tree[i])
        missing = 0
        for a in range(N_neighbors):
            j = bond_tree[i][a]
            if n_missing[j] > 0:
                missing += 1
        missing_neighbors.append(missing)
    return missing_neighbors


def bond_profile(self):
    """Obtain bonding topology of molecule"""
    
    # determine bond topology from covalent radii
    bonds = []
    for i in range(self.natoms()):
        for j in range(i + 1, self.natoms()):
            # below line used to be multipled by * psi_bohr2angstroms
            bdist = dist(self, i, j) 
            bonded_dist = BOND_FACTOR * (psi_cov_radii[self.symbol(i)] + psi_cov_radii[self.symbol(j)])
            if bonded_dist > bdist:
                bonds.append([i, j, 1])

    # determine bond order from number of bonds
    N_atoms = self.natoms()
    N_bonds = len(bonds)
    at_types = [self.symbol(i) for i in range(self.natoms())]
    bond_tree = [[] for i in range(N_atoms)]
    for i in range(N_bonds):
        at1 = bonds[i][0]
        at2 = bonds[i][1]
        bond_tree[at1].append(at2)
        bond_tree[at2].append(at1)

    # determine bond order for all bonds from bond tree and element types
    n_missing = missing_bonds(bonds, bond_tree, at_types)
    n_neighbors_missing = missing_neighbors(bond_tree, n_missing)

    # add double / triple bonds if only one neighbor missing bonds
    N_left = math.floor(sum(n_missing) / 2)
    N_left_previous = N_left + 1
    N_iter = 0
    while N_left > 0:
        N_iter += 1

        if N_left == N_left_previous:
            neighbor_min += 1
        else:
            neighbor_min = 1

        N_left_previous = N_left

        # add a multiple bond to a deficient atom with the fewest number of deficient neighbors
        BREAK_LOOP = False
        for i in range(N_atoms):
            if n_missing[i] > 0 and n_neighbors_missing[i] == neighbor_min:
                N_neighbors = len(bond_tree[i])
                for a in range(N_neighbors):
                    j = bond_tree[i][a]
                    if n_missing[j] > 0:
                        for p in range(N_bonds):
                            at1 = bonds[p][0]
                            at2 = bonds[p][1]
                            if (at1 == i and at2 == j) or (at1 == j and at2 == i):
                                bonds[p][2] += 1
                                n_missing[i] += -1
                                n_missing[j] += -1
                                n_neighbors_missing[i] += -1
                                n_neighbors_missing[j] += -1
                                N_left = math.floor(sum(n_missing) / 2)
                                BREAK_LOOP = True
                    if BREAK_LOOP:
                        break
            if BREAK_LOOP:
                break

        # recalculate incomplete bond topology
        n_missing = missing_bonds(bonds, bond_tree, at_types)
        n_neighbors_missing = missing_neighbors(bond_tree, n_missing)

        # break cycle if takes more than given number of iterations
        max_iter = 100
        if N_iter > max_iter:
            print("""Error: multiple bond determination not complete""")
            print("""  %i bonds unaccounted for""" % (N_left))
            break

    # bond order is number of bonds between each bonded atom pair
    bond_order = []
    for p in range(N_bonds):
        bond_order.append(bonds[p][2])
    for p in range(len(bond_order)):
        bonds[p][2] = bond_order[p]

    return [bonds, bond_tree]
