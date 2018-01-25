import numpy as np
from psiomm import molecule
from psiomm import psi_omm as po

TEST_BONDS = [ [[0, 1, 1], [1, 2, 2], [1, 9, 1], [2, 3, 1], [2, 4, 1],
               [4, 5, 1], [4, 6, 2], [6, 7, 1], [6, 8, 1], [8, 9, 2],
               [8, 11, 1], [9, 10, 1], [11, 12, 1]] ,
               [[1], [0, 2, 9], [1, 3, 4], [2], [2, 5, 6], [4],
               [4, 7, 8], [6], [6, 9, 11], [1, 8, 10], [9], [8, 12], [11]] ]

TEST_ATOM_TYPES = ['ha', 'ca', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha',
                   'ca', 'ca', 'ha', 'oh', 'ho']

# charges with Mulliken protocol, SCF/STO-3G
TEST_CHARGES = [ 0.06315342, -0.04909911, -0.07689912,  0.05832837,
                 -0.05027252,  0.06357228, -0.08233831,  0.07057471,
                 0.13355706, -0.10097644,  0.05400022, -0.29912363, 0.21552307]

test_phenol = """ H      1.7344      0.1731      2.1387
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
test_mol = molecule.Molecule.from_xyz_string(test_phenol)

# Assign bonds, atom types, charges (using Psi4)
test_mol.generate_bonds()
test_mol.generate_atom_types()
test_mol.generate_charges()

# Test if generated and reference data are equal/nearly equal
if test_mol.bonds != TEST_BONDS:
    raise Exception("Bonds no longer match expected values")

if (test_mol.atom_types == TEST_ATOM_TYPES).all() is False:
    raise Exception("Atom types no longer match expected values")

if np.allclose(np.asarray(test_mol.charges), np.asarray(TEST_CHARGES), atol=.001) is False:
    raise Exception("Bonds no longer match expected values")
