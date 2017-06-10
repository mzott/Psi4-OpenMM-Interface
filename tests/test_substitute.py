import numpy as np

from psiomm import molecule
from psiomm import psi_omm as po

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


benzene = """
  H      1.2194     -0.1652      2.1600
  C      0.6825     -0.0924      1.2087
  C     -0.7075     -0.0352      1.1973
  H     -1.2644     -0.0630      2.1393
  C     -1.3898      0.0572     -0.0114
  H     -2.4836      0.1021     -0.0204
  C     -0.6824      0.0925     -1.2088
  H     -1.2194      0.1652     -2.1599
  C      0.7075      0.0352     -1.1973
  H      1.2641      0.0628     -2.1395
  C      1.3899     -0.0572      0.0114
  H      2.4836 -0.1022 0.0205
"""

hydroxy_str = """
O 0.0 0.0 0.0
H 0.96 0.05 0.05
"""

methyl_str = """
C      0.0000      0.0000      0.0000
H      0.2051      0.8240     -0.6786
H -1.0685 -0.0537 0.1921
H      0.3345     -0.9314     -0.4496
"""

ether_str = """
  H      2.8816      0.0336      1.6217
  C      2.2891     -0.0221      0.7000
  H      2.6101      0.8010      0.0481
  H      2.5511     -0.9598      0.1926
  C      0.8085      0.0537      1.0218
  H      0.4993     -0.7786      1.6846
  H      0.5596      0.9983      1.5447
  O      0.1076     -0.0194     -0.2103
  C     -1.3023      0.0395     -0.0564
  H     -1.6493     -0.7929      0.5873
  H     -1.5903      0.9839      0.4464
  C     -1.9085     -0.0506     -1.4441
  H     -1.5780      0.7728     -2.0907
  H     -1.6404     -0.9880     -1.9487
H -3.0032 -0.0070 -1.3841
"""

hydroxy = molecule.Molecule.from_xyz_string(hydroxy_str)
methyl = molecule.Molecule.from_xyz_string(methyl_str)
ether = molecule.Molecule.from_xyz_string(ether_str)


mol1 = molecule.Molecule.from_xyz_string(benzene)
mol2 = molecule.Molecule.from_xyz_string(benzene)
mol3 = molecule.Molecule.from_xyz_string(benzene)

mol1.substitute(3, hydroxy, group_ix=0)
mol2.substitute(3, methyl, group_ix=0)
mol3.substitute(3, ether, group_ix=7)

mol1 = po.find_mm_min(mol1)

mol1.to_xyz_file('hydroxylated.xyz')
mol2.to_xyz_file('methylated.xyz')
mol3.to_xyz_file('ether_substituted.xyz')


