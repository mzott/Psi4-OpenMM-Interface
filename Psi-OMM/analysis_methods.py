import numpy as np
from psi4.driver.qcdb import physconst


def boltzmann_weight(E, temp, unit_conversion=physconst.psi_hartree2J):
    """
    Method to return a value in its Boltzmann factor i.e.
    E -> exp[-E/kB/T]

    E : float
        Energy.
    temp : float
        Temperature in Kelvin.
    unit_conversion : float
        Conversion for energy such that it is in Joules. Psi4 provides
        many conversions. Default is Hartree to Joule.

    Returns
    -------
    Boltzmann weighted value
    """
    return np.exp(-E*unit_conversion / kb / temp)
    

