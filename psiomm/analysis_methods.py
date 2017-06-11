import numpy as np
from psi4.driver.qcdb import physconst

"""
These are methods which are helpful for internal methods, but
users may also want to call them for analysis of their systems.
"""

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
    
def vector_angle(v1, v2):
    """
    Method to find the angle between any two vectors via

    theta = arccos( [v1 dot v2] / ||v1|| * ||v2| )

    v1, v2 : Numpy (3,) array

    Returns
    -------
    Float - angle between vectors in radians
    """
    cos_theta = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    # Clip ensures that cos_theta is within -1 to 1 by rounding say -1.000001 to -1 to fix numerical issues
    angle = np.arccos(np.clip(cos_theta, -1, 1))

    return angle

def rotation_matrix(axis, theta):
    """
    Code for Euler-Rodrigues formula taken from user unutbu from Stack Exchange:
    https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector    

    Generate the rotation matrix that will rotate vector v counterclockwise
    about axis by theta radians i.e. Rv = v' where v' is the rotated vector.

    Useful for rotating an molecule/group about a known axis.

    axis : Numpy array (3,)
        Axis to rotate around counterclockwise
    theta : Float
        RADIAN angle to rotate

    Returns
    -------
    Rotation matrix associated with rotation in 3D space.
    """
    axis = np.asarray(axis)
    axis = axis/np.linalg.norm(axis)
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

#TODO: Add method to find a rotation for a matrix onto another matrix; find_rotation rotates only single vectors
def find_rotation(a, b):
    """
    Find the rotation matrix that will rotate a onto b.
    
    Algorithm stolen from user Jur van den Berg at Stack Exchange:
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    Notation below follows his answer. 

    Useful for aligning systems that can be represented orientationally by single vectors.

    m1, m2 : Numpy arrays of size (3,)
        
    Returns
    -------
    Rotation matrix, R, that rotates m1 onto m2; Rm1 = m2

    """
    a.shape = (3,)
    b.shape = (3,)

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    
    v = np.cross(a, b)
    
    angle_AB = -1*vector_angle(a, b) 
    
    print(angle_AB)
    s = np.linalg.norm(v) * np.sin(angle_AB)
    
    c = np.dot(a, b) * np.cos(angle_AB)
    
    # Rotation matrix, R = I + Vx + Vx^2 * (1-c)/s^2
    I = np.identity(3)
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    
    R = I + Vx + np.linalg.matrix_power(Vx, 2) / (1+c)
    return R 

def SVD_rotate(m1, m2):
    """
    Method to find the rotation that takes points from m1 onto points in m2.

    Uses singular value decomposition algorithm taken from Nghia Ho,
    http://nghiaho.com/?page_id=671.

    m1, m2: Numpy (3,N) arrays where N is the number of atoms. 
        The arrays that represent geometries. The geometries must have
        the same number of atoms. 

    Returns
    -------
    The rotation matrix that will take m1 onto m2; Rm1 = m2.
    """
    assert m1.shape[0] == m2.shape[0]

    # Find the centroids of m1, m2
    centroid1 = np.mean(m1, axis=0)
    centroid2 = np.mean(m2, axis=0)

    # Build the covariance matrix
    H = np.dot((m1 - centroid1).T, (m2 - centroid2))

    U, S, V = np.linalg.svd(H)

    # Middle matrix is to ensure that matrix yields a rotation, not reflection
    R = np.dot(V.T, np.array([ [1,0,0] , [0,1,0], [0,0, np.linalg.det(np.dot(V.T,U.T))] ]) ) 
    R = np.dot(R, U.T)

    # Find translation 
    t = -np.dot(R, centroid1) + centroid2
 
    return (R, t)
    
def rmsd(a1, a2):
    """
    Return the root mean square deviation between arrays 1 and 2.

    a1, a2 : Numpy array

    Returns
    -------
    Float - root mean square deviation
    """
    # Check that the arrays are the same size
    if np.shape(a1) != np.shape(a2):
        raise Exception("""The shape of array 1 is not the same as the shape of array 2. If these arrays represent
                         geometries, the molecules are not the same!""")

    # Rotate and translate a2 so that it is as close to a1 as possible
    R, t = SVD_rotate(a1, a2)
    a2 = np.dot(R, a2.T)
    a2 = a2.T + t 

    return np.sqrt(np.mean((a1-a2)**2))

