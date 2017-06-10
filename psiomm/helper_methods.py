import os
import pandas as pd
import numpy as np

"""
These are methods which are helpful for some of the internal methods, but not likely to 
be called by users.
"""

def which(program):
    """
    Method to find the path of an executable.

    Ripped from user Jay at Stack Overflow in thread
    https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python

    program : string
        Name of executable to search for in path.

    Returns
    -------
    string or None : if path is found, returns string; else, returns None
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None



def align_antechamber(xyz_string, antechamber_output):
    # Initial method written by Daniel Smith; @dgasmith
    # Read in both files
    # Skip first two lines where number of atoms and comment are
    inp = pd.read_csv(xyz_string, skiprows=2, sep=' +', engine='python', header=None)
    inp.columns = ['Atom', 'X', 'Y', 'Z']

    # Find which lines to skip - only lines with 8 columns are desired
    ant_o = open(antechamber_output, 'r')
    ant_data = ant_o.readlines()
    ant_o.close()
    bad_rows = [x for x in range(len(ant_data)) if len(ant_data[x].strip().split()) != 8]
    out = pd.read_csv(antechamber_output, skiprows=bad_rows, sep=' +', engine='python', header=None)
    out.columns = ['#', 'Sym', 'GAFF', '??', 'X', 'Y', 'Z', 'Charge?']

    # Parse out dummy atoms
    out = out[out['Sym'] != 'DUMM']

    ### Figure out mapping, easy way:
    inp_xyz = np.array(inp[['X', 'Y', 'Z']])
    out_xyz = np.array(out[['X', 'Y', 'Z']])

    inp_size = inp_xyz.shape[0]
    out_size = out_xyz.shape[0]

    # Sizes must be equal
    if inp_size != out_size:
        raise ValueError("Warning input and output sizes are not the same")

    translation_indices = np.zeros((inp_size), dtype=np.int)

    for irow in range(inp_size):
        found = 0
        idx = None
        for orow in range(out_size):

            # Careful! It looks like they round antechamer digits
            norm = np.linalg.norm(inp_xyz[irow] - out_xyz[orow])
            if norm < 1.e-3:
                found += 1
                idx = orow

        # Should only be one
        if found > 1:
            raise ValueError("More than one output index matches a input index")
        if idx is None:
            raise ValueError("Did not find a matching output index for row %d" % irow)

        translation_indices[irow] = idx

    # Reorganize antechamber output
    out = out.iloc[translation_indices]
    out.reset_index()

    # The norm of this should now be small
    inp_xyz = np.array(inp[['X', 'Y', 'Z']])
    out_xyz = np.array(out[['X', 'Y', 'Z']])

    # Maximum norm per vector
    if np.max(np.linalg.norm(inp_xyz - out_xyz, axis=1)) > 0.002:
        raise ValueError("Warning alignment mismatch")

    inp['GAFF'] = out['GAFF'].values


    #inp.to_csv('parsed_output.csv')
    return np.asarray(inp['GAFF'])

