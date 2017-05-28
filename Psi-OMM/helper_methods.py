import os
import pandas as pd
import numpy as np

def which(program):
    import os
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
    inp = pd.read_csv(xyz_string, skiprows=2, sep=' +', engine='python', header=None)
    inp.columns = ['Atom', 'X', 'Y', 'Z']

    out = pd.read_csv(antechamber_output, sep=' +', engine='python', header=None)
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

    # print translation_indices

    # Reorganize antechamer output
    out = out.iloc[translation_indices]
    out.reset_index()

    # The norm of this should now be small
    inp_xyz = np.array(inp[['X', 'Y', 'Z']])
    out_xyz = np.array(out[['X', 'Y', 'Z']])

    # Maximum norm per vector
    if np.max(np.linalg.norm(inp_xyz - out_xyz, axis=1)) > 0.002:
        raise ValueError("Warning alignment mismatch")

    inp['GAFF'] = out['GAFF'].values

    #print inp['GAFF']
    for x in range(len(inp['GAFF'])):
        print( x , ":", inp['GAFF'][x] )

    #inp.to_csv('parsed_output.csv')
    return np.asarray(inp['GAFF'])

align_antechamber('llll.xyz', 'llll.TEMP_ATS')
