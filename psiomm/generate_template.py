from simtk.openmm.app import forcefield

def generateTemplate(forcefield, topology, molecule):
    [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(topology)
    if templates:
        for index, atom in enumerate(templates[0].atoms):
            atom.type = molecule.atom_types[index]
            atom.parameters = {'charge':molecule.charges[index]}
        forcefield.registerResidueTemplate(templates[0])
    return forcefield
