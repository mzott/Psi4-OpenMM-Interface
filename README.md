Psi4-OpenMM Interface
=====================
Minimal interface between Psi4 and OpenMM that allows passing of systems between each program. Additionally, some helpful tools are included such as spherical addition of solvent (because adding solvent other than water doesn't work with addSolvent in OpenMM), calculation of bond lengths, angles, dihedrals, and unique conformation searching. 

Example Usages
==============
Accounting for solvent interactions is critical in translating computational results to experimental data where reactions were carried out in solvent. In order to account for solvent dynamically, a molecular mechanics simulation could be carried out with OpenMM and various solvent conformations could be found. When the single point energy is calculated with quantum mechanical methods (probably DFT if you have many solvent molecules), especially stable (important) conformations can be identified. An example implementation of such a workflow is found [here](https://github.com/mzott/Psi4-OpenMM-Interface/blob/master/tests/workflow_model.py). 

Here, a Shimizu torsion balance is modeled in chloroform. One of the most stable configurations for a single chloroform molecule and the torsion balance per B3LYP is represented as solvent accessible surfaces using VMD to make visually apparent the close fit of the chloroform into the torsion balance. This application has been represented in the Psi4 1.1 paper [doi: 10.1021/acs.jctc.7b00174](http://dx.doi.org/10.1021/acs.jctc.7b00174) and Georgia Tech news as well in [an article about Google's quantum chemistry project](https://www.cos.gatech.edu/hg/item/598564 "GT CoS News").

<p align="center">

<br>
<img src="https://github.com/mzott/Psi4-OpenMM-Interface/blob/master/media/open_1_minE_surface.png" alt="Torsion balance with chloroform" height=400> <br>
</p>

How to Install
==============
Using this package is very easy thanks to the capabilities provided by both Github and Conda. "Installing" this package is (hopefully) simple. Before any of the following steps, if you do not have Conda, please install it [here](https://www.anaconda.com/download/ "Conda"). Although Conda is not necessary to use this interface, it greatly simplifies the installation of all of the dependencies. Advanced users can choose to simply clone this repo (steps 1. and 2.) and add the ` psiomm ` directory to their ` PYTHONPATH ` and install the other programs as they wish.

1. Get a copy of this repo: 

    ` git clone https://github.com/mzott/Psi4-OpenMM-Interface.git `

2. Add the ` psiomm ` directory to your ` PYTHONPATH `:
* Bash: 

    ` export PYTHONPATH="/path/to/location/of/interface/Psi4-OpenMM-Interface:$PYTHONPATH" `
* tcsh: 

    ` setenv PYTHONPATH "/path/to/location/of/interface/Psi4-OpenMM-Interface:$PYTHONPATH" `
 
3. Install Psi4 and OpenMM if you do not have them already (only Conda approach presented here):
* ` conda install -c psi4 psi4 `
* ` conda install -c omnia openmm `

4. Install additional dependencies:
* Antechamber is used to calculate atom types automatically; if you have AmberTools installed, you should have Antechamber already and this step is unnecessary. If you need Antechamber:

    ` conda install -c omnia ambermini=16.16.0 `
* OpenBabel is also used to help switch between file formats:

    ` conda install -c openbabel openbabel `
* Pandas is currently used as well:

    `  conda install -c anaconda pandas `

5. [Let me know](mailto:mzott3@gatech.edu) if you are using this interface or have any questions!

Additional Resources
====================
* **Psi4 website**  www.psicode.org

* **OpenMM website** http://openmm.org/

* **Psi4 Github** https://github.com/psi4/psi4 

* **OpenMM Github** https://github.com/pandegroup/openmm

* **Psi4 Manual**  http://psicode.org/psi4manual/master/index.html

* **OpenMM Manual** http://docs.openmm.org/7.0.0/userguide/index.html

* **Psi4 Tutorial** http://psicode.org/psi4manual/master/tutorial.html

* **OpenMM Tutorial** http://docs.openmm.org/7.0.0/userguide/library.html#openmm-tutorials
