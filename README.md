# IonDiff

<img src=docs/logo.svg width="20%">

Despite playing a central role in the design of high performance solid-state electrolytes, little is known about the processes governing ionic diffusion in these materials and the spatio-temporal correlations acting on migrating particles. Computer simulations can reproduce the trajectories of individual diffusing ions in real time with extraordinary accuracy, thus providing incredibly valuable atomistic data that in practice cannot be resolved by experiments.

However, the identification of hopping events in computer simulations typically relies on active supervision and definition of arbitrary material-dependent geometrical parameters, thus frustrating high throughput screenings of diffusing paths and mechanisms across simulation databases and the assessment of many-diffusing-ion correlations.   

Here, we introduce a novel approach for analysing ion hopping events in molecular dynamics (MD) simulations in a facile and totally unsupervised manner, what would allow the extraction of completely new descriptors related to these diffusions. Our approach relies on the k-means clustering algorithm and allows to identify with precision which and when particles diffuse in a simulation and the exact migrating paths that they follow as well.

Documentation showing functionality and usage of the code are provided one the [`docs`](https://pypi.org/project/IonDiff/) site. Please be aware that the code is under active development, bug reports are welcomed in the GitHub issues!

## Installation

IonDiff can be installed from PyPI:

```bash
pip3 install IonDiff
```

or used directly from source without explicit installation:

```bash
git clone https://github.com/IonRepo/IonDiff.git
cd IonDiff
pip3 install -r docs/requirements.txt
```

## Execution

To extract the diffusion paths from a **XDATCAR** simulation file (with its corresponding **INCAR** file) located at *examples* folder, from the IonDiff folder run:

```bash
python3 cli.py identify_diffusion --MD_path examples
```

To analyze temporal correlations among the diffusions of different simulations, from the IonDiff folder run:

```bash
python3 cli.py analyze_correlations
```

and to extract atomistic descriptors from the simulation and diffusion events run:

```bash
python3 cli.py analyze_descriptors
```

where it has to be provided a file named **DIFFUSION_paths**, as in *examples* folder, for which each line represents the relative path to a simulation folder which is to be considered, name of the compound, its stoichiometricity/polymorf and the temperature of simulation. Each folder must contain a **XDATCAR** simulation file (with its corresponding **INCAR** file). 

An *ab initio* MD simulation based on density functional theory of non-stoichiometric Li<sub>7</sub>La<sub>3</sub>Zr<sub>2</sub>O<sub>12</sub> (LLZO) fast-ion conductor at a temperature of 400K is provided to run as an example:
 - [`examples/INCAR`](examples/INCAR): Basic parameters of the simulation (only **POTIM** and **NBLOCK** flags are considered).
 - [`examples/XDATCAR`](examples/XDATCAR): Concatenation of all simulated configurations (recorded each **NBLOCK** simulation steps).
 - [`examples/README.md`](examples/README.md): More specific information regarding these files.

## Authors

IonDiff is being developed by:

 - Cibrán López
 - Riccardo Rurali
 - Claudio Cazorla

## Contact, questions and contributing

If you have questions, please don't hesitate to reach out at: cibran.lopez@upc.edu
