IonDiff
=======

.. image:: logo.svg
   :width: 20%

Despite playing a central role in the design of high performance solid-state electrolytes, little is known about the processes governing ionic diffusion in these materials and the spatio-temporal correlations acting on migrating particles. Computer simulations can reproduce the trajectories of individual diffusing ions in real time with extraordinary accuracy, thus providing incredibly valuable atomistic data that in practice cannot be resolved by experiments.

However, the identification of hopping events in computer simulations typically relies on active supervision and definition of arbitrary material-dependent geometrical parameters, thus frustrating high throughput screenings of diffusing paths and mechanisms across simulation databases and the assessment of many-diffusing-ion correlations.

Here, we introduce a novel approach for analysing ion hopping events in molecular dynamics (MD) simulations in a facile and totally unsupervised manner, which allows the extraction of completely new descriptors related to these diffusions. Our approach relies on the k-means clustering algorithm and allows to identify with precision which and when particles diffuse in a simulation and the exact migrating paths that they follow as well.

Please be aware that the code is under active development, bug reports are welcomed in the GitHub issues!

Installation
------------

IonDiff can be installed from PyPI::

    pip3 install IonDiff

or directly from source::

    git clone https://github.com/IonRepo/IonDiff.git
    cd IonDiff
    pip3 install -r requirements.txt

Execution
---------

To extract the diffusion paths from a **XDATCAR** simulation file (with its corresponding **INCAR** file) located in the *examples* folder, from the IonDiff folder run::

    python3 cli.py identify_diffusion --MD_path examples

To analyze temporal correlations among the diffusions of different simulations, from the IonDiff folder run::

    python3 cli.py analyze_correlations

and to extract atomistic descriptors from the simulation and diffusion events run::

    python3 cli.py analyze_descriptors

where it has to be provided a file named **DIFFUSION_paths**, as in the *examples* folder, for which each line represents the relative path to a simulation folder which is to be considered, name of the compound, its stoichiometricity/polymorph and the temperature of the simulation. Each folder must contain a **XDATCAR** simulation file (with its corresponding **INCAR** file).

An *ab initio* MD simulation based on density functional theory of non-stoichiometric Li\ :sub:`7`\ La\ :sub:`3`\ Zr\ :sub:`2`\ O\ :sub:`12`\ (LLZO) fast-ion conductor at a temperature of 400K is provided to run as an example:

- `examples/INCAR <examples/INCAR>`_: Basic parameters of the simulation (only **POTIM** and **NBLOCK** flags are considered).
- `examples/XDATCAR <examples/XDATCAR>`_: Concatenation of all simulated configurations (recorded each **NBLOCK** simulation steps).
- `examples/README.md <examples/README.md>`_: More specific information regarding these files.

Authors
-------

IonDiff is being developed by:

- Cibrán López
- Riccardo Rurali
- Claudio Cazorla

Contact, questions and contributing
-----------------------------------

If you have questions, please don't hesitate to reach out at: cibran.lopez@upc.edu
