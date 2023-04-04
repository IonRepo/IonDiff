---
title: 'IonDiff: command-line tool to identify diffusion events from molecular dynamics simulations'
tags:
  - Python
  - molecular dynamics
  - solid-state electrolytes
authors:
  - name: Cibrán López
    orcid: 0000-0003-3949-5058
    affiliation: "1, 2"
  - name: Riccardo Rurali
    orcid: 0000-0002-4086-4191
    affiliation: "3"
  - name: Claudio Cazorla
    orcid: 0000-0002-6501-4513
    corresponding: true
    affiliation: "1, 2"
affiliations:
 - name: Departament de Física, Universitat Politècnica de Catalunya, 08034 Barcelona, Spain.
   index: 1
 - name: Barcelona Research Center in Multiscale Science and Egineering, Universitat Politècnica de Catalunya, 08019 Barcelona, Spain.
   index: 2
 - name: Institut de Ciència de Materials de Barcelona, ICMAB-CSIC, Campus UAB, 08193 Bellaterra, Spain.
   index: 3
date: 16 March 2023
bibliography: paper.bib
---

# Summary

Molecular dynamics (MD) simulations render the positions of atoms over time inside materials. However, extracting meaningful insights from this data is often a challenge, as most common analysis rely on active supervision of the simulations and definition of arbitrary material-dependent parameters, thus frustrating high throughput screenings. In particular, to the best of our knowledge, determining the exact migrating paths of diffusive particles has not been previously addressed systematically, despite of its central role in the understanding and design of high performance solid-state electrolytes (SSE).

Here, we introduce a completely unsupervised approach for analysing ion-hopping events in MD simulations. Based on k-means clustering, our algorithm identifies with precision which and when particles diffuse in a simulation, as well as their exact migrating paths. This analysis allows for the quantification of correlations between many diffusing ions as well as of original atomistic descriptors like the duration/length of diffusion events and residence times, to cite some examples.

# IonDiff

**IonDiff** consists on a repository of fully functional Python scripts designed for extracting and analyzing the exact migrating paths of diffusive particles from MD simulations.

The repository is divided into independent functionalities:

- *identify_diffusion*: extracts the migrating paths from a given MD simulation, generating a file named **DIFFUSION** with all the necessary information in the folder containing the given simulation.
- *analyze_correlations*: analyzes the correlations between the diffusion events of a series of simulations (the **DIFFUSION** file for each of these simulations will be generated if it does not exist yet).
- *analyze_descriptors*: extracts and analyzes spatio-temporal descriptors for the diffusions of a simulation (under active development).

The minimal input needed (besides the file containing the actual atomistic trajectories) consists in a **INCAR** file with **POTIM** and **NBLOCK** flags (indicating the simulation time step and the frequency with which the configurations are recorded, respectively). After installation, all routines are easily controlled from the command line. More detailed information can be found in the documentation of the project (including specific **README**s within each folder).

The script allows graphing the identified diffusion paths for each simulated particle and provides the confidence interval associated to the results retrieved by the algorithm. An example of the analysis performed on an *ab initio* MD (AIMD) simulation based on density functional theory (DFT) is shown in \autoref{fig:diffusion-detection}. The AIMD configurations file employed in this example is available online at [@database], along with many other AIMD simulations comprehensively analyzed in a previous work [@horizons].

![Example of the performance of our unsupervised algorithm at extracting the diffusive path for one random particle of an AIMD simulation of Li\textsubscript{7}La\textsubscript{3}Zr\textsubscript{2}O\textsubscript{12} at a temperature of 400K.\label{fig:diffusion-detection}](figure.svg){width=60%}

Moreover, users may find information regarding their previous executions of the scripts in the *logs* folder, which should be used to track possible errors. Finally, a number of tests for checking out all **IonDiff** functions can be found in the *tests* folder.

Mainly, our code is based on the sklearn [@scikit] implementation of k-means clustering, although numpy [@numpy] and matplotlib [@matplotlib] are used for numerical analysis and plotting, respectively.

The current version is only able to read information from VASP [@vasp] simulations, although future releases (already under active development) will extend its scope to simulations from LAMMPS [@lammps] and other molecular dynamics (either *ab initio* or classical) software packages.

Future releases will also include libraries for analyzing these diffusions with the extraction of key descriptors and the inclusion of novel statistical analysis.

# Methods

K-means algorithm conforms spherical groups, given that, for every subgroup $G = \{G_1, G_2, \dots, G_k\}$ in a dataset, it minimizes the sum of squares:

\begin{equation}
    \sum_{i = 1}^N \min_{\boldsymbol{\mu}_j \in G} \left( \| \mathbf{x}_i - \boldsymbol{\mu}_j \|^2 \right)
\end{equation}

where $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N$ are the $N$ data points and $\boldsymbol{\mu}_j$ the mean at $G_j$.

Therefore, this approach is specially suitable for solid crystals, as the atoms tend to vibrate in spherical groups. If it was pretended to study a material which did not verify this statement, the present algorithm allows choosing other clustering schemes such as spectral clustering, which is know for creating non-spherical groups (but rather connected in terms of adjacency).

Moreover, the optimal number of clusters maximizes the average silhouette’s ratio (that measures the similarity of a point in its own cluster and its dissimilarity in comparison to the others), defined as:

\begin{equation}
    S(k) = \frac{b(k) - a(k)}{\max{(a(k), b(k))}}
\end{equation}

where:

\begin{equation}
    \begin{gathered}
        a(k) = \frac{1}{|G_{I}| - 1} \sum_{j = 1}^k \| \mathbf{x}_k - \mathbf{x}_j \|^2 \\
        b(k) = \min_{J \neq I} \frac{1}{|G_{J}|} \sum_{j = 1}^k \| \mathbf{x}_k - \mathbf{x}_j \|^2
    \end{gathered}
\end{equation}


# Acknowledgements

We acknowledge financial support from the MCIN/AEI/10.13039/501100011033 under the grants PID2020-119777GB-I00, PID2020-112975GB-I00 and TED2021-130265B-C22, the “Ramón y Cajal” fellowship RYC2018-024947-I, the Severo Ochoa Centres of Excellence Program (CEX2019-000917-S), and the Generalitat de Catalunya under Grant No.2017SGR1506.

# References
