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

Moreover, this implementation introduces an optimized code for computing the full ion diffusion coefficient, this is, considering cross terms in the mean-squared-displacements, above the dilute limit approximation.

# IonDiff

Fast-ion conductors consist on a new family of SSE, in which some particles diffuse, generation an electronic drift which allows storing energy efficiently. However, although DFT simulations accurately describe the behaviour of these materials, there are not exhaustive tools for analyzing the atomic behaviour in SSE, thus highly underutilizing this simulations. **IonDiff** efficiently addresses this challenge, providinga a repository of fully functional Python scripts designed for extracting and analyzing the exact migrating paths of diffusive particles, which are tracked from MD simulations.

The repository is divided into independent functionalities:

- *identify_diffusion*: extraction of the migrating paths from a given MD simulation. It generates in the folder containing the given simulation a file named **DIFFUSION**, with all the necessary atomistic information for next analysis of the diffusion events than have been found.
- *analyze_correlations*: analysis of correlations between diffusion events from a series of simulations (the **DIFFUSION** file for each of these simulations will be generated if it does not exist yet).
- *analyze_descriptors*: extraction and analysis of spatio-temporal descriptors from diffusion events of a simulation. Within this library, we implemented an optimized approach for computing the full ionic diffusion coefficient (which includes cross correlations, proven to exist in previous works (our work, and others)).

The (full) ionic diffusion coefficient consists on two parts [@kozinsky, @tateyama]: the first considering the mean-squared-displacement (MSD) of a particle with itself (MSD_{self}), and this with cross terms (MSD_{distinct}). However less accurate, the ionic diffusion coefficient is usually approximated to only consider the *self* part given that it was much faster in previous implementations. Recently, correlations among different atoms have been proved to be present [@arxiv], thus undermining the reliability of this approach. Therefore, on the contrary, we present a novel implementation of the full ionic diffusion coefficient which outperforms previous codes, exploiting the matricial representation of this calculation. The time required for computing *self* or *distinct* parts here is roughly the same. 

The minimal input needed (besides the file containing the actual atomistic trajectories) consists in a **INCAR** file with **POTIM** and **NBLOCK** flags (indicating the simulation time step and the frequency with which the configurations are recorded, respectively). After installation, all routines are easily controlled from command line. More detailed information can be found in the documentation of the project (including specific **README**s within each folder).

The script allows graphing the identified diffusion paths for each simulated particle and provides the confidence interval associated to the results retrieved by the algorithm. An example of the analysis performed on an *ab initio* MD (AIMD) simulation based on density functional theory (DFT) is shown in \autoref{fig:diffusion-detection}. The AIMD configurations file employed in this example is available online at [@database], along with many other AIMD simulations comprehensively analyzed in a previous work [@horizons].

![Example of the performance of our unsupervised algorithm at extracting the diffusive path for one random particle of an AIMD simulation of Li\textsubscript{7}La\textsubscript{3}Zr\textsubscript{2}O\textsubscript{12} at a temperature of 400K.\label{fig:diffusion-detection}](figure.svg){width=60%}

Moreover, users may find information regarding their previous executions of the scripts in the *logs* folder, which should be used to track possible errors on data format and more. Finally, a number of tests for checking out all **IonDiff** functions can be found in the *tests* folder.

Mainly, our code is based on the sklearn [@scikit] implementation of k-means clustering, although numpy [@numpy] and matplotlib [@matplotlib] are used for numerical analysis and plotting, respectively.

The current version reads information from VASP [@vasp] simulations, although future releases (already under active development) will extend its scope to simulations from other molecular dynamics software packages (either *ab initio* or classical).

# Methods

## Machine learning outline

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
        a(k) = \frac{1}{|G_{I}| - 1} \sum_{j = 1}^k \| \mathbf{x}_k - \mathbf{x}_j \|^2  \\
        b(k) = \min_{J \neq I} \frac{1}{|G_{J}|} \sum_{j = 1}^k \| \mathbf{x}_k - \mathbf{x}_j \|^2
    \end{gathered}
\end{equation}

## Ionic conductivity

The ionic conductivity ($\sigma$) computes from [@tateyama]:

\begin{equation}
    \begin{gathered}
        \sigma = \lim_{\Delta t \to \infty} \frac{e^2}{2 d V k_B T} \left[ \sum_i z_i^2 \langle \left[ x_i(t_0 + \Delta t) - x_i(t_0) \right]^2 \rangle_{t_0} + \right. \\
        \left. + \sum_{i, j \neq i} z_i z_j \langle \left[ x_i(t_0 + \Delta t) - x_i(t_0) \right] \cdot \left[ x_j(t_0 + \Delta t) - x_j(t_0) \right] \rangle_{t_0} \right]
    \end{gathered}
\end{equation}

where $e$, $V$, $k_B$, and $T$ are the elementary charge, system volume, Boltzmann constant, and temperature of the MD simulation, respectively. The $z_i$ and $r_i$ are the charge and position (in cartesian coordinates) of particle $i$ and $d$ is the dimension of $r_i$. $\Delta t$ is the time window and $t_0$ the temporal offset of $\Delta t$. Thus, for those simulations in which one only species diffusses, the ionic diffusion coefficient reads: 

\begin{equation}
    \begin{gathered}
        D = \lim_{\Delta t \to \infty} \frac{1}{6 \Delta t} \left[ \sum_i \langle \left[ x_i(t_0 + \Delta t) - x_i(t_0) \right]^2 \rangle_{t_0} + \right. \\
        \left. + \sum_{i, j \neq i} \langle \left[ x_i(t_0 + \Delta t) - x_i(t_0) \right] \cdot \left[ x_j(t_0 + \Delta t) - x_j(t_0) \right] \rangle_{t_0} \right] = \right. \\
        \left. = \lim_{\Delta t \to \infty} \frac{1}{6 \Delta t} \left[ \Delta r_{self} (\Delta t) + \Delta r_{distinc} (\Delta t) \right]
    \end{gathered}
\end{equation}

As a result, all these displacements can be computed just once and stored in a three-dimensional tensor, what allows simple vectorization and runs much faster in libraries such as Numpy compared to traditional loops. Then, for a simulation of $n_{atoms}$ number of atoms for the diffusive species and $\tau$ temporal duration, we only need to compute:

\begin{equation}
    M (\Delta t, p_i, d) = \frac{1}{t_{sim} - \Delta t} \sum_{t_0 = 0}^{t_{sim} - \Delta t - \tau} \left[ r (t_0 + \Delta t, p_i, d) - r (t_0, p_i, d) \right]
\end{equation}

being $M(\Delta t, p_i, d)$ a three dimensional tensor of shape $N_t \times N_t \times N_p$ storing all mean displacements of temporal length $\Delta t$ for particle $p_i$ in catersian dimension $d$. This leads to:

\begin{equation}
    \Delta r_{self} (\Delta t) = \frac{1}{n_{atoms}} \sum_{d} \sum_{i = 1}^{n_{atoms}} M (\Delta t, p_i, d) \cdot M (\Delta t, p_i, d)
\end{equation}

\begin{equation}
    \Delta r_{distinc} (\Delta t) = \frac{1}{n_{atoms} (n_{atoms}-1)} \sum_{d} \sum_{i = 1}^{n_{atoms}} \sum_{j = i+1}^{n_{atoms}} M (\Delta t, p_i, d) \cdot M (\Delta t, p_j, d)
\end{equation}

Note that we keep $D_{self}$ and $D_{distinct}$ separate as this allows easily analising the contribution of crossed terms to $D$ without adding any code complication.

# Acknowledgements

We acknowledge financial support from the MCIN/AEI/10.13039/501100011033 under the grants PID2020-119777GB-I00, PID2020-112975GB-I00 and TED2021-130265B-C22, the “Ramón y Cajal” fellowship RYC2018-024947-I, the Severo Ochoa Centres of Excellence Program (CEX2019-000917-S), and the Generalitat de Catalunya under Grant No.2017SGR1506.

# References
