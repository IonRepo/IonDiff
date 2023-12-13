---
title: 'IonDiff: command-line tool to identify diffusion events from molecular dynamics simulations'
tags:
  - Python
  - Molecular dynamics
  - Solid-state electrolytes
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
date: 16 December 2023
bibliography: paper.bib
---

# Summary

Molecular dynamics (MD) simulations of superionic materials render the trajectories of the atoms conforming them. However, extracting meaningful insights from this data is often a challenge since most common analysis rely on active supervision of the simulations and definition of arbitrary material-dependent parameters, thus frustrating high throughput screenings. In particular, to the best of our knowledge, determining exact ionic migrating paths and the level of coordination between mobile particles in diffusive events have not been previously addressed in a systematic and quantitative manner, despite of its central role in the understanding and design of high performance solid-state electrolytes. Here, we introduce a completely unsupervised approach for analysing ion-hopping events in MD simulations. Based on k-means clustering, our algorithm identifies with precision which and when particles diffuse during a simulation, thus identifying their exact migrating paths. This analysis allows also for the quantification of correlations between many diffusing ions as well as of key atomistic descriptors like the duration/length of diffusion events and residence times, to cite some examples. Moreover, the present implementation introduces an optimized code for computing the full ion diffusion coefficient, that is, entirely considering ionic correlations, thus going beyond the dilute limit approximation.

# IonDiff

Fast-ion conductors (FIC) are materials in which some of their constituent  atoms diffuse with large drift velocities comparable to those found in liquids. FIC are the pillars of many energy conversion and storage technologies like solid-state electrochemical batteries and fuel cells.   Molecular dynamics (MD) simulations is a computational method that employs Newton's laws to evaluate the trajectory of ions in complex atomic and molecular systems. MD simulations of FIC are highly valuable since can accurately describe the diffusion and vibration of the ions conforming them. Nevertheless, there are not versatile tools at hand for analyzing the outputs of FIC MD simulations in an unsupervised and materials-independent manner, thus frustrating the fundamental understanding and possible rational design of FIC.

 **IonDiff** efficiently addresses the challenge described above by implementing unsupervised machine learning approaches in a repository of Python scripts designed to extract the exact migrating paths of diffusive particles from MD simulations, along with other physically relevant quantities like the degree of correlation between diffusive ions, ionic residence times in metastable positions and the length and duration of ionic hops.

The repository is divided into three independent functionalities:

- *identify_diffusion*: extraction of the migrating paths from a given MD simulation. It generates a **DIFFUSION** file in the folder containing the inputs and outputs of the MD simulation. This file contains all the necessary atomistic information for the following analysis of ionic diffusion events.
- *analyze_correlations*: analysis of the correlations between ionic diffusion events extracted from a series of MD simulations (the **DIFFUSION** file for each of these simulations will be generated if it does not exist yet).
- *analyze_descriptors*: extraction and analysis of spatio-temporal descriptors involving the ionic diffusion events identified in the MD simulations. In this library, an optimized approach for computing the full ionic diffusion coefficient (i.e., including ionic cross correlations, proven to be non-negligible in FIC [@kozinsky; @tateyama; @arxiv]) is implemented.

The (full) ionic diffusion coefficient consists on two parts [@kozinsky; @tateyama], one that involves the mean-square displacement of a particle with itself (MSD$_{self}$) and another the mean-square displacement of a particle with all others (MSD$_{distinct}$). Typically, the distinct part of the MSD is neglected in order to accelerate the estimation and convergence of diffusion coefficients. However, many-ions correlations have been recently demonstrated to be essential in FIC [@arxiv] hence should not be disregarded in practice. IonDiff provides a novel implementation of the full ionic diffusion coefficient which outperforms previous codes, exploiting the matricial representation of this calculation. The time required for computing the *self* or *distinct* parts of the diffusion coefficient roughly are the same here.

The minimal input needed (besides the file containing the actual atomistic trajectories) consists in an **INCAR** file with **POTIM** and **NBLOCK** flags (indicating the simulation time step and the frequency with which the configurations are written, respectively). After installation, all routines are easily controlled from the command line. More detailed information can be found in the documentation of the project (including specific **README**s within each folder).

The script allows graphing the identified diffusion paths for each simulated particle and provides the confidence interval associated to the results retrieved by the algorithm. An example of the analysis performed on an *ab initio* MD (AIMD) simulation based on density functional theory (DFT) is shown in \autoref{fig:diffusion-detection}. The AIMD configurations file employed in this example is available online at [@database], along with many other AIMD simulations comprehensively analyzed in a previous work [@horizons].

![Example of the performance of our unsupervised algorithm at extracting the diffusive path for one random particle of an AIMD simulation of Li\textsubscript{7}La\textsubscript{3}Zr\textsubscript{2}O\textsubscript{12} at a temperature of 400K.\label{fig:diffusion-detection}](figure.svg){width=60%}

Moreover, users may find information regarding their previous executions of the scripts in the *logs* folder, which should be used to track possible errors on the data format and more. Finally, a number of tests for checking out all **IonDiff** functions can be found in the *tests* folder.

Mainly, our code is based on the sklearn [@scikit] implementation of k-means clustering, although numpy [@numpy] and matplotlib [@matplotlib] are used for numerical analysis and plotting, respectively. The current version reads information from VASP [@vasp] simulations, although future releases (already under active development) will extend its scope to simulation data obtained from other quantum and classical molecular dynamics packages.

# Methods

## Machine learning outline

K-means algorithm conforms spherical groups given that, for every subgroup $G = \{G_1, G_2, \dots, G_k\}$ in a dataset, it minimizes the sum of squares:

\begin{equation}
    \sum_{i = 1}^N \min_{\boldsymbol{\mu}_j \in G} \left( \| \mathbf{x}_i - \boldsymbol{\mu}_j \|^2 \right)
\end{equation}

where $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N$ are the $N$ data points and $\boldsymbol{\mu}_j$ the mean at $G_j$.

Therefore, this approach is particularly well suited for crystals since the atoms tend to fluctuate isotropically around their equilibrium positions. For the study of materials in which atoms vibrate strongly anisotropically, the present algorithm also allows choosing other clustering schemes like spectral clustering, which is known to perform well for cases in which group adjacency is relevant.

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

The ionic conductivity ($\sigma$) is computed like [@tateyama]:

\begin{equation}
    \begin{gathered}
        \sigma = \lim_{\Delta t \to \infty} \frac{e^2}{2 n_d V k_B T} \left[ \sum_i z_i^2 \langle \left[ \mathbf{r}_i(t_0 + \Delta t) - \mathbf{r}_i(t_0) \right]^2 \rangle_{t_0} + \right. \\
        \left. + \sum_{i, j \neq i} z_i z_j \langle \left[ \mathbf{r}_i(t_0 + \Delta t) - \mathbf{r}_i(t_0) \right] \cdot \left[ \mathbf{r}_j(t_0 + \Delta t) - \mathbf{r}_j(t_0) \right] \rangle_{t_0} \right]
    \end{gathered}
\end{equation}

where $e$, $V$, $k_B$, and $T$ are the elementary charge, system volume, Boltzmann constant, and temperature of the MD simulation, respectively, $z_i$ the ionic charge and $\mathbf{r}_i = x_{1i} \hat{i} + x_{2i} \hat{j} + x_{3i} \hat{k}$ the cartesian position of particle $i$, $n_d$ the number of spatial dimensions, $\Delta t$ the time window, and $t_0$ the temporal offset of $\Delta t$. Thus, for those simulations in which only one atomic species diffusses, the ionic diffusion coefficient reads: 

\begin{equation}
    \begin{gathered}
        D = \lim_{\Delta t \to \infty} \frac{1}{6 \Delta t} \left[ \sum_i \langle \left[ \mathbf{r}_i(t_0 + \Delta t) - \mathbf{r}_i(t_0) \right]^2 \rangle_{t_0} + \right. \\
        \left. + \sum_{i, j \neq i} \langle \left[ \mathbf{r}_i(t_0 + \Delta t) - \mathbf{r}_i(t_0) \right] \cdot \left[ \mathbf{r}_j(t_0 + \Delta t) - \mathbf{r}_j(t_0) \right] \rangle_{t_0} \right] = \\
        = \lim_{\Delta t \to \infty} \frac{1}{6 \Delta t} \left[ \text{MSD}_{self} (\Delta t) + \text{MSD}_{distinct} (\Delta t) \right]
    \end{gathered}
\end{equation}

All the ionic displacements appearing in Eq. (5) can be computed just once and stored in a four-dimensional tensor thus allowing for simple vectorization and very much fast processing with python libraries (e.g., Numpy) as compared to traditional calculation loops. Then, for a simulation with $n_t$ time steps, $n_{\Delta t}$ temporal windows, and $n_p$ number of atoms for the diffusive species, we only need to compute:

\begin{equation}
    \Delta x (\Delta t, i, d, t_0) = x_{di} (t_0 + \Delta t) - x_{di} (t_0)
\end{equation}

being $\Delta x(\Delta t, i, d, t_0)$ a four rank tensor of dimension $n_{\Delta t} \times n_t \times n_p \times n_d$ that stores all mean displacements of temporal length $\Delta t$ for particle $i$ in space dimension $d$. This leads to:

\begin{equation}
    \text{MSD}_{self} (\Delta t) = \frac{1}{n_p} \sum_{i = 1}^{n_p} \langle \sum_{d} \Delta x (\Delta t, i, d, t_0) \cdot \Delta x (\Delta t, i, d, t_0) \rangle_{t_0}
\end{equation}
open 
\begin{equation}
    \text{MSD}_{distinct} (\Delta t) = \frac{2}{n_p (n_p-1)} \sum_{i = 1}^{n_p} \sum_{j = i+1}^{n_p} \langle \sum_{d} \Delta x (\Delta t, i, d, t_0) \cdot \Delta x (\Delta t, j, d, t_0) \rangle_{t_0}
\end{equation}

Note that we keep $D_{self}$ and $D_{distinct}$ separate since this allows for an straightforward evaluation of the $D$ contributions resulting from the ionic correlations without increasing the code complexity. 

This implementation scales linearly with the maximum shape of the temporal window, the lenght of the simulation and the number of mobile ions in terms of memory resources.

# Acknowledgements

We acknowledge financial support from the MCIN/AEI/10.13039/501100011033 under the grants PID2020-119777GB-I00, PID2020-112975GB-I00 and TED2021-130265B-C22, the “Ramón y Cajal” fellowship RYC2018-024947-I, the Severo Ochoa Centres of Excellence Program (CEX2019-000917-S), and the Generalitat de Catalunya under Grant No.2017SGR1506.

# References
