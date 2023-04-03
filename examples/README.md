# Examples

The following files may be found in the present folder:

- Datafiles from an ab initio molecular dynamics simulations of non-stoichiometric LLZO at a temperature of 400K (**XDATCAR** with the simulated configurations and **INCAR** with the simulation parameters).
- **DIFFUSION** file with the resulting diffusive paths. The first column represents the index of the diffusive particle with respect to the **XDATCAR** file, and the second and third columns are the starting and ending configurations of the diffusion process also with respect to the **XDATCAR** file. Results from different runs might slightly differ, due to the random initialization of the clusters in the k-means algorithm
- Example of a protoypical *DIFFUSION_paths* file, with paths and information of each simulations to be considered in the correlation analysis.
