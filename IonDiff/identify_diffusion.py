#!/usr/bin/env python
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import sys

from IonDiff import common_library as CL

sns.set_theme()

"""Definition of the class to extract the diffusion information. Only VASP simulations are considered as yet.
"""

# Defining the class
class xdatcar:
    """Python Class for loading information from VASP simulations and identifying diffusions.

    Attributes:
        kmeans_kwargs   (dict):  Parameters for K-means clustering algorithm.
        spectral_kwargs (dict):  Parameters for Spectral clustering algorithm.
        time_step       (float): Time step between consecutive XDATCAR configurations.
        n_ions          (int):   Number of ions in the simulation.

    Methods:
        __init__(self, args):
            Initializes the XDATCAR class.
        read_simulation(self, args):
            Reads VASP XDATCAR files.
        calculate_silhouette(self, coordinates, method, n_attempts, silhouette_thd):
            Calculates silhouette scores for different numbers of clusters and selects the optimal number.
        calculate_clusters(self, coordinates, n_clusters, method, distance_thd):
            Calculates clusters and related information based on the chosen method and number of clusters.
        get_diffusion(self, args):
            Obtains diffusion information from the simulation data.
    """

    def __init__(self, args):
        """Initialize the XDATCAR class.

        Args:
            args: Command line arguments containing MD_path and other parameters.

        Raises:
            exit: If required files are missing.
        """
        
        # Loading intervals step and number of simulation steps between records
        self.delta_t, self.n_steps = CL.read_INCAR(args.MD_path)
        
        # Time step between consecutive XDATCAR configurations
        self.time_step = self.n_steps * self.delta_t
        
        # Reading the simulation data
        try:
            self.read_simulation(args)
        except:
            sys.exit('Error: simulation not found')


    def read_simulation(self, args):
        """Reads VASP XDATCAR files.
        """

        cell, self.n_ions, self.compounds, self.concentration, position = CL.read_XDATCAR(args.MD_path)
        
        self.positionC = np.zeros_like(position)
        self.n_iter    = position.shape[0]
        
        # Getting the variation in positions and applying periodic boundary condition
        dpos = np.diff(position, axis=0)
        dpos[dpos >  0.5] -= 1.0
        dpos[dpos < -0.5] += 1.0
        
        # Getting the positions and variations in cell units
        for i in range(self.n_iter-1):
            self.positionC[i] = np.dot(position[i], cell)
            dpos[i]           = np.dot(dpos[i],     cell)

        self.positionC[-1] = np.dot(position[-1], cell)
        self.dpos          = dpos

        expanded_dimensions        = np.expand_dims(self.positionC[0], 0)
        self.cartesian_coordinates = np.concatenate([expanded_dimensions, self.dpos], axis=0)
        self.cartesian_coordinates = np.cumsum(self.cartesian_coordinates, axis=0)
        
        # Defining the attribute of window=1 variation in position and velocity
        self.velocity = self.dpos / self.time_step

    
    def get_diffusion(self, args):
        """Obtains diffusion information from the simulation data.

        Args:
            args: Command line arguments.

        Returns:
            list: List of diffusion events, each represented as [particle index, start step, end step].
        """
        
        hoppings = []
        if args.make_plot:
            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes(projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        
        for particle in range(self.n_ions):
            coordinates = self.cartesian_coordinates[:, particle]
            
            # Recommended number of clusters
            n_clusters = CL.calculate_silhouette(coordinates,
                                                 args.classifier,
                                                 args.n_attempts,
                                                 args.silhouette_thd)
            
            # Information from the clustering
            centers, classification, vibration, cluster_change = CL.calculate_clusters(coordinates,
                                                                                       n_clusters,
                                                                                       args.classifier,
                                                                                       args.distance_thd)
            
            # Whenever any group change is found,
            # the initial and ending configurations are obtained regarding the distance threshold
            if cluster_change.size:
                for change in cluster_change:
                    idx_0 = np.where(vibration[1:change] != vibration[:change-1])[0]
                    if idx_0.size: idx_0 = idx_0[-1] + 1
                    else:          idx_0 = 0
                    
                    idx_1 = np.where(vibration[change+1:] != vibration[change:-1])[0]
                    if idx_1.size: idx_1 = idx_1[0] + change
                    else:          idx_1 = -1
                    
                    # Checking that the new diffusion process is not already saved
                    # This can happen due to the distance threshold,
                    # gathering two consecutive, spatially-close diffusions
                    new_hoppings = [particle, idx_0, idx_1]
                    if new_hoppings not in hoppings:
                        hoppings.append(new_hoppings)
                        
                        if args.make_plot:
                            ax.scatter(coordinates[idx_0:idx_1, 0],
                                       coordinates[idx_0:idx_1, 1],
                                       coordinates[idx_0:idx_1, 2],
                                       color=np.random.rand(3),
                                       marker='o',
                                       label=f'Diffusion')
        
            if args.make_plot:
                for i in range(n_clusters):
                    color = np.random.rand(3)
                    positions = np.where(classification == i)[0]
                    
                    for position in positions:
                        ax.scatter(coordinates[position, 0], coordinates[position, 1], coordinates[position, 2],
                                   color=color, marker='.')
                    ax.scatter(centers[i, 0], centers[i, 1], centers[i, 2],
                               s=200,
                               color='black',
                               marker='x')
                
                if cluster_change.size:  # Otherwise there are no diffusive events to label
                    plt.legend(loc='best')
                plt.show()
        return hoppings

    def get_diffusion_coefficient(self, args, initial_ratio=0.4, ending_ratio=1, contributions='full', n_delta_t=None, axis=None):
        """Calculates the diffusion coefficient from atomic trajectories using Mean Square Displacement (MSD) method.

            Args:
                args:                            Arguments containing parameters like `diffusion_particle` and `make_plot`.
                initial_ratio (float, optional): Ratio of the initial point for fitting the MSD curve. Defaults to 0.4.
                ending_ratio  (float, optional): Ratio of the ending point for fitting the MSD curve. Defaults to -1.
                contributions (str, optional):   Specifies which contributions to consider ('full', 'self', 'distinct'). Defaults to 'full'.
                n_delta_t     (int, optional):   Number of time steps to consider. If None, uses the entire simulation. Defaults to None.
                axis          (int, optional):   Axis along which to compute the ionic diffusion coefficient. Default to average
            """

        # Identify the index of the diffusive particle
        for i in range(np.sum(self.concentration)):
            if self.compounds[i] == args.diffusion_particle:
                diffusive_idx = i
                break

        # Cumulative sum of concentration with an initial zero
        concentration_cumsum = np.insert(np.cumsum(self.concentration), 0, 0)

        # Exact indexes of the diffusive particles
        diffusive_indexes = np.arange(concentration_cumsum[diffusive_idx],
                                      concentration_cumsum[diffusive_idx + 1],
                                      dtype=int)

        # Number of diffusive particles
        n_diff_atoms = len(diffusive_indexes)

        # Consider only diffusive atoms
        diff_cartesian_coordinates = self.cartesian_coordinates[:, diffusive_indexes]

        # In case the number of ionic steps to be considered is not given, it is taken as the full AIMD simulation
        if n_delta_t is None:
            n_delta_t = self.n_iter

        # Generate tensor of atomic differences over time windows
        differences_tensor_mean = np.ones((n_delta_t, self.n_iter, n_diff_atoms, 3)) * np.NaN

        # We vectorize in terms of n_atoms (only possibility here)
        for delta_t in np.arange(1, n_delta_t):
            # Number of windows which are used for screening distances
            n_windows = self.n_iter - delta_t

            # Generate mean over windows
            for t_0 in range(n_windows):
                # Distance between two configurations of a same particle
                # td (atom_i, dim_i) = cc (t_0 + delta_t, atom_i, dim_k) - cc (t_0, atom_i, dim_k)
                temporal_dist = diff_cartesian_coordinates[t_0 + delta_t] - diff_cartesian_coordinates[t_0]

                # Add to temporal variable
                differences_tensor_mean[delta_t, t_0] = temporal_dist

        # Compute mean-square displacements
        if (contributions == 'full') or (contributions == 'self'):  # Compute self part
            # Scalar product between self particles
            self_scalar_product = differences_tensor_mean * differences_tensor_mean

            # Diffusion coefficient in one dimension (axis) or average
            if axis is not None:
                self_scalar_product = self_scalar_product[:, :, :, axis]
            else:
                self_scalar_product = np.sum(self_scalar_product, axis=-1)

            # Apply average for particles
            particles_avg = np.nanmean(self_scalar_product, axis=-1)  # Raises warning due to NaNs in some positions

            # Apply average for windows
            MSD_self = np.nanmean(particles_avg, axis=-1)
        if (contributions == 'full') or (contributions == 'distinct'):  # Compute distinct part
            n_distinct = (n_diff_atoms * (n_diff_atoms - 1)) / 2

            particles_sum = np.zeros((n_delta_t, self.n_iter))
            for idx_i in np.arange(n_diff_atoms):
                # All remaining indexes
                idx_j = np.arange(idx_i + 1, n_diff_atoms)

                # Extract data
                diff_i = differences_tensor_mean[:, :, np.newaxis, idx_i]  # np.newaxis to allow dot product
                diff_j = differences_tensor_mean[:, :, idx_j]

                # Scalar product between distinct particles
                distinct_scalar_product = diff_i * diff_j

                # Diffusion coefficient in one dimension (axis) or average
                if axis is not None:
                    distinct_scalar_product = distinct_scalar_product[:, :, :, axis]
                else:
                    distinct_scalar_product = np.sum(distinct_scalar_product, axis=-1)

                # Apply sum for particles
                particles_sum += np.sum(distinct_scalar_product, axis=-1)

            # Apply average for particles
            particles_avg = particles_sum / n_distinct

            # Apply average for windows
            MSD_distinct = np.nanmean(particles_avg, axis=-1)

        # Add contribution if necessary
        if contributions == 'full':
            # Add both contributions
            MSD = MSD_self + MSD_distinct
        elif contributions == 'self':
            # Just self contribution
            MSD = MSD_self
        elif contributions == 'distinct':
            # Just distinct contribution
            MSD = MSD_distinct
        else:
            sys.exit('Error: contribution not understood')

        # Define the array of time simulation in pico-seconds
        delta_t_array = np.arange(n_delta_t) * (self.n_steps * self.delta_t * 1e-3)

        initial_point = int(initial_ratio * n_delta_t)
        ending_point  = int(ending_ratio  * n_delta_t)

        _beta_ = CL.weighted_regression(delta_t_array[initial_point:ending_point],
                                        MSD[initial_point:ending_point],
                                        CL.D_function).beta

        print(f'Ionic-diffusion coefficient: {_beta_[1]:.4f}')

        # Plot the mean-squared displacements if required
        if args.make_plot:
            # All MSD over time windows
            plt.plot(delta_t_array,
                     MSD,
                     label=contributions)

            # MSD used for the extraction of the ionic-diffusion coefficient
            plt.plot(delta_t_array[initial_point:ending_point],
                     MSD[initial_point:ending_point],
                     label='fit')

            plt.xlabel(r'$\Delta t$ (ps)')
            plt.ylabel(r'$MSD$ ($\mathregular{Å^2}$)')
            plt.legend(loc='best')
            plt.show()
