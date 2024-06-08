#!/usr/bin/env python
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import sys
import os

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
        delta_t, n_steps = CL.read_INCAR(args.MD_path)
        
        # Time step between consecutive XDATCAR configurations
        self.time_step = n_steps * delta_t
        
        # Reading the simulation data
        try:
            self.read_simulation(args)
        except:
            sys.exit('Error: simulation not found')


    def read_simulation(self, args):
        """Reads VASP XDATCAR files.
        """
        
        cell, self.n_ions, _, _, position = CL.read_XDATCAR(args.MD_path)
        
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
        
        # Defining the attribute of window=1 variation in position and velocity
        self.velocity = dpos / self.time_step
        self.d1pos    = dpos
    
    
    def get_diffusion(self, args):
        """Obtains diffusion information from the simulation data.

        Args:
            args: Command line arguments.

        Returns:
            list: List of diffusion events, each represented as [particle index, start step, end step].
        """
        
        hoppings = []
        
        full_coordinates = np.concatenate([np.expand_dims(self.positionC[0], 0), self.d1pos], axis=0)
        full_coordinates = np.cumsum(full_coordinates, axis=0)
        
        if args.make_plot:
            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes(projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        
        for particle in range(self.n_ions):
            coordinates = full_coordinates[:, particle]
            
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
