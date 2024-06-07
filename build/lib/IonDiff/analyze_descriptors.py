#!/usr/bin/env python
import numpy as np
import json
import sys

import IonDiff.common_library as CL

"""Definition of the class to analyse correlations among descriptors of diffusive paths.
   A database with more than one simulation must be provided, as well as paths to every
   interesting DIFFUSION file, extracted from the simulation.
"""

# Defining the class
class descriptors:
    """Python Class for loading information from VASP simulations
       and analyzing correlations among descriptors of diffusive paths.

    Methods:
        __init__(self, args):
            Initializes the Descriptors class.
    """

    def __init__(self, args):
        """Initialize the Database class.

        Args:
            args: Command line arguments containing MD_path and other parameters.

        Raises:
            exit: If required files are missing.
        """
        
        # Loading the data
        coordinates, self.hoppings, self.cell, self.compounds, self.concentration = CL.load_data(args.MD_path)
        (self.n_conf, self.n_particles, _) = np.shape(coordinates)

        # Getting diffusion coordinates
        _, nan_hoppings = CL.get_expanded_hoppings(self.n_conf, self.n_particles, self.concentration, self.compounds,
                                                   self.hoppings, method='original')
        nan_hoppings[nan_hoppings == 0] = np.NaN

        self.coordinates = np.stack([nan_hoppings, nan_hoppings, nan_hoppings], axis=2) * coordinates

        # Expanding the hoppings avoiding non-diffusive particles
        _, self.expanded_hoppings = CL.get_expanded_hoppings(self.n_conf, self.n_particles, self.concentration,
                                                             self.compounds, self.hoppings, method='separate')
    

    def time_until_diffusion(self, index=0, outer=None):
        """
        Calculates an array with the number of steps until the start of the first diffusion of a particle.
        Expanded matrix in 'separate' or 'cleaned' format is mandatory.
        We seek the first one for each particle and append it to the list (indexes are in agreement with 'key').
        Index = {0, -1} variable allows obtaining the first or last diffusive position of a diffusion track.
        Calculate Time Until Diffusion

        Args:
            expanded_matrix (numpy.ndarray): Expanded matrix of diffusive/non-diffusive processes.
            index           (int, optional): Index to obtain the first or last diffusive position (default is 0).
            outer           (optional):      Value for handling NaN values (default is None).

        Returns:
            numpy.ndarray: Array with time until diffusion for each particle.
        """
        n_particles = np.shape(self.expanded_matrix)[1]
        initial_times = np.zeros(n_particles)
        for i in range(n_particles):
            particle_track = self.expanded_matrix[:, i]
            if outer is None:
                initial_times[i] = np.where(particle_track)[0][index]
            elif outer == 'nan':
                initial_times[i] = np.where(~np.isnan(particle_track))[0][index]
        return initial_times
    
    
    def duration_of_diffusion(self):
        """
        Calculates an array with the number of steps of diffusion of a particle.
        Expanded hoppings in 'separate' format is mandatory.
        We seek the first one for each particle and append it to the list (indexes are in agreement with 'key').
        Calculate Duration of Diffusion

        Args:
            expanded_hoppings (numpy.ndarray): Expanded hopping data.

        Returns:
            numpy.ndarray: Array with the duration of diffusion for each particle.
        """

        diffusion_duration = time_until_diffusion(self.expanded_hoppings, index=-1) - time_until_diffusion(self.expanded_hoppings, index=0)
        return diffusion_duration


    def length_of_diffusion(self, outer='nan'):
        """
        Calculates an array with the diffusion length of a particle.
        Coordinates must be supplied multiplied by the hoppings mask and with NaNs.
        We expand the coordinates (only one diffusion per column/particle).
        Returns the distance between initial and ending points.
        Periodic boundary conditions are considered.
        Calculate Length of Diffusion

        Args:
            coordinates (numpy.ndarray): Input coordinates.
            cell        (numpy.ndarray): Cell information.
            outer       (str, optional): Handling for NaN values (default is 'nan').

        Returns:
            numpy.ndarray: Array with the length of diffusion for each particle.
        """

        # Expanding the coordinates with separate mode

        expanded_coordinates, _, starts, ends, _, n_particles = CL.get_expanded_coordinates(self.coordinates,
                                                                                            outer=outer)
        
        # Computing the distance between initial and final diffusion coordinates (considering PBC)

        diffusion_length = np.zeros(n_particles)
        for particle in range(n_particles):
            start = int(starts[particle])
            end   = int(ends[particle])

            distance = np.abs(expanded_coordinates[start, particle] - expanded_coordinates[end, particle])
            distance[distance >= 0.5] -= 1
            distance = np.dot(distance, self.cell)
            diffusion_length[particle] = np.linalg.norm(distance, axis=0)
        return diffusion_length


    def n_diffusive_events(self):
        """
        Returns the number of detected diffusion events from hoppings array.
        Count Diffusive Events

        Args:
            n_conf        (int):           Number of configurations.
            n_particles   (int):           Number of particles.
            concentration (list):          Particle concentration.
            compounds     (list):          Compounds.
            hoppings      (numpy.ndarray): Hopping data.

        Returns:
            numpy.ndarray: Number of detected diffusion events for each particle.
        """
        
        _, cleaned_hoppings = CL.get_expanded_hoppings(self.n_conf, self.n_particles, self.concentration,
                                                       self.compounds, self.hoppings, method='cleaned')
        cleaned_hoppings[cleaned_hoppings == 0] = np.NaN

        n_columns = np.shape(cleaned_hoppings)[1]

        n_diffusive_events = np.zeros(n_columns)
        for i in range(n_columns):
            n_diffusive_events[i] = len(find_groups(cleaned_hoppings[:, i]))
        return n_diffusive_events


    def residence_time(self, args, threshold=1):
        """Returns the mean time (in pico-seconds) that a particle stay in meta-stable positions.
        The centers of vibration are extracted with IonDiff functions, which are compared with POSCAR positions.
        Diffusion events are considered as part of residence time when it happens (we do not discard those points).
        
        Args:
            MD_path        (str):   Path to current simulation.
            reference_path (str):   Path stoichiometric simulation.
            threshold      (float): Minimum distance threshold between centers of vibration and
                                    stoichiometric positions to be considered as meta-stable positions.
            
        Returns:
            residence_time (float): Proportion of mesta-stable positions
        """
        
        # Read INCAR settings
        delta_t, n_steps = read_INCAR(args.MD_path)

        # A soichiometric path is required
        if args.reference_path is None:
            sys.exit('Error: stoichiometric not available for comparing')
        
        # Load simulation data
        cartesian_coordinates = get_cartesian_coordinates(self.coordinates, self.cell)
        
        # Compute inverse cell
        inv_cell = np.linalg.inv(self.cell)

        # Load POSCAR
        _, _, _, stc_positions = MPL.information_from_VASPfile(args.reference_path, 'POSCAR')

        # Number of simulations steps in some meta-stable position
        metastable     = []
        non_metastable = []
        residence_time         = 0
        residence_time_counter = 0
        for particle in range(self.n_particles):
            #print(100 * (particle+1) / self.n_particles)
            coordinates_i = cartesian_coordinates[:, particle]
            n_clusters    = calculate_silhouette('K-means', coordinates_i, False)
            
            if n_clusters > 1:
                centers, classification, _, _ = calculate_clusters('K-means', coordinates_i, n_clusters)

                for k in range(n_clusters):  # classification_idx = i by definition
                    center = centers[k]

                    # Check if the center is a metastable position or not
                    # Convert cartesian coordinates to direct (fractional) coordinates
                    direct_center = np.dot(center, inv_cell)

                    # Get the distance between every positions of the POSCAR and the center
                    diff = np.abs(stc_positions - direct_center)

                    # Apply pbc
                    while np.any(diff > 0.5):
                        diff[diff > 0.5] -= 1

                    # Convert to cartesian distances
                    for i in range(len(diff)):
                        diff[i] = np.dot(diff[i], self.cell)

                    # Get distance
                    diff = np.linalg.norm(diff, axis=1)

                    # It is meta-stable if it is far from a position of the POSCAR
                    if np.min(diff) > threshold:
                        # Sum the number of positions linked to that center
                        residence_time         += np.sum(classification == k)
                        residence_time_counter += 1
                        metastable.append(direct_center)
                    else:
                        non_metastable.append(direct_center)

        # Average and pass to ps
        average_residence_time = residence_time * delta_t * n_steps * 1e-3 / residence_time_counter
        return average_residence_time, metastable, non_metastable
