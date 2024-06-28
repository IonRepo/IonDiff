#!/usr/bin/env python
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import multiprocess      as mp
import sys
import os

import IonDiff.common_library as CL

from scipy.optimize import curve_fit

sns.set_theme()

"""Definition of the class to analyse correlations among diffusive paths. A database with more than one
   simulation must be provided, as well as paths to every interesting DIFFUSION file, extracted from the simulation.
"""

fontsize = 10
dpi      = 100

# Defining the class
class database:
    """Python Class for loading information from VASP simulations and analyzing correlations among diffusive paths.

    Attributes:
        fontsize (int): Font size for plot labels and ticks.
        dpi      (int): Dots per inch for saving plots.
    
    Methods:
        __init__(self, args):
            Initializes the Database class.
        exp_function(self, x, A, B, C):
            Defines a decreasing and always positive exponential function.
        parallel_calculation(self, element, args):
            Auxiliar function to parallelize calculations.
    """

    def __init__(self, args):
        """Initialize the Database class.

        Args:
            args: Command line arguments containing MD_path and other parameters.

        Raises:
            exit: If required files are missing.
        """
        
        correlations_matrix_path = f'{args.MD_path}/temp_matrix'
        DIFFUSION_paths          = f'{args.MD_path}/DIFFUSION_paths'
        
        if os.path.exists(correlations_matrix_path):
            print('As temp_matrix already exists, its information is being loaded.')
            
            with open(correlations_matrix_path, 'r') as file:
                lines = file.readlines()
            
            n = int(len(lines) / 3)
            length = len(lines[0].split())

            corr_matrix = np.zeros((n, length))
            temp_matrix = np.zeros((n, length))
            fami_matrix = np.zeros((n, length), dtype=object)
            for i in range(n):
                corr_matrix[i] = np.array(lines[i].split(), dtype=float)
                temp_matrix[i] = np.array(lines[i+n].split(), dtype=float)
                fami_matrix[i] = np.array(lines[i+2*n].split(), dtype=object)
        
        elif os.path.exists(DIFFUSION_paths):
            print('As temp_matrix does not exist, it is being computed.')
            
            with open(DIFFUSION_paths, 'r') as file:
                paths_to_DIFFUSION = file.readlines()
            
            # The computation of correlations for each element are parallelized
            pool    = mp.Pool(mp.cpu_count())  # Number of CPUs in the PC
            metrics = [pool.apply(self.parallel_calculation, (element, args,)) for element in paths_to_DIFFUSION]
            pool.close()
            
            n_events = 0
            for lines in metrics:
                line = lines[0]
                if len(line) > n_events:
                    n_events = len(line)

            n_simulations = len(metrics)

            corr_matrix = np.zeros((n_simulations, n_events))
            temp_matrix = np.ones((n_simulations, n_events)) * np.NaN
            fami_matrix = np.zeros((n_simulations, n_events), object)

            for i in range(n_simulations):
                corr_temporal = np.array(metrics[i][0])
                temp_temporal = np.array(metrics[i][1])
                fami_temporal = np.array(metrics[i][2])
                
                corr_matrix[i] = np.hstack([corr_temporal, np.zeros(n_events - len(corr_temporal))])
                temp_matrix[i] = np.hstack([temp_temporal, np.ones(n_events  - len(temp_temporal)) * np.NaN])
                fami_matrix[i] = np.hstack([fami_temporal, ['0']*(n_events - len(fami_temporal))])
            
            np.savetxt(correlations_matrix_path, np.vstack([corr_matrix, temp_matrix, fami_matrix]), fmt='%s')
        
        else:
            sys.exit(f'File with previous data ({correlations_matrix_path}) and file with paths to DIFFUSIONs ({DIFFUSION_paths}) are missing.')
        
        # Extracting main data from the correlation matrix
        n_simulations, n_events = np.shape(corr_matrix)

        x = np.arange(1, n_events+1)  # Number of correlated bodies (list)
        y = np.sum(corr_matrix, axis=0)  # Number of particles with which each particle correlates

        # Representing some elements and passing to probability
        x = x[1:args.max_corr]
        y = y[1:args.max_corr]

        # Normalizing y
        y = y / np.sum(y)

        # Fitting an exponential function to the the distribution of correlated bodies
        [A, B, C], _ = curve_fit(self.exp_function, x, y, p0=[0.01, 0.1, 0.1])

        # Plotting fitting and computed points
        ran_c = np.arange(min(x), max(x), 0.01)
        exp_c = self.exp_function(ran_c, A, B, C)
        
        plt.plot(ran_c, exp_c,  label='Exponential fitting')
        plt.plot(x,     y, 'o', label='Computed groups')

        plt.tick_params(axis='x', labelsize=fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)

        plt.xlabel(f'Correlated bodies',            fontsize=fontsize)
        plt.ylabel(f'Probability density function', fontsize=fontsize)

        plt.legend(loc='best', fontsize=fontsize)
        plt.savefig(f'{args.MD_path}/PDOS_correlations.eps', dpi=dpi, bbox_inches='tight')
        plt.show()


    def exp_function(self, x, A, B, C):
        """Defines a decreasing and always positive exponential function.

        Args:
            x (numpy.ndarray): Input array.
            A (float):         Exponential parameter.
            B (float):         Exponential parameter.
            C (float):         Exponential parameter.

        Returns:
            numpy.ndarray: Exponential function values for the given input.
        """
        
        A = np.abs(A)
        return A + B * np.exp(-C * x)


    def parallel_calculation(self, element, args):
        """Auxiliar function to parallelize calculations.
        Obtains the distribution and compares it with a random distribution.

        Args:
            element (str): Path to the simulation, material, and temperature.
            args:          Command line arguments.

        Returns:
            tuple: Tuple containing correlation_cumulative, temperature_cumulative, and family_cumulative.
        """
        
        path_to_simulation, material, temperature = element.split()
        
        # Loading the data
        coordinates, hoppings, cell, compounds, concentration = CL.load_data(path_to_simulation)
        (n_conf, n_particles, _) = np.shape(coordinates)

        # Expanding the hoppings
        key, expanded_hoppings = CL.get_expanded_hoppings(n_conf, n_particles, concentration,
                                                          compounds, hoppings, 'separate')

        # No filter
        Z_ngs, _ = CL.get_correlation_matrix(expanded_hoppings, gaussian_smoothing=False)

        # Gaussian smoothing
        Z, Z_corr = CL.get_correlation_matrix(expanded_hoppings, gaussian_smoothing=True)
        
        if args.threshold == 2:
            threshold_aux = []  # Cumulative of thresholds
            for _ in range(200):
                # Defining a uniformly-random distributed position of hoppings
                n_random_particles = np.shape(expanded_hoppings)[1]

                diffusion_length = int(np.mean(np.sum(Z_ngs, axis=0)))

                random_positions = np.random.randint(0, n_conf-diffusion_length, n_random_particles)

                Z_random_ngs = np.zeros((n_conf, n_random_particles))
                for i in range(n_random_particles):
                    pos1 = random_positions[i]
                    pos2 = pos1 + diffusion_length

                    Z_random_ngs[pos1:pos2, i] = 1

                # No filter
                Z_random_ngs, _ = CL.get_correlation_matrix(Z_random_ngs, gaussian_smoothing=False)

                # Gaussian smoothing
                Z_random, Z_random_corr = CL.get_correlation_matrix(Z_random_ngs, gaussian_smoothing=True)

                # Calculating the threshold
                Z_random_corr[np.diag_indices(len(Z_random_corr))] = np.NaN
                aux = Z_random_corr.flatten()
                aux = aux[~np.isnan(aux)]
                threshold_aux.append(np.mean(aux))
            
            # A threshold is defined in correspondence to a random distribution,
            # so less-correlated particles are not considered
            threshold = np.mean(threshold_aux)
        
        else:
            threshold = args.threshold

        # Applying the threshold
        binary_Z_corr = np.zeros_like(Z_corr)
        binary_Z_corr[Z_corr >  threshold] = 1
        binary_Z_corr[Z_corr <= threshold] = 0

        # The number of particles with which each particle correlates is obtained from the binary matrix of correlations
        binary_sum = np.sum(binary_Z_corr, axis=0)
        n = int(max(binary_sum))  # Maximum number of correlated bodies
        
        # Defining the matrixes with information regarding the correlations,
        # and temperatures and diffusive families of the respective simulations
        corr_cum = np.zeros(n)
        temp_cum = np.ones(n) * np.NaN
        fami_cum = np.zeros(n, object)
        for i in range(n):
            corr_cum[i] = np.sum(binary_sum == (i+1))  # Number of bodies exhibiting the respective correlation
            if np.sum(binary_sum == (i+1)) > 0:  # Else, np.NaN
                temp_cum[i] = float(temperature[:-1])
                fami_cum[i] = CL.obtain_diffusive_family(material)
        return list(corr_cum), list(temp_cum), list(fami_cum)
