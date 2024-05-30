#!/usr/bin/env python
import numpy as np
import argparse
import logging
import settings

from IonDiff import identify_diffusion   as ID_library
from IonDiff import analyze_correlations as AC_library
from IonDiff import analyze_descriptors  as AD_library

from datetime import datetime

"""
The command line variables are read. Three options are considered:
    - Identification of diffusive paths from a molecular dynamics simulation.
    - Analysis of correlations among diffusive paths.
    - Analysis of atomistic descriptors extracted from the diffusive paths (under active development).

At the input folder, a XDATCAR file with all the configurations of the system through simulation is required.
Optionally, a POSCAR can be supplied with the initial configuration.
As well, an INCAR specifying POTIM (simulation time step) and NBLOCK (number of simulation steps between
consecutive configurations in the XDATCAR) is necessary.
"""

# Preparing the interpretation of input (command line) variables
parser = argparse.ArgumentParser()

task_subparser = parser.add_subparsers(
    help='Task to perform.',
    dest='task'
)

ID_parser = task_subparser.add_parser('identify_diffusion')  # Identification of diffusive paths (ID)

ID_parser.add_argument(
    '--MD_path',
    default='.',
    help='Path to the input molecular dynamics simulation files.',
)
ID_parser.add_argument(
    '--classifier',
    default='K-means',
    help='Name of the classifier used to group non-diffusive particles ("K-means" or "Spectral").',
)
ID_parser.add_argument(
    '--distance_thd',
    type=float,
    default=0.4,
    help='Distance threshold from the diffusive path to the vibrational center.',
)
ID_parser.add_argument(
    '--silhouette_thd',
    type=float,
    default=0.7,
    help='Silhouette value threshold for selecting the correct number of vibrational groups.',
)
ID_parser.add_argument(
    '--make_plot',
    type=bool,
    default=False,
    help='Whether to plot the clustering calculations or not.',
)
ID_parser.add_argument(
    '--n_attempts',
    type=int,
    default=10,
    help='Number of considered possible diffusive events during a simulation.',
)

AC_parser = task_subparser.add_parser('analyze_correlations')  # Analysis of correlations (AC)

AC_parser.add_argument(
    '--MD_path',
    default='.',
    help='Path to the input molecular dynamics simulation files.',
)
AC_parser.add_argument(
    '--max_corr',
    type=int,
    default='20',
    help='Maximum number of correlated bodies considered for representation (max_corr=-1 to select all of them).',
)
AC_parser.add_argument(
    '--threshold',
    type=float,
    default='2',
    help='Threshold for removing random correlations among particles lower than the'
         'threshold, between [-1, 1] (threshold=2 to automatically compute it).',
)

AD_parser = task_subparser.add_parser('analyze_descriptors')  # Analysis of descriptors (AD)

AD_parser.add_argument(
    '--MD_path',
    default=None,
    help='Path to the input molecular dynamics simulation files.',
)

AD_parser.add_argument(
    '--reference_path',
    default=None,
    help='Path to a folder with a stoichiometric POSCAR structure file for the simulation of the given material.',
)

# Computing the vibrational paths
if __name__ == '__main__':
    # Reading the input variables
    args = parser.parse_args()
    
    # Configuring loggin information
    logging.basicConfig(
        filename=settings.LOGS_PATH / f'{args.task}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.log',
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        level=logging.INFO,
    )
    
    # Performing the specified task
    if args.task == 'identify_diffusion':
        # Logging update
        logging.info(f'Task: Extracting diffusive paths from MD simulation at {args.MD_path}.')
        
        # Calling the library and loading the class
        inp = ID_library.xdatcar(args)
        
        # Logging update
        logging.info(f'Simulation successfully loaded.')
        
        # Computing the diffusive paths
        diffusive_paths = inp.get_diffusion(args)
        
        # Saving the results
        np.savetxt(f'{args.MD_path}/DIFFUSION', diffusive_paths)
        
        # Logging update
        logging.info(f'Diffusive information successfully extracted and saved.')
    
    elif args.task == 'analyze_correlations':
        # Saving logging information
        logging.info(f'Task: Analysing N-body correlations from MD simulations database at {args.MD_path}.')
        
        # Calling the library and executing the analysis
        AC_library.database(args)
    
    elif args.task == 'analyze_descriptors':
        # Saving logging information
        logging.info(f'Task: Analysing atomistic descriptors from MD simulations database at {args.MD_path}.')
        
        # Calling the library and loading the class
        inp = AD_library.descriptors(args)
        
        # Computing descriptors
        time_interval      = inp.time_until_diffusion()
        temporal_duration  = inp.duration_of_diffusion()
        spatial_length     = inp.length_of_diffusion(outer='nan')
        n_diffusive_events = inp.n_diffusive_events()
        residence_time     = inp.residence_time(args)[0] if args.reference_path is not None else None
        
        # Save descriptors as dictionary
        descriptors = {
            MD_path:      args.MD_path,
            delta_t_min:  np.min(time_interval),
            delta_t_max:  np.max(time_interval),
            delta_t_mean: np.mean(time_interval),
            delta_r_min:  np.min(temporal_duration),
            delta_r_max:  np.max(temporal_duration),
            delta_r_mean: np.mean(temporal_duration),
            gamma:        residence_time
        }
        
        # Logging update
        logging.info(f'Descriptors successfully extracted.')
        
        # Write the dictionary to the file in JSON format
        with open(f'{args.MD_path}/atomistic_descriptors.json', 'w') as json_file:
            json.dump(descriptors, json_file)
        
        # Logging update
        logging.info(f'Descriptors successfully saved.')
