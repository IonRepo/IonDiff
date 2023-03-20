#!/usr/bin/env python
import argparse
import logging
import settings

import numpy as np

from libraries import identify_diffusion as ID_library
from datetime  import datetime
from os        import system, getcwd, chdir, mkdir, remove, path

"""
The command line variables are read. Three options are considered:
    - Identification of diffusive paths from a molecular dynamics simulation.
    - Analysis of correlations among diffusive paths.
    - Analysis of atomistic descriptors extracted from the diffusive paths.

At the input folder, and XDATCAR file with all the configurations of the system through simulation is required. Optionally, a POSCAR can be supplied with the initial configuration. As well, an INCAR specifyin POTIM (simulation step) and NBLOCK (number of simulation steps between consecutive configurations in the XDATCAR) are necessary.
"""

# Preparing the interpretation of input (command line) variables

parser = argparse.ArgumentParser()

task_subparser = parser.add_subparsers(
    help='Task to perform.',
    dest='task'
)

ID_parser = task_subparser.add_parser('identify_diffusion')  # Identication of diffusive paths (ID)

ID_parser.add_argument(
    '--MD_path',
    default='.',
    help='Path to the input molecular dynamics simulation files.',
)
ID_parser.add_argument(
    '--classifier',
    default='K-means',
    help='Name of the classifier used to group non-diffusive particles.',
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
    '--n_attemps',
    type=int,
    default=10,
    help='Number of considered possible diffusive events during a simulation.',
)

AC_parser = task_subparser.add_parser('analyze_correlations')  # Analysis of correlations (AC)

AC_parser.add_argument(
    '--data_path',
    default='.',
    help='Path to the input simulations dynamics simulation files.',
)

AD_parser = task_subparser.add_parser('analyze_descriptors')  # Analysis of descriptors (AD)

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
        
        logging.info(f'Diffive information successfully extracted and saved.')
    
    elif args.task == 'analyze_correlations':
        # Saving logging information
        
        logging.info(f'Task: Analysing N-body correlations from MD simulation at {args.MD_path}.')
        
        # Calling the library and loading the class
        
        something
    
    elif args.task == 'analyze_descriptors':
        # Saving logging information
        
        logging.info(f'Task: Analysing atomistic descriptors from MD simulation at {args.MD_path}.')
        
        # Calling the library and loading the class
        
        something
