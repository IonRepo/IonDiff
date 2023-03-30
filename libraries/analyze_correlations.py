#!/usr/bin/env python
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns

from os              import path
from sys             import argv, exit
from scipy.fftpack   import fft, fftfreq
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

sns.set_theme()

"""Definition of the class to analyse correlations among diffusive paths. A database with more than one simulation must be provided.
"""

# Defining the information lines of the file

scale_line         = 1  # Line for the scale of the simulation box
s_cell_line        = 2  # Start of the definition of the simulation box
e_cell_line        = 4  # End of the definition of the simulation box
name_line          = 5  # Composition of the compound
concentration_line = 6  # Concentration of the compound
x_line             = 7  # Start of the simulation data

# Defining the basic parameters for k-means and spectral clustering algorithms

kmeans_kwargs   = dict(init='random', n_init=10, max_iter=300, tol=1e-04,                          random_state=0)
spectral_kwargs = dict(affinity='nearest_neighbors', n_neighbors=1000, assign_labels='cluster_qr', random_state=0)

# Defining the class

class database:
    """Python Class for loading information from VASP simulations and analysing correlations among diffusive paths.
    """

    def __init__(self, args):
        """Important variables added to the class and reading of the simulation data.
        """
        
        # Loading intervals step and number of simulation steps between records
        
        delta_t, n_steps = self.read_INCAR(args)
        
        # Time step between consecutive XDATCAR configurations
        
        self.time_step = n_steps * delta_t
        
        # Reading the simulation data
        
        if path.exists(f'{args.MD_path}/XDATCAR'):
            self.read_simulation(args)
    
    def read_INCAR(self, args):
        """Reads VASP INCAR files. It is always expected to find these parameters.
        """
        
        # Predefining the variable, so later we check if they were found
        
        delta_t = None
        n_steps = None
        
        # Loading the INCAR file
        
        with open(f'{args.MD_path}/INCAR', 'r') as INCAR_file:
            INCAR_lines = INCAR_file.readlines()
        
        # Looking for delta_t and n_steps
        
        for line in INCAR_lines:
            split_line = line.split('=')
            if len(split_line) > 1:  # Skipping empty lines
                label = split_line[0].split()[0]
                value = split_line[1].split()[0]
                
                if   label == 'POTIM':  delta_t = float(value)
                elif label == 'NBLOCK': n_steps = float(value)
        
        # Checking if they were found
        
        if (delta_t is None) or (n_steps is None):
            exit('POTIM or NBLOCK are not correctly defined in the INCAR file.')
        return delta_t, n_steps
     
    def read_simulation(self, args):
        """Reads VASP XDATCAR files.
        """
        
        # Loading data from XDATCAR file
        
        with open(f'{args.MD_path}/XDATCAR', 'r') as POSCAR_file:
            inp = POSCAR_file.readlines()
        
        # Extracting the data
        
        try:
            scale = float(inp[scale_line])
        except:
            exit('Wrong definition of the scale in the XDATCAR.')
        
        try:
            self.cell = np.array([line.split() for line in inp[s_cell_line:e_cell_line+1]], dtype=float)
            self.cell *= scale
        except:
            exit('Wrong definition of the cell in the XDATCAR.')

        self.TypeName = inp[name_line].split()
        self.Nelem    = np.array(inp[concentration_line].split(), dtype=int)
        
        if len(self.TypeName) != len(self.Nelem):
            exit('Wrong definition of the composition of the compound in the XDATCAR.')
        
        self.Ntype = len(self.TypeName)
        self.Nions = self.Nelem.sum()
        
        # Shaping the configurations data into the positions attribute
        
        pos = np.array([line.split() for line in inp[x_line:] if not line.split()[0][0].isalpha()], dtype=float)
        
        # Checking if the number of configurations is correct
        
        if not (len(pos) / self.Nions).is_integer():
            exit('The number of lines is not correct in the XDATCAR file.')
        
        self.position  = pos.ravel().reshape((-1, self.Nions, 3))
        self.positionC = np.zeros_like(self.position)
        self.Niter     = self.position.shape[0]
        
        # Getting the variation in positions and applying periodic boundary condition
        
        dpos = np.diff(self.position, axis=0)
        dpos[dpos > 0.5]  -= 1.0
        dpos[dpos < -0.5] += 1.0
        
        # Getting the positions and variations in cell units
        
        for i in range(self.Niter-1):
            self.positionC[i] = np.dot(self.position[i], self.cell)
            dpos[i]           = np.dot(dpos[i],          self.cell)
        
        self.positionC[-1] = np.dot(self.position[-1], self.cell)
        
        # Defining the attribute of window=1 variation in position and velocity
        
        self.velocity = dpos / self.time_step
        self.d1pos    = dpos
