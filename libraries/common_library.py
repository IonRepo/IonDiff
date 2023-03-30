import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import re                as re

from scipy.ndimage   import gaussian_filter1d
from scipy.stats     import pearsonr, spearmanr
from os              import path, stat


def obtain_diffusive_family(element):
    """
    Returns the family of an element using the obtain_diffusive_information function.
    Automatically detects components from upper cases and numbers.
    """

    composition = []
    split_compound = re.split('(\d+)', element)  # Splitting by numbers
    for string in split_compound:  # Splitting by upper-cases
        composition += re.findall('[a-zA-Z][^A-Z]*', string)

    DiffTypeName, _, _ = obtain_diffusive_information(composition, [0]*len(composition))

    diffusive_family = ''.join(DiffTypeName)
    for diffusive_element in DiffTypeName:
        if diffusive_element in ['Cl', 'I', 'Br', 'F']:
            diffusive_family = 'Halide'

    return f'{diffusive_family}-based'

def load_data(path_to_simulation, calculate_hoppings=False):
    """
    Returns all needed data in a suitable shape from a simulation.
    Allows requiring hoppings file to exist.
    POSCAR and XDATCAR lines consider only first three rows.
    """

    # Loading data from XDATCAR file
    
    with open(f'{path_to_simulation}/XDATCAR', 'r') as XDATCAR_file:
        XDATCAR_lines = XDATCAR_file.readlines()
    
    scale = float(POSCAR_lines[1])
    cell = scale * np.array([line.split() for line in POSCAR_lines[2:5]], dtype=float)

    simulation_box = np.array([cell[0, 0], cell[1, 1], cell[2, 2]], dtype=float)
    compounds = np.array(POSCAR_lines[5].split())
    concentration = np.array(POSCAR_lines[6].split(), dtype=int)

    n_ions = sum(concentration)
    
        coordinates = np.array([line.split()[:3] for line in XDATCAR_lines[8:] if not line.split()[0][0].isalpha()],
                               dtype=float).ravel().reshape((-1, n_ions, 3))
    
    # Loading hoppings data, if available
    
    hoppings = np.NaN
    if path.exists(f'{path_to_simulation}/DIFFUSION'):
        if stat(f'{path_to_simulation}/DIFFUSION').st_size:
            hoppings = np.loadtxt(f'{path_to_simulation}/DIFFUSION')
    else:
        hoppings = obtain_binary_diffusion(f'{path_to_simulation}/DIFFUSION', graph=True)

    return coordinates, hoppings, cell, compounds, concentration

def paths_to_simulations(temperature, material, mode):
    """Returns the path to the simulation, with our structure for the database.
    """
    
    family = obtain_diffusive_family(material)
    path_to_simulation = f'../Data/{family}/{material}/AIMD/{mode}/{temperature}'
    return path_to_simulation

def get_expanded_hoppings(n_conf, n_particles, concentration, particle_type, hoppings, method='original'):
    """
    Expands the compressed information of hoppings into a matrix of (time x particle).
    'Key' stands for the index of the corresponding particle.

    Separate: treats every diffusive event as from different particles. Non-diffusive particles are not considered.
    Cleaned:  non-diffusive particles are not considered.
    Original: just expanded as the coordinates matrix.
    """

    # Converting the compressed data into a matrix of diffusive/non-diffusive processes

    aux = np.zeros((n_conf, n_particles))
    for line in hoppings:
        line = np.array(line, dtype=int)
        aux[line[1]+1:line[2]+1, line[0]] = 1
    hoppings = aux

    # Diffusions of a same particle can be supposed to be independent

    if method == 'separate':
        hoppings[hoppings == 0] = np.NaN
        key, expanded_hoppings = get_separated_groups(hoppings)
        expanded_hoppings[np.isnan(expanded_hoppings)] = 0
        
        cumsum = np.cumsum(concentration)
        n_types = len(particle_type)

        for i in range(n_types):
            aux = np.where((key < cumsum[i]) & (key >= cumsum[i] - concentration[i]))[0]
            if np.any(aux):
                key[aux] = i
    elif method == 'cleaned':
        key = np.where(hoppings.any(0))[0]
        expanded_hoppings = hoppings[:, hoppings.any(0)]
    elif method == 'original':
        return None, hoppings
    return key.flatten(), expanded_hoppings

def get_correlation_matrix(expanded_hoppings, correlation_method='pearson', gaussian_smoothing=True, sigma=None):
    """
    Returns the correlation matrix.
    Gaussian smoothing is allowed.
    If sigma is None, it is calculated as sigma = delta T / 2.355 (FWHM).
    """

    # Converting into DataFrame

    matrix = pd.DataFrame(expanded_hoppings).copy()

    # Applying Gaussian smoothing

    if gaussian_smoothing:
        for i in range(np.shape(matrix)[1]):
            aux = np.array(matrix.iloc[:, i], dtype=float)
            if sigma is None:
                delta_T = np.count_nonzero(aux)
                sigma = delta_T / 2.355
            
            blurred = gaussian_filter1d(aux, sigma=sigma)
            matrix.iloc[:, i] = blurred * np.max(aux) / np.max(blurred)

    # Getting the correlations with the previously-specified method

    correlation_matrix = np.array(matrix.corr(method=correlation_method))
    return matrix, correlation_matrix
