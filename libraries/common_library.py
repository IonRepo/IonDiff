import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import re

from sys             import exit
from scipy.ndimage   import gaussian_filter1d
from scipy.stats     import pearsonr, spearmanr
from os              import path, stat, system

"""Set of common functions for many libraries, with a general purpose.
"""

# Defining the information lines of the file

scale_line         = 1  # Line for the scale of the simulation box
s_cell_line        = 2  # Start of the definition of the simulation box
e_cell_line        = 4  # End of the definition of the simulation box
name_line          = 5  # Composition of the compound
concentration_line = 6  # Concentration of the compound
x_line             = 7  # Start of the simulation data

def obtain_diffusive_information(composition, concentration, DiffTypeName=None):
    """
    Gets the diffusive and non-diffusive elements.
    Various diffusive elements are allowed. In that case, they shall be specified in DiffTypeName.
    """

    if not DiffTypeName:
        # Getting the diffusive element following the implicit order of preference

        for diff_component in ['Li', 'Na', 'Ag', 'Cu', 'Cl', 'I', 'Br', 'F', 'O']:
            if diff_component in composition:
                DiffTypeName = [diff_component]
                break

    NonDiffTypeName = composition.copy()
    for diff_element in DiffTypeName:
        NonDiffTypeName.remove(diff_element)

    # Getting the positions of the diffusive elements regarding the XDATCAR file

    diffusion_information = []
    for DiffTypeName_value in DiffTypeName:
        for TypeName_index in range(len(composition)):
            if composition[TypeName_index] == DiffTypeName_value:
                diffusion_information.append([TypeName_index, np.sum(concentration[:TypeName_index]),
                                              concentration[TypeName_index]])
    return DiffTypeName, NonDiffTypeName, diffusion_information

def obtain_diffusive_family(element):
    """Returns the family of an element using the obtain_diffusive_information function.
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

def read_INCAR(path_to_simulation):
    """Reads VASP INCAR files. It is always expected to find these parameters.
    """
    
    # Predefining the variable, so later we check if they were found
    
    delta_t = None
    n_steps = None
    
    # Loading the INCAR file
    
    if not path.exists(f'{path_to_simulation}/INCAR'):
        exit('INCAR file is not available.')
    
    with open(f'{path_to_simulation}/INCAR', 'r') as INCAR_file:
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

def read_XDATCAR(path_to_simulation):
    """Reads cell data and coordinates in direct units.
    """

    # Loading data from XDATCAR file
    
    if not path.exists(f'{path_to_simulation}/XDATCAR'):
        exit('XDATCAR file is not available.')
    
    with open(f'{path_to_simulation}/XDATCAR', 'r') as XDATCAR_file:
        XDATCAR_lines = XDATCAR_file.readlines()
    
    # Extracting the data
    
    try:
        scale = float(XDATCAR_lines[scale_line])
    except:
        exit('Wrong definition of the scale in the XDATCAR.')
    
    try:
        cell = np.array([line.split() for line in XDATCAR_lines[s_cell_line:e_cell_line+1]], dtype=float)
        cell *= scale
    except:
        exit('Wrong definition of the cell in the XDATCAR.')

    compounds     = XDATCAR_lines[name_line].split()
    concentration = np.array(XDATCAR_lines[concentration_line].split(), dtype=int)
    
    if len(compounds) != len(concentration):
        exit('Wrong definition of the composition of the compound in the XDATCAR.')
    
    n_ions = sum(concentration)
    
    # Shaping the configurations data into the positions attribute
    
    coordinates = np.array([line.split() for line in XDATCAR_lines[x_line:] if not line.split()[0][0].isalpha()], dtype=float)
    
    # Checking if the number of configurations is correct

    if not (len(coordinates) / n_ions).is_integer():
        exit('The number of lines is not correct in the XDATCAR file.')
    
    coordinates  = coordinates.ravel().reshape((-1, n_ions, 3))
    return cell, n_ions, compounds, concentration, coordinates
    
def load_data(path_to_simulation):
    """Returns all needed data in a suitable shape from a simulation.
    Allows requiring DIFFUSION file to exist.
    POSCAR and XDATCAR lines consider only first three rows.
    """
    
    cell, _, compounds, concentration, coordinates = read_XDATCAR(path_to_simulation)
    
    # Loading hoppings data, if available
    
    hoppings = np.NaN
    
    DIFFUSION_path = f'{path_to_simulation}/DIFFUSION'
    
    if path.exists(DIFFUSION_path):
        if stat(DIFFUSION_path).st_size:
            hoppings = np.loadtxt(DIFFUSION_path)
    else:
        print(f'DIFFUSION file is being computed at {path_to_simulation}.')
        system(f'python3 cli.py identify_diffusion --MD_path {path_to_simulation}')

    return coordinates, hoppings, cell, compounds, concentration

def find_groups(array):
    """Gives the ranges where an array does not have NaN values.
    """

    # Create an array that is 1 where a is `value`, and pad each end with an extra 0.

    isvalue = np.concatenate(([0], np.isfinite(array).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isvalue))

    # Runs start and end where absdiff is 1.

    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def get_separated_groups(matrix, outer=np.NaN):
    """These regions with non-NaNs are found and split.
    The original value is put in the corresponding place.
    """

    (n_conf, n_particles) = np.shape(matrix)
    separated_matrix = np.ones(n_conf) * outer
    key = []

    for particle in range(n_particles):
        matrix_row = matrix[:, particle]
        groups = find_groups(matrix_row)

        if groups.size:
            for i in range(len(groups)):
                aux = np.ones(n_conf) * outer
                aux[groups[i, 0]:groups[i, 1]] = matrix_row[groups[i, 0]:groups[i, 1]]
                separated_matrix = np.vstack([separated_matrix, aux])
                key.append(particle)
    return np.array(key), separated_matrix[1:].T  # First row was initialized with zeros, and it was transposed

def get_expanded_hoppings(n_conf, n_particles, concentration, particle_type, hoppings, method='original'):
    """Expands the compressed information of hoppings into a matrix of (time x particle).
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
    """Returns the correlation matrix.
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
