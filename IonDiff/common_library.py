import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import re
import os
import sys

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.ndimage   import gaussian_filter1d
from scipy.stats     import pearsonr, spearmanr

# Defining the basic parameters for k-means and spectral clustering algorithms
kmeans_kwargs   = dict(init='random', n_init=10, max_iter=300, tol=1e-04,                          random_state=0)
spectral_kwargs = dict(affinity='nearest_neighbors', n_neighbors=1000, assign_labels='cluster_qr', random_state=0)

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
    """Gets the diffusive and non-diffusive elements.
    
    Args:
        composition   (list):           List of element symbols in the compound.
        concentration (list):           List of element concentrations.
        DiffTypeName  (list, optional): List of diffusive element symbols. Defaults to None.
    
    Returns:
        tuple: Tuple containing DiffTypeName (list), NonDiffTypeName (list), and diffusion_information (list).
    
    Example:
        Obtain the diffusive information from a given composition and concentration.
    
        Usage:
            obtain_diffusive_information(['Li', 'O'], [1, 2])
    
    Note:
        If DiffTypeName is not provided, the function automatically selects diffusive elements based on an implicit order of preference.
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
    
    Args:
        element (str): Element symbol.
    
    Returns:
        str: Diffusive family of the element.
    
    Example:
        Obtain the diffusive family of a given element.
    
        Usage:
            obtain_diffusive_family('Li')
    
    Note:
        The function automatically detects components from upper cases and numbers.
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
    
    Args:
        path_to_simulation (str): Path to the simulation directory.
    
    Returns:
        tuple: Tuple containing delta_t (float) and n_steps (float).
    
    Example:
        Read INCAR file from a simulation directory.
    
        Usage:
            read_INCAR('/path/to/simulation')
    
    Note:
        The function expects the presence of specific parameters (POTIM, NBLOCK) in the INCAR file.
    """
    
    # Predefining the variable, so later we check if they were found
    delta_t = None
    n_steps = None
    
    # Loading the INCAR file
    if not os.path.exists(f'{path_to_simulation}/INCAR'):
        sys.exit('INCAR file is not available.')
    
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
        sys.exit('POTIM or NBLOCK are not correctly defined in the INCAR file.')
    return delta_t, n_steps


def read_XDATCAR(path_to_simulation):
    """Reads cell data and coordinates in direct units.
    
    Args:
        path_to_simulation (str): Path to the simulation directory.
    
    Returns:
        tuple: Tuple containing cell (numpy.ndarray), n_ions (int), compounds (list), concentration (numpy.ndarray), and coordinates (numpy.ndarray).
    
    Example:
        Read XDATCAR file from a simulation directory.
    
        Usage:
            read_XDATCAR('/path/to/simulation')
    
    Note:
        The function expects a specific format in the XDATCAR file.
    """

    # Loading data from XDATCAR file
    if not os.path.exists(f'{path_to_simulation}/XDATCAR'):
        sys.exit('XDATCAR file is not available.')
    
    with open(f'{path_to_simulation}/XDATCAR', 'r') as XDATCAR_file:
        XDATCAR_lines = XDATCAR_file.readlines()
    
    # Extracting the data
    try:
        scale = float(XDATCAR_lines[scale_line])
    except:
        sys.exit('Wrong definition of the scale in the XDATCAR.')
    
    try:
        cell = np.array([line.split() for line in XDATCAR_lines[s_cell_line:e_cell_line+1]], dtype=float)
        cell *= scale
    except:
        sys.exit('Wrong definition of the cell in the XDATCAR.')

    compounds     = XDATCAR_lines[name_line].split()
    concentration = np.array(XDATCAR_lines[concentration_line].split(), dtype=int)
    
    if len(compounds) != len(concentration):
        sys.exit('Wrong definition of the composition of the compound in the XDATCAR.')
    
    n_ions = sum(concentration)
    
    # Shaping the configurations data into the positions attribute
    coordinates = np.array([line.split() for line in XDATCAR_lines[x_line:] if not line.split()[0][0].isalpha()], dtype=float)
    
    # Checking if the number of configurations is correct
    if not (len(coordinates) / n_ions).is_integer():
        sys.exit('The number of lines is not correct in the XDATCAR file.')
    
    coordinates  = coordinates.ravel().reshape((-1, n_ions, 3))
    return cell, n_ions, compounds, concentration, coordinates


def load_data(path_to_simulation):
    """Returns all needed data in a suitable shape from a simulation.
    
    Args:
        path_to_simulation (str): Path to the simulation directory.
    
    Returns:
        tuple: Tuple containing coordinates (numpy.ndarray), hoppings (numpy.ndarray), cell (numpy.ndarray), compounds (list), and concentration (numpy.ndarray).
    
    Example:
        Load simulation data from a directory.
    
        Usage:
            load_data('/path/to/simulation')
    
    Note:
        The function requires the existence of specific files such as 'DIFFUSION' and 'XDATCAR'.
    """
    
    cell, _, compounds, concentration, coordinates = read_XDATCAR(path_to_simulation)
    
    # Loading hoppings data, if available
    hoppings = np.NaN
    
    DIFFUSION_path = f'{path_to_simulation}/DIFFUSION'
    
    if os.path.exists(DIFFUSION_path):
        if os.stat(DIFFUSION_path).st_size:
            hoppings = np.loadtxt(DIFFUSION_path)
    else:
        print(f'DIFFUSION file is being computed at {path_to_simulation}.')
        os.system(f'python3 cli.py identify_diffusion --MD_path {path_to_simulation}')

    return coordinates, hoppings, cell, compounds, concentration


def find_groups(array):
    """Gives the ranges where an array does not have NaN values.
    
    Args:
        array (numpy.ndarray): Input array.
    
    Returns:
        numpy.ndarray: Array containing start and end indices of non-NaN regions.
    
    Example:
        Find non-NaN regions in an array.
    
        Usage:
            find_groups(np.array([1, 2, np.NaN, 4, 5, np.NaN, 7]))
    
    Note:
        The function is useful for identifying continuous regions without NaN values.
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
    
    Args:
        matrix (numpy.ndarray): Input matrix.
        outer  (optional):      Value to use for outer regions. Defaults to np.NaN.
    
    Returns:
        tuple: Tuple containing key (numpy.ndarray) and separated_matrix (numpy.ndarray).
    
    Example:
        Separate regions with non-NaNs in a matrix.
    
        Usage:
            get_separated_groups(np.array([[1, 2, 3], [4, 5, 6], [7, np.NaN, 9]]))
    
    Note:
        The function is useful for splitting regions with non-NaN values in a matrix.
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
    
    Args:
        n_conf        (int):           Number of configurations.
        n_particles   (int):           Number of particles.
        concentration (numpy.ndarray): Array of particle concentrations.
        particle_type (list):          List of particle types.
        hoppings      (numpy.ndarray): Array of hoppings.
        method        (str, optional): Expansion method. Defaults to 'original'.
    
    Returns:
        tuple: Tuple containing key (numpy.ndarray) and expanded_hoppings (numpy.ndarray).
    
    Example:
        Expand compressed hopping information into a matrix.
    
        Usage:
            get_expanded_hoppings(100, 3, np.array([2, 1]), ['A', 'B', 'C'], np.array([[1, 0, 1], [0, 1, 1]]))
    
    Note:
        The function provides flexibility in expanding hopping information based on different methods.
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
    
    Args:
        expanded_hoppings  (numpy.ndarray):   Expanded hopping matrix.
        correlation_method (str, optional):   Correlation method. Defaults to 'pearson'.
        gaussian_smoothing (bool, optional):  Enable Gaussian smoothing. Defaults to True.
        sigma              (float, optional): Standard deviation for Gaussian smoothing. Defaults to None.
    
    Returns:
        tuple: Tuple containing matrix (pandas.DataFrame) and correlation_matrix (numpy.ndarray).
    
    Example:
        Calculate the correlation matrix from expanded hopping data.
    
        Usage:
            get_correlation_matrix(np.array([[1, 0, 1], [0, 1, 1]]), correlation_method='spearman')
    
    Note:
        The function allows customization of correlation calculations and Gaussian smoothing.
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


def information_from_VASPfile(path_to_VASPfile, file='POSCAR'):
    """Returns cell, composition, concentration, and positions of particles from a VASP file.

    Args:
        path_to_VASPfile (str): The path to the directory containing the VASP file.
        file             (str, optional): The VASP file type, 'POSCAR' or 'XDATCAR'. Default to POSCAR file.

    Returns:
        cell          (ndarray): The lattice cell matrix.
        composition   (list):    A list of chemical composition in the form of elements.
        concentration (ndarray): A list of concentrations corresponding to the elements.
        positions     (ndarray): An array of atomic positions.
    """

    with open(f'{path_to_VASPfile}/{file}', 'r') as VASP_file:
        VASPfile_lines = VASP_file.readlines()

    try:
        scale = float(VASPfile_lines[1])
    except:
        exit(f'Wrong definition of the scale in the {file}.')
    
    try:
        cell  = np.array([line.split() for line in VASPfile_lines[2:5]], dtype=float) * scale
    except:
        exit(f'Wrong definition of the cell in the {file}.')

    composition   = VASPfile_lines[5].split()
    concentration = np.array(VASPfile_lines[6].split(), dtype=int)
    total_particles = np.sum(concentration)
    
    # Checking if the number of compounds and concentrations is correct
    
    if len(composition) != len(concentration):
        exit(f'Wrong definition of the composition of the compound in the {file}.')
    
    # Shaping the configurations data into the positions attribute
    
    if file == 'XDATCAR':
        positions = np.array([line.split() for line in VASPfile_lines[7:] if not line.split()[0][0].isalpha()], dtype=float)
        
        # Checking if the number of configurations is correct
        
        if not (len(positions) / total_particles).is_integer():
            exit(f'The number of lines is not correct in the {file} file.')
        
        positions = positions.ravel().reshape((-1, total_particles, 3))
    else:
        positions = np.array([line.split()[:3] for line in VASPfile_lines[8:8 + total_particles]], dtype=float)
    return cell, composition, concentration, positions


def calculate_silhouette(coordinates, method, n_attempts, silhouette_thd):
        """Calculates silhouette scores for different numbers of clusters and selects the optimal number.

        Args:
            coordinates    (numpy.ndarray): Input array of coordinates.
            method         (str):           Clustering method ('K-means' or 'Spectral').
            n_attempts     (int):           Number of attempts for clustering.
            silhouette_thd (float):         Silhouette score threshold for selecting the number of clusters.

        Returns:
            int: Optimal number of clusters.
        """
        
        all_clusters = np.arange(2, n_attempts+1)
        silhouette_averages = []

        # Iterating over each number of clusters
        for n_clusters in all_clusters:
            if   method == 'K-means':  clustering = KMeans(n_clusters=n_clusters,             **kmeans_kwargs)
            elif method == 'Spectral': clustering = SpectralClustering(n_clusters=n_clusters, **spectral_kwargs)
            else:                      sys.exit('Error: clustering method not recognized')
            
            # Getting the labels
            labels = clustering.fit_predict(coordinates)
            
            # Calculate distortion for a range of number of cluster
            silhouette_averages.append(silhouette_score(coordinates, labels))
        
        # Checking if one cluster is selected
        n_clusters = all_clusters[np.argmax(silhouette_averages)]
        if (np.max(silhouette_averages) < silhouette_thd):
            n_clusters = 1
            
        print(f'Number of clusters: {n_clusters} with SA = {np.max(silhouette_averages)}')
        return n_clusters


def calculate_clusters(coordinates, n_clusters, method, distance_thd):
        """Calculates clusters and related information based on the chosen method and number of clusters.

        Args:
            coordinates  (numpy.ndarray): Input array of coordinates.
            n_clusters   (int):           Number of clusters.
            method       (str):           Clustering method ('K-means' or 'Spectral').
            distance_thd (float):         Distance threshold for identifying vibrations.

        Returns:
            tuple: Tuple containing centers, classification, vibration, and cluster_change.
        """
        
        if   method == 'K-means':  clustering = KMeans(n_clusters=n_clusters,             **kmeans_kwargs)
        elif method == 'Spectral': clustering = SpectralClustering(n_clusters=n_clusters, **spectral_kwargs)
        else:                      sys.exit('Error: clustering method not recognized')

        # Getting the labels and centers
        classification = clustering.fit_predict(coordinates)
        cluster_change = np.where(classification[1:] != classification[:-1])[0]
        
        centers = np.zeros((n_clusters, 3))
        distances_to_center = np.zeros(len(coordinates))
        
        for i in range(n_clusters):
            positions = np.where(classification == i)[0]
            centers[i] = np.mean(coordinates[positions], axis=0)
            distances_to_center[positions] = np.linalg.norm(coordinates[positions] - centers[i], axis=1)

        vibration = distances_to_center < distance_thd
        return centers, classification, vibration, cluster_change