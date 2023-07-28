
import umap
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def generate_data(n_obs, true_dims, n_redundant_per_true, true_gen_func, redundant_gen_noise_func, sd_ratio):
    """
    Generates data matrix with true dimensions and redundant dimensions.
    
    Args:
    - n_obs (int): Number of observations
    - true_dims (int): Number of true dimensions
    - n_redundant_per_true (int): Number of redundant dimensions per true dimension
    - true_gen_func (function): Function to generate the main matrix
    - redundant_gen_noise_func (function): Function to generate noise for redundant dimensions
    - sd_ratio (float): Ratio for scaling noise
    
    Returns:
    - out_mat (numpy matrix): Final data matrix
    """
    
    # Generate the main matrix
    main_mat = true_gen_func(n_obs, true_dims)
    
    # Measure the standard deviation of the main matrix
    main_mat_std = np.std(main_mat)
    
    # Generate the redundant matrix
    redundant_mat = np.hstack([redundant_gen_noise_func(n_obs, true_dims) * sd_ratio * main_mat_std 
                               for _ in range(n_redundant_per_true)])
    
    # Combine the matrices
    out_mat = np.hstack([main_mat, redundant_mat])
    
    return out_mat

def dim_reduction(in_mat, dim_red_func_list, dim_red_names, final_dims):
    """
    Reduces the dimensions of the input matrix using specified functions.
    
    Args:
    - in_mat (numpy matrix): Input data matrix
    - dim_red_func_list (list): List of dimension reduction functions
    - final_dims (int): Number of dimensions after reduction
    
    Returns:
    - dim_red_mat (numpy matrix): Dimension reduced matrix
    """
    
    # Placeholder for the results from each dimension reduction function
    results = []
    
    for func in dim_red_func_list:
        result = func(in_mat, final_dims)
        results.append(result)
        
    return dim_red_mat



def tsne_wrapper(data, n_components):
    """
    Wrapper for t-SNE.
    
    Args:
    - data (numpy array): Input data matrix
    - n_components (int): Number of dimensions after reduction
    
    Returns:
    - Reduced data (numpy array)
    """
    tsne = TSNE(n_components=n_components)
    return tsne.fit_transform(data)

def umap_wrapper(data, n_components):
    """
    Wrapper for UMAP.
    
    Args:
    - data (numpy array): Input data matrix
    - n_components (int): Number of dimensions after reduction
    
    Returns:
    - Reduced data (numpy array)
    """
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit_transform(data)


def true_gen_func(n_obs, true_dims):
    """
    Example function to generate the main matrix.
    
    Args:
    - n_obs (int): Number of observations
    - true_dims (int): Number of true dimensions
    
    Returns:
    - Main matrix (numpy array)
    """
    return np.random.randn(n_obs, true_dims)


def redundant_gen_noise_func(n_obs, true_dims):
    """
    Example function to generate noise for redundant dimensions.
    
    Args:
    - n_obs (int): Number of observations
    - true_dims (int): Number of true dimensions
    
    Returns:
    - Noise matrix (numpy array)
    """
    return np.random.randn(n_obs, true_dims)



# Parameters for the experiment
n_obs = 1000
true_dims = 2
n_redundant_per_true = 100
sd_ratios = []
for sd_ratio in sd_ratios:
    sd_ratio = 0.01
    final_dims = true_dims  # This is just an example; adjust as needed

    # Generate data
    data = generate_data(n_obs, true_dims, n_redundant_per_true, true_gen_func, redundant_gen_noise_func, sd_ratio)

    # Perform dimension reduction
    dim_red_funcs = [tsne_wrapper, umap_wrapper]
    reduced_data = dim_reduction(data, dim_red_funcs, ["tSNE","UMAP"], final_dims)

    # For now, the result will only contain the reduced data from the first function (tSNE in this case).
    # You can extend this to handle and analyze results from multiple dimension reduction functions.

