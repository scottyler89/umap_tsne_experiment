
import matplotlib.pyplot as plt
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
    - main_mat (numpy matrix): Matrix of true dimensions
    - redundant_mat (numpy matrix): Matrix of redundant dimensions
    """

    # Generate the main matrix
    main_mat = true_gen_func(n_obs, true_dims)

    # Placeholder for the redundant dimensions
    redundant_dims = []

    for i in range(true_dims):
        # Standard deviation for this dimension in main_mat
        dim_std = np.std(main_mat[:, i])

        # Create n_redundant_per_true redundant dimensions seeded at main_mat[:, i] values
        for _ in range(n_redundant_per_true):
            noise = redundant_gen_noise_func(n_obs, 1)
            redundant_dim = main_mat[:, i][:,
                                           np.newaxis] + noise * sd_ratio * dim_std

            # Standardize the redundant dimension
            redundant_dim = redundant_dim / np.std(redundant_dim)

            redundant_dims.append(redundant_dim)

    # Stack all redundant dimensions horizontally
    redundant_mat = np.hstack(redundant_dims)

    return main_mat, redundant_mat


def dim_reduction(in_mat, dim_red_func_list, dim_red_names, final_dims):
    """
    Reduces the dimensions of the input matrix using specified functions.
    
    Args:
    - in_mat (numpy matrix): Input data matrix
    - dim_red_func_list (list): List of dimension reduction functions
    - final_dims (int): Number of dimensions after reduction
    
    Returns:
    - results (dict): dictionary of dim_red_names and their results
    """
    
    # Placeholder for the results from each dimension reduction function
    results = {}
    
    for func, name in zip(dim_red_func_list, dim_red_names):
        result = func(in_mat, final_dims)
        results[name]=result
        
    return results



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


#########################################


def plot_dim_reductions(true_dim_dict, results_dict):
    """
    Plots scatter plots of true dimensions and results of dimension reduction methods.
    
    Args:
    - true_dim_dict (dict): Dictionary of true dimensions for each sd_ratio
    - results_dict (dict): Dictionary of results for each sd_ratio and each dimension reduction method
    
    """
    # Number of rows is the number of sd_ratios
    n_rows = len(results_dict)

    # Number of columns is 1 (for true dimensions) + number of dimension reduction methods
    n_cols = 1 + len(next(iter(results_dict.values())))

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Loop through each sd_ratio and plot
    for i, sd_ratio in enumerate(results_dict):
        # Plot true dimensions
        axes[i, 0].scatter(true_dim_dict[sd_ratio][:, 0],
                           true_dim_dict[sd_ratio][:, 1], alpha=0.6)
        axes[i, 0].set_title(f"True Dimensions ({sd_ratio})")

        # Loop through each dimension reduction method and plot
        for j, method in enumerate(results_dict[sd_ratio]):
            axes[i, j+1].scatter(results_dict[sd_ratio][method][:, 0],
                                 results_dict[sd_ratio][method][:, 1], alpha=0.6)
            axes[i, j+1].set_title(f"{method} ({sd_ratio})")

    plt.tight_layout()
    plt.show()




#########################################

# Parameters for the experiment
np.random.seed(123456)
n_obs = 1000
true_dims = 2
n_redundant_per_true = 100
sd_ratios = [0.01, 0.05, 0.25]
true_dim_dict = {}
results_dict = {}
for sd_ratio in sd_ratios:
    sd_name = "SD ratio:"+str(sd_ratio)
    final_dims = true_dims  # This is just an example; adjust as needed
    # Generate data
    true_dim_data, obs_data = generate_data(n_obs, true_dims, n_redundant_per_true, true_gen_func, redundant_gen_noise_func, sd_ratio)

    # log the true dimensions
    true_dim_dict[sd_name] = true_dim_data

    # Perform dimension reduction
    dim_red_funcs = [tsne_wrapper, umap_wrapper]
    dim_red_names = ["tSNE","UMAP"]
    results_dict[sd_name] = dim_reduction(obs_data, dim_red_funcs, dim_red_names, final_dims)


# Call the function to plot
plot_dim_reductions(true_dim_dict, results_dict)


