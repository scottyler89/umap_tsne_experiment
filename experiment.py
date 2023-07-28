import skdim
import seaborn as sns
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
    tsne = TSNE(n_components=n_components)
    return tsne.fit_transform(data)

def umap_wrapper(data, n_components):
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

    # Adjust the spacing between subplots
    # Adjust these values as needed for desired spacing
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # Loop through each sd_ratio and plot
    for i, sd_ratio in enumerate(results_dict):
        # Row title for sd_ratio
        if n_cols > 1:
            axes[i, 0].set_ylabel(sd_ratio, fontsize=28,
                                  rotation=90, labelpad=50, va="center")
        else:
            axes.set_ylabel(sd_ratio, fontsize=28, rotation=90,
                            labelpad=50, va="center")

        # Plot true dimensions
        axes[i, 0].scatter(true_dim_dict[sd_ratio][:, 0],
                           true_dim_dict[sd_ratio][:, 1], alpha=0.6, s=5)
        #axes[i, 0].set_title(f"True Dimensions", fontsize=12)

        # Loop through each dimension reduction method and plot
        for j, method in enumerate(results_dict[sd_ratio]):
            axes[i, j+1].scatter(results_dict[sd_ratio][method][:, 0],
                                 results_dict[sd_ratio][method][:, 1], alpha=0.6, s=5)
            #axes[i, j+1].set_title(f"{method}", fontsize=12)

    # Add column titles
    col_titles = ['Intrinsic Dimensions that\ncreated input vals+noise'] + \
        list(next(iter(results_dict.values())).keys())
    for ax, col in zip(axes[0], col_titles):
        ax.annotate(col, (0.5, 1.15), xycoords='axes fraction', ha='center',
                    va='center', fontsize=28, textcoords='offset points')
    for ax_row in axes:
        for ax in ax_row:
            for spine in ax.spines.values():
                spine.set_linewidth(2)
    plt.savefig("assets/true_dims_with_noise_vs_dim_reduction.png", dpi=300)


def rotate_point_around_origin(point, angle):
    """
    Rotates a point around the origin by a given angle.
    
    Args:
    - point (tuple): (x, y) coordinates of the point
    - angle (float): Angle in radians to rotate
    
    Returns:
    - (x', y'): Rotated coordinates of the point
    """
    x, y = point
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    return x_rot, y_rot


def plot_obs_data_heatmap(gt_data_dict, obs_data_dict, danco_dict):
    """
    Plots heatmaps of obs_data for each sd_ratio.
    
    Args:
    - gt_data_dict (dict): Dictionary of ground truth data matrices for each sd_ratio
    - obs_data_dict (dict): Dictionary of obs_data matrices for each sd_ratio
    
    """
    # Number of rows is the number of sd_ratios
    n_rows = len(obs_data_dict)

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows))
    fig.subplots_adjust(wspace=0.35, hspace=0.38)

    if n_rows == 1:
        axes = [axes]

    # Loop through each sd_ratio and plot
    for i, sd_ratio in enumerate(obs_data_dict.keys()):
        # Extract ground truth and observation data for scatter plot
        temp_gt_data = gt_data_dict[sd_ratio]
        temp_obs_data = obs_data_dict[sd_ratio]
        estimated_dims = danco_dict[sd_ratio]

        # Plot the heatmap
        sns.heatmap(temp_obs_data, ax=axes[i, 0], cmap="YlGnBu", cbar=False)
        axes[i, 0].set_title(f"DANCo dim\nestimate {estimated_dims:.2f}", fontsize=23)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(sd_ratio, fontsize=26, rotation=90,
                              labelpad=50, va="center")

        # Scatter plot
        axes[i, 1].scatter(temp_obs_data[:, 0], temp_gt_data[:, 0], alpha=0.5)
        #axes[i, 1].set_title(f"Scatter for {sd_ratio}", fontsize=14)
        axes[i, 1].set_xlabel("Dim-1 Redundant + noise", fontsize=23)
        axes[i, 1].set_ylabel("Ground-truth Dim-1", fontsize=23)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)

        # Add correlation line
        m, b = np.polyfit(temp_obs_data[:, 0], temp_gt_data[:, 0], 1)
        axes[i, 1].plot(temp_obs_data[:, 0], m *
                        temp_obs_data[:, 0] + b, color='red', linewidth=2)
    for ax_row in axes:
        for ax in ax_row:
            for spine in ax.spines.values():
                spine.set_linewidth(2)
    plt.savefig("assets/heatmap_and_scatters.png", dpi=300)


#########


def plot_intrinsic_dimensionality(sd_lookup, intrinsic_dim_estimate_dict):
    """
    Plots the estimated intrinsic dimensionality against noise levels.
    
    Args:
    - sd_lookup (dict): Dictionary mapping sd_ratio names to their respective values
    - intrinsic_dim_estimate_dict (dict): Dictionary of estimated intrinsic dimensionality for each sd_ratio
    """
    # Extract data
    sd_values = [sd_lookup[key] for key in intrinsic_dim_estimate_dict.keys()]
    dim_estimates = list(intrinsic_dim_estimate_dict.values())

    # Create a scatter plot with loess fit curve
    plt.figure(figsize=(10, 6))
    sns.regplot(x=sd_values, y=dim_estimates, lowess=True, scatter_kws={
                's': 100, 'alpha': 0.6}, line_kws={'color': 'red', 'lw': 2})
    plt.xlabel("Noise Level (SD Ratio)")
    plt.ylabel("Estimated Intrinsic Dimensionality")
    plt.title("Intrinsic Dimensionality vs. Noise Level")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("assets/intrinsic_dims_increase_with_noise.png", dpi=300)


#########
#########################################

# Parameters for the experiment
np.random.seed(123456)
n_obs = 1000
true_dims = 2
n_redundant_per_true = 100
sd_ratios = [0.01, 0.05, 0.25, 0.5, 1.]
intrinsic_dim_estimate_dict = {}
true_dim_dict = {}
obs_data_dict = {}
results_dict = {}
sd_lookup = {}
for sd_ratio in sd_ratios:
    sd_name = "SD ratio:"+str(sd_ratio)
    sd_lookup[sd_name] = sd_ratio
    final_dims = true_dims  # This is just an example; adjust as needed
    # Generate data
    true_dim_data, obs_data = generate_data(n_obs, true_dims, n_redundant_per_true, true_gen_func, redundant_gen_noise_func, sd_ratio)

    # Estimates of intrinsic dimensionality.
    # Interesting note here, but it actually identifies
    # that as noise dimensions are added, and the size of the noise relative to
    #  dims are 'real dims.' This fits with the model of it finding 
    # that added noise in one dimension is actually adding its own dimension, even if the 'real'
    # varyation was already accounted for by prior dims. It's not like this is incorrect or anything...
    # It's just that noise is a dimension. The hard part is figuring out which dims are "meaningful"!
    ## https: // doi.org/10.48550/arXiv.1206.3881
    danco = skdim.id.DANCo().fit(obs_data)
    print(danco.dimension_)
    intrinsic_dim_estimate_dict[sd_name] = danco.dimension_

    # log the data
    true_dim_dict[sd_name] = true_dim_data
    obs_data_dict[sd_name] = obs_data

    # Perform dimension reduction
    dim_red_funcs = [tsne_wrapper, umap_wrapper]
    dim_red_names = ["tSNE","UMAP"]
    results_dict[sd_name] = dim_reduction(obs_data, dim_red_funcs, dim_red_names, final_dims)


# Call the function to plot
plot_dim_reductions(true_dim_dict, results_dict)



# Call the function to plot
plot_obs_data_heatmap(true_dim_dict, obs_data_dict,
                      intrinsic_dim_estimate_dict)


# Call the function to plot
plot_intrinsic_dimensionality(sd_lookup, intrinsic_dim_estimate_dict)

