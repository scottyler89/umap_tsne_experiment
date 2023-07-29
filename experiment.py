from scipy.stats import pearsonr
from scipy.spatial import distance_matrix
from sklearn.decomposition import NMF
from minisom import MiniSom
import skdim
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def get_main_mat(true_gen_func, n_obs, true_dims, separation=0):
    """
    Generate the main matrix with an optional separation between two clusters.
    
    Args:
    - true_gen_func (function): Function to generate the main matrix
    - n_obs (int): Number of observations
    - true_dims (int): Number of true dimensions
    - separation (float): Multiple of standard deviation for separating clusters
    
    Returns:
    - main_mat (numpy array): Generated main matrix
    """
    if separation == 0:
        return true_gen_func(n_obs, true_dims)

    # Generate half of the observations
    half_obs = n_obs // 2
    cluster_1 = true_gen_func(half_obs, true_dims)

    # Generate the other half with an offset in the first dimension
    cluster_2 = true_gen_func(half_obs, true_dims)
    offset = separation * np.std(cluster_1[:, 0])
    cluster_2[:, 0] += offset
    # Concatenate the two clusters vertically
    main_mat = np.vstack([cluster_1, cluster_2])

    return main_mat




def generate_data(n_obs, true_dims, n_redundant_per_true, true_gen_func, redundant_gen_noise_func, sd_ratio, separation=0):
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
    
    main_mat = get_main_mat(true_gen_func, n_obs, true_dims, separation=separation)
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


####################################################
def tsne_wrapper(data, n_components):
    tsne = TSNE(n_components=n_components)
    return tsne.fit_transform(data)


def umap_wrapper(data, n_components):
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit_transform(data)


def pca_wrapper(data, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


def nmf_wrapper(data, n_components, epsilon = 1e-8):
    """
    Use NMF for dimensionality reduction.
    
    Parameters:
    - data: input data
    - n_components: number of components for the reduced dimension
    
    Returns:
    - transformed_data: numpy array of shape (n_samples, n_components)
    """
    data -= np.min(data)
    data += epsilon
    nmf = NMF(n_components=n_components, init='random', random_state=0)
    transformed_data = nmf.fit_transform(data)
    return transformed_data


def som_wrapper(data, n_components=2):
    """
    Use MiniSom for SOM.
    Note: For SOM, n_components is expected to be 2 since we are using a 2D grid.
    
    Parameters:
    - data: input data
    - n_components: dimensions of the output (expected to be 2 for a 2D grid)
    
    Returns:
    - positions: numpy array of shape (n_samples, n_components) representing positions on the grid
    """
    assert n_components == 2, "For SOM, n_components should be 2."

    x_size, y_size = 50, 50  # You can adjust these values based on your needs
    som = MiniSom(x_size, y_size, data.shape[1])
    som.train_random(data, 5000)

    positions = np.array([som.winner(d) for d in data])
    return positions


#####################################################

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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5 * n_rows))

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


def plot_distance_correlations(true_dim_dict, results_dict):
    """
    Plots scatter plots of true pairwise distances and pairwise distances from dimension reduction methods.
    
    Args:
    - true_dim_dict (dict): Dictionary of true dimensions for each sd_ratio
    - results_dict (dict): Dictionary of results for each sd_ratio and each dimension reduction method
    
    """
    # Number of rows is the number of sd_ratios
    n_rows = len(results_dict)

    # Number of columns is 1 (for true dimensions) + number of dimension reduction methods
    n_cols = 1 + len(next(iter(results_dict.values())))

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5 * n_rows))

    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # Loop through each sd_ratio and plot
    for i, sd_ratio in enumerate(results_dict):
        # Compute true pairwise distances and flatten
        true_distances = distance_matrix(
            true_dim_dict[sd_ratio], true_dim_dict[sd_ratio])
        true_distances_flat = true_distances.flatten()

        # Plot true distances against themselves
        axes[i, 0].scatter(true_distances_flat,
                           true_distances_flat, alpha=0.6, s=5)
        r_val, p_val = pearsonr(true_distances_flat, true_distances_flat)
        axes[i, 0].set_title(f"r={r_val:.2f}, p={p_val:.2e}", fontsize=22)

        # Loop through each dimension reduction method and plot
        for j, method in enumerate(results_dict[sd_ratio]):
            # Compute pairwise distances for reduced data and flatten
            reduced_distances = distance_matrix(
                results_dict[sd_ratio][method], results_dict[sd_ratio][method])
            reduced_distances_flat = reduced_distances.flatten()

            # Plot true distances against reduced distances
            axes[i, j+1].scatter(true_distances_flat,
                                 reduced_distances_flat, alpha=0.025, s=5)

            # Compute correlation
            r_val, p_val = pearsonr(
                true_distances_flat, reduced_distances_flat)
            axes[i, j +
                 1].set_title(f"r={r_val:.2f}, p={p_val:.2e}", fontsize=22)

    # Add column titles
    col_titles = ['True Distances'] + \
        list(next(iter(results_dict.values())).keys())
    for ax, col in zip(axes[0], col_titles):
        ax.annotate(col, (0.5, 1.15), xycoords='axes fraction', ha='center',
                    va='center', fontsize=26, textcoords='offset points')

    for ax_row in axes:
        for ax in ax_row:
            for spine in ax.spines.values():
                spine.set_linewidth(2)

    # Save the figure
    plt.savefig("assets/distance_correlations.png", dpi=300)



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
separation_vect = [0,4]
sep_dict = {}
intrinsic_dim_estimate_dict = {}
true_dim_dict = {}
obs_data_dict = {}
results_dict = {}
sd_lookup = {}
for sep in separation_vect:
    sep_name = "Clust Sep:"+str()
for sd_ratio in sd_ratios:
    sd_name = "SD ratio:"+str(sd_ratio)
    sd_lookup[sd_name] = sd_ratio
    final_dims = true_dims  # This is just an example; adjust as needed
    # Generate data
    true_dim_data, obs_data = generate_data(n_obs, true_dims, n_redundant_per_true, true_gen_func, redundant_gen_noise_func, sd_ratio, separation=4)

    # Estimates of intrinsic dimensionality.
    # Interesting note here, but it actually identifies
    # that as noise dimensions are added, and the size of the noise relative to
    # dims are 'real dims.' This fits with the model of it finding 
    # that added noise in one dimension is actually adding its own dimension, even if the 'real'
    # variation was already accounted for by prior dims. It's not like this is incorrect or anything...
    # It's just that noise is a dimension. The hard part is figuring out which dims are "meaningful"!
    ## https: // doi.org/10.48550/arXiv.1206.3881
    danco = skdim.id.DANCo().fit(obs_data)
    print(danco.dimension_)
    intrinsic_dim_estimate_dict[sd_name] = danco.dimension_

    # log the data
    true_dim_dict[sd_name] = true_dim_data
    obs_data_dict[sd_name] = obs_data

    # Perform dimension reduction
    dim_red_funcs = [pca_wrapper, nmf_wrapper, tsne_wrapper, umap_wrapper, som_wrapper]
    dim_red_names = ["PCA", "NMF", "tSNE", "UMAP", "SOM"]
    results_dict[sd_name] = dim_reduction(obs_data, dim_red_funcs, dim_red_names, final_dims)


# Call the function to plot
plot_dim_reductions(true_dim_dict, results_dict)



# Call the function to plot
plot_obs_data_heatmap(true_dim_dict, obs_data_dict,
                      intrinsic_dim_estimate_dict)


# Call the function to plot
plot_intrinsic_dimensionality(sd_lookup, intrinsic_dim_estimate_dict)


plot_distance_correlations(true_dim_dict, results_dict)

