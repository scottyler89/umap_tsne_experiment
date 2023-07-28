# Do tSNE and UMAP overfit their intrinsic dimensionality (Yes)

This repository presents an analysis of the intrinsic dimensionality of data that has true dimensions and additional noise dimensions. The primary goal is to explore how various noise levels influence the perceived dimensionality and how different dimensionality reduction methods represent the data.

## Intention

The main intention behind this analysis is to:
1. Generate synthetic data with known "true dimensions" and introduce redundant noisy dimensions.
2. Observe how different dimension reduction techniques, namely t-SNE and UMAP, represent the data with increasing noise.
3. Estimate the intrinsic dimensionality as noise levels increase and analyze its implications.

## Methods Walkthrough

1. **Data Generation**: 
   - We started by creating synthetic data with known dimensions using a random number generator.
   - For each true dimension, redundant dimensions were added. These dimensions were seeded from the true data but had noise introduced, which was controlled by a specified standard deviation ratio (`sd_ratio`).

2. **Dimension Reduction**:
   - For each noise level, dimensionality was reduced using two popular methods: t-SNE and UMAP.
   - The reduced dimensions were then visualized against the true dimensions to observe the effects of noise.

3. **Intrinsic Dimensionality Estimation**:
   - At each noise level, the intrinsic dimensionality of the data was estimated using the DANCo method.
   - This helped us understand how perceived dimensionality changes as noise increases.

## Results

Okay - so folks have griped that the "false creation of struction from nothingness" might only happen when you "reduce" from 2 dimentions to 2 dimentions. Of course, that's not the point of dimension reduction - the point is to reduce dimetions. So - what if the "intrinsic" dimentionality is 2, but there are lots of redundant dimentions? In this situation, we have 2 "real dimentions" that could explain most of the variation in the data. We'll first simulate those 2 "real dimentions" (left hand column in the plots below), then we'll create 100 redundant dimentions per real dimention for our 1000 observations (+variable amounts of noise for the redundant dimentions: rows). So here, the input fed into the tSNE and UMAP algs are actually 1000 rows (observations), with 200 features (columns). But they are generated from 2 main features + variable amounts of noise. Do we still see structure from nothing when we are actually performing dimensionality reduction from 200 features to 2, knowing that the underlying 2 main features are unrelated? Yes. 

![True Dimensions vs Dimension Reduction](assets/true_dims_with_noise_vs_dim_reduction.png)

From the above figure, we can see that t-SNE and UMAP (default params) create the appearance of structure from random Gaussian distributions, even when doing a 100 fold dimension reduction, knowing what the true dimensions are. We also see that as noise increases, this structure gets blurier (usurprising, we'll circle back to that).

![Heatmap and Scatters](assets/heatmap_and_scatters.png)

This visualization provides a heatmap of observed data against the ground truth data. The scatter plots showcase the relationship between the noisy dimensions and the original data. A red line of best fit is added for clarity.

![Intrinsic Dimensions Increase with Noise](assets/intrinsic_dims_increase_with_noise.png)

This plot highlights an interesting phenomenon. As noise levels (or the `sd_ratio`) increase, the estimated intrinsic dimensionality also rises. This resonates with the notion that added noise in one dimension is perceived as adding its own dimension. The challenge lies in deciphering which dimensions are "meaningful" and which are mere noise.

## Conclusion

Noise, when added to data, can complicate the structure and perceived dimensionality of the dataset. While dimensionality reduction techniques like t-SNE and UMAP strive to represent this complexity, it's crucial to understand the source of the data and the nature of the noise when interpreting the results.

## References

- [DANCo: Dimensionality from Angle and Norm Concentration. Camastra & Vinciarelli, 2012](https://doi.org/10.48550/arXiv.1206.3881)
