# Clustering Comparison on Amazon Reviews

This project performs a comprehensive comparison of clustering algorithms (K-Means and HDBSCAN) on Amazon product review data. The analysis evaluates different clustering configurations using various clustering quality metrics, comparing them against ground truth topic labels.

## Overview

The project takes existing clustered Amazon review data and systematically evaluates multiple clustering algorithms with different parameter configurations to find the optimal clustering approach.

## Features

- **Multiple Clustering Algorithms**: K-Means and HDBSCAN
- **Parameter Search**: Systematic grid search across algorithm hyperparameters
- **Semantic Embeddings**: Uses SentenceTransformer for generating high-quality text embeddings
- **Custom Metrics**: All clustering metrics implemented from scratch using NumPy
- **Comprehensive Evaluation**: Both internal and external clustering quality metrics

## Files

### Main Scripts

- **`clustering_comparison.py`**: Main script that orchestrates the clustering comparison
  - Loads Amazon review data with existing topic labels
  - Generates SentenceTransformer embeddings from review text
  - Performs parameter search for K-Means (varying k)
  - Performs parameter search for HDBSCAN (varying min_cluster_size and min_samples)
  - Evaluates all configurations against ground truth
  - Identifies best performing models
  - Exports results to CSV files

- **`metrics.py`**: Custom implementation of clustering evaluation metrics
  - **Internal metrics** (no ground truth needed):
    - Silhouette Score
    - Davies-Bouldin Index
  - **External metrics** (compared to ground truth):
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    - Homogeneity Score
    - Completeness Score
    - V-Measure

### Data Files

- **`reviews_clustering_results_with_topics.csv`**: Input data containing Amazon reviews with existing topic labels (used as ground truth)
- **`kmeans_comparison_results.csv`**: Results from K-Means parameter search
- **`hdbscan_comparison_results.csv`**: Results from HDBSCAN parameter search
- **`comparison_summary.csv`**: Summary comparison of best models

## Key Methods

### Embedding Generation
- Uses `SentenceTransformer` (all-MiniLM-L6-v2 model) to generate semantic embeddings
- Combines review title and text for richer representation
- Optionally incorporates rating as an additional feature

### K-Means Parameter Search
- Tests multiple values of k (number of clusters)
- Range typically set around the ground truth number of clusters (±5-10)
- Evaluates each configuration using all metrics

### HDBSCAN Parameter Search
- Tests combinations of:
  - `min_cluster_size`: [5, 10, 15, 20, 30, 50]
  - `min_samples`: [1, 5, 10, 15, 20]
- Handles noise points (-1 labels) appropriately
- Filters results with too few clusters or excessive noise

## Metrics Explained

### Internal Metrics (No Ground Truth Needed)

**Silhouette Score** (Range: [-1, 1], higher is better)
- Measures cluster cohesion and separation
- +1: Clusters are well-separated
- 0: Overlapping clusters
- -1: Points may be misclassified

**Davies-Bouldin Index** (Range: [0, ∞), lower is better)
- Measures cluster separation based on within-cluster scatter
- 0: Perfect clustering (theoretical minimum)
- Higher values indicate poor separation

### External Metrics (Require Ground Truth)

**Adjusted Rand Index (ARI)** (Range: [-1, 1], higher is better)
- Measures agreement between predicted and ground truth clustering
- +1: Perfect agreement
- 0: Random chance level
- Negative: Worse than random

**Normalized Mutual Information (NMI)** (Range: [0, 1], higher is better)
- Information-theoretic measure of clustering quality
- 1: Perfect agreement
- 0: No mutual information

**Homogeneity** (Range: [0, 1], higher is better)
- Whether each cluster contains only members of a single class

**Completeness** (Range: [0, 1], higher is better)
- Whether all members of a given class are in the same cluster

**V-Measure** (Range: [0, 1], higher is better)
- Harmonic mean of homogeneity and completeness

## Usage

### Basic Usage

```python
python clustering_comparison.py
```

### Expected Workflow

1. Script loads the reviews data from `reviews_clustering_results_with_topics.csv`
2. Generates SentenceTransformer embeddings for all reviews
3. Performs K-Means parameter search over multiple k values
4. Performs HDBSCAN parameter search over parameter grid
5. Evaluates each configuration using all metrics
6. Identifies and displays best performing models
7. Exports detailed results to CSV files

### Output

The script prints:
- Progress updates during embedding generation
- Results for each parameter configuration tested
- Best model configurations for each algorithm
- Winner declaration based on Adjusted Rand Index
- Summary statistics comparing best models

Generated CSV files contain complete results for further analysis and visualization.

## Dependencies

```python
pandas
numpy
scikit-learn
sentence-transformers
hdbscan
scipy
```

## Installation

```bash
pip install pandas numpy scikit-learn sentence-transformers hdbscan scipy
```

## Implementation Notes

### Why SentenceTransformer?
- Produces semantically meaningful embeddings
- Captures context better than TF-IDF
- Pre-trained on large corpora for better generalization
- Model `all-MiniLM-L6-v2` offers good balance of speed and quality

### Custom Metrics Implementation
- All metrics implemented from scratch using NumPy
- No reliance on sklearn's metric implementations
- Educational value: understanding metric calculations
- Full control over computation details

### HDBSCAN Considerations
- Hierarchical density-based clustering
- Automatically determines number of clusters
- Can identify noise points (outliers)
- More robust to cluster shape than K-Means
- Requires careful parameter tuning

## Results Interpretation

### What to Look For

1. **High ARI/NMI**: Strong agreement with ground truth topic labels
2. **High Silhouette**: Well-separated, cohesive clusters
3. **Low Davies-Bouldin**: Good cluster separation
4. **Balance of Homogeneity/Completeness**: Neither too fine-grained nor too coarse

### Trade-offs

- **K-Means**: Fast, simple, requires k specification, assumes spherical clusters
- **HDBSCAN**: Flexible cluster shapes, finds optimal k, handles noise, slower, more parameters

## Future Enhancements

- Visualization of clustering results (t-SNE/UMAP projections)
- Additional algorithms (Gaussian Mixture Models, Spectral Clustering)
- Automatic optimal parameter selection
- Topic modeling integration (LDA comparison)
- Interactive cluster exploration dashboard

## Author Notes

This project demonstrates best practices in clustering evaluation:
- Systematic parameter search
- Multiple evaluation metrics
- Comparison to ground truth
- Clean, modular code structure
- Comprehensive documentation
