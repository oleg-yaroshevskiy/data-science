"""
Clustering Comparison: K-Means and HDBSCAN with Parameter Search

This script loads your existing clustering results and compares various
K-Means and HDBSCAN configurations against your original clustering
(treated as ground truth).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import hdbscan
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Import our custom metrics
from metrics import evaluate_clustering, print_metrics


def load_data(filepath: str = 'reviews_clustering_results_with_topics.csv'):
    """
    Load the clustering results CSV file.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        DataFrame with the clustering results
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} reviews with {df['topic'].nunique()} unique topics")
    return df


def prepare_features(df: pd.DataFrame, text_column: str = 'text', 
                     model_name: str = 'all-MiniLM-L6-v2'):
    """
    Prepare feature matrix from text data using SentenceTransformer embeddings.
    
    SentenceTransformer produces semantically meaningful embeddings that
    capture context better than TF-IDF for clustering tasks.
    
    Args:
        df: DataFrame with text data
        text_column: Name of column containing text
        model_name: Name of the SentenceTransformer model
                   'all-MiniLM-L6-v2' - fast and good quality (384 dim)
                   'all-mpnet-base-v2' - higher quality but slower (768 dim)
    
    Returns:
        Feature matrix (numpy array) and model
    """
    print(f"\nPreparing SentenceTransformer embeddings using {model_name}...")
    
    # Load pre-trained model
    model = SentenceTransformer(model_name)
    
    # Combine text and title if available
    if 'title' in df.columns:
        text_data = (df['title'].fillna('') + ' ' + df[text_column].fillna('')).values
    else:
        text_data = df[text_column].fillna('').values
    
    # Generate embeddings
    print(f"Encoding {len(text_data)} texts...")
    embeddings = model.encode(text_data, show_progress_bar=True, batch_size=32)
    
    # Optionally include rating as a feature
    if 'rating' in df.columns:
        rating_scaled = StandardScaler().fit_transform(df[['rating']].fillna(0))
        X = np.hstack([embeddings, rating_scaled])
    else:
        X = embeddings
    
    print(f"Feature matrix shape: {X.shape}")
    return X, model


def kmeans_parameter_search(X: np.ndarray, ground_truth: np.ndarray, 
                            k_range: range = range(2, 21)):
    """
    Perform K-Means clustering with different values of k.
    
    Args:
        X: Feature matrix
        ground_truth: Original cluster labels (treated as ground truth)
        k_range: Range of k values to try
    
    Returns:
        DataFrame with results for each k value
    """
    print("\n" + "="*80)
    print("K-MEANS PARAMETER SEARCH")
    print("="*80)
    
    results = []
    
    for k in k_range:
        print(f"\nTesting K-Means with k={k}...")
        
        # Fit K-Means
        kmeans = KMeans(
            n_clusters=k, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        metrics = evaluate_clustering(X, labels, ground_truth)
        
        # Store results
        result = {
            'algorithm': 'KMeans',
            'n_clusters': k,
            'inertia': kmeans.inertia_
        }
        result.update(metrics)
        results.append(result)
        
        # Print brief summary
        print(f"  ARI: {result['adjusted_rand_index']:.4f}, "
              f"NMI: {result['normalized_mutual_info']:.4f}, "
              f"Homogeneity: {result['homogeneity']:.4f}, "
              f"Completeness: {result['completeness']:.4f}")
    
    return pd.DataFrame(results)


def hdbscan_parameter_search(X: np.ndarray, ground_truth: np.ndarray,
                             min_cluster_sizes: list = [5, 10, 15, 20, 30],
                             min_samples_list: list = [1, 5, 10, 15]):
    """
    Perform HDBSCAN clustering with different parameter combinations.
    
    Args:
        X: Feature matrix
        ground_truth: Original cluster labels (treated as ground truth)
        min_cluster_sizes: List of min_cluster_size values to try
        min_samples_list: List of min_samples values to try
    
    Returns:
        DataFrame with results for each parameter combination
    """
    print("\n" + "="*80)
    print("HDBSCAN PARAMETER SEARCH")
    print("="*80)
    
    results = []
    
    # Try different parameter combinations
    param_combinations = list(product(min_cluster_sizes, min_samples_list))
    
    for min_cluster_size, min_samples in param_combinations:
        print(f"\nTesting HDBSCAN with min_cluster_size={min_cluster_size}, "
              f"min_samples={min_samples}...")
        
        try:
            # Fit HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(X)
            
            # Count clusters (excluding noise points labeled as -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"  Found {n_clusters} clusters, {n_noise} noise points")
            
            # Skip if too few clusters or all noise
            if n_clusters < 2:
                print("  Skipping: too few clusters")
                continue
            
            # For metrics, filter out noise points
            mask = labels != -1
            if np.sum(mask) < 10:
                print("  Skipping: too many noise points")
                continue
            
            X_filtered = X[mask]
            labels_filtered = labels[mask]
            ground_truth_filtered = ground_truth[mask]
            
            # Calculate metrics
            metrics = evaluate_clustering(X_filtered, labels_filtered, 
                                         ground_truth_filtered)
            
            # Store results
            result = {
                'algorithm': 'HDBSCAN',
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / len(labels)
            }
            result.update(metrics)
            results.append(result)
            
            # Print brief summary
            print(f"  ARI: {result['adjusted_rand_index']:.4f}, "
                  f"NMI: {result['normalized_mutual_info']:.4f}, "
                  f"Homogeneity: {result['homogeneity']:.4f}, "
                  f"Completeness: {result['completeness']:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return pd.DataFrame(results)


def find_best_models(kmeans_results: pd.DataFrame, 
                     hdbscan_results: pd.DataFrame,
                     metric: str = 'adjusted_rand_index'):
    """
    Find the best performing models based on a specific metric.
    
    Args:
        kmeans_results: DataFrame with K-Means results
        hdbscan_results: DataFrame with HDBSCAN results
        metric: Metric to use for comparison
    
    Returns:
        Dictionary with best models
    """
    print("\n" + "="*80)
    print(f"BEST MODELS (based on {metric})")
    print("="*80)
    
    # Find best K-Means
    best_kmeans_idx = kmeans_results[metric].idxmax()
    best_kmeans = kmeans_results.loc[best_kmeans_idx]
    
    print("\nBest K-Means Configuration:")
    print(f"  k = {int(best_kmeans['n_clusters'])}")
    print(f"  {metric} = {best_kmeans[metric]:.4f}")
    print(f"  Silhouette Score = {best_kmeans['silhouette_score']:.4f}")
    print(f"  NMI = {best_kmeans['normalized_mutual_info']:.4f}")
    print(f"  Homogeneity = {best_kmeans['homogeneity']:.4f}")
    print(f"  Completeness = {best_kmeans['completeness']:.4f}")
    
    # Find best HDBSCAN (if results available)
    if len(hdbscan_results) > 0:
        best_hdbscan_idx = hdbscan_results[metric].idxmax()
        best_hdbscan = hdbscan_results.loc[best_hdbscan_idx]
        
        print("\nBest HDBSCAN Configuration:")
        print(f"  min_cluster_size = {int(best_hdbscan['min_cluster_size'])}")
        print(f"  min_samples = {int(best_hdbscan['min_samples'])}")
        print(f"  n_clusters = {int(best_hdbscan['n_clusters'])}")
        print(f"  {metric} = {best_hdbscan[metric]:.4f}")
        print(f"  Silhouette Score = {best_hdbscan['silhouette_score']:.4f}")
        print(f"  NMI = {best_hdbscan['normalized_mutual_info']:.4f}")
        print(f"  Homogeneity = {best_hdbscan['homogeneity']:.4f}")
        print(f"  Completeness = {best_hdbscan['completeness']:.4f}")
    
    return {
        'best_kmeans': best_kmeans,
        'best_hdbscan': best_hdbscan if len(hdbscan_results) > 0 else None
    }


def compare_to_ground_truth(df: pd.DataFrame, X: np.ndarray, 
                           ground_truth_column: str = 'topic'):
    """
    Main comparison function.
    
    Args:
        df: DataFrame with original clustering results
        X: Feature matrix
        ground_truth_column: Column name with ground truth labels
    """
    # Encode ground truth labels as integers
    ground_truth = pd.Categorical(df[ground_truth_column]).codes
    n_true_clusters = len(np.unique(ground_truth))
    
    print(f"\nGround truth has {n_true_clusters} unique clusters")
    print(f"Cluster distribution:")
    print(df[ground_truth_column].value_counts())
    
    # K-Means parameter search
    # Test around the true number of clusters
    k_min = max(2, n_true_clusters - 5)
    k_max = n_true_clusters + 10
    kmeans_results = kmeans_parameter_search(X, ground_truth, 
                                             range(k_min, k_max + 1))
    
    # HDBSCAN parameter search
    hdbscan_results = hdbscan_parameter_search(
        X, ground_truth,
        min_cluster_sizes=[5, 10, 15, 20, 30, 50],
        min_samples_list=[1, 5, 10, 15, 20]
    )
    
    # Save results
    kmeans_results.to_csv('kmeans_comparison_results.csv', index=False)
    print("\nK-Means results saved to 'kmeans_comparison_results.csv'")
    
    if len(hdbscan_results) > 0:
        hdbscan_results.to_csv('hdbscan_comparison_results.csv', index=False)
        print("HDBSCAN results saved to 'hdbscan_comparison_results.csv'")
    
    # Find and display best models
    best_models = find_best_models(kmeans_results, hdbscan_results)
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON TO GROUND TRUTH")
    print("="*80)
    
    print(f"\nGround Truth Clusters: {n_true_clusters}")
    print(f"\nBest K-Means:")
    print(f"  k = {int(best_models['best_kmeans']['n_clusters'])}")
    print(f"  ARI = {best_models['best_kmeans']['adjusted_rand_index']:.4f}")
    print(f"  Silhouette = {best_models['best_kmeans']['silhouette_score']:.4f}")
    print(f"  Homogeneity = {best_models['best_kmeans']['homogeneity']:.4f}")
    print(f"  Completeness = {best_models['best_kmeans']['completeness']:.4f}")
    
    if best_models['best_hdbscan'] is not None:
        print(f"\nBest HDBSCAN:")
        print(f"  clusters = {int(best_models['best_hdbscan']['n_clusters'])}")
        print(f"  min_cluster_size = {int(best_models['best_hdbscan']['min_cluster_size'])}")
        print(f"  min_samples = {int(best_models['best_hdbscan']['min_samples'])}")
        print(f"  ARI = {best_models['best_hdbscan']['adjusted_rand_index']:.4f}")
        print(f"  Silhouette = {best_models['best_hdbscan']['silhouette_score']:.4f}")
        print(f"  Homogeneity = {best_models['best_hdbscan']['homogeneity']:.4f}")
        print(f"  Completeness = {best_models['best_hdbscan']['completeness']:.4f}")
        
        # Declare winner
        kmeans_ari = best_models['best_kmeans']['adjusted_rand_index']
        hdbscan_ari = best_models['best_hdbscan']['adjusted_rand_index']
        
        print(f"\n{'🏆 WINNER: ' if hdbscan_ari > kmeans_ari else ''}HDBSCAN" if hdbscan_ari > kmeans_ari 
              else f"\n{'🏆 WINNER: ' if kmeans_ari > hdbscan_ari else ''}K-Means")
        print(f"  (ARI: {max(kmeans_ari, hdbscan_ari):.4f} vs {min(kmeans_ari, hdbscan_ari):.4f})")
    
    # Create comparison plot data
    comparison_data = []
    for _, row in kmeans_results.iterrows():
        comparison_data.append({
            'Algorithm': 'K-Means',
            'k': row['n_clusters'],
            'ARI': row['adjusted_rand_index'],
            'NMI': row['normalized_mutual_info'],
            'Silhouette': row['silhouette_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('comparison_summary.csv', index=False)
    print("\nComparison summary saved to 'comparison_summary.csv'")
    
    return kmeans_results, hdbscan_results, best_models


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("CLUSTERING COMPARISON: K-Means & HDBSCAN vs Ground Truth")
    print("="*80)
    
    # Load data
    df = load_data('reviews_clustering_results_with_topics.csv')
    
    # Prepare features using SentenceTransformer (better than TF-IDF)
    X, model = prepare_features(df, text_column='text', 
                                model_name='all-MiniLM-L6-v2')
    
    # Run comparison
    kmeans_results, hdbscan_results, best_models = compare_to_ground_truth(
        df, X, ground_truth_column='topic'
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - kmeans_comparison_results.csv")
    print("  - hdbscan_comparison_results.csv")
    print("  - comparison_summary.csv")
    print("\nYou can analyze these results further or visualize the metrics!")


if __name__ == "__main__":
    main()
