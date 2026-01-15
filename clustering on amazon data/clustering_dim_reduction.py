"""
Domain-Adaptive Dimensionality Reduction for Clustering

This script implements unsupervised domain adaptation techniques by:
1. Computing PCA on the domain corpus
2. Domain-specific whitening (covariance reweighting)
3. Variance-based dimension weighting

Tests if these approaches improve clustering performance on K-Means and HDBSCAN.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import hdbscan
import warnings
warnings.filterwarnings('ignore')

# Import custom metrics
from metrics import evaluate_clustering


def load_data(filepath: str = 'reviews_clustering_results_with_topics.csv'):
    """Load the clustering results CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} reviews with {df['topic'].nunique()} unique topics")
    return df


def prepare_embeddings(df: pd.DataFrame, text_column: str = 'text', 
                       model_name: str = 'all-MiniLM-L6-v2'):
    """
    Generate embeddings using SentenceTransformer.
    
    Returns:
        X: Raw embeddings matrix
        model: The SentenceTransformer model
    """
    print(f"\nGenerating embeddings using {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Combine text and title if available
    if 'title' in df.columns:
        text_data = (df['title'].fillna('') + ' ' + df[text_column].fillna('')).values
    else:
        text_data = df[text_column].fillna('').values
    
    embeddings = model.encode(text_data, show_progress_bar=True, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    return embeddings, model


def apply_pca_projection(X: np.ndarray, n_components: int = None, variance_ratio: float = 0.95):
    """
    Apply PCA to keep top components that explain variance_ratio of variance.
    
    This keeps the domain-specific subspace with maximum variance.
    
    Args:
        X: Original embeddings
        n_components: Explicit number of components (if None, use variance_ratio)
        variance_ratio: Keep components that explain this much variance
    
    Returns:
        X_transformed: PCA-projected embeddings
        pca: Fitted PCA object
    """
    if n_components is None:
        pca = PCA(n_components=variance_ratio, random_state=42)
    else:
        pca = PCA(n_components=n_components, random_state=42)
    
    X_transformed = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_.sum()
    n_comp = pca.n_components_
    
    print(f"  PCA: kept {n_comp} components (explaining {explained_var:.2%} variance)")
    
    return X_transformed, pca


def apply_domain_whitening(X: np.ndarray):
    """
    Apply domain-specific whitening transformation.
    
    This downweights dominant "global" directions and amplifies
    rare but informative directions specific to the domain.
    
    Transform: x' = Σ^{-1/2} (x - μ)
    
    Args:
        X: Original embeddings
    
    Returns:
        X_whitened: Whitened embeddings
        mean: Domain mean
        cov: Domain covariance
    """
    # Compute mean and covariance
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    # Compute covariance
    cov = np.cov(X_centered.T)
    
    # Eigendecomposition for whitening
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Avoid division by zero
    epsilon = 1e-5
    eigenvalues = np.maximum(eigenvalues, epsilon)
    
    # Whitening transform: Σ^{-1/2}
    whitening_transform = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    
    # Apply whitening
    X_whitened = X_centered @ whitening_transform
    
    print(f"  Domain Whitening: transformed to zero mean, unit covariance")
    
    return X_whitened, mean, cov


def apply_variance_weighting(X: np.ndarray, percentile_threshold: float = 10):
    """
    Apply variance-based dimension weighting.
    
    Downweight or remove dimensions with low variance in the domain corpus.
    These dimensions are either constant or noisy for this specific domain.
    
    Args:
        X: Original embeddings
        percentile_threshold: Drop dimensions below this variance percentile
    
    Returns:
        X_weighted: Variance-weighted embeddings
        weights: Per-dimension weights
    """
    # Compute per-dimension variance
    variances = np.var(X, axis=0)
    
    # Compute variance threshold
    threshold = np.percentile(variances, percentile_threshold)
    
    # Create weights: sqrt(variance) for high-variance dims, 0 for low-variance
    weights = np.where(variances > threshold, np.sqrt(variances), 0)
    
    # Normalize weights
    weights = weights / (np.linalg.norm(weights) + 1e-8)
    
    # Apply weights
    X_weighted = X * weights
    
    # L2 normalize
    X_weighted = X_weighted / (np.linalg.norm(X_weighted, axis=1, keepdims=True) + 1e-8)
    
    n_kept = np.sum(weights > 0)
    print(f"  Variance Weighting: kept {n_kept}/{len(weights)} dimensions (>{percentile_threshold}th percentile)")
    
    return X_weighted, weights


def apply_hybrid_approach(X: np.ndarray, pca_components: int = 200):
    """
    Hybrid approach: PCA first, then variance weighting.
    
    Args:
        X: Original embeddings
        pca_components: Number of PCA components to keep
    
    Returns:
        X_transformed: Transformed embeddings
    """
    # First apply PCA to reduce to manageable dimensions
    X_pca, _ = apply_pca_projection(X, n_components=pca_components)
    
    # Then apply variance weighting on PCA components
    X_transformed, _ = apply_variance_weighting(X_pca, percentile_threshold=10)
    
    print(f"  Hybrid: PCA({pca_components}) + Variance Weighting")
    
    return X_transformed


def test_clustering(X: np.ndarray, ground_truth: np.ndarray, 
                   method_name: str, n_clusters: int):
    """
    Test both K-Means and HDBSCAN on given embeddings.
    
    Args:
        X: Feature matrix
        ground_truth: True cluster labels
        method_name: Name of the transformation method
        n_clusters: Number of clusters for K-Means
    
    Returns:
        Dictionary with results for both algorithms
    """
    results = {'method': method_name}
    
    # Test K-Means
    print(f"\n  Testing K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_metrics = evaluate_clustering(X, kmeans_labels, ground_truth)
    
    results['kmeans'] = {
        'labels': kmeans_labels,
        'metrics': kmeans_metrics
    }
    
    print(f"    ARI: {kmeans_metrics['adjusted_rand_index']:.4f}, "
          f"Silhouette: {kmeans_metrics['silhouette_score']:.4f}, "
          f"NMI: {kmeans_metrics['normalized_mutual_info']:.4f}")
    
    # Test HDBSCAN with parameters that worked best in previous comparison
    # From hdbscan_comparison_results.csv: min_cluster_size=5, min_samples=1 gave best results
    print(f"  Testing HDBSCAN...")
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        hdbscan_labels = clusterer.fit_predict(X)
        
        # Filter out noise points for metrics
        mask = hdbscan_labels != -1
        n_clusters_found = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
        n_noise = np.sum(~mask)
        
        print(f"    Found {n_clusters_found} clusters, {n_noise} noise points")
        
        if np.sum(mask) > 10 and n_clusters_found >= 2:
            hdbscan_metrics = evaluate_clustering(
                X[mask], hdbscan_labels[mask], ground_truth[mask]
            )
            
            results['hdbscan'] = {
                'labels': hdbscan_labels,
                'metrics': hdbscan_metrics,
                'n_clusters': n_clusters_found,
                'n_noise': n_noise
            }
            
            print(f"    ARI: {hdbscan_metrics['adjusted_rand_index']:.4f}, "
                  f"Silhouette: {hdbscan_metrics['silhouette_score']:.4f}, "
                  f"NMI: {hdbscan_metrics['normalized_mutual_info']:.4f}")
        else:
            print(f"    Skipped: too few valid clusters or too many noise points")
            results['hdbscan'] = None
            
    except Exception as e:
        print(f"    HDBSCAN failed: {e}")
        results['hdbscan'] = None
    
    return results


def run_comparison_experiment(X_original: np.ndarray, ground_truth: np.ndarray, 
                              n_true_clusters: int):
    """
    Run comprehensive comparison experiment with granular parameter testing.
    
    Args:
        X_original: Original embeddings
        ground_truth: True cluster labels
        n_true_clusters: Number of clusters in ground truth
    
    Returns:
        Dictionary with all results
    """
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE COMPARISON EXPERIMENT")
    print("="*80)
    
    all_results = []
    exp_num = 1
    
    # 1. Baseline: Original embeddings
    print(f"\n[{exp_num}] Testing BASELINE (Original Embeddings)")
    results = test_clustering(X_original, ground_truth, 
                             'Baseline', n_true_clusters)
    all_results.append(results)
    exp_num += 1
    
    # 2. PCA with different variance thresholds
    print("\n" + "="*80)
    print("PCA EXPERIMENTS - Variance Ratios")
    print("="*80)
    for variance_ratio in [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]:
        print(f"\n[{exp_num}] Testing PCA ({variance_ratio*100:.0f}% variance)")
        X_pca, pca_obj = apply_pca_projection(X_original, variance_ratio=variance_ratio)
        results = test_clustering(X_pca, ground_truth, 
                                 f'PCA_{variance_ratio*100:.0f}%', n_true_clusters)
        all_results.append(results)
        exp_num += 1
    
    # 3. PCA with fixed number of components
    print("\n" + "="*80)
    print("PCA EXPERIMENTS - Fixed Components")
    print("="*80)
    for n_comp in [50, 100, 150, 200, 250, 300]:
        if n_comp < X_original.shape[1]:
            print(f"\n[{exp_num}] Testing PCA ({n_comp} components)")
            X_pca, _ = apply_pca_projection(X_original, n_components=n_comp)
            results = test_clustering(X_pca, ground_truth, 
                                     f'PCA_{n_comp}d', n_true_clusters)
            all_results.append(results)
            exp_num += 1
    
    # 4. Domain Whitening
    print("\n" + "="*80)
    print("DOMAIN WHITENING EXPERIMENTS")
    print("="*80)
    print(f"\n[{exp_num}] Testing Domain Whitening")
    X_whitened, _, _ = apply_domain_whitening(X_original)
    results = test_clustering(X_whitened, ground_truth, 
                             'Whitening', n_true_clusters)
    all_results.append(results)
    exp_num += 1
    
    # 5. Variance Weighting with different thresholds
    print("\n" + "="*80)
    print("VARIANCE WEIGHTING EXPERIMENTS")
    print("="*80)
    for percentile in [5, 10, 15, 20, 25, 30]:
        print(f"\n[{exp_num}] Testing Variance Weighting (>{percentile}th percentile)")
        X_var_weighted, weights = apply_variance_weighting(X_original, 
                                                           percentile_threshold=percentile)
        results = test_clustering(X_var_weighted, ground_truth, 
                                 f'VarWeight_p{percentile}', n_true_clusters)
        all_results.append(results)
        exp_num += 1
    
    # 6. Hybrid approaches with different PCA components
    print("\n" + "="*80)
    print("HYBRID EXPERIMENTS (PCA + Variance Weighting)")
    print("="*80)
    for pca_comp in [100, 150, 200, 250]:
        if pca_comp < X_original.shape[1]:
            for var_percentile in [10, 15, 20]:
                print(f"\n[{exp_num}] Testing Hybrid (PCA={pca_comp}, VarWeight>{var_percentile}%)")
                # PCA first
                X_pca, _ = apply_pca_projection(X_original, n_components=pca_comp)
                # Then variance weighting
                X_hybrid, _ = apply_variance_weighting(X_pca, 
                                                       percentile_threshold=var_percentile)
                results = test_clustering(X_hybrid, ground_truth, 
                                         f'Hybrid_PCA{pca_comp}_VW{var_percentile}', 
                                         n_true_clusters)
                all_results.append(results)
                exp_num += 1
    
    # 7. PCA followed by whitening
    print("\n" + "="*80)
    print("PCA + WHITENING EXPERIMENTS")
    print("="*80)
    for pca_comp in [100, 150, 200]:
        if pca_comp < X_original.shape[1]:
            print(f"\n[{exp_num}] Testing PCA ({pca_comp}) + Whitening")
            X_pca, _ = apply_pca_projection(X_original, n_components=pca_comp)
            X_pca_whitened, _, _ = apply_domain_whitening(X_pca)
            results = test_clustering(X_pca_whitened, ground_truth, 
                                     f'PCA{pca_comp}_Whitening', n_true_clusters)
            all_results.append(results)
            exp_num += 1
    
    print(f"\n\nTotal experiments completed: {len(all_results)}")
    return all_results


def print_summary(all_results: list):
    """Print comprehensive summary of all experiments."""
    print("\n" + "="*80)
    print("SUMMARY: CLUSTERING PERFORMANCE COMPARISON")
    print("="*80)
    
    # Prepare summary table
    summary_data = []
    
    for result in all_results:
        method = result['method']
        
        # K-Means metrics
        kmeans_metrics = result['kmeans']['metrics']
        row_kmeans = {
            'Method': method,
            'Algorithm': 'K-Means',
            'ARI': kmeans_metrics['adjusted_rand_index'],
            'NMI': kmeans_metrics['normalized_mutual_info'],
            'Silhouette': kmeans_metrics['silhouette_score'],
            'Homogeneity': kmeans_metrics['homogeneity'],
            'Completeness': kmeans_metrics['completeness'],
            'DB_Index': kmeans_metrics['davies_bouldin_index']
        }
        summary_data.append(row_kmeans)
        
        # HDBSCAN metrics (if available)
        if result['hdbscan'] is not None:
            hdbscan_metrics = result['hdbscan']['metrics']
            row_hdbscan = {
                'Method': method,
                'Algorithm': 'HDBSCAN',
                'ARI': hdbscan_metrics['adjusted_rand_index'],
                'NMI': hdbscan_metrics['normalized_mutual_info'],
                'Silhouette': hdbscan_metrics['silhouette_score'],
                'Homogeneity': hdbscan_metrics['homogeneity'],
                'Completeness': hdbscan_metrics['completeness'],
                'DB_Index': hdbscan_metrics['davies_bouldin_index']
            }
            summary_data.append(row_hdbscan)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by ARI for each algorithm
    kmeans_df = summary_df[summary_df['Algorithm'] == 'K-Means'].copy()
    kmeans_df = kmeans_df.sort_values('ARI', ascending=False)
    
    hdbscan_df = summary_df[summary_df['Algorithm'] == 'HDBSCAN'].copy()
    if len(hdbscan_df) > 0:
        hdbscan_df = hdbscan_df.sort_values('ARI', ascending=False)
    
    # Print TOP 10 K-Means results
    print("\n🔝 TOP 10 K-Means Results (sorted by ARI):")
    print("-" * 100)
    top_kmeans = kmeans_df.head(10)
    print(top_kmeans[['Method', 'ARI', 'NMI', 'Silhouette', 'Homogeneity', 'Completeness']].to_string(index=False))
    
    # Print TOP 10 HDBSCAN results
    if len(hdbscan_df) > 0:
        print("\n\n🔝 TOP 10 HDBSCAN Results (sorted by ARI):")
        print("-" * 100)
        top_hdbscan = hdbscan_df.head(10)
        print(top_hdbscan[['Method', 'ARI', 'NMI', 'Silhouette', 'Homogeneity', 'Completeness']].to_string(index=False))
    
    # Find best methods by different metrics
    print("\n" + "="*80)
    print("BEST PERFORMING METHODS BY DIFFERENT METRICS")
    print("="*80)
    
    # K-Means best methods
    print("\n📊 K-MEANS:")
    print("-" * 80)
    
    baseline_kmeans = kmeans_df[kmeans_df['Method'] == 'Baseline'].iloc[0]
    
    for metric in ['ARI', 'NMI', 'Silhouette', 'Homogeneity']:
        best = kmeans_df.loc[kmeans_df[metric].idxmax()]
        baseline_val = baseline_kmeans[metric]
        improvement = ((best[metric] - baseline_val) / abs(baseline_val + 1e-9)) * 100
        
        print(f"\n  Best by {metric}: {best['Method']}")
        print(f"    Value: {best[metric]:.4f} (baseline: {baseline_val:.4f}, {improvement:+.1f}%)")
    
    # HDBSCAN best methods
    if len(hdbscan_df) > 0:
        print("\n\n📊 HDBSCAN:")
        print("-" * 80)
        
        baseline_hdbscan_rows = hdbscan_df[hdbscan_df['Method'] == 'Baseline']
        has_baseline = len(baseline_hdbscan_rows) > 0
        
        for metric in ['ARI', 'NMI', 'Silhouette', 'Homogeneity']:
            best = hdbscan_df.loc[hdbscan_df[metric].idxmax()]
            
            if has_baseline:
                baseline_val = baseline_hdbscan_rows.iloc[0][metric]
                improvement = ((best[metric] - baseline_val) / abs(baseline_val + 1e-9)) * 100
                print(f"\n  Best by {metric}: {best['Method']}")
                print(f"    Value: {best[metric]:.4f} (baseline: {baseline_val:.4f}, {improvement:+.1f}%)")
            else:
                print(f"\n  Best by {metric}: {best['Method']}")
                print(f"    Value: {best[metric]:.4f} (no baseline for comparison)")
    
    # Overall winners
    print("\n" + "="*80)
    print("🏆 OVERALL WINNERS")
    print("="*80)
    
    best_kmeans_overall = kmeans_df.iloc[0]
    print(f"\n✨ Best K-Means Configuration: {best_kmeans_overall['Method']}")
    print(f"   ARI: {best_kmeans_overall['ARI']:.4f}")
    print(f"   NMI: {best_kmeans_overall['NMI']:.4f}")
    print(f"   Silhouette: {best_kmeans_overall['Silhouette']:.4f}")
    improvement = ((best_kmeans_overall['ARI'] - baseline_kmeans['ARI']) / abs(baseline_kmeans['ARI'] + 1e-9)) * 100
    print(f"   Improvement over baseline: {improvement:+.2f}%")
    
    if len(hdbscan_df) > 0:
        best_hdbscan_overall = hdbscan_df.iloc[0]
        print(f"\n✨ Best HDBSCAN Configuration: {best_hdbscan_overall['Method']}")
        print(f"   ARI: {best_hdbscan_overall['ARI']:.4f}")
        print(f"   NMI: {best_hdbscan_overall['NMI']:.4f}")
        print(f"   Silhouette: {best_hdbscan_overall['Silhouette']:.4f}")
        
        if has_baseline:
            improvement = ((best_hdbscan_overall['ARI'] - baseline_hdbscan_rows.iloc[0]['ARI']) / 
                          abs(baseline_hdbscan_rows.iloc[0]['ARI'] + 1e-9)) * 100
            print(f"   Improvement over baseline: {improvement:+.2f}%")
    
    # Save results
    summary_df.to_csv('dimensionality_reduction_results.csv', index=False)
    print("\n" + "="*80)
    print("Results saved to 'dimensionality_reduction_results.csv'")
    print("="*80)
    
    return summary_df


def main():
    """Main execution function."""
    print("="*80)
    print("DOMAIN-ADAPTIVE DIMENSIONALITY REDUCTION FOR CLUSTERING")
    print("="*80)
    
    # Load data
    df = load_data('reviews_clustering_results_with_topics.csv')
    
    # Prepare embeddings
    X_original, model = prepare_embeddings(df, text_column='text')
    
    # Prepare ground truth
    ground_truth = pd.Categorical(df['topic']).codes
    n_true_clusters = len(np.unique(ground_truth))
    
    print(f"\nGround truth: {n_true_clusters} unique clusters")
    print(f"Original embedding dimension: {X_original.shape[1]}")
    
    # Run experiments
    all_results = run_comparison_experiment(X_original, ground_truth, n_true_clusters)
    
    # Print summary
    summary_df = print_summary(all_results)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print("  - Tested multiple PCA variance thresholds: 80%, 85%, 90%, 95%, 98%, 99%")
    print("  - Tested multiple PCA fixed dimensions: 50, 100, 150, 200, 250, 300")
    print("  - Tested variance weighting with different percentiles: 5, 10, 15, 20, 25, 30")
    print("  - Tested hybrid approaches combining PCA + variance weighting")
    print("  - Tested PCA + whitening combinations")
    print(f"  - Total experiments: {len(summary_df)}")
    print("\n  Check 'dimensionality_reduction_results.csv' for full results!")
    print("  Look for consistent patterns across multiple configurations.")


if __name__ == "__main__":
    main()
