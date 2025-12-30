"""
Essential Clustering Evaluation Metrics - Implemented from Scratch

This module provides essential metrics to evaluate clustering quality.
All metrics are implemented from scratch using only numpy.

Internal metrics: Silhouette Score, Davies-Bouldin Index
External metrics: Adjusted Rand Index, Normalized Mutual Information
"""

import numpy as np
from typing import Optional, Dict, Any
from scipy.spatial.distance import cdist


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Silhouette Score - measures cluster cohesion and separation.
    
    For each sample i:
    - a(i) = average distance to other points in same cluster
    - b(i) = average distance to points in nearest different cluster
    - s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    Final score is the mean of all s(i).
    
    Range: [-1, 1], where higher is better.
    - 1: clusters are well separated, points far from neighboring clusters
    - 0: clusters are overlapping, points on decision boundary
    - -1: points may be assigned to wrong clusters
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels for each sample
    
    Returns:
        float: Mean silhouette coefficient across all samples
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0.0
    
    n_samples = X.shape[0]
    silhouette_values = np.zeros(n_samples)
    
    # Calculate pairwise distances between all points
    distances = cdist(X, X, metric='euclidean')
    
    for i in range(n_samples):
        cluster_i = labels[i]
        
        # Points in same cluster
        same_cluster_mask = labels == cluster_i
        same_cluster_distances = distances[i, same_cluster_mask]
        
        # a(i): mean distance to points in same cluster (excluding self)
        if len(same_cluster_distances) > 1:
            a_i = np.mean(same_cluster_distances[same_cluster_distances > 0])
        else:
            a_i = 0.0
        
        # b(i): smallest mean distance to points in different clusters
        b_i = np.inf
        for other_cluster in unique_labels:
            if other_cluster != cluster_i:
                other_cluster_mask = labels == other_cluster
                other_cluster_distances = distances[i, other_cluster_mask]
                mean_dist = np.mean(other_cluster_distances)
                b_i = min(b_i, mean_dist)
        
        # Silhouette value for point i
        if max(a_i, b_i) > 0:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_values[i] = 0.0
    
    return np.mean(silhouette_values)


def davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Davies-Bouldin Index - measures cluster separation.
    
    For each cluster, finds the cluster that is most similar to it.
    Similarity is based on within-cluster scatter and between-cluster separation.
    Lower values indicate better clustering (well-separated clusters).
    
    Range: [0, ∞), where lower is better.
    - 0: perfect clustering (theoretical minimum)
    - Higher values indicate overlapping or poorly separated clusters
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels for each sample
    
    Returns:
        float: Davies-Bouldin index
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        return float('inf')
    
    # Calculate cluster centroids and within-cluster scatter
    centroids = np.zeros((n_clusters, X.shape[1]))
    scatter = np.zeros(n_clusters)
    
    for idx, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        centroids[idx] = np.mean(cluster_points, axis=0)
        
        # Within-cluster scatter (average distance to centroid)
        distances = np.linalg.norm(cluster_points - centroids[idx], axis=1)
        scatter[idx] = np.mean(distances)
    
    # Calculate Davies-Bouldin index
    db_values = []
    for i in range(n_clusters):
        max_ratio = 0.0
        for j in range(n_clusters):
            if i != j:
                # Distance between centroids
                centroid_distance = np.linalg.norm(centroids[i] - centroids[j])
                
                if centroid_distance > 0:
                    # Ratio of within-cluster scatter to between-cluster separation
                    ratio = (scatter[i] + scatter[j]) / centroid_distance
                    max_ratio = max(max_ratio, ratio)
        
        db_values.append(max_ratio)
    
    return np.mean(db_values)


def adjusted_rand_index(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Calculate Adjusted Rand Index (ARI) - measures clustering agreement.
    
    Counts pairs of points that are:
    - In same cluster in both labelings (true positive)
    - In different clusters in both labelings (true negative)
    
    Adjusted for chance (expected value of random labeling is 0).
    
    Range: [-1, 1], where higher is better.
    - 1: perfect agreement between clusterings
    - 0: agreement equal to random chance
    - Negative: agreement worse than random
    
    Args:
        true_labels: Ground truth cluster labels
        predicted_labels: Predicted cluster labels
    
    Returns:
        float: Adjusted Rand Index
    """
    # Build contingency table
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(predicted_labels)
    
    contingency = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)
    
    for i, true_class in enumerate(true_classes):
        for j, pred_class in enumerate(pred_classes):
            contingency[i, j] = np.sum((true_labels == true_class) & 
                                       (predicted_labels == pred_class))
    
    # Sum of combinations of pairs in each cell
    sum_comb_c = np.sum(contingency * (contingency - 1) / 2)
    
    # Sum over rows and columns
    sum_rows = np.sum(contingency, axis=1)
    sum_cols = np.sum(contingency, axis=0)
    
    # Sum of combinations for rows and columns
    sum_comb_rows = np.sum(sum_rows * (sum_rows - 1) / 2)
    sum_comb_cols = np.sum(sum_cols * (sum_cols - 1) / 2)
    
    # Total number of samples and combinations
    n = np.sum(contingency)
    sum_comb_n = n * (n - 1) / 2
    
    # Calculate ARI
    expected_index = sum_comb_rows * sum_comb_cols / sum_comb_n
    max_index = (sum_comb_rows + sum_comb_cols) / 2
    
    if max_index - expected_index == 0:
        return 0.0
    
    ari = (sum_comb_c - expected_index) / (max_index - expected_index)
    return ari


def entropy(labels: np.ndarray) -> float:
    """
    Calculate entropy of a label distribution.
    
    Entropy measures the uncertainty in the labels.
    
    Args:
        labels: Array of cluster labels
    
    Returns:
        float: Entropy value
    """
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    
    return -np.sum(probabilities * np.log2(probabilities))


def mutual_information(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Calculate Mutual Information between two labelings.
    
    Measures how much information is shared between the two labelings.
    
    Args:
        true_labels: Ground truth cluster labels
        predicted_labels: Predicted cluster labels
    
    Returns:
        float: Mutual information value
    """
    # Build contingency table
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(predicted_labels)
    
    contingency = np.zeros((len(true_classes), len(pred_classes)), dtype=np.float64)
    
    for i, true_class in enumerate(true_classes):
        for j, pred_class in enumerate(pred_classes):
            contingency[i, j] = np.sum((true_labels == true_class) & 
                                       (predicted_labels == pred_class))
    
    n = len(true_labels)
    
    # Calculate MI
    mi = 0.0
    for i in range(len(true_classes)):
        for j in range(len(pred_classes)):
            if contingency[i, j] > 0:
                # Joint probability
                p_ij = contingency[i, j] / n
                # Marginal probabilities
                p_i = np.sum(contingency[i, :]) / n
                p_j = np.sum(contingency[:, j]) / n
                
                mi += p_ij * np.log2(p_ij / (p_i * p_j))
    
    return mi


def normalized_mutual_info(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Calculate Normalized Mutual Information (NMI).
    
    Normalizes mutual information by the arithmetic mean of entropies.
    This makes the metric bounded between 0 and 1.
    
    Range: [0, 1], where higher is better.
    - 1: perfect correlation between labelings
    - 0: no correlation (independent labelings)
    
    Args:
        true_labels: Ground truth cluster labels
        predicted_labels: Predicted cluster labels
    
    Returns:
        float: Normalized Mutual Information
    """
    mi = mutual_information(true_labels, predicted_labels)
    
    h_true = entropy(true_labels)
    h_pred = entropy(predicted_labels)
    
    # Normalize by arithmetic mean of entropies
    if h_true + h_pred == 0:
        return 0.0
    
    nmi = 2 * mi / (h_true + h_pred)
    return nmi


def homogeneity_score(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Calculate Homogeneity Score.
    
    Homogeneity measures whether each cluster contains only members of a single class.
    A clustering result satisfies homogeneity if all clusters contain only data points
    which are members of a single class.
    
    Formula: h = 1 - H(C|K) / H(C)
    where H(C|K) is conditional entropy of classes given clusters
    and H(C) is entropy of classes
    
    Range: [0, 1], where higher is better.
    - 1: perfectly homogeneous labeling
    - 0: no homogeneity
    
    Args:
        true_labels: Ground truth cluster labels
        predicted_labels: Predicted cluster labels
    
    Returns:
        float: Homogeneity score
    """
    # If only one cluster or one class, homogeneity is meaningless but set to 1
    if len(np.unique(true_labels)) == 1 or len(np.unique(predicted_labels)) == 1:
        return 1.0
    
    # Calculate H(C) - entropy of true classes
    h_c = entropy(true_labels)
    
    if h_c == 0:
        return 1.0
    
    # Calculate H(C|K) - conditional entropy of classes given clusters
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(predicted_labels)
    
    # Build contingency table
    contingency = np.zeros((len(true_classes), len(pred_classes)), dtype=np.float64)
    
    for i, true_class in enumerate(true_classes):
        for j, pred_class in enumerate(pred_classes):
            contingency[i, j] = np.sum((true_labels == true_class) & 
                                       (predicted_labels == pred_class))
    
    n = len(true_labels)
    
    # Calculate conditional entropy H(C|K)
    h_c_given_k = 0.0
    for j in range(len(pred_classes)):
        # Probability of cluster j
        p_k = np.sum(contingency[:, j]) / n
        
        if p_k > 0:
            # Entropy within cluster j
            entropy_k = 0.0
            for i in range(len(true_classes)):
                if contingency[i, j] > 0:
                    p_c_given_k = contingency[i, j] / np.sum(contingency[:, j])
                    entropy_k -= p_c_given_k * np.log2(p_c_given_k)
            
            h_c_given_k += p_k * entropy_k
    
    # Homogeneity = 1 - H(C|K) / H(C)
    homogeneity = 1.0 - (h_c_given_k / h_c) if h_c > 0 else 1.0
    
    return homogeneity


def completeness_score(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Calculate Completeness Score.
    
    Completeness measures whether all members of a given class are assigned
    to the same cluster. A clustering result satisfies completeness if all
    data points that are members of a given class are elements of the same cluster.
    
    Formula: c = 1 - H(K|C) / H(K)
    where H(K|C) is conditional entropy of clusters given classes
    and H(K) is entropy of clusters
    
    Range: [0, 1], where higher is better.
    - 1: perfectly complete labeling
    - 0: no completeness
    
    Args:
        true_labels: Ground truth cluster labels
        predicted_labels: Predicted cluster labels
    
    Returns:
        float: Completeness score
    """
    # If only one cluster or one class, completeness is meaningless but set to 1
    if len(np.unique(true_labels)) == 1 or len(np.unique(predicted_labels)) == 1:
        return 1.0
    
    # Calculate H(K) - entropy of predicted clusters
    h_k = entropy(predicted_labels)
    
    if h_k == 0:
        return 1.0
    
    # Calculate H(K|C) - conditional entropy of clusters given classes
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(predicted_labels)
    
    # Build contingency table
    contingency = np.zeros((len(true_classes), len(pred_classes)), dtype=np.float64)
    
    for i, true_class in enumerate(true_classes):
        for j, pred_class in enumerate(pred_classes):
            contingency[i, j] = np.sum((true_labels == true_class) & 
                                       (predicted_labels == pred_class))
    
    n = len(true_labels)
    
    # Calculate conditional entropy H(K|C)
    h_k_given_c = 0.0
    for i in range(len(true_classes)):
        # Probability of class i
        p_c = np.sum(contingency[i, :]) / n
        
        if p_c > 0:
            # Entropy within class i
            entropy_c = 0.0
            for j in range(len(pred_classes)):
                if contingency[i, j] > 0:
                    p_k_given_c = contingency[i, j] / np.sum(contingency[i, :])
                    entropy_c -= p_k_given_c * np.log2(p_k_given_c)
            
            h_k_given_c += p_c * entropy_c
    
    # Completeness = 1 - H(K|C) / H(K)
    completeness = 1.0 - (h_k_given_c / h_k) if h_k > 0 else 1.0
    
    return completeness


def evaluate_clustering(X: np.ndarray, 
                       predicted_labels: np.ndarray,
                       true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate clustering using all available metrics.
    
    Computes internal metrics (always) and external metrics (if ground truth provided).
    
    Args:
        X: Feature matrix (n_samples, n_features)
        predicted_labels: Predicted cluster labels
        true_labels: Ground truth cluster labels (optional)
    
    Returns:
        dict: Dictionary containing all metric scores
    """
    results = {}
    
    # Internal metrics (don't need ground truth)
    try:
        results['silhouette_score'] = silhouette_score(X, predicted_labels)
    except Exception as e:
        results['silhouette_score'] = None
        print(f"Error calculating silhouette score: {e}")
    
    try:
        results['davies_bouldin_index'] = davies_bouldin_index(X, predicted_labels)
    except Exception as e:
        results['davies_bouldin_index'] = None
        print(f"Error calculating Davies-Bouldin index: {e}")
    
    # External metrics (need ground truth)
    if true_labels is not None:
        try:
            results['adjusted_rand_index'] = adjusted_rand_index(true_labels, predicted_labels)
        except Exception as e:
            results['adjusted_rand_index'] = None
            print(f"Error calculating ARI: {e}")
        
        try:
            results['normalized_mutual_info'] = normalized_mutual_info(true_labels, predicted_labels)
        except Exception as e:
            results['normalized_mutual_info'] = None
            print(f"Error calculating NMI: {e}")
        
        try:
            results['homogeneity'] = homogeneity_score(true_labels, predicted_labels)
        except Exception as e:
            results['homogeneity'] = None
            print(f"Error calculating homogeneity: {e}")
        
        try:
            results['completeness'] = completeness_score(true_labels, predicted_labels)
        except Exception as e:
            results['completeness'] = None
            print(f"Error calculating completeness: {e}")
    
    return results


def print_metrics(metrics: Dict[str, float], title: str = "Clustering Metrics"):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Title for the report
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Internal metrics
    internal = ['silhouette_score', 'davies_bouldin_index']
    has_internal = any(m in metrics for m in internal)
    
    if has_internal:
        print("\nINTERNAL METRICS:")
        print("-" * 60)
        for metric in internal:
            if metric in metrics and metrics[metric] is not None:
                print(f"  {metric:.<40} {metrics[metric]:>8.4f}")
    
    # External metrics
    external = ['adjusted_rand_index', 'normalized_mutual_info', 'homogeneity', 'completeness']
    has_external = any(m in metrics for m in external)
    
    if has_external:
        print("\nEXTERNAL METRICS (vs ground truth):")
        print("-" * 60)
        for metric in external:
            if metric in metrics and metrics[metric] is not None:
                print(f"  {metric:.<40} {metrics[metric]:>8.4f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                          cluster_std=0.60, random_state=42)
    
    # Cluster with k-means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # Evaluate
    metrics = evaluate_clustering(X, y_pred, y_true)
    print_metrics(metrics, "K-Means Clustering Evaluation")
