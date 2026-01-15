"""
Domain-Adaptive Dimensionality Reduction for Information Retrieval

Uses BEIR datasets for evaluation of domain adaptation techniques.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

BEIR_AVAILABLE = True
try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
except ImportError:
    BEIR_AVAILABLE = False
    print("WARNING: BEIR not installed. Install with: pip install beir")


def compute_retrieval_metrics(query_embeddings, doc_embeddings, qrels, k_values=[10, 100]):
    """
    Compute retrieval metrics: NDCG@k, MAP, Recall@k
    
    Args:
        query_embeddings: Query embeddings (n_queries, dim)
        doc_embeddings: Document embeddings (n_docs, dim)
        qrels: Dict mapping query_id -> {doc_id: relevance_score}
        k_values: List of k values for metrics
    
    Returns:
        Dictionary of metrics
    """
    # Compute similarity matrix
    similarities = cosine_similarity(query_embeddings, doc_embeddings)
    
    # Get ranked doc indices for each query
    ranked_docs = np.argsort(-similarities, axis=1)
    
    metrics = {}
    ndcg_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    ap_scores = []
    
    for query_idx, (query_id, relevant_docs) in enumerate(qrels.items()):
        if not relevant_docs:
            continue
        
        # Get ranked list for this query
        ranked_list = ranked_docs[query_idx]
        
        # Calculate metrics for each k
        for k in k_values:
            top_k = ranked_list[:k]
            
            # NDCG@k
            dcg = 0.0
            for pos, doc_idx in enumerate(top_k):
                if doc_idx in relevant_docs:
                    rel = relevant_docs[doc_idx]
                    dcg += (2**rel - 1) / np.log2(pos + 2)
            
            # Ideal DCG
            ideal_rels = sorted(relevant_docs.values(), reverse=True)[:k]
            idcg = sum((2**rel - 1) / np.log2(pos + 2) for pos, rel in enumerate(ideal_rels))
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores[k].append(ndcg)
            
            # Recall@k
            relevant_retrieved = len(set(top_k) & set(relevant_docs.keys()))
            recall = relevant_retrieved / len(relevant_docs)
            recall_scores[k].append(recall)
        
        # Average Precision (for MAP)
        num_relevant = len(relevant_docs)
        precision_at_k_list = []
        relevant_found = 0
        
        for pos, doc_idx in enumerate(ranked_list[:1000]):  # Limit to top 1000
            if doc_idx in relevant_docs:
                relevant_found += 1
                precision_at_k = relevant_found / (pos + 1)
                precision_at_k_list.append(precision_at_k)
        
        ap = sum(precision_at_k_list) / num_relevant if num_relevant > 0 else 0.0
        ap_scores.append(ap)
    
    # Aggregate metrics
    for k in k_values:
        metrics[f'NDCG@{k}'] = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0.0
        metrics[f'Recall@{k}'] = np.mean(recall_scores[k]) if recall_scores[k] else 0.0
    
    metrics['MAP'] = np.mean(ap_scores) if ap_scores else 0.0
    
    return metrics


def load_retrieval_dataset(task_name):
    """
    Load a retrieval dataset from MTEB/BEIR.
    
    Returns:
        queries: Dict of {query_id: query_text}
        corpus: Dict of {doc_id: doc_text}
        qrels: Dict of {query_id: {doc_id: relevance}}
    """
    print(f"    Loading {task_name} dataset...")
    
    # Use BEIR datasets which are more reliable
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        
        # Map to BEIR dataset names
        beir_names = {
            'scifact': 'scifact',
            'nfcorpus': 'nfcorpus',
            'fiqa': 'fiqa',
            'arguana': 'arguana'
        }
        
        task_lower = task_name.lower().replace('2018', '')
        beir_name = beir_names.get(task_lower, task_lower)
        
        # Download and load using BEIR
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{beir_name}.zip"
        data_path = util.download_and_unzip(url, "datasets")
        
        # Load corpus, queries, and qrels
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        
        # Convert corpus format (dict of dicts to dict of strings)
        corpus_texts = {}
        for doc_id, doc_data in corpus.items():
            title = doc_data.get('title', '')
            text = doc_data.get('text', '')
            corpus_texts[doc_id] = f"{title} {text}".strip() if title else text
        
        print(f"    ✓ Loaded: {len(queries)} queries, {len(corpus_texts)} docs")
        return queries, corpus_texts, qrels
        
    except ImportError:
        print(f"    ✗ BEIR not installed. Install with: pip install beir")
        return None, None, None
    except Exception as e:
        print(f"    ✗ Error loading {task_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def load_and_encode_task(base_model, task_name):
    """
    Load dataset and encode all documents and queries once.
    Cache in memory to avoid re-encoding for each adaptation.
    
    Args:
        base_model: SentenceTransformer model
        task_name: Name of the retrieval task
    
    Returns:
        Tuple of (query_embeddings, doc_embeddings, indexed_qrels, task_info)
    """
    print(f"\n  Loading and encoding {task_name}...")
    
    # Load dataset
    queries, corpus, qrels = load_retrieval_dataset(task_name)
    
    if queries is None:
        return None
    
    # Encode queries ONCE
    print(f"    Encoding {len(queries)} queries...")
    query_texts = list(queries.values())
    query_embeddings = base_model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Encode corpus ONCE
    print(f"    Encoding {len(corpus)} documents...")
    doc_texts = list(corpus.values())
    doc_embeddings = base_model.encode(
        doc_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Build indexed qrels
    query_id_to_idx = {qid: idx for idx, qid in enumerate(queries.keys())}
    doc_id_to_idx = {did: idx for idx, did in enumerate(corpus.keys())}
    
    indexed_qrels = {}
    for query_id, rel_docs in qrels.items():
        if query_id in query_id_to_idx:
            q_idx = query_id_to_idx[query_id]
            indexed_qrels[q_idx] = {}
            for doc_id, score in rel_docs.items():
                if doc_id in doc_id_to_idx:
                    d_idx = doc_id_to_idx[doc_id]
                    indexed_qrels[q_idx][d_idx] = score
    
    task_info = {
        'name': task_name,
        'n_queries': len(queries),
        'n_docs': len(corpus)
    }
    
    print(f"    ✓ Cached embeddings for {task_name}")
    
    return query_embeddings, doc_embeddings, indexed_qrels, task_info


def evaluate_with_adaptation(query_embeddings, doc_embeddings, indexed_qrels, 
                             adaptation_fn, config_name):
    """
    Apply adaptation to cached embeddings and evaluate.
    
    Args:
        query_embeddings: Pre-computed query embeddings
        doc_embeddings: Pre-computed document embeddings
        indexed_qrels: Relevance judgments with integer indices
        adaptation_fn: Function to adapt embeddings (or None for baseline)
        config_name: Name of the adaptation configuration
    
    Returns:
        Dictionary of metrics
    """
    # Apply adaptation if provided
    # CRITICAL: Fit on documents FIRST (the domain corpus), then transform queries
    if adaptation_fn is not None:
        adapted_docs = adaptation_fn(doc_embeddings.copy())  # Fits transformation on documents
        adapted_queries = adaptation_fn(query_embeddings.copy())  # Applies fitted transformation
    else:
        adapted_queries = query_embeddings
        adapted_docs = doc_embeddings
    
    # Compute metrics
    metrics = compute_retrieval_metrics(adapted_queries, adapted_docs, indexed_qrels)
    
    return metrics


def apply_pca_projection(n_components=None, variance_ratio=0.95):
    """
    Create PCA adaptation function.
    
    Args:
        n_components: Explicit number of components
        variance_ratio: Keep components explaining this much variance
    
    Returns:
        Function that applies PCA transformation
    """
    pca = None
    
    def adapt(X):
        nonlocal pca
        if pca is None:
            # Fit PCA on first batch (domain corpus)
            if n_components is None:
                pca = PCA(n_components=variance_ratio, random_state=42)
            else:
                pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X)
            print(f"    PCA fitted: {pca.n_components_} components "
                  f"({pca.explained_variance_ratio_.sum():.2%} variance)")
        
        # Transform
        X_transformed = pca.transform(X)
        # L2 normalize
        X_transformed = X_transformed / (np.linalg.norm(X_transformed, axis=1, keepdims=True) + 1e-8)
        return X_transformed
    
    return adapt


def apply_variance_weighting(percentile_threshold=10):
    """
    Create variance weighting adaptation function.
    
    Args:
        percentile_threshold: Drop dimensions below this variance percentile
    
    Returns:
        Function that applies variance weighting
    """
    weights = None
    
    def adapt(X):
        nonlocal weights
        if weights is None:
            # Compute weights from first batch (domain corpus)
            variances = np.var(X, axis=0)
            threshold = np.percentile(variances, percentile_threshold)
            weights = np.where(variances > threshold, np.sqrt(variances), 0)
            weights = weights / (np.linalg.norm(weights) + 1e-8)
            n_kept = np.sum(weights > 0)
            print(f"    Variance weighting: kept {n_kept}/{len(weights)} dimensions")
        
        # Apply weights
        X_weighted = X * weights
        # L2 normalize
        X_weighted = X_weighted / (np.linalg.norm(X_weighted, axis=1, keepdims=True) + 1e-8)
        return X_weighted
    
    return adapt


def apply_hybrid_adaptation(pca_components=200, var_percentile=10):
    """
    Create hybrid PCA + variance weighting adaptation.
    
    Args:
        pca_components: Number of PCA components
        var_percentile: Variance percentile threshold
    
    Returns:
        Function that applies hybrid transformation
    """
    pca = None
    weights = None
    
    def adapt(X):
        nonlocal pca, weights
        
        # Step 1: PCA
        if pca is None:
            pca = PCA(n_components=pca_components, random_state=42)
            pca.fit(X)
            print(f"    Hybrid PCA: {pca.n_components_} components "
                  f"({pca.explained_variance_ratio_.sum():.2%} variance)")
        
        X_pca = pca.transform(X)
        
        # Step 2: Variance weighting
        if weights is None:
            variances = np.var(X_pca, axis=0)
            threshold = np.percentile(variances, var_percentile)
            weights = np.where(variances > threshold, np.sqrt(variances), 0)
            weights = weights / (np.linalg.norm(weights) + 1e-8)
            n_kept = np.sum(weights > 0)
            print(f"    Hybrid VarWeight: kept {n_kept}/{len(weights)} dimensions")
        
        X_weighted = X_pca * weights
        # L2 normalize
        X_weighted = X_weighted / (np.linalg.norm(X_weighted, axis=1, keepdims=True) + 1e-8)
        return X_weighted
    
    return adapt


def apply_domain_whitening():
    """
    Create domain whitening adaptation function.
    
    Returns:
        Function that applies whitening transformation
    """
    mean = None
    whitening_transform = None
    
    def adapt(X):
        nonlocal mean, whitening_transform
        
        if mean is None:
            # Compute whitening from first batch
            mean = X.mean(axis=0)
            X_centered = X - mean
            cov = np.cov(X_centered.T)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            epsilon = 1e-5
            eigenvalues = np.maximum(eigenvalues, epsilon)
            
            # Whitening transform
            whitening_transform = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            print(f"    Domain whitening fitted")
        
        # Apply whitening
        X_centered = X - mean
        X_whitened = X_centered @ whitening_transform
        # L2 normalize
        X_whitened = X_whitened / (np.linalg.norm(X_whitened, axis=1, keepdims=True) + 1e-8)
        return X_whitened
    
    return adapt


def apply_rocchio_prf(k_top=10, alpha=1.0, beta=0.4, gamma=0.0, k_negative=0):
    """
    Create Rocchio Pseudo-Relevance Feedback adaptation.
    
    This modifies query embeddings based on top retrieved documents.
    Query-side feedback in embedding space.
    
    Args:
        k_top: Number of top documents to use as pseudo-relevant
        alpha: Weight for original query
        beta: Weight for positive centroid
        gamma: Weight for negative centroid
        k_negative: Number of near-miss docs (set to 0 to skip negatives)
    
    Returns:
        Function that applies Rocchio PRF
    """
    doc_embeddings_cache = None
    
    def adapt(X):
        nonlocal doc_embeddings_cache
        
        # First call: store document embeddings (fitted on corpus)
        if doc_embeddings_cache is None:
            doc_embeddings_cache = X.copy()
            print(f"    Rocchio PRF: cached {len(X)} documents")
            return X
        
        # Second call: update query embeddings using stored docs
        query_embeddings = X.copy()
        updated_queries = []
        
        # Compute similarities
        similarities = cosine_similarity(query_embeddings, doc_embeddings_cache)
        
        for i, q in enumerate(query_embeddings):
            # Get top-k indices
            top_k_indices = np.argsort(-similarities[i])[:k_top]
            
            # Positive centroid
            positive_centroid = doc_embeddings_cache[top_k_indices].mean(axis=0)
            
            # Negative centroid (if requested)
            if gamma > 0 and k_negative > 0:
                negative_indices = np.argsort(-similarities[i])[k_top:k_top + k_negative]
                negative_centroid = doc_embeddings_cache[negative_indices].mean(axis=0)
            else:
                negative_centroid = 0.0
            
            # Rocchio update
            q_updated = alpha * q + beta * positive_centroid - gamma * negative_centroid
            
            # L2 normalize
            q_updated = q_updated / (np.linalg.norm(q_updated) + 1e-8)
            updated_queries.append(q_updated)
        
        updated_queries = np.array(updated_queries)
        print(f"    Rocchio PRF: updated {len(updated_queries)} queries (k={k_top}, β={beta})")
        return updated_queries
    
    return adapt


def apply_remove_top_pcs(n_components=1):
    """
    Remove the top N principal components (all-but-the-top).
    
    This targets anisotropy/hubness without reducing dimensionality.
    Removes dominant directions that cause generic documents to become hubs.
    
    Args:
        n_components: Number of top PCs to remove (1-5 typical)
    
    Returns:
        Function that applies top-PC removal
    """
    doc_pcs = None
    query_pcs = None
    
    def adapt(X):
        nonlocal doc_pcs, query_pcs
        
        # First call: compute PCs on documents
        if doc_pcs is None:
            # Center the data
            mean = X.mean(axis=0)
            X_centered = X - mean
            
            # Compute covariance and eigendecomposition
            cov = np.cov(X_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Get top N components (largest eigenvalues)
            top_indices = np.argsort(-eigenvalues)[:n_components]
            doc_pcs = eigenvectors[:, top_indices]
            
            print(f"    Remove Top PCs: computed {n_components} dominant components from docs")
            
            # Remove these components from documents
            for pc in doc_pcs.T:
                X = X - np.outer(X @ pc, pc)
            
            # L2 normalize
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            return X
        
        # Second call: compute PCs on queries and remove
        if query_pcs is None:
            mean = X.mean(axis=0)
            X_centered = X - mean
            cov = np.cov(X_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            top_indices = np.argsort(-eigenvalues)[:n_components]
            query_pcs = eigenvectors[:, top_indices]
            
            print(f"    Remove Top PCs: computed {n_components} dominant components from queries")
        
        # Remove top components from queries
        for pc in query_pcs.T:
            X = X - np.outer(X @ pc, pc)
        
        # L2 normalize
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return X
    
    return adapt


def apply_csls(k_neighbors=10):
    """
    Cross-domain Similarity Local Scaling (CSLS) for hubness reduction.
    
    CSLS penalizes "hub" documents that are near neighbors to many queries.
    Originally from bilingual embedding alignment, but works for IR too.
    
    Args:
        k_neighbors: Number of neighbors for computing local scaling
    
    Returns:
        Function that applies CSLS rescoring
    """
    doc_embeddings_cache = None
    r_d_cache = None  # Mean similarity to k nearest docs
    
    def adapt(X):
        nonlocal doc_embeddings_cache, r_d_cache
        
        # First call: store documents and compute r_d for each doc
        if doc_embeddings_cache is None:
            doc_embeddings_cache = X.copy()
            
            # Compute pairwise doc-doc similarities
            doc_similarities = cosine_similarity(doc_embeddings_cache, doc_embeddings_cache)
            
            # For each doc, compute mean similarity to k nearest neighbors
            r_d_cache = []
            for i in range(len(doc_embeddings_cache)):
                # Get k nearest (excluding self)
                top_k_indices = np.argsort(-doc_similarities[i])[1:k_neighbors+1]
                r_d = doc_similarities[i][top_k_indices].mean()
                r_d_cache.append(r_d)
            
            r_d_cache = np.array(r_d_cache)
            print(f"    CSLS: computed local scaling for {len(doc_embeddings_cache)} docs (k={k_neighbors})")
            return X
        
        # Second call: apply CSLS to queries
        # NOTE: CSLS is a scoring function, not an embedding transformation
        # We'll approximate by adjusting query embeddings based on r_q
        query_embeddings = X.copy()
        
        # Compute query-doc similarities
        similarities = cosine_similarity(query_embeddings, doc_embeddings_cache)
        
        # For each query, compute r_q
        adjusted_queries = []
        for i, q in enumerate(query_embeddings):
            # Get top-k docs for this query
            top_k_indices = np.argsort(-similarities[i])[:k_neighbors]
            r_q = similarities[i][top_k_indices].mean()
            
            # CSLS adjustment: penalize by (r_q + r_d) / 2
            # Since we can't modify similarities directly in this framework,
            # we'll use a heuristic: boost query toward low-r_d docs
            # This is an approximation of CSLS for the adaptation framework
            
            # Weight by inverse r_d (prefer low-hub docs)
            weights = 1.0 / (r_d_cache + 0.1)  # +0.1 to avoid div by zero
            weights = weights / weights.sum()
            
            # Adjust query slightly toward low-hub docs
            adjustment = (doc_embeddings_cache.T @ weights).T
            q_adjusted = 0.9 * q + 0.1 * adjustment
            
            # L2 normalize
            q_adjusted = q_adjusted / (np.linalg.norm(q_adjusted) + 1e-8)
            adjusted_queries.append(q_adjusted)
        
        adjusted_queries = np.array(adjusted_queries)
        print(f"    CSLS: adjusted {len(adjusted_queries)} queries for hubness reduction")
        return adjusted_queries
    
    return adapt


def create_adaptation_configs():
    """
    Create all adaptation configurations to test.
    
    Returns:
        List of (name, adaptation_fn) tuples
    """
    configs = [
        ("Baseline", None),
    ]
    
    # ============================================================================
    # NEW METHODS FROM FURTHER IDEAS
    # These are suggested to work better than PCA/whitening
    # ============================================================================
    
    # 1. Rocchio PRF (Query-side pseudo-relevance feedback)
    #    Most practical no-training method from the document
    for k in [5, 10, 20]:
        for beta in [0.2, 0.4, 0.6]:
            configs.append((
                f"Rocchio_k{k}_β{beta}",
                apply_rocchio_prf(k_top=k, alpha=1.0, beta=beta, gamma=0.0)
            ))
    
    # 1b. Rocchio with negatives
    for beta in [0.4, 0.6]:
        configs.append((
            f"Rocchio_k10_β{beta}_γ0.1",
            apply_rocchio_prf(k_top=10, alpha=1.0, beta=beta, gamma=0.1, k_negative=50)
        ))
    
    # 2. Remove Top PCs (all-but-the-top)
    #    Targets anisotropy without throwing away dimensions
    for n_comp in [1, 2, 3, 5]:
        configs.append((
            f"RemoveTopPCs_{n_comp}",
            apply_remove_top_pcs(n_components=n_comp)
        ))
    
    # 3. CSLS (hubness reduction)
    #    Down-ranks hub documents
    for k in [10, 20, 50]:
        configs.append((
            f"CSLS_k{k}",
            apply_csls(k_neighbors=k)
        ))
    
    # ============================================================================
    # ORIGINAL METHODS (baseline comparisons)
    # PCA/whitening approaches - kept for comprehensive evaluation
    # ============================================================================
    
    # 4. PCA with different variance ratios
    for var_ratio in [0.80, 0.85, 0.90, 0.95, 0.98]:
        configs.append((
            f"PCA_{int(var_ratio*100)}%",
            apply_pca_projection(variance_ratio=var_ratio)
        ))
    
    # 5. PCA with fixed components
    for n_comp in [50, 100, 150, 200]:
        configs.append((
            f"PCA_{n_comp}d",
            apply_pca_projection(n_components=n_comp)
        ))
    
    # 6. Variance weighting
    for percentile in [5, 10, 15, 20, 25]:
        configs.append((
            f"VarWeight_p{percentile}",
            apply_variance_weighting(percentile_threshold=percentile)
        ))
    
    # 7. Hybrid approaches (PCA + Variance Weighting)
    for pca_comp in [100, 150, 200]:
        for var_perc in [10, 15, 20]:
            configs.append((
                f"Hybrid_PCA{pca_comp}_VW{var_perc}",
                apply_hybrid_adaptation(pca_components=pca_comp, var_percentile=var_perc)
            ))
    
    # 8. Domain whitening
    configs.append((
        "Whitening",
        apply_domain_whitening()
    ))
    
    return configs


def run_retrieval_evaluation(model_name="all-MiniLM-L6-v2", 
                            tasks=None,
                            output_dir="retrieval_results",
                            sample_size=None):
    """
    Run retrieval evaluation with different adaptations.
    Optimized: Encodes each dataset only once, then applies adaptations.
    
    Args:
        model_name: Base model name
        tasks: List of task names
        output_dir: Output directory for results
        sample_size: Optional limit on queries per task (for speed)
    
    Returns:
        DataFrame with results
    """
    if not BEIR_AVAILABLE:
        print("ERROR: BEIR package not installed.")
        print("Install with: pip install beir")
        return None
    
    print("="*80)
    print("RETRIEVAL BENCHMARK WITH DOMAIN ADAPTATION")
    print("="*80)
    
    # Default to multiple BEIR tasks
    if tasks is None:
        tasks = ["scifact", "nfcorpus", "fiqa", "trec-covid", "arguana"]
    
    print(f"\nBase model: {model_name}")
    print(f"Tasks: {', '.join(tasks)}")
    if sample_size:
        print(f"Sample size: {sample_size} queries per task")
    
    # Load base model
    print(f"\nLoading base model...")
    base_model = SentenceTransformer(model_name)
    
    # Get adaptation configurations
    adaptation_configs = create_adaptation_configs()
    
    print(f"\n🚀 Optimization: Each dataset will be encoded only ONCE")
    print(f"   Then all {len(adaptation_configs)} adaptations will be applied to cached embeddings")
    print(f"\nTotal configurations to test: {len(adaptation_configs)}")
    print(f"Total evaluations: {len(adaptation_configs)} configs × {len(tasks)} tasks")
    
    all_results = []
    
    # OUTER LOOP: Tasks (encode once per task)
    for task_idx, task_name in enumerate(tasks, 1):
        print(f"\n{'='*80}")
        print(f"TASK [{task_idx}/{len(tasks)}]: {task_name.upper()}")
        print(f"{'='*80}")
        
        # Load and encode this task ONCE
        task_data = load_and_encode_task(base_model, task_name)
        
        if task_data is None:
            print(f"  ✗ Skipping {task_name} due to loading error")
            continue
        
        query_embeddings, doc_embeddings, indexed_qrels, task_info = task_data
        
        print(f"\n  ✓ Embeddings cached in memory!")
        print(f"    Query embeddings: {query_embeddings.shape}")
        print(f"    Document embeddings: {doc_embeddings.shape}")
        print(f"\n  Now testing {len(adaptation_configs)} adaptation methods...")
        
        # INNER LOOP: Adaptations (use cached embeddings)
        for i, (config_name, adaptation_fn) in enumerate(adaptation_configs, 1):
            print(f"\n  [{i}/{len(adaptation_configs)}] {config_name}...", end=" ")
            
            try:
                metrics = evaluate_with_adaptation(
                    query_embeddings,
                    doc_embeddings,
                    indexed_qrels,
                    adaptation_fn,
                    config_name
                )
                
                if metrics:
                    result_row = {
                        'Method': config_name,
                        'Task': task_name,
                        **metrics
                    }
                    
                    all_results.append(result_row)
                    
                    print(f"NDCG@10: {metrics['NDCG@10']:.4f}, "
                          f"MAP: {metrics['MAP']:.4f}, "
                          f"Recall@100: {metrics['Recall@100']:.4f}")
                    
            except Exception as e:
                print(f"ERROR: {str(e)[:100]}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\nERROR: No results collected!")
        return None
    
    # Save raw results
    results_df.to_csv(f'{output_dir}_raw.csv', index=False)
    print(f"\n{'='*80}")
    print(f"✓ Raw results saved to '{output_dir}_raw.csv'")
    print(f"{'='*80}")
    
    return results_df


def analyze_results(results_df):
    """
    Analyze and summarize results.
    
    Args:
        results_df: DataFrame with evaluation results
    
    Returns:
        Summary DataFrame
    """
    if results_df is None or len(results_df) == 0:
        return None
    
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    # Aggregate across tasks (mean performance)
    summary = results_df.groupby('Method').agg({
        'NDCG@10': 'mean',
        'MAP': 'mean',
        'Recall@100': 'mean'
    }).reset_index()
    
    # Sort by NDCG@10
    summary = summary.sort_values('NDCG@10', ascending=False)
    
    # Calculate improvement over baseline
    baseline_row = summary[summary['Method'] == 'Baseline']
    if len(baseline_row) > 0:
        baseline_ndcg = baseline_row['NDCG@10'].values[0]
        baseline_map = baseline_row['MAP'].values[0]
        baseline_recall = baseline_row['Recall@100'].values[0]
        
        summary['NDCG_Improvement_%'] = ((summary['NDCG@10'] - baseline_ndcg) / (baseline_ndcg + 1e-9)) * 100
        summary['MAP_Improvement_%'] = ((summary['MAP'] - baseline_map) / (baseline_map + 1e-9)) * 100
        summary['Recall_Improvement_%'] = ((summary['Recall@100'] - baseline_recall) / (baseline_recall + 1e-9)) * 100
    
    # Print top 15 methods
    print("\n🏆 TOP 15 Methods (by NDCG@10, averaged across tasks):")
    print("-" * 120)
    top_15 = summary.head(15)
    display_cols = ['Method', 'NDCG@10', 'MAP', 'Recall@100']
    if 'NDCG_Improvement_%' in summary.columns:
        display_cols.extend(['NDCG_Improvement_%', 'MAP_Improvement_%'])
    print(top_15[display_cols].to_string(index=False))
    
    # Best by each metric
    print("\n" + "="*80)
    print("BEST METHODS BY METRIC")
    print("="*80)
    
    for metric in ['NDCG@10', 'MAP', 'Recall@100']:
        best = summary.loc[summary[metric].idxmax()]
        print(f"\n  Best by {metric}: {best['Method']}")
        print(f"    Value: {best[metric]:.4f}")
        if f"{metric.split('@')[0]}_Improvement_%" in summary.columns:
            imp_col = f"{metric.split('@')[0]}_Improvement_%"
            if imp_col in best.index:
                print(f"    Improvement: {best[imp_col]:+.2f}%")
    
    # Per-task breakdown for top 5 methods
    print("\n" + "="*80)
    print("TOP 5 METHODS: PER-TASK BREAKDOWN")
    print("="*80)
    
    top_5_methods = summary.head(5)['Method'].values
    for method in top_5_methods:
        method_results = results_df[results_df['Method'] == method]
        print(f"\n{method}:")
        for _, row in method_results.iterrows():
            print(f"  {row['Task']:20s} - NDCG@10: {row['NDCG@10']:.4f}, MAP: {row['MAP']:.4f}, Recall@100: {row['Recall@100']:.4f}")
    
    # Save summary
    summary.to_csv('retrieval_adaptation_summary.csv', index=False)
    print("\n" + "="*80)
    print("Summary saved to 'retrieval_adaptation_summary.csv'")
    print("="*80)
    
    return summary


def visualize_results(results_df, summary_df):
    """
    Create visualizations of results.
    
    Args:
        results_df: Raw results DataFrame
        summary_df: Summary DataFrame
    """
    if results_df is None or summary_df is None:
        return
    
    print("\nCreating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Top 15 methods by NDCG@10
    ax1 = axes[0, 0]
    top_15 = summary_df.head(15).copy()
    top_15 = top_15.sort_values('NDCG@10', ascending=True)  # For horizontal bar
    
    colors = ['#2ecc71' if m == 'Baseline' else '#3498db' for m in top_15['Method']]
    ax1.barh(range(len(top_15)), top_15['NDCG@10'], color=colors)
    ax1.set_yticks(range(len(top_15)))
    ax1.set_yticklabels(top_15['Method'], fontsize=9)
    ax1.set_xlabel('NDCG@10 (avg across tasks)', fontsize=11)
    ax1.set_title('Top 15 Methods by NDCG@10', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add baseline line
    baseline_ndcg = summary_df[summary_df['Method'] == 'Baseline']['NDCG@10'].values[0]
    ax1.axvline(baseline_ndcg, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
    ax1.legend()
    
    # 2. Improvement over baseline
    ax2 = axes[0, 1]
    if 'NDCG_Improvement_%' in summary_df.columns:
        top_15_imp = summary_df[summary_df['Method'] != 'Baseline'].head(14).copy()
        top_15_imp = top_15_imp.sort_values('NDCG_Improvement_%', ascending=True)
        
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_15_imp['NDCG_Improvement_%']]
        ax2.barh(range(len(top_15_imp)), top_15_imp['NDCG_Improvement_%'], color=colors)
        ax2.set_yticks(range(len(top_15_imp)))
        ax2.set_yticklabels(top_15_imp['Method'], fontsize=9)
        ax2.set_xlabel('Improvement over Baseline (%)', fontsize=11)
        ax2.set_title('NDCG@10 Improvement over Baseline', fontsize=13, fontweight='bold')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        ax2.grid(axis='x', alpha=0.3)
    
    # 3. Per-task heatmap (top 10 methods)
    ax3 = axes[1, 0]
    top_10_methods = summary_df.head(10)['Method'].values
    tasks = results_df['Task'].unique()
    
    heatmap_data = []
    for method in top_10_methods:
        row = []
        for task in tasks:
            value = results_df[(results_df['Method'] == method) & (results_df['Task'] == task)]['NDCG@10']
            row.append(value.values[0] if len(value) > 0 else 0)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, columns=tasks, index=top_10_methods)
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax3, cbar_kws={'label': 'NDCG@10'})
    ax3.set_title('NDCG@10 per Task (Top 10 Methods)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Task', fontsize=11)
    ax3.set_ylabel('Method', fontsize=11)
    
    # 4. Method category comparison (extract method type)
    ax4 = axes[1, 1]
    
    def categorize_method(method_name):
        if method_name == 'Baseline':
            return 'Baseline'
        elif method_name.startswith('PCA_') and not 'VW' in method_name:
            return 'PCA Only'
        elif method_name.startswith('VarWeight'):
            return 'Variance Weight'
        elif method_name.startswith('Hybrid'):
            return 'Hybrid (PCA+VW)'
        elif method_name == 'Whitening':
            return 'Whitening'
        else:
            return 'Other'
    
    summary_df['Category'] = summary_df['Method'].apply(categorize_method)
    category_stats = summary_df.groupby('Category').agg({
        'NDCG@10': ['mean', 'std', 'max'],
        'MAP': ['mean', 'std', 'max']
    }).reset_index()
    
    categories = category_stats['Category'].values
    ndcg_means = category_stats[('NDCG@10', 'mean')].values
    ndcg_stds = category_stats[('NDCG@10', 'std')].values
    
    x_pos = np.arange(len(categories))
    ax4.bar(x_pos, ndcg_means, yerr=ndcg_stds, capsize=5, 
            color=['#e74c3c' if c == 'Baseline' else '#3498db' for c in categories],
            alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('NDCG@10 (mean ± std)', fontsize=11)
    ax4.set_title('Performance by Method Category', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('retrieval_adaptation_results.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to 'retrieval_adaptation_results.png'")
    
    plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("DOMAIN ADAPTATION FOR RETRIEVAL BENCHMARKS")
    print("="*80)
    
    if not BEIR_AVAILABLE:
        print("\n❌ ERROR: BEIR package not installed!")
        print("\nPlease install with:")
        print("  pip install beir")
        return
    
    # Configuration
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Test on multiple BEIR datasets for robust evaluation
    tasks = ["scifact", "nfcorpus", "fiqa", "arguana", "trec-covid"]
    
    print(f"\n📋 Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"\nBEIR Tasks:")
    print(f"  - SciFact: Scientific claim verification (~300 queries)")
    print(f"  - NFCorpus: Medical information retrieval (~323 queries)")
    print(f"  - FiQA: Financial question answering (~648 queries)")
    print(f"  - ArguAna: Argument retrieval (~1406 queries)")
    print(f"  - TREC-COVID: COVID-19 research (~50 queries)")
    print(f"\n⚡ Adaptation methods to test:")
    print(f"  NEW methods from further_ideas.md:")
    print(f"    1. Rocchio PRF (query-side feedback)")
    print(f"    2. Remove Top PCs (all-but-the-top)")
    print(f"    3. CSLS (hubness reduction)")
    print(f"  BASELINE methods for comparison:")
    print(f"    4. PCA with variance ratios")
    print(f"    5. PCA with fixed components")
    print(f"    6. Variance weighting")
    print(f"    7. Hybrid PCA + Variance Weighting")
    print(f"    8. Domain whitening")
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    print("\n⚡ Optimized version: ~8-12 minutes with 5 tasks")
    
    results_df = run_retrieval_evaluation(
        model_name=model_name,
        tasks=tasks,
        output_dir="retrieval_results",
        sample_size=None  # Use all queries (set to e.g. 50 for faster testing)
    )
    
    if results_df is None:
        print("\n❌ Evaluation failed!")
        return
    
    # Analyze results
    summary_df = analyze_results(results_df)
    
    if summary_df is None:
        print("\n❌ Analysis failed!")
        return
    
    # Create visualizations
    visualize_results(results_df, summary_df)
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*80)
    print("\n📊 Generated files:")
    print("  - retrieval_results_raw.csv (all results)")
    print("  - retrieval_adaptation_summary.csv (aggregated results)")
    print("  - retrieval_adaptation_results.png (visualizations)")
    print("\n💡 Key insights:")
    print("  - Tested 8 adaptation method categories (42 total configs)")
    print("  - NEW methods (1-3): target query mismatch, anisotropy, and hubness")
    print("  - BASELINE methods (4-8): PCA/whitening approaches for comparison")
    print("  - Best configurations are in the summary table")


if __name__ == "__main__":
    main()
