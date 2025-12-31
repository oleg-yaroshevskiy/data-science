import numpy as np

# Custom implementations of recommendation/retrieval metrics for interview preparation

def precision_at_k(relevant_items, recommended_items, k):
    """
    Precision@K: Fraction of recommended items in top-K that are relevant.
    
    Formula: (# of relevant items in top-K) / K
    
    Args:
        relevant_items: set or list of relevant item IDs
        recommended_items: ordered list of recommended item IDs
        k: number of top recommendations to consider
    
    Returns:
        Precision@K score (float between 0 and 1)
    """
    if k <= 0 or len(recommended_items) == 0:
        return 0.0
    
    # Take top K recommendations
    top_k = recommended_items[:k]
    
    # Count how many are relevant
    relevant_count = sum(1 for item in top_k if item in relevant_items)
    
    return relevant_count / k


def recall_at_k(relevant_items, recommended_items, k):
    """
    Recall@K: Fraction of relevant items that appear in top-K recommendations.
    
    Formula: (# of relevant items in top-K) / (total # of relevant items)
    
    Args:
        relevant_items: set or list of relevant item IDs
        recommended_items: ordered list of recommended item IDs
        k: number of top recommendations to consider
    
    Returns:
        Recall@K score (float between 0 and 1)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    # Take top K recommendations
    top_k = recommended_items[:k]
    
    # Count how many relevant items are retrieved
    relevant_count = sum(1 for item in top_k if item in relevant_items)
    
    return relevant_count / len(relevant_items)


def average_precision(relevant_items, recommended_items):
    """
    Average Precision (AP): Average of precision values at positions where relevant items occur.
    
    Formula: AP = (1/R) * sum(Precision@k * rel(k))
    where R is the total number of relevant items,
    k is the rank position, and rel(k) is 1 if item at position k is relevant, else 0.
    
    Args:
        relevant_items: set or list of relevant item IDs
        recommended_items: ordered list of recommended item IDs
    
    Returns:
        Average Precision score (float between 0 and 1)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            num_hits += 1.0
            # Precision at this position
            precision_at_i = num_hits / (i + 1.0)
            score += precision_at_i
    
    # Average over all relevant items
    return score / len(relevant_items)


def mean_average_precision(relevant_items_list, recommended_items_list):
    """
    Mean Average Precision (MAP): Mean of Average Precision across all queries.
    
    Formula: MAP = (1/Q) * sum(AP_q) for all queries q
    
    Args:
        relevant_items_list: list of sets/lists, each containing relevant items for a query
        recommended_items_list: list of lists, each containing ordered recommendations for a query
    
    Returns:
        MAP score (float between 0 and 1)
    """
    if len(relevant_items_list) == 0:
        return 0.0
    
    ap_scores = []
    for relevant, recommended in zip(relevant_items_list, recommended_items_list):
        ap = average_precision(relevant, recommended)
        ap_scores.append(ap)
    
    return np.mean(ap_scores)


def reciprocal_rank(relevant_items, recommended_items):
    """
    Reciprocal Rank (RR): 1 / (rank of first relevant item)
    
    Args:
        relevant_items: set or list of relevant item IDs
        recommended_items: ordered list of recommended item IDs
    
    Returns:
        Reciprocal Rank score (float between 0 and 1)
    """
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            return 1.0 / (i + 1.0)
    
    return 0.0


def mean_reciprocal_rank(relevant_items_list, recommended_items_list):
    """
    Mean Reciprocal Rank (MRR): Mean of Reciprocal Rank across all queries.
    
    Formula: MRR = (1/Q) * sum(RR_q) for all queries q
    
    Args:
        relevant_items_list: list of sets/lists, each containing relevant items for a query
        recommended_items_list: list of lists, each containing ordered recommendations for a query
    
    Returns:
        MRR score (float between 0 and 1)
    """
    if len(relevant_items_list) == 0:
        return 0.0
    
    rr_scores = []
    for relevant, recommended in zip(relevant_items_list, recommended_items_list):
        rr = reciprocal_rank(relevant, recommended)
        rr_scores.append(rr)
    
    return np.mean(rr_scores)


def dcg_at_k(relevant_items, recommended_items, k, relevance_scores=None):
    """
    Discounted Cumulative Gain at K (DCG@K).
    Measures ranking quality with position discount.
    
    Formula: DCG@K = sum(rel_i / log2(i + 1)) for i in [1, k]
    
    Args:
        relevant_items: set or list of relevant item IDs
        recommended_items: ordered list of recommended item IDs
        k: number of top recommendations to consider
        relevance_scores: dict mapping item_id -> relevance score (default: binary 1 or 0)
    
    Returns:
        DCG@K score
    """
    if k <= 0:
        return 0.0
    
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if relevance_scores is not None:
            rel = relevance_scores.get(item, 0.0)
        else:
            rel = 1.0 if item in relevant_items else 0.0
        
        # Discount factor: log2(position + 1), position is 1-indexed
        dcg += rel / np.log2(i + 2.0)
    
    return dcg


def ndcg_at_k(relevant_items, recommended_items, k, relevance_scores=None):
    """
    Normalized Discounted Cumulative Gain at K (NDCG@K).
    DCG normalized by ideal DCG (IDCG).
    
    Formula: NDCG@K = DCG@K / IDCG@K
    
    Args:
        relevant_items: set or list of relevant item IDs
        recommended_items: ordered list of recommended item IDs
        k: number of top recommendations to consider
        relevance_scores: dict mapping item_id -> relevance score (default: binary 1 or 0)
    
    Returns:
        NDCG@K score (float between 0 and 1)
    """
    dcg = dcg_at_k(relevant_items, recommended_items, k, relevance_scores)
    
    # Calculate ideal DCG (IDCG) - best possible ranking
    if relevance_scores is not None:
        # Sort relevant items by their relevance scores
        ideal_items = sorted(
            relevant_items, 
            key=lambda x: relevance_scores.get(x, 0.0), 
            reverse=True
        )
    else:
        # All relevant items have same relevance (binary case)
        ideal_items = list(relevant_items)
    
    idcg = dcg_at_k(relevant_items, ideal_items, k, relevance_scores)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(relevant_items, recommended_items, k):
    """
    Hit Rate@K: Binary metric indicating if at least one relevant item is in top-K.
    Also known as Recall@K for single relevant item.
    
    Args:
        relevant_items: set or list of relevant item IDs
        recommended_items: ordered list of recommended item IDs
        k: number of top recommendations to consider
    
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    top_k = recommended_items[:k]
    
    for item in top_k:
        if item in relevant_items:
            return 1.0
    
    return 0.0


def mean_hit_rate_at_k(relevant_items_list, recommended_items_list, k):
    """
    Mean Hit Rate@K: Average of Hit Rate@K across all queries.
    
    Args:
        relevant_items_list: list of sets/lists, each containing relevant items for a query
        recommended_items_list: list of lists, each containing ordered recommendations for a query
        k: number of top recommendations to consider
    
    Returns:
        Mean Hit Rate@K score (float between 0 and 1)
    """
    if len(relevant_items_list) == 0:
        return 0.0
    
    hit_scores = []
    for relevant, recommended in zip(relevant_items_list, recommended_items_list):
        hit = hit_rate_at_k(relevant, recommended, k)
        hit_scores.append(hit)
    
    return np.mean(hit_scores)


def coverage(recommended_items_list, catalog_size):
    """
    Catalog Coverage: Fraction of catalog items that appear in recommendations.
    Measures diversity of the recommendation system.
    
    Args:
        recommended_items_list: list of lists, each containing recommendations
        catalog_size: total number of items in the catalog
    
    Returns:
        Coverage score (float between 0 and 1)
    """
    if catalog_size == 0:
        return 0.0
    
    # Collect all unique recommended items
    all_recommended = set()
    for recommended in recommended_items_list:
        all_recommended.update(recommended)
    
    return len(all_recommended) / catalog_size


def f1_at_k(relevant_items, recommended_items, k):
    """
    F1@K: Harmonic mean of Precision@K and Recall@K.
    
    Args:
        relevant_items: set or list of relevant item IDs
        recommended_items: ordered list of recommended item IDs
        k: number of top recommendations to consider
    
    Returns:
        F1@K score (float between 0 and 1)
    """
    precision = precision_at_k(relevant_items, recommended_items, k)
    recall = recall_at_k(relevant_items, recommended_items, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)
