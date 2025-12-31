import numpy as np
import matplotlib.pyplot as plt
from metrics import (
    precision_at_k, recall_at_k, average_precision, mean_average_precision,
    reciprocal_rank, mean_reciprocal_rank, dcg_at_k, ndcg_at_k,
    hit_rate_at_k, mean_hit_rate_at_k, coverage, f1_at_k
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("RECOMMENDATION/RETRIEVAL SYSTEM METRICS - INTERVIEW PREPARATION")
print("=" * 80)
print()

# =============================================================================
# SIMULATE RECOMMENDATION SYSTEM DATA
# =============================================================================

# Parameters
NUM_QUERIES = 50  # Number of user queries
CATALOG_SIZE = 1000  # Total number of items in catalog
NUM_RECOMMENDATIONS = 20  # Number of items recommended per query
AVG_RELEVANT_ITEMS = 5  # Average number of relevant items per query

print("=" * 80)
print("GENERATING SYNTHETIC DATA")
print("=" * 80)
print(f"Number of queries:               {NUM_QUERIES}")
print(f"Catalog size:                    {CATALOG_SIZE}")
print(f"Recommendations per query:       {NUM_RECOMMENDATIONS}")
print(f"Avg relevant items per query:    {AVG_RELEVANT_ITEMS}")
print()

# Generate synthetic data
relevant_items_list = []
recommended_items_list = []

for query_idx in range(NUM_QUERIES):
    # Generate relevant items for this query (ground truth)
    num_relevant = np.random.poisson(AVG_RELEVANT_ITEMS) + 1  # At least 1 relevant item
    relevant_items = set(np.random.choice(CATALOG_SIZE, size=min(num_relevant, CATALOG_SIZE), replace=False))
    relevant_items_list.append(relevant_items)
    
    # Generate recommendations (with some relevant items mixed in)
    # Simulate imperfect recommender: some relevant items appear early, some late, some not at all
    recommendations = []
    
    # First, add some relevant items (with probability)
    relevant_in_recs = np.random.choice(
        list(relevant_items), 
        size=min(len(relevant_items), np.random.randint(0, len(relevant_items) + 1)), 
        replace=False
    )
    recommendations.extend(relevant_in_recs)
    
    # Fill rest with random items (may include some relevant items by chance)
    remaining = NUM_RECOMMENDATIONS - len(recommendations)
    if remaining > 0:
        non_relevant = np.random.choice(CATALOG_SIZE, size=remaining, replace=False)
        recommendations.extend(non_relevant)
    
    # Shuffle to simulate different ranking quality
    # Better queries have relevant items ranked higher (controlled by beta distribution)
    ranking_quality = np.random.beta(2, 2)  # 0 to 1, centered around 0.5
    
    # Create scores: relevant items get higher scores
    scores = []
    for item in recommendations:
        if item in relevant_items:
            # Relevant items get higher scores (with some noise)
            score = ranking_quality * 0.7 + np.random.uniform(0, 0.3)
        else:
            # Non-relevant items get lower scores
            score = (1 - ranking_quality) * 0.4 + np.random.uniform(0, 0.1)
        scores.append(score)
    
    # Sort recommendations by scores (descending)
    sorted_indices = np.argsort(scores)[::-1]
    recommendations = [recommendations[i] for i in sorted_indices]
    
    recommended_items_list.append(recommendations[:NUM_RECOMMENDATIONS])

# =============================================================================
# EXAMPLE 1: SINGLE QUERY METRICS
# =============================================================================

print("=" * 80)
print("EXAMPLE 1: SINGLE QUERY ANALYSIS")
print("=" * 80)

# Select a representative query
query_idx = 0
relevant = relevant_items_list[query_idx]
recommended = recommended_items_list[query_idx]

print(f"Query ID:                        {query_idx}")
print(f"Number of relevant items:        {len(relevant)}")
print(f"Number of recommendations:       {len(recommended)}")
print()

# Show which recommended items are relevant
print("Recommendations (✓ = relevant):")
for i, item in enumerate(recommended[:10]):  # Show top 10
    marker = "✓" if item in relevant else "✗"
    print(f"  Rank {i+1:2d}: Item {item:4d} {marker}")
print(f"  ... ({len(recommended) - 10} more)")
print()

# Calculate metrics for different K values
k_values = [1, 3, 5, 10, 20]

print("Metrics at different K values:")
print(f"{'K':<5} {'P@K':<8} {'R@K':<8} {'F1@K':<8} {'NDCG@K':<8} {'Hit@K':<8}")
print("-" * 50)

for k in k_values:
    p_at_k = precision_at_k(relevant, recommended, k)
    r_at_k = recall_at_k(relevant, recommended, k)
    f1_k = f1_at_k(relevant, recommended, k)
    ndcg_k = ndcg_at_k(relevant, recommended, k)
    hit_k = hit_rate_at_k(relevant, recommended, k)
    
    print(f"{k:<5} {p_at_k:<8.4f} {r_at_k:<8.4f} {f1_k:<8.4f} {ndcg_k:<8.4f} {hit_k:<8.1f}")

print()

# Single query aggregate metrics
ap = average_precision(relevant, recommended)
rr = reciprocal_rank(relevant, recommended)

print("Aggregate metrics for this query:")
print(f"  Average Precision (AP):        {ap:.4f}")
print(f"  Reciprocal Rank (RR):          {rr:.4f}")
print()

# =============================================================================
# EXAMPLE 2: MULTI-QUERY METRICS (MEAN METRICS)
# =============================================================================

print("=" * 80)
print("EXAMPLE 2: MULTI-QUERY ANALYSIS (ALL QUERIES)")
print("=" * 80)

# Calculate mean metrics across all queries
map_score = mean_average_precision(relevant_items_list, recommended_items_list)
mrr_score = mean_reciprocal_rank(relevant_items_list, recommended_items_list)

print(f"Mean Average Precision (MAP):    {map_score:.4f}")
print(f"Mean Reciprocal Rank (MRR):      {mrr_score:.4f}")
print()

# Mean metrics at different K values
print(f"Mean metrics across {NUM_QUERIES} queries at different K values:")
print(f"{'K':<5} {'Mean P@K':<12} {'Mean R@K':<12} {'Mean F1@K':<12} {'Mean NDCG@K':<12} {'Mean Hit@K':<12}")
print("-" * 70)

k_values_extended = [1, 3, 5, 10, 15, 20]
mean_precision_list = []
mean_recall_list = []
mean_f1_list = []
mean_ndcg_list = []
mean_hit_list = []

for k in k_values_extended:
    # Calculate metrics for all queries
    precision_scores = [precision_at_k(rel, rec, k) for rel, rec in zip(relevant_items_list, recommended_items_list)]
    recall_scores = [recall_at_k(rel, rec, k) for rel, rec in zip(relevant_items_list, recommended_items_list)]
    f1_scores = [f1_at_k(rel, rec, k) for rel, rec in zip(relevant_items_list, recommended_items_list)]
    ndcg_scores = [ndcg_at_k(rel, rec, k) for rel, rec in zip(relevant_items_list, recommended_items_list)]
    
    mean_p = np.mean(precision_scores)
    mean_r = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_ndcg = np.mean(ndcg_scores)
    mean_hit = mean_hit_rate_at_k(relevant_items_list, recommended_items_list, k)
    
    mean_precision_list.append(mean_p)
    mean_recall_list.append(mean_r)
    mean_f1_list.append(mean_f1)
    mean_ndcg_list.append(mean_ndcg)
    mean_hit_list.append(mean_hit)
    
    print(f"{k:<5} {mean_p:<12.4f} {mean_r:<12.4f} {mean_f1:<12.4f} {mean_ndcg:<12.4f} {mean_hit:<12.4f}")

print()

# Catalog coverage
catalog_coverage = coverage(recommended_items_list, CATALOG_SIZE)
print(f"Catalog Coverage:                {catalog_coverage:.4f}")
print(f"  ({catalog_coverage * CATALOG_SIZE:.0f} out of {CATALOG_SIZE} items recommended)")
print()

# =============================================================================
# METRIC INTERPRETATIONS
# =============================================================================

print("=" * 80)
print("METRIC INTERPRETATIONS")
print("=" * 80)
print("""
1. PRECISION@K
   - Measures: What fraction of top-K recommendations are relevant?
   - Range: [0, 1], higher is better
   - Use case: When you want to ensure recommended items are mostly relevant
   
2. RECALL@K
   - Measures: What fraction of relevant items appear in top-K?
   - Range: [0, 1], higher is better
   - Use case: When you want to ensure you don't miss relevant items
   
3. F1@K
   - Measures: Harmonic mean of Precision@K and Recall@K
   - Range: [0, 1], higher is better
   - Use case: When you want to balance precision and recall
   
4. AVERAGE PRECISION (AP)
   - Measures: Average precision at each relevant item position
   - Range: [0, 1], higher is better
   - Use case: Considers both relevance and ranking quality
   
5. MEAN AVERAGE PRECISION (MAP)
   - Measures: Mean of AP across all queries
   - Range: [0, 1], higher is better
   - Use case: Standard metric for evaluating ranking quality
   
6. RECIPROCAL RANK (RR)
   - Measures: 1 / (rank of first relevant item)
   - Range: [0, 1], higher is better
   - Use case: When first relevant result matters most (e.g., search engines)
   
7. MEAN RECIPROCAL RANK (MRR)
   - Measures: Mean of RR across all queries
   - Range: [0, 1], higher is better
   - Use case: Evaluating systems where first result is critical
   
8. NORMALIZED DISCOUNTED CUMULATIVE GAIN (NDCG@K)
   - Measures: Ranking quality with position-based discount
   - Range: [0, 1], higher is better
   - Use case: When position in ranking matters (early results more important)
   
9. HIT RATE@K
   - Measures: Binary - was at least one relevant item in top-K?
   - Range: [0, 1], higher is better
   - Use case: Simple metric for "did we get at least one right?"
   
10. CATALOG COVERAGE
    - Measures: Fraction of catalog items that get recommended
    - Range: [0, 1], higher means more diversity
    - Use case: Evaluating recommendation diversity (avoid filter bubbles)
""")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Figure 1: Metrics vs K
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Precision, Recall, F1 vs K
axes[0, 0].plot(k_values_extended, mean_precision_list, 'o-', label='Precision@K', linewidth=2, markersize=8)
axes[0, 0].plot(k_values_extended, mean_recall_list, 's-', label='Recall@K', linewidth=2, markersize=8)
axes[0, 0].plot(k_values_extended, mean_f1_list, '^-', label='F1@K', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('K (Number of Recommendations)', fontsize=11)
axes[0, 0].set_ylabel('Score', fontsize=11)
axes[0, 0].set_title('Precision, Recall, and F1 at Different K', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1.05])

# Plot 2: NDCG vs K
axes[0, 1].plot(k_values_extended, mean_ndcg_list, 'o-', color='purple', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('K (Number of Recommendations)', fontsize=11)
axes[0, 1].set_ylabel('NDCG@K', fontsize=11)
axes[0, 1].set_title('NDCG at Different K', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 1.05])

# Plot 3: Hit Rate vs K
axes[1, 0].plot(k_values_extended, mean_hit_list, 'o-', color='green', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('K (Number of Recommendations)', fontsize=11)
axes[1, 0].set_ylabel('Hit Rate@K', fontsize=11)
axes[1, 0].set_title('Hit Rate at Different K', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1.05])

# Plot 4: Summary bar chart of key metrics
metric_names = ['MAP', 'MRR', 'P@10', 'R@10', 'NDCG@10', 'Hit@10']
metric_values = [
    map_score, 
    mrr_score, 
    mean_precision_list[3],  # P@10 (index 3 in k_values_extended)
    mean_recall_list[3],     # R@10
    mean_ndcg_list[3],       # NDCG@10
    mean_hit_list[3]         # Hit@10
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bars = axes[1, 1].bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Score', fontsize=11)
axes[1, 1].set_title('Summary of Key Metrics', fontsize=12, fontweight='bold')
axes[1, 1].set_ylim([0, 1.05])
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, metric_values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('recommendation_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: recommendation_metrics.png")

# Figure 2: Distribution of metrics across queries
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Calculate distributions
ap_scores = [average_precision(rel, rec) for rel, rec in zip(relevant_items_list, recommended_items_list)]
rr_scores = [reciprocal_rank(rel, rec) for rel, rec in zip(relevant_items_list, recommended_items_list)]
p10_scores = [precision_at_k(rel, rec, 10) for rel, rec in zip(relevant_items_list, recommended_items_list)]
r10_scores = [recall_at_k(rel, rec, 10) for rel, rec in zip(relevant_items_list, recommended_items_list)]
ndcg10_scores = [ndcg_at_k(rel, rec, 10) for rel, rec in zip(relevant_items_list, recommended_items_list)]
hit10_scores = [hit_rate_at_k(rel, rec, 10) for rel, rec in zip(relevant_items_list, recommended_items_list)]

# Plot distributions
distributions = [
    (ap_scores, 'Average Precision', axes[0, 0]),
    (rr_scores, 'Reciprocal Rank', axes[0, 1]),
    (p10_scores, 'Precision@10', axes[0, 2]),
    (r10_scores, 'Recall@10', axes[1, 0]),
    (ndcg10_scores, 'NDCG@10', axes[1, 1]),
    (hit10_scores, 'Hit Rate@10', axes[1, 2])
]

for scores, title, ax in distributions:
    ax.hist(scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
    ax.set_xlabel('Score', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Distribution of {title}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('metric_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metric_distributions.png")

# Figure 3: Precision-Recall curve at different K values
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(mean_recall_list, mean_precision_list, 'o-', linewidth=2, markersize=10, color='darkblue')

# Annotate each point with K value
for i, k in enumerate(k_values_extended):
    ax.annotate(f'K={k}', 
                (mean_recall_list[i], mean_precision_list[i]),
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.set_xlabel('Recall@K', fontsize=12)
ax.set_ylabel('Precision@K', fontsize=12)
ax.set_title('Precision-Recall Trade-off at Different K', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: precision_recall_curve.png")

print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Analyzed {NUM_QUERIES} queries with {CATALOG_SIZE} items in catalog")
print(f"✓ Implemented and calculated 10+ recommendation metrics")
print(f"✓ Generated 3 visualization files:")
print("    1. recommendation_metrics.png - Metrics vs K and summary")
print("    2. metric_distributions.png - Distribution of metrics across queries")
print("    3. precision_recall_curve.png - Precision-Recall trade-off")
print()
print("KEY INSIGHTS:")
print(f"  • MAP (Mean Average Precision):      {map_score:.4f}")
print(f"  • MRR (Mean Reciprocal Rank):        {mrr_score:.4f}")
print(f"  • NDCG@10:                           {mean_ndcg_list[3]:.4f}")
print(f"  • Catalog Coverage:                  {catalog_coverage:.4f}")
print()
print("All metrics are implemented in pure NumPy and ready for interviews!")
print("=" * 80)
