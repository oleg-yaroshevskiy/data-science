"""
One-Sample T-Test: Validating Filtered Sample

Use Case: Testing if filtered movie reviews are still representative of the original dataset

This script demonstrates how to validate that filtering hasn't introduced bias
by comparing the filtered sample mean to the original population mean.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)

def validate_filtered_sample():
    """Test if filtered sample is still representative"""
    
    print_separator()
    print("FILTERED SAMPLE VALIDATION: MOVIE REVIEWS")
    print_separator()
    print()
    
    print("SCENARIO:")
    print("-" * 80)
    print("We have 2000 movie reviews with an average rating of 4.1")
    print("After applying filters (e.g., verified purchases, recent reviews),")
    print("we have 100 reviews with an average rating of 3.9")
    print()
    print("QUESTION: Did our filtering introduce bias, or is this difference")
    print("          just random variation?")
    print()
    
    # Original population parameters
    original_n = 2000
    original_mean = 4.1
    
    # Simulate filtered sample (assuming similar std dev)
    # In reality, you would use your actual filtered data
    filtered_n = 100
    filtered_std = 0.9  # Standard deviation of filtered sample
    
    # Generate sample data centered around 3.9
    # (In practice, you'd use your actual filtered reviews)
    filtered_reviews = np.random.normal(loc=3.9, scale=filtered_std, size=filtered_n)
    
    # Ensure ratings are in valid range [1, 5]
    filtered_reviews = np.clip(filtered_reviews, 1, 5)
    
    filtered_mean = np.mean(filtered_reviews)
    filtered_std = np.std(filtered_reviews, ddof=1)
    
    print("DATA SUMMARY:")
    print("-" * 80)
    print(f"Original dataset:")
    print(f"  Sample size:             {original_n}")
    print(f"  Mean rating:             {original_mean:.2f} ⭐")
    print()
    print(f"Filtered dataset:")
    print(f"  Sample size:             {filtered_n}")
    print(f"  Mean rating:             {filtered_mean:.2f} ⭐")
    print(f"  Std deviation:           {filtered_std:.2f}")
    print()
    print(f"Observed difference:       {filtered_mean - original_mean:.2f}")
    print()
    
    # Formulate hypotheses
    print("HYPOTHESES:")
    print("-" * 80)
    print(f"H₀ (Null Hypothesis):      μ_filtered = {original_mean}")
    print("                           (Filtering didn't introduce bias)")
    print()
    print(f"H₁ (Alternative Hypothesis): μ_filtered ≠ {original_mean}")
    print("                           (Filtering introduced bias)")
    print()
    print("Test Type:                 Two-tailed one-sample t-test")
    print("Significance Level (α):    0.05")
    print()
    
    # Perform one-sample t-test
    t_statistic, p_value = stats.ttest_1samp(filtered_reviews, original_mean)
    
    # Degrees of freedom
    df = filtered_n - 1
    
    # Critical value
    alpha = 0.05
    critical_value = stats.t.ppf(1 - alpha/2, df)
    
    # Standard error
    se = filtered_std / np.sqrt(filtered_n)
    
    print("TEST RESULTS:")
    print("-" * 80)
    print(f"T-statistic:               {t_statistic:.4f}")
    print(f"Degrees of freedom:        {df}")
    print(f"P-value:                   {p_value:.4f}")
    print(f"Critical value (±):        ±{critical_value:.4f}")
    print(f"Standard error:            {se:.4f}")
    print()
    
    # Confidence interval
    ci = stats.t.interval(0.95, df, loc=filtered_mean, scale=se)
    
    print("CONFIDENCE INTERVAL:")
    print("-" * 80)
    print(f"95% CI for filtered mean:  [{ci[0]:.2f}, {ci[1]:.2f}]")
    print()
    print(f"Does CI contain original mean ({original_mean})? ", end="")
    if ci[0] <= original_mean <= ci[1]:
        print("✓ YES")
        print("The original mean is within the confidence interval.")
    else:
        print("✗ NO")
        print("The original mean is outside the confidence interval.")
    print()
    
    # Decision
    print("STATISTICAL DECISION:")
    print("-" * 80)
    if p_value < alpha:
        print(f"✓ REJECT H₀ (p = {p_value:.4f} < {alpha})")
        print()
        print("INTERPRETATION:")
        print("⚠️  Your filtering appears to have introduced BIAS.")
        print(f"    The filtered sample mean ({filtered_mean:.2f}) is significantly")
        print(f"    different from the original mean ({original_mean:.2f}).")
        print()
        print("RECOMMENDATIONS:")
        print("- Review your filtering criteria")
        print("- Check if filters are removing specific rating groups")
        print("- Consider adjusting filters or noting the bias in analysis")
    else:
        print(f"✗ FAIL TO REJECT H₀ (p = {p_value:.4f} ≥ {alpha})")
        print()
        print("INTERPRETATION:")
        print("✓ Your filtered sample is REPRESENTATIVE.")
        print(f"   The difference between filtered mean ({filtered_mean:.2f})")
        print(f"   and original mean ({original_mean:.2f}) can be explained")
        print("   by random sampling variation.")
        print()
        print("CONCLUSION:")
        print("Your filtering does not appear to have introduced significant bias.")
        print("The filtered sample is a valid representation of the original data.")
    
    print()
    
    # Effect size
    cohens_d = (filtered_mean - original_mean) / filtered_std
    
    print("EFFECT SIZE:")
    print("-" * 80)
    print(f"Cohen's d:                 {cohens_d:.4f}")
    
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    print(f"Effect size interpretation: {effect_interpretation}")
    print(f"Practical significance:     ", end="")
    
    if abs(filtered_mean - original_mean) < 0.2:
        print("Minimal practical difference (< 0.2 stars)")
    elif abs(filtered_mean - original_mean) < 0.5:
        print("Moderate practical difference")
    else:
        print("Large practical difference (≥ 0.5 stars)")
    
    print()
    print_separator()
    
    # Create visualizations
    create_visualizations(filtered_reviews, original_mean, filtered_mean, ci,
                         t_statistic, p_value, df, original_n)
    
    return {
        'filtered_reviews': filtered_reviews,
        'filtered_mean': filtered_mean,
        'original_mean': original_mean,
        'p_value': p_value,
        't_statistic': t_statistic,
        'ci': ci
    }

def create_visualizations(filtered_reviews, original_mean, filtered_mean, ci,
                         t_statistic, p_value, df, original_n):
    """Create visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Filtered Sample Validation: Movie Reviews', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Distribution of filtered reviews
    ax1 = axes[0, 0]
    ax1.hist(filtered_reviews, bins=20, alpha=0.7, color='skyblue', 
            edgecolor='black', density=True)
    
    # Add density curve
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(filtered_reviews)
    x_range = np.linspace(1, 5, 100)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Density curve')
    
    # Mark means
    ax1.axvline(original_mean, color='green', linestyle='--', linewidth=2.5,
               label=f'Original mean = {original_mean:.2f}')
    ax1.axvline(filtered_mean, color='red', linestyle='-', linewidth=2.5,
               label=f'Filtered mean = {filtered_mean:.2f}')
    
    ax1.set_xlabel('Rating (1-5 stars)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Distribution of Filtered Reviews', fontsize=12, fontweight='bold')
    ax1.set_xlim(1, 5)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Mean comparison with CI
    ax2 = axes[0, 1]
    
    # Plot filtered mean with CI
    ax2.errorbar([1], [filtered_mean], 
                yerr=[[filtered_mean - ci[0]], [ci[1] - filtered_mean]], 
                fmt='o', markersize=12, color='blue', capsize=15, capthick=3, 
                elinewidth=3, label='Filtered sample\n(95% CI)')
    
    # Plot original mean
    ax2.plot([0.7, 1.3], [original_mean, original_mean], 'g--', linewidth=3,
            label=f'Original mean ({original_mean:.2f})')
    
    # Shade region
    in_ci = ci[0] <= original_mean <= ci[1]
    if in_ci:
        ax2.axhspan(ci[0], ci[1], alpha=0.2, color='green', 
                   label='CI contains original')
    else:
        ax2.axhspan(ci[0], ci[1], alpha=0.2, color='red',
                   label='CI excludes original')
    
    ax2.set_xlim(0.5, 1.5)
    ax2.set_ylim(1, 5)
    ax2.set_ylabel('Rating', fontsize=11)
    ax2.set_title('Mean Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Filtered\nSample'])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis='y')
    
    # Plot 3: T-distribution
    ax3 = axes[1, 0]
    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df)
    
    ax3.plot(x, y, 'b-', linewidth=2, label='t-distribution')
    ax3.fill_between(x, 0, y, where=(x <= -abs(t_statistic)), 
                    alpha=0.3, color='red', label='Rejection region')
    ax3.fill_between(x, 0, y, where=(x >= abs(t_statistic)), 
                    alpha=0.3, color='red')
    
    ax3.axvline(t_statistic, color='red', linestyle='-', linewidth=2,
               label=f't = {t_statistic:.2f}')
    
    critical_value = stats.t.ppf(0.975, df)
    ax3.axvline(-critical_value, color='orange', linestyle='--', linewidth=1.5,
               label=f'Critical = ±{critical_value:.2f}')
    ax3.axvline(critical_value, color='orange', linestyle='--', linewidth=1.5)
    
    ax3.set_xlabel('t-value', fontsize=11)
    ax3.set_ylabel('Probability Density', fontsize=11)
    ax3.set_title(f'T-Distribution (df={df}, p={p_value:.4f})', 
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # Plot 4: Sample size comparison
    ax4 = axes[1, 1]
    
    categories = ['Original\nDataset', 'Filtered\nSample']
    sizes = [original_n, len(filtered_reviews)]
    means = [original_mean, filtered_mean]
    colors = ['lightgreen', 'lightblue']
    
    # Create bar chart with two y-axes
    ax4_twin = ax4.twinx()
    
    # Sample sizes on left axis
    bars1 = ax4.bar([0], [sizes[0]], width=0.35, color=colors[0], 
                   edgecolor='black', linewidth=1.5, label='Sample size')
    bars2 = ax4.bar([1], [sizes[1]], width=0.35, color=colors[1], 
                   edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('Sample Size (n)', fontsize=11, color='black')
    ax4.set_ylim(0, max(sizes) * 1.2)
    
    # Add sample size labels
    for i, (bar, size) in enumerate(zip([bars1, bars2], sizes)):
        height = bar[0].get_height()
        ax4.text(i, height, f'n={size}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # Means on right axis (as line plot)
    x_pos = [0, 1]
    ax4_twin.plot(x_pos, means, 'ro-', linewidth=3, markersize=12, 
                 label='Mean rating')
    ax4_twin.set_ylabel('Mean Rating', fontsize=11, color='red')
    ax4_twin.set_ylim(1, 5)
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    # Add mean labels
    for i, mean in enumerate(means):
        ax4_twin.text(i + 0.1, mean, f'{mean:.2f}⭐', ha='left', va='center',
                     fontweight='bold', fontsize=10, color='red')
    
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(categories)
    ax4.set_title('Dataset Comparison', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    filename = 'filtered_sample_validation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{filename}'")
    
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXAMPLE: Validating Filtered Movie Reviews Sample")
    print("="*80)
    print()
    print("This example demonstrates how to test if your filtering criteria")
    print("have introduced bias into your sample.")
    print()
    print("IMPORTANT: In practice, you would replace the simulated data")
    print("with your actual filtered reviews.")
    print()
    
    results = validate_filtered_sample()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS:")
    print("="*80)
    print()
    print("✓ Use one-sample t-test to validate filtered samples")
    print("✓ Check both statistical significance (p-value) and practical significance")
    print("✓ Examine if the CI includes the original mean")
    print("✓ Consider effect size (Cohen's d) for practical interpretation")
    print()
    print("If filtering introduces bias, you should either:")
    print("  1. Adjust your filtering criteria")
    print("  2. Acknowledge and document the bias in your analysis")
    print("  3. Use the original dataset if representativeness is critical")
    print()
