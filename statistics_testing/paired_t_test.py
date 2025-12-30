"""
Paired T-Test Example: Before-After Treatment Analysis

Use Case: Does a fitness program reduce body weight?

This script demonstrates a paired t-test (dependent samples) to compare
measurements from the same subjects before and after a treatment.
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

def perform_paired_t_test():
    """Perform paired t-test on before-after data"""
    
    print_separator()
    print("PAIRED T-TEST: FITNESS PROGRAM WEIGHT LOSS ANALYSIS")
    print_separator()
    print()
    
    # Generate sample data
    n_participants = 30
    
    # Before weights (kg)
    weight_before = np.random.normal(loc=85, scale=12, size=n_participants)
    
    # After weights (simulating weight loss of ~3 kg with some variation)
    # Each person loses different amount, but on average 3 kg
    weight_loss = np.random.normal(loc=3, scale=2, size=n_participants)
    weight_after = weight_before - weight_loss
    
    # Calculate differences
    differences = weight_after - weight_before
    
    print("SCENARIO:")
    print("-" * 80)
    print(f"We measured the weight of {n_participants} participants before and after")
    print("a 12-week fitness program. We want to test if the program")
    print("significantly reduces body weight.")
    print()
    print("Each participant serves as their own control (paired design).")
    print()
    
    # Display sample of data
    print("SAMPLE DATA (First 5 participants):")
    print("-" * 80)
    print(f"{'ID':<5} {'Before (kg)':<15} {'After (kg)':<15} {'Difference':<15}")
    print("-" * 80)
    for i in range(min(5, n_participants)):
        print(f"{i+1:<5} {weight_before[i]:<15.2f} {weight_after[i]:<15.2f} {differences[i]:<15.2f}")
    print()
    
    # Calculate statistics
    mean_before = np.mean(weight_before)
    mean_after = np.mean(weight_after)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(n_participants)
    
    print("DESCRIPTIVE STATISTICS:")
    print("-" * 80)
    print(f"Sample size (n):           {n_participants}")
    print()
    print(f"Before Program:")
    print(f"  Mean weight:             {mean_before:.2f} kg")
    print(f"  Std deviation:           {np.std(weight_before, ddof=1):.2f} kg")
    print()
    print(f"After Program:")
    print(f"  Mean weight:             {mean_after:.2f} kg")
    print(f"  Std deviation:           {np.std(weight_after, ddof=1):.2f} kg")
    print()
    print(f"Differences (After - Before):")
    print(f"  Mean difference:         {mean_diff:.2f} kg")
    print(f"  Std deviation:           {std_diff:.2f} kg")
    print(f"  Standard error:          {se_diff:.2f} kg")
    print()
    
    # Formulate hypotheses
    print("HYPOTHESES:")
    print("-" * 80)
    print("H₀ (Null Hypothesis):      μ_diff = 0")
    print("                           (The program has no effect on weight)")
    print()
    print("H₁ (Alternative Hypothesis): μ_diff < 0")
    print("                           (The program reduces weight)")
    print()
    print("Test Type:                 One-tailed paired t-test (left-tailed)")
    print("Significance Level (α):    0.05")
    print()
    
    # Perform paired t-test
    # One-tailed test: alternative='less' because we expect weight_after < weight_before
    t_statistic, p_value_two_tailed = stats.ttest_rel(weight_after, weight_before)
    
    # For one-tailed test (left tail)
    p_value = p_value_two_tailed / 2 if t_statistic < 0 else 1 - p_value_two_tailed / 2
    
    # Degrees of freedom
    df = n_participants - 1
    
    # Critical value for one-tailed test
    alpha = 0.05
    critical_value = stats.t.ppf(alpha, df)  # Left tail
    
    print("TEST RESULTS:")
    print("-" * 80)
    print(f"T-statistic:               {t_statistic:.4f}")
    print(f"Degrees of freedom:        {df}")
    print(f"P-value (one-tailed):      {p_value:.4f}")
    print(f"Critical value:            {critical_value:.4f}")
    print()
    
    # Calculate confidence interval for the mean difference
    ci_level = 0.95
    ci = stats.t.interval(ci_level, df, loc=mean_diff, scale=se_diff)
    
    print("CONFIDENCE INTERVAL:")
    print("-" * 80)
    print(f"95% CI for mean difference: [{ci[0]:.2f}, {ci[1]:.2f}] kg")
    print(f"Interpretation:            We are 95% confident that the true mean")
    print(f"                           weight change is between {ci[0]:.2f} and {ci[1]:.2f} kg")
    print()
    
    # Decision
    print("DECISION:")
    print("-" * 80)
    if p_value < alpha:
        print(f"✓ REJECT the null hypothesis (p = {p_value:.4f} < {alpha})")
        print()
        print("CONCLUSION:")
        print(f"There is sufficient evidence to conclude that the fitness program")
        print(f"significantly reduces body weight.")
        print(f"Average weight loss: {abs(mean_diff):.2f} kg")
    else:
        print(f"✗ FAIL TO REJECT the null hypothesis (p = {p_value:.4f} ≥ {alpha})")
        print()
        print("CONCLUSION:")
        print(f"There is insufficient evidence to conclude that the fitness program")
        print(f"significantly reduces body weight.")
    
    print()
    
    # Effect size (Cohen's d for paired samples)
    cohens_d = mean_diff / std_diff
    
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
    print()
    
    # Check normality of differences
    shapiro_stat, shapiro_p = stats.shapiro(differences)
    print("NORMALITY CHECK (Shapiro-Wilk Test):")
    print("-" * 80)
    print(f"Shapiro-Wilk statistic:    {shapiro_stat:.4f}")
    print(f"P-value:                   {shapiro_p:.4f}")
    
    if shapiro_p > 0.05:
        print("Result:                    Differences appear normally distributed (p > 0.05)")
        print("                           Paired t-test is appropriate")
    else:
        print("Result:                    Differences may not be normally distributed (p ≤ 0.05)")
        print("                           Consider non-parametric alternative (Wilcoxon)")
    
    print()
    print_separator()
    
    # Create visualizations
    create_visualizations(weight_before, weight_after, differences, mean_diff, ci,
                         t_statistic, p_value, df)
    
    return {
        'before': weight_before,
        'after': weight_after,
        'differences': differences,
        't_statistic': t_statistic,
        'p_value': p_value
    }

def create_visualizations(before, after, differences, mean_diff, ci, t_statistic, 
                         p_value, df):
    """Create and save visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Paired T-Test: Fitness Program Weight Loss Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Before-After comparison with connecting lines
    ax1 = axes[0, 0]
    
    n_show = min(len(before), 30)  # Show up to 30 participants
    x_positions = [1, 2]
    
    # Plot individual lines
    for i in range(n_show):
        ax1.plot(x_positions, [before[i], after[i]], 'o-', color='gray', 
                alpha=0.4, linewidth=1)
    
    # Plot means
    mean_before = np.mean(before)
    mean_after = np.mean(after)
    ax1.plot(x_positions, [mean_before, mean_after], 'ro-', linewidth=3, 
            markersize=12, label='Mean', zorder=10)
    
    ax1.set_xlim(0.5, 2.5)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(['Before', 'After'])
    ax1.set_ylabel('Weight (kg)', fontsize=11)
    ax1.set_title('Individual Weight Changes', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    
    # Add annotations for means
    ax1.text(1, mean_before, f'{mean_before:.1f} kg', ha='right', va='bottom', 
            fontweight='bold')
    ax1.text(2, mean_after, f'{mean_after:.1f} kg', ha='left', va='bottom', 
            fontweight='bold')
    
    # Plot 2: Histogram of differences
    ax2 = axes[0, 1]
    ax2.hist(differences, bins=12, alpha=0.7, color='skyblue', edgecolor='black', 
            density=True)
    
    # Overlay normal distribution
    x_range = np.linspace(differences.min(), differences.max(), 100)
    ax2.plot(x_range, stats.norm.pdf(x_range, mean_diff, np.std(differences, ddof=1)), 
             'r-', linewidth=2, label='Normal fit')
    
    # Mark zero and mean
    ax2.axvline(0, color='green', linestyle='--', linewidth=2, label='No change (H₀)')
    ax2.axvline(mean_diff, color='red', linestyle='-', linewidth=2, 
               label=f'Mean diff = {mean_diff:.2f} kg')
    
    ax2.set_xlabel('Weight Difference (After - Before) kg', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Distribution of Weight Changes', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Box plot of before and after
    ax3 = axes[1, 0]
    bp = ax3.boxplot([before, after], labels=['Before', 'After'],
                     patch_artist=True, widths=0.5)
    
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    # Add mean markers
    means = [np.mean(before), np.mean(after)]
    ax3.plot([1, 2], means, 'rd', markersize=10, label='Mean', zorder=10)
    
    ax3.set_ylabel('Weight (kg)', fontsize=11)
    ax3.set_title('Weight Distribution Before and After', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: Mean difference with confidence interval
    ax4 = axes[1, 1]
    
    # Plot mean difference with CI
    ax4.errorbar([1], [mean_diff], yerr=[[mean_diff - ci[0]], [ci[1] - mean_diff]], 
                fmt='o', markersize=12, color='blue', capsize=15, capthick=3, 
                elinewidth=3, label='Mean difference ± 95% CI')
    
    # Add horizontal line at zero
    ax4.axhline(0, color='red', linestyle='--', linewidth=2, label='No effect (H₀)')
    
    # Shade region below zero (weight loss)
    if mean_diff < 0:
        ax4.axhspan(ax4.get_ylim()[0], 0, alpha=0.2, color='green', 
                   label='Weight loss region')
    
    ax4.set_xlim(0.5, 1.5)
    ax4.set_ylabel('Weight Difference (kg)', fontsize=11)
    ax4.set_title(f'Mean Weight Change\n(p = {p_value:.4f})', 
                 fontsize=12, fontweight='bold')
    ax4.set_xticks([1])
    ax4.set_xticklabels(['Difference\n(After - Before)'])
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    # Add value annotation
    ax4.text(1.05, mean_diff, f'{mean_diff:.2f} kg', ha='left', va='center', 
            fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    filename = 'paired_t_test.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{filename}'")
    
    plt.show()

if __name__ == "__main__":
    results = perform_paired_t_test()
