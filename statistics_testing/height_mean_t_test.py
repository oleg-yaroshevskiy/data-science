"""
One-Sample T-Test Example: Height Analysis

Use Case: Is the average height different from 170 cm?

This script demonstrates a one-sample t-test to determine if the sample mean
is significantly different from a hypothesized population mean.
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

def perform_one_sample_t_test():
    """Perform one-sample t-test on height data"""
    
    print_separator()
    print("ONE-SAMPLE T-TEST: HEIGHT ANALYSIS")
    print_separator()
    print()
    
    # Generate sample data: heights in cm (simulated data)
    # True population mean is 172 cm with std dev of 8 cm
    sample_size = 50
    heights = np.random.normal(loc=172, scale=8, size=sample_size)
    
    # Hypothesized population mean
    mu_0 = 170
    
    print("SCENARIO:")
    print("-" * 80)
    print(f"We collected height measurements from {sample_size} individuals.")
    print(f"We want to test if the average height is different from {mu_0} cm.")
    print()
    
    # Calculate sample statistics
    sample_mean = np.mean(heights)
    sample_std = np.std(heights, ddof=1)  # Using sample standard deviation
    sample_sem = sample_std / np.sqrt(sample_size)
    
    print("SAMPLE STATISTICS:")
    print("-" * 80)
    print(f"Sample size (n):           {sample_size}")
    print(f"Sample mean (x̄):           {sample_mean:.2f} cm")
    print(f"Sample std dev (s):        {sample_std:.2f} cm")
    print(f"Standard error (SEM):      {sample_sem:.2f} cm")
    print()
    
    # Formulate hypotheses
    print("HYPOTHESES:")
    print("-" * 80)
    print("H₀ (Null Hypothesis):      μ = 170 cm")
    print("                           (The population mean height is 170 cm)")
    print()
    print("H₁ (Alternative Hypothesis): μ ≠ 170 cm")
    print("                           (The population mean height is different from 170 cm)")
    print()
    print("Test Type:                 Two-tailed test")
    print("Significance Level (α):    0.05")
    print()
    
    # Perform t-test
    t_statistic, p_value = stats.ttest_1samp(heights, mu_0)
    
    # Degrees of freedom
    df = sample_size - 1
    
    # Critical value for two-tailed test at alpha = 0.05
    alpha = 0.05
    critical_value = stats.t.ppf(1 - alpha/2, df)
    
    print("TEST RESULTS:")
    print("-" * 80)
    print(f"T-statistic:               {t_statistic:.4f}")
    print(f"Degrees of freedom:        {df}")
    print(f"P-value:                   {p_value:.4f}")
    print(f"Critical value (±):        ±{critical_value:.4f}")
    print()
    
    # Calculate confidence interval
    ci_level = 0.95
    ci = stats.t.interval(ci_level, df, loc=sample_mean, scale=sample_sem)
    
    print("CONFIDENCE INTERVAL:")
    print("-" * 80)
    print(f"95% Confidence Interval:   [{ci[0]:.2f}, {ci[1]:.2f}] cm")
    print(f"Interpretation:            We are 95% confident that the true population")
    print(f"                           mean height lies between {ci[0]:.2f} and {ci[1]:.2f} cm")
    print()
    
    # Decision
    print("DECISION:")
    print("-" * 80)
    if p_value < alpha:
        print(f"✓ REJECT the null hypothesis (p = {p_value:.4f} < {alpha})")
        print()
        print("CONCLUSION:")
        print(f"There is sufficient evidence to conclude that the average height")
        print(f"is significantly different from {mu_0} cm.")
        print(f"The sample mean is {sample_mean:.2f} cm.")
    else:
        print(f"✗ FAIL TO REJECT the null hypothesis (p = {p_value:.4f} ≥ {alpha})")
        print()
        print("CONCLUSION:")
        print(f"There is insufficient evidence to conclude that the average height")
        print(f"is different from {mu_0} cm.")
    
    print()
    
    # Effect size (Cohen's d)
    cohens_d = (sample_mean - mu_0) / sample_std
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
    print_separator()
    
    # Create visualizations
    create_visualizations(heights, mu_0, sample_mean, ci, t_statistic, df)
    
    return {
        'heights': heights,
        'sample_mean': sample_mean,
        'mu_0': mu_0,
        'p_value': p_value,
        't_statistic': t_statistic,
        'ci': ci
    }

def create_visualizations(heights, mu_0, sample_mean, ci, t_statistic, df):
    """Create and save visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('One-Sample T-Test: Height Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Histogram with distribution
    ax1 = axes[0, 0]
    ax1.hist(heights, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Overlay normal distribution
    x_range = np.linspace(heights.min(), heights.max(), 100)
    ax1.plot(x_range, stats.norm.pdf(x_range, sample_mean, np.std(heights, ddof=1)), 
             'r-', linewidth=2, label='Normal fit')
    
    # Mark hypothesized mean
    ax1.axvline(mu_0, color='green', linestyle='--', linewidth=2, label=f'H₀: μ = {mu_0} cm')
    ax1.axvline(sample_mean, color='red', linestyle='-', linewidth=2, label=f'Sample mean = {sample_mean:.2f} cm')
    
    ax1.set_xlabel('Height (cm)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Distribution of Heights', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Box plot with CI
    ax2 = axes[0, 1]
    bp = ax2.boxplot(heights, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    
    # Add confidence interval
    ax2.plot([0.8, 1.2], [ci[0], ci[0]], 'r-', linewidth=2)
    ax2.plot([0.8, 1.2], [ci[1], ci[1]], 'r-', linewidth=2)
    ax2.plot([1, 1], [ci[0], ci[1]], 'r-', linewidth=2, label='95% CI')
    
    # Add hypothesized mean
    ax2.axhline(mu_0, color='green', linestyle='--', linewidth=2, label=f'H₀: μ = {mu_0} cm')
    
    ax2.set_ylabel('Height (cm)', fontsize=11)
    ax2.set_title('Box Plot with Confidence Interval', fontsize=12, fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Heights'])
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    # Plot 3: T-distribution with test statistic
    ax3 = axes[1, 0]
    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df)
    
    ax3.plot(x, y, 'b-', linewidth=2, label='t-distribution')
    ax3.fill_between(x, 0, y, where=(x <= -abs(t_statistic)), alpha=0.3, color='red', label='Rejection region')
    ax3.fill_between(x, 0, y, where=(x >= abs(t_statistic)), alpha=0.3, color='red')
    
    # Mark test statistic
    ax3.axvline(t_statistic, color='red', linestyle='-', linewidth=2, label=f't = {t_statistic:.2f}')
    
    # Mark critical values
    critical_value = stats.t.ppf(0.975, df)
    ax3.axvline(-critical_value, color='orange', linestyle='--', linewidth=1.5, label=f'Critical values = ±{critical_value:.2f}')
    ax3.axvline(critical_value, color='orange', linestyle='--', linewidth=1.5)
    
    ax3.set_xlabel('t-value', fontsize=11)
    ax3.set_ylabel('Probability Density', fontsize=11)
    ax3.set_title(f'T-Distribution (df={df})', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Mean comparison with error bars
    ax4 = axes[1, 1]
    
    # Plot sample mean with CI
    ax4.errorbar([1], [sample_mean], yerr=[[sample_mean - ci[0]], [ci[1] - sample_mean]], 
                 fmt='o', markersize=10, color='blue', capsize=10, capthick=2, 
                 elinewidth=2, label='Sample mean ± 95% CI')
    
    # Plot hypothesized mean
    ax4.plot([0.7, 1.3], [mu_0, mu_0], 'g--', linewidth=2, label=f'Hypothesized mean (μ₀ = {mu_0} cm)')
    
    ax4.set_xlim(0.5, 1.5)
    ax4.set_ylabel('Height (cm)', fontsize=11)
    ax4.set_title('Mean Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks([1])
    ax4.set_xticklabels(['Sample'])
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    filename = 'height_mean_t_test.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{filename}'")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    results = perform_one_sample_t_test()
