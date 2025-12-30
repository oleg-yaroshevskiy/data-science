"""
Bootstrap Confidence Interval Example

Use Case: Estimating confidence intervals using bootstrap resampling

This script demonstrates bootstrap methods for:
1. Confidence intervals for mean
2. Confidence intervals for median
3. Confidence intervals for difference between two means
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

def bootstrap_statistic(data, statistic_func, n_bootstrap=10000, confidence_level=0.95):
    """
    Perform bootstrap resampling to estimate confidence interval
    
    Parameters:
    -----------
    data : array-like
        The original sample data
    statistic_func : callable
        Function to compute the statistic (e.g., np.mean, np.median)
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level for the interval
        
    Returns:
    --------
    dict with bootstrap results
    """
    n = len(data)
    bootstrap_samples = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_samples[i] = statistic_func(resample)
    
    # Calculate confidence interval using percentile method
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
    
    return {
        'samples': bootstrap_samples,
        'mean': np.mean(bootstrap_samples),
        'std': np.std(bootstrap_samples),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def example_1_mean_ci():
    """Example 1: Bootstrap confidence interval for mean"""
    
    print_separator()
    print("EXAMPLE 1: BOOTSTRAP CONFIDENCE INTERVAL FOR MEAN")
    print_separator()
    print()
    
    # Generate sample data: reaction times in milliseconds
    sample_data = np.random.gamma(shape=2, scale=50, size=35)
    
    print("SCENARIO:")
    print("-" * 80)
    print("We measured reaction times (in ms) from 35 participants.")
    print("The data appears to be right-skewed (not normally distributed).")
    print("We want to estimate a confidence interval for the mean reaction time.")
    print()
    
    print("SAMPLE STATISTICS:")
    print("-" * 80)
    print(f"Sample size:               {len(sample_data)}")
    print(f"Sample mean:               {np.mean(sample_data):.2f} ms")
    print(f"Sample median:             {np.median(sample_data):.2f} ms")
    print(f"Sample std dev:            {np.std(sample_data, ddof=1):.2f} ms")
    print(f"Skewness:                  {stats.skew(sample_data):.2f}")
    print()
    
    # Parametric confidence interval (assumes normality)
    parametric_ci = stats.t.interval(0.95, len(sample_data)-1, 
                                     loc=np.mean(sample_data),
                                     scale=stats.sem(sample_data))
    
    print("PARAMETRIC CONFIDENCE INTERVAL (t-distribution):")
    print("-" * 80)
    print(f"95% CI:                    [{parametric_ci[0]:.2f}, {parametric_ci[1]:.2f}] ms")
    print("Assumption:                Data are normally distributed")
    print()
    
    # Bootstrap confidence interval
    print("BOOTSTRAP METHOD:")
    print("-" * 80)
    print("Performing 10,000 bootstrap resamples...")
    
    bootstrap_result = bootstrap_statistic(sample_data, np.mean, n_bootstrap=10000)
    
    print(f"Bootstrap mean:            {bootstrap_result['mean']:.2f} ms")
    print(f"Bootstrap std error:       {bootstrap_result['std']:.2f} ms")
    print(f"95% Bootstrap CI:          [{bootstrap_result['ci_lower']:.2f}, {bootstrap_result['ci_upper']:.2f}] ms")
    print()
    print("Advantage:                 No normality assumption required")
    print()
    
    # Comparison
    print("COMPARISON:")
    print("-" * 80)
    print(f"Parametric CI width:       {parametric_ci[1] - parametric_ci[0]:.2f} ms")
    print(f"Bootstrap CI width:        {bootstrap_result['ci_upper'] - bootstrap_result['ci_lower']:.2f} ms")
    print()
    print_separator()
    
    return sample_data, bootstrap_result, parametric_ci

def example_2_median_ci():
    """Example 2: Bootstrap confidence interval for median"""
    
    print("\n")
    print_separator()
    print("EXAMPLE 2: BOOTSTRAP CONFIDENCE INTERVAL FOR MEDIAN")
    print_separator()
    print()
    
    # Generate sample data with outliers
    main_data = np.random.normal(loc=50, scale=10, size=40)
    outliers = np.array([100, 110, 95])  # Add some outliers
    sample_data = np.concatenate([main_data, outliers])
    
    print("SCENARIO:")
    print("-" * 80)
    print("We have income data (in thousands) with some outliers.")
    print("The median is more robust to outliers than the mean.")
    print("We want a confidence interval for the median income.")
    print()
    
    print("SAMPLE STATISTICS:")
    print("-" * 80)
    print(f"Sample size:               {len(sample_data)}")
    print(f"Sample mean:               ${np.mean(sample_data):.2f}k")
    print(f"Sample median:             ${np.median(sample_data):.2f}k")
    print(f"Note:                      Mean is inflated by outliers")
    print()
    
    # Bootstrap confidence interval for median
    print("BOOTSTRAP METHOD FOR MEDIAN:")
    print("-" * 80)
    print("Performing 10,000 bootstrap resamples...")
    
    bootstrap_result = bootstrap_statistic(sample_data, np.median, n_bootstrap=10000)
    
    print(f"Bootstrap median:          ${bootstrap_result['mean']:.2f}k")
    print(f"Bootstrap std error:       ${bootstrap_result['std']:.2f}k")
    print(f"95% Bootstrap CI:          [${bootstrap_result['ci_lower']:.2f}k, ${bootstrap_result['ci_upper']:.2f}k]")
    print()
    print("Interpretation:            We are 95% confident the true median income")
    print(f"                           is between ${bootstrap_result['ci_lower']:.2f}k and ${bootstrap_result['ci_upper']:.2f}k")
    print()
    print_separator()
    
    return sample_data, bootstrap_result

def example_3_difference_means():
    """Example 3: Bootstrap confidence interval for difference between means"""
    
    print("\n")
    print_separator()
    print("EXAMPLE 3: BOOTSTRAP CI FOR DIFFERENCE BETWEEN TWO MEANS")
    print_separator()
    print()
    
    # Generate two samples
    group_a = np.random.exponential(scale=20, size=30)
    group_b = np.random.exponential(scale=25, size=35)
    
    print("SCENARIO:")
    print("-" * 80)
    print("We want to compare average response times between two groups.")
    print("The data are right-skewed (exponential distribution).")
    print("We want a CI for the difference in means.")
    print()
    
    print("SAMPLE STATISTICS:")
    print("-" * 80)
    print(f"Group A: n={len(group_a)}, mean={np.mean(group_a):.2f}, std={np.std(group_a, ddof=1):.2f}")
    print(f"Group B: n={len(group_b)}, mean={np.mean(group_b):.2f}, std={np.std(group_b, ddof=1):.2f}")
    print(f"Observed difference (B - A): {np.mean(group_b) - np.mean(group_a):.2f}")
    print()
    
    # Bootstrap for difference
    print("BOOTSTRAP METHOD FOR DIFFERENCE:")
    print("-" * 80)
    print("Performing 10,000 bootstrap resamples...")
    
    n_bootstrap = 10000
    bootstrap_diffs = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        resample_a = np.random.choice(group_a, size=len(group_a), replace=True)
        resample_b = np.random.choice(group_b, size=len(group_b), replace=True)
        bootstrap_diffs[i] = np.mean(resample_b) - np.mean(resample_a)
    
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    print(f"Bootstrap mean difference: {np.mean(bootstrap_diffs):.2f}")
    print(f"Bootstrap std error:       {np.std(bootstrap_diffs):.2f}")
    print(f"95% Bootstrap CI:          [{ci_lower:.2f}, {ci_upper:.2f}]")
    print()
    
    if ci_lower > 0:
        print("INTERPRETATION:")
        print("The confidence interval does not contain zero.")
        print("There is evidence that Group B has a higher mean than Group A.")
    elif ci_upper < 0:
        print("INTERPRETATION:")
        print("The confidence interval does not contain zero.")
        print("There is evidence that Group A has a higher mean than Group B.")
    else:
        print("INTERPRETATION:")
        print("The confidence interval contains zero.")
        print("There is no strong evidence of a difference between groups.")
    
    print()
    print_separator()
    
    return group_a, group_b, bootstrap_diffs, ci_lower, ci_upper

def create_visualizations(data1, bootstrap1, parametric_ci, data2, bootstrap2,
                         group_a, group_b, bootstrap_diffs, ci_lower, ci_upper):
    """Create comprehensive visualization of bootstrap results"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Bootstrap Confidence Intervals: Comprehensive Examples', 
                 fontsize=16, fontweight='bold')
    
    # Example 1: Original data and bootstrap distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(data1, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(data1), color='red', linestyle='--', linewidth=2, 
               label=f'Sample mean = {np.mean(data1):.2f}')
    ax1.set_xlabel('Reaction Time (ms)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Example 1: Original Data (Skewed)', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(bootstrap1['samples'], bins=50, alpha=0.7, color='lightgreen', 
            edgecolor='black', density=True)
    ax2.axvline(bootstrap1['ci_lower'], color='red', linestyle='--', linewidth=2)
    ax2.axvline(bootstrap1['ci_upper'], color='red', linestyle='--', linewidth=2, 
               label=f"95% CI: [{bootstrap1['ci_lower']:.1f}, {bootstrap1['ci_upper']:.1f}]")
    ax2.axvline(np.mean(data1), color='blue', linestyle='-', linewidth=2, 
               label=f'Original mean = {np.mean(data1):.2f}')
    ax2.set_xlabel('Bootstrap Mean (ms)', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title('Bootstrap Distribution of Mean', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    means_comparison = ['Parametric\n(t-dist)', 'Bootstrap']
    ci_widths = [parametric_ci[1] - parametric_ci[0], 
                 bootstrap1['ci_upper'] - bootstrap1['ci_lower']]
    colors = ['lightblue', 'lightgreen']
    bars = ax3.bar(means_comparison, ci_widths, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('CI Width (ms)', fontsize=10)
    ax3.set_title('Comparison of CI Widths', fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    for bar, width in zip(bars, ci_widths):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{width:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Example 2: Median bootstrap
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(data2, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.axvline(np.mean(data2), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean = {np.mean(data2):.2f}k')
    ax4.axvline(np.median(data2), color='green', linestyle='--', linewidth=2, 
               label=f'Median = {np.median(data2):.2f}k')
    ax4.set_xlabel('Income ($k)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Example 2: Data with Outliers', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(bootstrap2['samples'], bins=50, alpha=0.7, color='plum', 
            edgecolor='black', density=True)
    ax5.axvline(bootstrap2['ci_lower'], color='red', linestyle='--', linewidth=2)
    ax5.axvline(bootstrap2['ci_upper'], color='red', linestyle='--', linewidth=2,
               label=f"95% CI: [{bootstrap2['ci_lower']:.1f}, {bootstrap2['ci_upper']:.1f}]")
    ax5.axvline(np.median(data2), color='green', linestyle='-', linewidth=2,
               label=f'Original median = {np.median(data2):.2f}k')
    ax5.set_xlabel('Bootstrap Median ($k)', fontsize=10)
    ax5.set_ylabel('Density', fontsize=10)
    ax5.set_title('Bootstrap Distribution of Median', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    bp = ax6.boxplot(data2, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][0].set_edgecolor('black')
    ax6.axhline(np.median(data2), color='green', linestyle='--', linewidth=2, 
               label='Median')
    ax6.axhline(np.mean(data2), color='blue', linestyle='--', linewidth=2, 
               label='Mean')
    ax6.set_ylabel('Income ($k)', fontsize=10)
    ax6.set_title('Box Plot Shows Outliers', fontsize=11, fontweight='bold')
    ax6.set_xticks([1])
    ax6.set_xticklabels(['Income'])
    ax6.legend()
    ax6.grid(alpha=0.3, axis='y')
    
    # Example 3: Difference between groups
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(group_a, bins=15, alpha=0.6, color='blue', label='Group A', 
            edgecolor='black', density=True)
    ax7.hist(group_b, bins=15, alpha=0.6, color='red', label='Group B', 
            edgecolor='black', density=True)
    ax7.axvline(np.mean(group_a), color='blue', linestyle='--', linewidth=2)
    ax7.axvline(np.mean(group_b), color='red', linestyle='--', linewidth=2)
    ax7.set_xlabel('Response Time', fontsize=10)
    ax7.set_ylabel('Density', fontsize=10)
    ax7.set_title('Example 3: Two Groups Comparison', fontsize=11, fontweight='bold')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(bootstrap_diffs, bins=50, alpha=0.7, color='orange', 
            edgecolor='black', density=True)
    ax8.axvline(ci_lower, color='red', linestyle='--', linewidth=2)
    ax8.axvline(ci_upper, color='red', linestyle='--', linewidth=2,
               label=f'95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]')
    ax8.axvline(0, color='green', linestyle='-', linewidth=2, 
               label='No difference')
    ax8.axvline(np.mean(bootstrap_diffs), color='blue', linestyle='--', linewidth=2,
               label=f'Mean diff = {np.mean(bootstrap_diffs):.2f}')
    ax8.set_xlabel('Difference (B - A)', fontsize=10)
    ax8.set_ylabel('Density', fontsize=10)
    ax8.set_title('Bootstrap Distribution of Difference', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, 2])
    groups = ['Group A', 'Group B']
    means = [np.mean(group_a), np.mean(group_b)]
    bars = ax9.bar(groups, means, color=['lightblue', 'lightcoral'], 
                  edgecolor='black', linewidth=1.5)
    ax9.set_ylabel('Mean Response Time', fontsize=10)
    ax9.set_title('Mean Comparison', fontsize=11, fontweight='bold')
    ax9.grid(alpha=0.3, axis='y')
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    filename = 'bootstrap_confidence_intervals.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{filename}'")
    
    plt.show()

if __name__ == "__main__":
    # Run all examples
    data1, bootstrap1, parametric_ci = example_1_mean_ci()
    data2, bootstrap2 = example_2_median_ci()
    group_a, group_b, bootstrap_diffs, ci_lower, ci_upper = example_3_difference_means()
    
    # Create comprehensive visualization
    create_visualizations(data1, bootstrap1, parametric_ci, data2, bootstrap2,
                         group_a, group_b, bootstrap_diffs, ci_lower, ci_upper)
