"""
Proportion Z-Test Examples

Use Cases:
1. One-sample proportion test: Is conversion rate different from expected?
2. Two-sample proportion test: Do two versions have different conversion rates?

This script demonstrates z-tests for proportions, commonly used in
A/B testing and categorical data analysis.
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

def one_sample_proportion_test():
    """Example 1: One-sample proportion test"""
    
    print_separator()
    print("ONE-SAMPLE PROPORTION Z-TEST")
    print_separator()
    print()
    
    # Sample data
    n_visitors = 1000
    n_conversions = 87
    p_sample = n_conversions / n_visitors
    
    # Hypothesized proportion (e.g., industry standard)
    p_0 = 0.10
    
    print("SCENARIO:")
    print("-" * 80)
    print("We launched a new website and want to test if our conversion rate")
    print("is different from the industry standard of 10%.")
    print()
    print(f"Visitors:                  {n_visitors}")
    print(f"Conversions:               {n_conversions}")
    print(f"Sample proportion:         {p_sample:.4f} ({p_sample*100:.2f}%)")
    print(f"Industry standard:         {p_0:.4f} ({p_0*100:.2f}%)")
    print()
    
    # Formulate hypotheses
    print("HYPOTHESES:")
    print("-" * 80)
    print(f"H₀ (Null Hypothesis):      p = {p_0:.2f}")
    print("                           (Our conversion rate equals industry standard)")
    print()
    print(f"H₁ (Alternative Hypothesis): p ≠ {p_0:.2f}")
    print("                           (Our conversion rate differs from standard)")
    print()
    print("Test Type:                 Two-tailed z-test for proportion")
    print("Significance Level (α):    0.05")
    print()
    
    # Check sample size requirements
    n_p0 = n_visitors * p_0
    n_1minusp0 = n_visitors * (1 - p_0)
    
    print("SAMPLE SIZE REQUIREMENTS:")
    print("-" * 80)
    print(f"n × p₀ =                   {n_p0:.1f} (should be ≥ 10)")
    print(f"n × (1-p₀) =               {n_1minusp0:.1f} (should be ≥ 10)")
    
    if n_p0 >= 10 and n_1minusp0 >= 10:
        print("✓ Requirements satisfied: Normal approximation is appropriate")
    else:
        print("✗ Requirements not satisfied: Consider exact test")
    print()
    
    # Calculate test statistic
    se = np.sqrt(p_0 * (1 - p_0) / n_visitors)
    z_statistic = (p_sample - p_0) / se
    
    # Calculate p-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    
    # Critical value
    alpha = 0.05
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    print("TEST RESULTS:")
    print("-" * 80)
    print(f"Standard error:            {se:.4f}")
    print(f"Z-statistic:               {z_statistic:.4f}")
    print(f"P-value:                   {p_value:.4f}")
    print(f"Critical value (±):        ±{z_critical:.4f}")
    print()
    
    # Confidence interval
    z_ci = stats.norm.ppf(0.975)
    se_sample = np.sqrt(p_sample * (1 - p_sample) / n_visitors)
    ci_lower = p_sample - z_ci * se_sample
    ci_upper = p_sample + z_ci * se_sample
    
    print("CONFIDENCE INTERVAL:")
    print("-" * 80)
    print(f"95% CI for proportion:     [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"                           [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print()
    
    # Decision
    print("DECISION:")
    print("-" * 80)
    if p_value < alpha:
        print(f"✓ REJECT the null hypothesis (p = {p_value:.4f} < {alpha})")
        print()
        print("CONCLUSION:")
        print(f"There is sufficient evidence that our conversion rate ({p_sample*100:.2f}%)")
        print(f"is significantly different from the industry standard ({p_0*100:.2f}%).")
    else:
        print(f"✗ FAIL TO REJECT the null hypothesis (p = {p_value:.4f} ≥ {alpha})")
        print()
        print("CONCLUSION:")
        print(f"There is insufficient evidence that our conversion rate differs")
        print(f"from the industry standard.")
    
    print()
    print_separator()
    
    return {
        'n': n_visitors,
        'x': n_conversions,
        'p_sample': p_sample,
        'p_0': p_0,
        'z_statistic': z_statistic,
        'p_value': p_value,
        'ci': (ci_lower, ci_upper)
    }

def two_sample_proportion_test():
    """Example 2: Two-sample proportion test"""
    
    print("\n")
    print_separator()
    print("TWO-SAMPLE PROPORTION Z-TEST")
    print_separator()
    print()
    
    # Sample data for two groups
    # Group A (Control)
    n_a = 1200
    x_a = 108  # conversions
    p_a = x_a / n_a
    
    # Group B (Treatment)
    n_b = 1150
    x_b = 138  # conversions
    p_b = x_b / n_b
    
    print("SCENARIO:")
    print("-" * 80)
    print("We're running an A/B test comparing two landing page designs.")
    print("We want to test if Version B has a different conversion rate than Version A.")
    print()
    
    print("GROUP A (Control):")
    print(f"  Visitors:                {n_a}")
    print(f"  Conversions:             {x_a}")
    print(f"  Conversion rate:         {p_a:.4f} ({p_a*100:.2f}%)")
    print()
    
    print("GROUP B (Treatment):")
    print(f"  Visitors:                {n_b}")
    print(f"  Conversions:             {x_b}")
    print(f"  Conversion rate:         {p_b:.4f} ({p_b*100:.2f}%)")
    print()
    
    print(f"Difference (B - A):        {(p_b - p_a):.4f} ({(p_b - p_a)*100:.2f} percentage points)")
    print(f"Relative lift:             {((p_b - p_a) / p_a * 100):.2f}%")
    print()
    
    # Formulate hypotheses
    print("HYPOTHESES:")
    print("-" * 80)
    print("H₀ (Null Hypothesis):      p_A = p_B")
    print("                           (Both versions have the same conversion rate)")
    print()
    print("H₁ (Alternative Hypothesis): p_A ≠ p_B")
    print("                           (Versions have different conversion rates)")
    print()
    print("Test Type:                 Two-tailed z-test for two proportions")
    print("Significance Level (α):    0.05")
    print()
    
    # Pooled proportion
    p_pooled = (x_a + x_b) / (n_a + n_b)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    
    # Z-statistic
    z_statistic = (p_b - p_a) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    
    # Critical value
    alpha = 0.05
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    print("TEST RESULTS:")
    print("-" * 80)
    print(f"Pooled proportion:         {p_pooled:.4f}")
    print(f"Standard error:            {se:.4f}")
    print(f"Z-statistic:               {z_statistic:.4f}")
    print(f"P-value:                   {p_value:.4f}")
    print(f"Critical value (±):        ±{z_critical:.4f}")
    print()
    
    # Confidence interval for difference
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    diff = p_b - p_a
    ci_lower = diff - z_critical * se_diff
    ci_upper = diff + z_critical * se_diff
    
    print("CONFIDENCE INTERVAL FOR DIFFERENCE:")
    print("-" * 80)
    print(f"Difference (B - A):        {diff:.4f}")
    print(f"95% CI for difference:     [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"                           [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print()
    
    # Decision
    print("DECISION:")
    print("-" * 80)
    if p_value < alpha:
        print(f"✓ REJECT the null hypothesis (p = {p_value:.4f} < {alpha})")
        print()
        print("CONCLUSION:")
        print(f"There is sufficient evidence that the two versions have")
        print(f"significantly different conversion rates.")
        
        if p_b > p_a:
            print(f"Version B ({p_b*100:.2f}%) outperforms Version A ({p_a*100:.2f}%).")
            print(f"Relative improvement: {((p_b - p_a) / p_a * 100):.2f}%")
        else:
            print(f"Version A ({p_a*100:.2f}%) outperforms Version B ({p_b*100:.2f}%).")
    else:
        print(f"✗ FAIL TO REJECT the null hypothesis (p = {p_value:.4f} ≥ {alpha})")
        print()
        print("CONCLUSION:")
        print(f"There is insufficient evidence that the conversion rates differ.")
        print(f"Continue with either version or collect more data.")
    
    print()
    
    # Effect size (relative risk and odds ratio)
    relative_risk = p_b / p_a
    odds_a = p_a / (1 - p_a)
    odds_b = p_b / (1 - p_b)
    odds_ratio = odds_b / odds_a
    
    print("EFFECT SIZE MEASURES:")
    print("-" * 80)
    print(f"Relative Risk (RR):        {relative_risk:.4f}")
    print(f"  Interpretation:          B is {relative_risk:.2f}x as likely to convert as A")
    print()
    print(f"Odds Ratio (OR):           {odds_ratio:.4f}")
    print()
    print_separator()
    
    return {
        'n_a': n_a, 'x_a': x_a, 'p_a': p_a,
        'n_b': n_b, 'x_b': x_b, 'p_b': p_b,
        'z_statistic': z_statistic,
        'p_value': p_value,
        'ci': (ci_lower, ci_upper)
    }

def create_visualizations(result1, result2):
    """Create visualization plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Proportion Z-Tests', fontsize=16, fontweight='bold')
    
    # Plot 1: One-sample test - proportion comparison
    ax1 = axes[0, 0]
    categories = ['Sample', 'Industry\nStandard']
    proportions = [result1['p_sample'], result1['p_0']]
    colors = ['lightblue', 'lightcoral']
    bars = ax1.bar(categories, proportions, color=colors, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Conversion Rate', fontsize=11)
    ax1.set_title('One-Sample: Rate Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(proportions) * 1.2)
    ax1.grid(alpha=0.3, axis='y')
    
    for bar, prop in zip(bars, proportions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{prop*100:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: One-sample test - confidence interval
    ax2 = axes[0, 1]
    ax2.errorbar([1], [result1['p_sample']], 
                yerr=[[result1['p_sample'] - result1['ci'][0]], 
                      [result1['ci'][1] - result1['p_sample']]], 
                fmt='o', markersize=12, color='blue', capsize=15, capthick=3, 
                elinewidth=3, label='Sample proportion')
    
    ax2.axhline(result1['p_0'], color='red', linestyle='--', linewidth=2, 
               label=f"H₀: p = {result1['p_0']:.2f}")
    
    ax2.set_ylabel('Proportion', fontsize=11)
    ax2.set_title('95% Confidence Interval', fontsize=12, fontweight='bold')
    ax2.set_xlim(0.5, 1.5)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Our Site'])
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    # Plot 3: One-sample test - z-distribution
    ax3 = axes[0, 2]
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)
    
    ax3.plot(x, y, 'b-', linewidth=2, label='Standard normal')
    ax3.fill_between(x, 0, y, where=(x <= -abs(result1['z_statistic'])), 
                    alpha=0.3, color='red', label='Rejection region')
    ax3.fill_between(x, 0, y, where=(x >= abs(result1['z_statistic'])), 
                    alpha=0.3, color='red')
    
    ax3.axvline(result1['z_statistic'], color='red', linestyle='-', linewidth=2,
               label=f"z = {result1['z_statistic']:.2f}")
    
    z_crit = stats.norm.ppf(0.975)
    ax3.axvline(-z_crit, color='orange', linestyle='--', linewidth=1.5,
               label=f'Critical values = ±{z_crit:.2f}')
    ax3.axvline(z_crit, color='orange', linestyle='--', linewidth=1.5)
    
    ax3.set_xlabel('z-value', fontsize=11)
    ax3.set_ylabel('Probability Density', fontsize=11)
    ax3.set_title(f'Z-Distribution (p={result1["p_value"]:.4f})', 
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # Plot 4: Two-sample test - conversion rates
    ax4 = axes[1, 0]
    groups = ['Version A\n(Control)', 'Version B\n(Treatment)']
    rates = [result2['p_a'], result2['p_b']]
    colors = ['lightblue', 'lightgreen']
    bars = ax4.bar(groups, rates, color=colors, edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('Conversion Rate', fontsize=11)
    ax4.set_title('Two-Sample: A/B Test Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(rates) * 1.2)
    ax4.grid(alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate*100:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Two-sample test - funnel visualization
    ax5 = axes[1, 1]
    
    # Create funnel data
    visitors_a, conv_a = result2['n_a'], result2['x_a']
    visitors_b, conv_b = result2['n_b'], result2['x_b']
    
    x = np.array([0, 1])
    y_a_visitors = np.array([visitors_a, visitors_a])
    y_a_conv = np.array([conv_a, conv_a])
    y_b_visitors = np.array([visitors_b, visitors_b])
    y_b_conv = np.array([conv_b, conv_b])
    
    width = 0.35
    ax5.bar(x - width/2, [visitors_a, visitors_b], width, label='Visitors', 
           color='lightgray', edgecolor='black')
    ax5.bar(x + width/2, [conv_a, conv_b], width, label='Conversions', 
           color=['lightblue', 'lightgreen'], edgecolor='black')
    
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Conversion Funnel', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Version A', 'Version B'])
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')
    
    # Plot 6: Two-sample test - difference CI
    ax6 = axes[1, 2]
    diff = result2['p_b'] - result2['p_a']
    ci_lower, ci_upper = result2['ci']
    
    ax6.errorbar([1], [diff], 
                yerr=[[diff - ci_lower], [ci_upper - diff]], 
                fmt='o', markersize=12, color='purple', capsize=15, capthick=3, 
                elinewidth=3, label='Difference (B - A)')
    
    ax6.axhline(0, color='red', linestyle='--', linewidth=2, 
               label='No difference (H₀)')
    
    # Shade the area
    if diff > 0:
        ax6.axhspan(0, ax6.get_ylim()[1], alpha=0.2, color='green', 
                   label='B better than A')
    
    ax6.set_ylabel('Difference in Conversion Rate', fontsize=11)
    ax6.set_title('Difference with 95% CI', fontsize=12, fontweight='bold')
    ax6.set_xlim(0.5, 1.5)
    ax6.set_xticks([1])
    ax6.set_xticklabels(['B - A'])
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3, axis='y')
    
    # Add value annotation
    ax6.text(1.05, diff, f'{diff*100:.2f}%', ha='left', va='center', 
            fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    filename = 'proportion_z_tests.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{filename}'")
    
    plt.show()

if __name__ == "__main__":
    result1 = one_sample_proportion_test()
    result2 = two_sample_proportion_test()
    
    create_visualizations(result1, result2)
