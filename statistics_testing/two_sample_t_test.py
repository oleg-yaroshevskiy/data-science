"""
Two-Sample T-Test Example: Treatment Effectiveness

Use Case: Does a new teaching method improve test scores compared to traditional method?

This script demonstrates an independent two-sample t-test to compare means between
two independent groups.
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

def perform_two_sample_t_test():
    """Perform independent two-sample t-test"""
    
    print_separator()
    print("TWO-SAMPLE T-TEST: TEACHING METHOD COMPARISON")
    print_separator()
    print()
    
    # Generate sample data
    # Control group: traditional teaching method (mean score: 75)
    control_group = np.random.normal(loc=75, scale=10, size=40)
    
    # Treatment group: new teaching method (mean score: 80)
    treatment_group = np.random.normal(loc=80, scale=9, size=45)
    
    print("SCENARIO:")
    print("-" * 80)
    print("We want to test whether a new teaching method improves test scores")
    print("compared to the traditional method.")
    print()
    print(f"Control Group (Traditional):  n = {len(control_group)} students")
    print(f"Treatment Group (New Method): n = {len(treatment_group)} students")
    print()
    
    # Calculate sample statistics
    mean_control = np.mean(control_group)
    mean_treatment = np.mean(treatment_group)
    std_control = np.std(control_group, ddof=1)
    std_treatment = np.std(treatment_group, ddof=1)
    
    print("SAMPLE STATISTICS:")
    print("-" * 80)
    print(f"Control Group:")
    print(f"  Sample size (n₁):         {len(control_group)}")
    print(f"  Sample mean (x̄₁):         {mean_control:.2f}")
    print(f"  Sample std dev (s₁):      {std_control:.2f}")
    print()
    print(f"Treatment Group:")
    print(f"  Sample size (n₂):         {len(treatment_group)}")
    print(f"  Sample mean (x̄₂):         {mean_treatment:.2f}")
    print(f"  Sample std dev (s₂):      {std_treatment:.2f}")
    print()
    print(f"Difference in means (x̄₂ - x̄₁): {mean_treatment - mean_control:.2f}")
    print()
    
    # Formulate hypotheses
    print("HYPOTHESES:")
    print("-" * 80)
    print("H₀ (Null Hypothesis):      μ₁ = μ₂")
    print("                           (Both methods have the same mean score)")
    print()
    print("H₁ (Alternative Hypothesis): μ₁ ≠ μ₂")
    print("                           (The methods have different mean scores)")
    print()
    print("Test Type:                 Two-tailed independent t-test")
    print("Significance Level (α):    0.05")
    print()
    
    # Check for equal variances using Levene's test
    levene_stat, levene_p = stats.levene(control_group, treatment_group)
    equal_var = levene_p > 0.05
    
    print("VARIANCE EQUALITY TEST (Levene's Test):")
    print("-" * 80)
    print(f"Levene statistic:          {levene_stat:.4f}")
    print(f"P-value:                   {levene_p:.4f}")
    
    if equal_var:
        print("Result:                    Equal variances assumed (p > 0.05)")
        print("                           Using standard two-sample t-test")
    else:
        print("Result:                    Unequal variances (p ≤ 0.05)")
        print("                           Using Welch's t-test")
    print()
    
    # Perform two-sample t-test
    t_statistic, p_value = stats.ttest_ind(control_group, treatment_group, equal_var=equal_var)
    
    # Calculate degrees of freedom
    if equal_var:
        df = len(control_group) + len(treatment_group) - 2
    else:
        # Welch-Satterthwaite equation
        s1_sq = std_control**2
        s2_sq = std_treatment**2
        n1 = len(control_group)
        n2 = len(treatment_group)
        df = ((s1_sq/n1 + s2_sq/n2)**2) / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
    
    # Critical value
    alpha = 0.05
    critical_value = stats.t.ppf(1 - alpha/2, df)
    
    print("TEST RESULTS:")
    print("-" * 80)
    print(f"T-statistic:               {t_statistic:.4f}")
    print(f"Degrees of freedom:        {df:.2f}")
    print(f"P-value:                   {p_value:.4f}")
    print(f"Critical value (±):        ±{critical_value:.4f}")
    print()
    
    # Calculate confidence interval for the difference in means
    se_diff = np.sqrt(std_control**2/len(control_group) + std_treatment**2/len(treatment_group))
    diff_means = mean_treatment - mean_control
    ci_margin = critical_value * se_diff
    ci_lower = diff_means - ci_margin
    ci_upper = diff_means + ci_margin
    
    print("CONFIDENCE INTERVAL FOR DIFFERENCE:")
    print("-" * 80)
    print(f"Difference (Treatment - Control): {diff_means:.2f}")
    print(f"Standard Error:            {se_diff:.2f}")
    print(f"95% CI for difference:     [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"Interpretation:            We are 95% confident that the true difference")
    print(f"                           in means is between {ci_lower:.2f} and {ci_upper:.2f}")
    print()
    
    # Decision
    print("DECISION:")
    print("-" * 80)
    if p_value < alpha:
        print(f"✓ REJECT the null hypothesis (p = {p_value:.4f} < {alpha})")
        print()
        print("CONCLUSION:")
        print(f"There is sufficient evidence to conclude that the two teaching methods")
        print(f"produce significantly different mean scores.")
        print(f"Treatment mean: {mean_treatment:.2f}, Control mean: {mean_control:.2f}")
        
        if mean_treatment > mean_control:
            print(f"The new teaching method shows improvement over the traditional method.")
        else:
            print(f"The traditional method shows better results than the new method.")
    else:
        print(f"✗ FAIL TO REJECT the null hypothesis (p = {p_value:.4f} ≥ {alpha})")
        print()
        print("CONCLUSION:")
        print(f"There is insufficient evidence to conclude that the teaching methods")
        print(f"produce different mean scores.")
    
    print()
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(control_group)-1)*std_control**2 + 
                          (len(treatment_group)-1)*std_treatment**2) / 
                         (len(control_group) + len(treatment_group) - 2))
    cohens_d = (mean_treatment - mean_control) / pooled_std
    
    print("EFFECT SIZE:")
    print("-" * 80)
    print(f"Pooled standard deviation: {pooled_std:.2f}")
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
    create_visualizations(control_group, treatment_group, mean_control, mean_treatment,
                         t_statistic, p_value, df, ci_lower, ci_upper)
    
    return {
        'control': control_group,
        'treatment': treatment_group,
        't_statistic': t_statistic,
        'p_value': p_value
    }

def create_visualizations(control, treatment, mean_control, mean_treatment,
                         t_statistic, p_value, df, ci_lower, ci_upper):
    """Create and save visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Two-Sample T-Test: Teaching Method Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Histograms
    ax1 = axes[0, 0]
    ax1.hist(control, bins=15, alpha=0.6, color='blue', label='Control (Traditional)', 
             edgecolor='black', density=True)
    ax1.hist(treatment, bins=15, alpha=0.6, color='red', label='Treatment (New Method)', 
             edgecolor='black', density=True)
    
    ax1.axvline(mean_control, color='blue', linestyle='--', linewidth=2, 
                label=f'Control mean = {mean_control:.2f}')
    ax1.axvline(mean_treatment, color='red', linestyle='--', linewidth=2, 
                label=f'Treatment mean = {mean_treatment:.2f}')
    
    ax1.set_xlabel('Test Score', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Distribution of Test Scores', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Box plots
    ax2 = axes[0, 1]
    bp = ax2.boxplot([control, treatment], labels=['Control', 'Treatment'],
                     patch_artist=True, widths=0.5)
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    ax2.set_ylabel('Test Score', fontsize=11)
    ax2.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Plot 3: Means with confidence intervals
    ax3 = axes[1, 0]
    groups = ['Control', 'Treatment']
    means = [mean_control, mean_treatment]
    
    # Calculate standard errors for each group
    se_control = np.std(control, ddof=1) / np.sqrt(len(control))
    se_treatment = np.std(treatment, ddof=1) / np.sqrt(len(treatment))
    
    ci_control = stats.t.ppf(0.975, len(control)-1) * se_control
    ci_treatment = stats.t.ppf(0.975, len(treatment)-1) * se_treatment
    
    errors = [ci_control, ci_treatment]
    
    bars = ax3.bar(groups, means, yerr=errors, capsize=10, alpha=0.7,
                   color=['blue', 'red'], edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Mean Test Score', fontsize=11)
    ax3.set_title('Mean Scores with 95% Confidence Intervals', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: T-distribution
    ax4 = axes[1, 1]
    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df)
    
    ax4.plot(x, y, 'b-', linewidth=2, label='t-distribution')
    ax4.fill_between(x, 0, y, where=(x <= -abs(t_statistic)), alpha=0.3, 
                     color='red', label='Rejection region')
    ax4.fill_between(x, 0, y, where=(x >= abs(t_statistic)), alpha=0.3, color='red')
    
    ax4.axvline(t_statistic, color='red', linestyle='-', linewidth=2, 
                label=f't = {t_statistic:.2f}')
    
    critical_value = stats.t.ppf(0.975, df)
    ax4.axvline(-critical_value, color='orange', linestyle='--', linewidth=1.5,
                label=f'Critical values = ±{critical_value:.2f}')
    ax4.axvline(critical_value, color='orange', linestyle='--', linewidth=1.5)
    
    ax4.set_xlabel('t-value', fontsize=11)
    ax4.set_ylabel('Probability Density', fontsize=11)
    ax4.set_title(f'T-Distribution (df={df:.1f}, p={p_value:.4f})', 
                  fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = 'two_sample_t_test.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{filename}'")
    
    plt.show()

if __name__ == "__main__":
    results = perform_two_sample_t_test()
