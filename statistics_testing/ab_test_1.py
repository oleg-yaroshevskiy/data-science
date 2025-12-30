"""
A/B Test Example 1: Email Campaign Click-Through Rate

Use Case: Testing whether a new email subject line improves click-through rates

This script demonstrates a complete A/B test workflow including:
- Sample size calculation
- Statistical testing
- Power analysis
- Practical significance assessment
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

def calculate_sample_size(p1, p2, alpha=0.05, power=0.8):
    """
    Calculate required sample size for each group
    
    Parameters:
    -----------
    p1 : float
        Baseline conversion rate
    p2 : float
        Expected conversion rate for treatment
    alpha : float
        Significance level
    power : float
        Desired statistical power (1 - β)
    """
    # Effect size
    effect_size = abs(p2 - p1)
    
    # Pooled proportion
    p_pooled = (p1 + p2) / 2
    
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Sample size per group
    n = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
         z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2 / effect_size**2
    
    return int(np.ceil(n))

def perform_ab_test():
    """Perform A/B test on email campaign"""
    
    print_separator()
    print("A/B TEST: EMAIL CAMPAIGN CLICK-THROUGH RATE")
    print_separator()
    print()
    
    print("BUSINESS CONTEXT:")
    print("-" * 80)
    print("We want to test if a new email subject line increases click-through rates.")
    print()
    print("Version A (Control):       'Special Offer Inside'")
    print("Version B (Treatment):     'Your Exclusive 20% Discount Awaits'")
    print()
    
    # Historical baseline
    baseline_ctr = 0.15
    
    # Minimum detectable effect (MDE)
    mde = 0.03  # 3 percentage points increase
    expected_ctr = baseline_ctr + mde
    
    print("EXPERIMENTAL DESIGN:")
    print("-" * 80)
    print(f"Baseline CTR (historical): {baseline_ctr:.2%}")
    print(f"Minimum Detectable Effect: {mde:.2%} ({mde/baseline_ctr:.1%} relative increase)")
    print(f"Expected Treatment CTR:    {expected_ctr:.2%}")
    print(f"Significance Level (α):    0.05")
    print(f"Desired Power (1-β):       0.80")
    print()
    
    # Calculate required sample size
    required_n = calculate_sample_size(baseline_ctr, expected_ctr, alpha=0.05, power=0.8)
    
    print("SAMPLE SIZE CALCULATION:")
    print("-" * 80)
    print(f"Required sample per group: {required_n}")
    print(f"Total required:            {required_n * 2}")
    print()
    
    # Generate experimental data
    # We'll simulate actual experiment with slightly better results
    n_control = 2000
    n_treatment = 2000
    
    # Simulate clicks (treatment actually has 17% CTR)
    clicks_control = np.random.binomial(n_control, 0.15)
    clicks_treatment = np.random.binomial(n_treatment, 0.17)
    
    ctr_control = clicks_control / n_control
    ctr_treatment = clicks_treatment / n_treatment
    
    print("EXPERIMENTAL RESULTS:")
    print("-" * 80)
    print(f"Group A (Control):")
    print(f"  Emails sent:             {n_control}")
    print(f"  Clicks:                  {clicks_control}")
    print(f"  CTR:                     {ctr_control:.2%}")
    print()
    print(f"Group B (Treatment):")
    print(f"  Emails sent:             {n_treatment}")
    print(f"  Clicks:                  {clicks_treatment}")
    print(f"  CTR:                     {ctr_treatment:.2%}")
    print()
    print(f"Absolute difference:       {(ctr_treatment - ctr_control):.2%}")
    print(f"Relative lift:             {((ctr_treatment - ctr_control) / ctr_control):.2%}")
    print()
    
    # Statistical test
    print("STATISTICAL TEST:")
    print("-" * 80)
    
    # Pooled proportion
    p_pooled = (clicks_control + clicks_treatment) / (n_control + n_treatment)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
    
    # Z-statistic
    z_stat = (ctr_treatment - ctr_control) / se
    
    # P-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    print(f"Z-statistic:               {z_stat:.4f}")
    print(f"P-value:                   {p_value:.4f}")
    print()
    
    # Confidence interval
    se_diff = np.sqrt(ctr_control * (1 - ctr_control) / n_control + 
                     ctr_treatment * (1 - ctr_treatment) / n_treatment)
    z_critical = stats.norm.ppf(0.975)
    diff = ctr_treatment - ctr_control
    ci_lower = diff - z_critical * se_diff
    ci_upper = diff + z_critical * se_diff
    
    print("CONFIDENCE INTERVAL:")
    print("-" * 80)
    print(f"95% CI for difference:     [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"                           [{ci_lower:.2%}, {ci_upper:.2%}]")
    print()
    
    # Decision
    alpha = 0.05
    print("STATISTICAL DECISION:")
    print("-" * 80)
    if p_value < alpha:
        print(f"✓ STATISTICALLY SIGNIFICANT (p = {p_value:.4f} < {alpha})")
        print()
        print("The treatment group has a significantly different CTR.")
    else:
        print(f"✗ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.4f} ≥ {alpha})")
        print()
        print("No significant difference detected.")
    print()
    
    # Business significance
    print("BUSINESS IMPACT ANALYSIS:")
    print("-" * 80)
    
    # Assume 100,000 emails per month
    monthly_emails = 100000
    avg_order_value = 50  # dollars
    conversion_rate = 0.10  # 10% of clicks convert to sales
    
    current_revenue = monthly_emails * ctr_control * conversion_rate * avg_order_value
    new_revenue = monthly_emails * ctr_treatment * conversion_rate * avg_order_value
    revenue_lift = new_revenue - current_revenue
    
    print(f"Monthly emails:            {monthly_emails:,}")
    print(f"Average order value:       ${avg_order_value:.2f}")
    print(f"Click-to-purchase rate:    {conversion_rate:.1%}")
    print()
    print(f"Current monthly revenue:   ${current_revenue:,.2f}")
    print(f"Projected new revenue:     ${new_revenue:,.2f}")
    print(f"Monthly revenue lift:      ${revenue_lift:,.2f}")
    print(f"Annual revenue lift:       ${revenue_lift * 12:,.2f}")
    print()
    
    # Recommendation
    print("RECOMMENDATION:")
    print("-" * 80)
    if p_value < alpha and revenue_lift > 0:
        print("✓ IMPLEMENT VERSION B")
        print()
        print("Rationale:")
        print("1. Statistically significant improvement detected")
        print(f"2. Estimated monthly revenue increase: ${revenue_lift:,.2f}")
        print(f"3. Relative CTR improvement: {((ctr_treatment - ctr_control) / ctr_control):.1%}")
        print("4. Low risk: Easy to implement and reverse if needed")
    else:
        print("✗ KEEP VERSION A OR CONTINUE TESTING")
        print()
        print("Rationale:")
        print("Results do not show sufficient evidence for change.")
    
    print()
    print_separator()
    
    # Create visualizations
    create_visualizations(n_control, clicks_control, ctr_control,
                         n_treatment, clicks_treatment, ctr_treatment,
                         ci_lower, ci_upper, p_value,
                         current_revenue, new_revenue)
    
    return {
        'n_control': n_control,
        'clicks_control': clicks_control,
        'ctr_control': ctr_control,
        'n_treatment': n_treatment,
        'clicks_treatment': clicks_treatment,
        'ctr_treatment': ctr_treatment,
        'p_value': p_value,
        'revenue_lift': revenue_lift
    }

def create_visualizations(n_control, clicks_control, ctr_control,
                         n_treatment, clicks_treatment, ctr_treatment,
                         ci_lower, ci_upper, p_value,
                         current_revenue, new_revenue):
    """Create comprehensive A/B test visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('A/B Test: Email Campaign Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: CTR comparison
    ax1 = axes[0, 0]
    versions = ['Version A\n(Control)', 'Version B\n(Treatment)']
    ctrs = [ctr_control, ctr_treatment]
    colors = ['lightblue', 'lightgreen']
    bars = ax1.bar(versions, ctrs, color=colors, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Click-Through Rate', fontsize=11)
    ax1.set_title('CTR Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(ctrs) * 1.2)
    ax1.grid(alpha=0.3, axis='y')
    
    for bar, ctr in zip(bars, ctrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{ctr:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 2: Conversion funnel
    ax2 = axes[0, 1]
    
    x = np.array([0, 1])
    width = 0.35
    
    sent = [n_control, n_treatment]
    clicked = [clicks_control, clicks_treatment]
    
    ax2.bar(x - width/2, sent, width, label='Emails Sent', 
           color='lightgray', edgecolor='black')
    ax2.bar(x + width/2, clicked, width, label='Clicks', 
           color=colors, edgecolor='black')
    
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Conversion Funnel', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Version A', 'Version B'])
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    # Plot 3: Confidence interval
    ax3 = axes[0, 2]
    diff = ctr_treatment - ctr_control
    
    ax3.errorbar([1], [diff], 
                yerr=[[diff - ci_lower], [ci_upper - diff]], 
                fmt='o', markersize=12, color='purple', capsize=15, capthick=3, 
                elinewidth=3)
    
    ax3.axhline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    
    if diff > 0:
        ax3.axhspan(0, ax3.get_ylim()[1], alpha=0.2, color='green', 
                   label='B better')
    
    ax3.set_ylabel('Difference in CTR', fontsize=11)
    ax3.set_title(f'Difference with 95% CI\n(p={p_value:.4f})', 
                 fontsize=12, fontweight='bold')
    ax3.set_xlim(0.5, 1.5)
    ax3.set_xticks([1])
    ax3.set_xticklabels(['B - A'])
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    
    ax3.text(1.05, diff, f'{diff:.2%}', ha='left', va='center', 
            fontweight='bold', fontsize=10)
    
    # Plot 4: Revenue projection
    ax4 = axes[1, 0]
    revenues = [current_revenue, new_revenue]
    revenue_labels = ['Current\n(Version A)', 'Projected\n(Version B)']
    bars = ax4.bar(revenue_labels, revenues, color=colors, edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('Monthly Revenue ($)', fontsize=11)
    ax4.set_title('Revenue Projection', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(revenues) * 1.2)
    ax4.grid(alpha=0.3, axis='y')
    
    for bar, rev in zip(bars, revenues):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${rev:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 5: Relative lift
    ax5 = axes[1, 1]
    relative_lift = (ctr_treatment - ctr_control) / ctr_control * 100
    
    ax5.barh(['Relative Lift'], [relative_lift], color='lightgreen', 
            edgecolor='black', linewidth=1.5, height=0.3)
    ax5.axvline(0, color='black', linewidth=1)
    
    ax5.set_xlabel('Percentage Increase (%)', fontsize=11)
    ax5.set_title('Relative CTR Improvement', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3, axis='x')
    
    ax5.text(relative_lift + 0.5, 0, f'{relative_lift:.1f}%', 
            va='center', fontweight='bold', fontsize=12)
    
    # Plot 6: Sample size adequacy
    ax6 = axes[1, 2]
    
    # Calculate achieved power
    effect_size = ctr_treatment - ctr_control
    pooled_p = (clicks_control + clicks_treatment) / (n_control + n_treatment)
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_control + 1/n_treatment))
    z_stat = effect_size / se
    achieved_power = 1 - stats.norm.cdf(1.96 - abs(z_stat))
    
    categories = ['Achieved\nPower', 'Target\nPower']
    powers = [achieved_power, 0.80]
    colors_power = ['lightcoral' if achieved_power < 0.80 else 'lightgreen', 'lightgray']
    bars = ax6.bar(categories, powers, color=colors_power, edgecolor='black', linewidth=1.5)
    
    ax6.set_ylabel('Statistical Power', fontsize=11)
    ax6.set_title('Power Analysis', fontsize=12, fontweight='bold')
    ax6.set_ylim(0, 1)
    ax6.axhline(0.80, color='red', linestyle='--', linewidth=2, label='Target: 80%')
    ax6.grid(alpha=0.3, axis='y')
    ax6.legend()
    
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{power:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    filename = 'ab_test_email_campaign.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{filename}'")
    
    plt.show()

if __name__ == "__main__":
    results = perform_ab_test()
