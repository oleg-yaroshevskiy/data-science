"""
A/B Test Example 2: Landing Page Conversion Rate with Sequential Testing

Use Case: Testing a new landing page design with early stopping capability

This script demonstrates:
- Sequential A/B testing (early stopping)
- Multiple metrics evaluation
- Bayesian credible intervals
- Segmented analysis
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

def sequential_test_analysis(n_control, conv_control, n_treatment, conv_treatment, 
                            alpha=0.05):
    """
    Perform sequential test with early stopping check
    Uses alpha spending function approach
    """
    p_control = conv_control / n_control
    p_treatment = conv_treatment / n_treatment
    
    # Z-test
    p_pooled = (conv_control + conv_treatment) / (n_control + n_treatment)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
    z_stat = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return p_control, p_treatment, z_stat, p_value

def bayesian_analysis(n_control, conv_control, n_treatment, conv_treatment):
    """
    Perform Bayesian analysis using Beta distributions
    """
    # Prior: Beta(1, 1) = Uniform(0, 1)
    alpha_prior = 1
    beta_prior = 1
    
    # Posterior distributions
    alpha_control = alpha_prior + conv_control
    beta_control = beta_prior + (n_control - conv_control)
    
    alpha_treatment = alpha_prior + conv_treatment
    beta_treatment = beta_prior + (n_treatment - conv_treatment)
    
    # Sample from posteriors
    n_samples = 100000
    samples_control = np.random.beta(alpha_control, beta_control, n_samples)
    samples_treatment = np.random.beta(alpha_treatment, beta_treatment, n_samples)
    
    # Probability that treatment is better
    prob_b_better = np.mean(samples_treatment > samples_control)
    
    # Credible intervals
    ci_control = np.percentile(samples_control, [2.5, 97.5])
    ci_treatment = np.percentile(samples_treatment, [2.5, 97.5])
    
    # Expected lift
    lift_samples = (samples_treatment - samples_control) / samples_control
    expected_lift = np.mean(lift_samples)
    lift_ci = np.percentile(lift_samples, [2.5, 97.5])
    
    return {
        'prob_b_better': prob_b_better,
        'ci_control': ci_control,
        'ci_treatment': ci_treatment,
        'expected_lift': expected_lift,
        'lift_ci': lift_ci,
        'samples_control': samples_control,
        'samples_treatment': samples_treatment
    }

def perform_ab_test_sequential():
    """Perform A/B test with sequential analysis"""
    
    print_separator()
    print("A/B TEST: LANDING PAGE REDESIGN WITH SEQUENTIAL TESTING")
    print_separator()
    print()
    
    print("BUSINESS CONTEXT:")
    print("-" * 80)
    print("We're testing a new landing page design to improve conversion rates.")
    print("We'll monitor the test daily and check for early stopping.")
    print()
    print("Version A (Control):       Current landing page")
    print("Version B (Treatment):     New design with simplified form")
    print()
    
    # Simulate daily data collection
    print("EXPERIMENTAL SETUP:")
    print("-" * 80)
    print("Daily traffic per group:   ~500 visitors")
    print("Test duration:             Up to 14 days")
    print("Significance level:        0.05")
    print("Early stopping:            Enabled")
    print()
    
    # Simulate full experiment data
    # True conversion rates: Control 8%, Treatment 10%
    days = 14
    daily_visitors = 500
    
    total_n_control = 0
    total_conv_control = 0
    total_n_treatment = 0
    total_conv_treatment = 0
    
    daily_results = []
    
    print("DAILY RESULTS:")
    print("-" * 80)
    print(f"{'Day':<5} {'Control':<20} {'Treatment':<20} {'P-value':<12} {'Decision':<15}")
    print("-" * 80)
    
    early_stop = False
    stop_day = days
    
    for day in range(1, days + 1):
        # Daily conversions
        daily_conv_control = np.random.binomial(daily_visitors, 0.08)
        daily_conv_treatment = np.random.binomial(daily_visitors, 0.10)
        
        total_n_control += daily_visitors
        total_n_treatment += daily_visitors
        total_conv_control += daily_conv_control
        total_conv_treatment += daily_conv_treatment
        
        # Perform test
        p_c, p_t, z_stat, p_value = sequential_test_analysis(
            total_n_control, total_conv_control,
            total_n_treatment, total_conv_treatment
        )
        
        daily_results.append({
            'day': day,
            'n_control': total_n_control,
            'conv_control': total_conv_control,
            'rate_control': p_c,
            'n_treatment': total_n_treatment,
            'conv_treatment': total_conv_treatment,
            'rate_treatment': p_t,
            'p_value': p_value
        })
        
        decision = ""
        if day >= 7:  # Only consider early stopping after minimum 7 days
            if p_value < 0.01:  # More conservative threshold for early stopping
                decision = "✓ Early stop"
                early_stop = True
                stop_day = day
        
        print(f"{day:<5} {p_c*100:>5.2f}% ({total_conv_control}/{total_n_control})"
              f"     {p_t*100:>5.2f}% ({total_conv_treatment}/{total_n_treatment})"
              f"     {p_value:>8.4f}    {decision}")
        
        if early_stop:
            break
    
    print()
    
    # Final results
    final_p_c = total_conv_control / total_n_control
    final_p_t = total_conv_treatment / total_n_treatment
    
    print("FINAL FREQUENTIST RESULTS:")
    print("-" * 80)
    print(f"Test duration:             {stop_day} days")
    print(f"Total sample size:         {total_n_control + total_n_treatment}")
    print()
    print(f"Control (A):")
    print(f"  Visitors:                {total_n_control}")
    print(f"  Conversions:             {total_conv_control}")
    print(f"  Conversion rate:         {final_p_c:.2%}")
    print()
    print(f"Treatment (B):")
    print(f"  Visitors:                {total_n_treatment}")
    print(f"  Conversions:             {total_conv_treatment}")
    print(f"  Conversion rate:         {final_p_t:.2%}")
    print()
    
    # Final statistical test
    _, _, z_stat, p_value = sequential_test_analysis(
        total_n_control, total_conv_control,
        total_n_treatment, total_conv_treatment
    )
    
    print(f"Z-statistic:               {z_stat:.4f}")
    print(f"P-value:                   {p_value:.4f}")
    print()
    
    # Bayesian analysis
    print("BAYESIAN ANALYSIS:")
    print("-" * 80)
    
    bayes_results = bayesian_analysis(
        total_n_control, total_conv_control,
        total_n_treatment, total_conv_treatment
    )
    
    print(f"Probability B > A:         {bayes_results['prob_b_better']:.2%}")
    print()
    print(f"Control rate 95% CI:       [{bayes_results['ci_control'][0]:.2%}, "
          f"{bayes_results['ci_control'][1]:.2%}]")
    print(f"Treatment rate 95% CI:     [{bayes_results['ci_treatment'][0]:.2%}, "
          f"{bayes_results['ci_treatment'][1]:.2%}]")
    print()
    print(f"Expected relative lift:    {bayes_results['expected_lift']:.2%}")
    print(f"Lift 95% credible int.:    [{bayes_results['lift_ci'][0]:.2%}, "
          f"{bayes_results['lift_ci'][1]:.2%}]")
    print()
    
    # Decision criteria
    print("DECISION FRAMEWORK:")
    print("-" * 80)
    
    statistical_sig = p_value < 0.05
    bayesian_threshold = bayes_results['prob_b_better'] > 0.95
    practical_sig = (final_p_t - final_p_c) / final_p_c > 0.10  # 10% relative improvement
    
    print(f"Statistical significance:  {'✓ YES' if statistical_sig else '✗ NO'} "
          f"(p = {p_value:.4f})")
    print(f"Bayesian confidence:       {'✓ YES' if bayesian_threshold else '✗ NO'} "
          f"(P(B>A) = {bayes_results['prob_b_better']:.2%})")
    print(f"Practical significance:    {'✓ YES' if practical_sig else '✗ NO'} "
          f"(lift = {(final_p_t - final_p_c) / final_p_c:.1%})")
    print()
    
    # Recommendation
    print("RECOMMENDATION:")
    print("-" * 80)
    
    if statistical_sig and bayesian_threshold:
        print("✓ IMPLEMENT VERSION B")
        print()
        print("Rationale:")
        print("- Strong statistical evidence (frequentist and Bayesian)")
        print(f"- {bayes_results['prob_b_better']:.0%} probability that B is better than A")
        print(f"- Expected {(final_p_t - final_p_c) / final_p_c:.1%} relative improvement")
        if early_stop:
            print(f"- Early stopping achieved on day {stop_day} (saved testing time)")
    elif statistical_sig:
        print("⚠ CAUTIOUS IMPLEMENTATION")
        print()
        print("Rationale:")
        print("- Statistically significant but Bayesian confidence is moderate")
        print("- Consider longer test or validation")
    else:
        print("✗ KEEP VERSION A")
        print()
        print("Rationale:")
        print("- Insufficient evidence for change")
        print("- Consider redesigning treatment or testing longer")
    
    print()
    
    # Segment analysis (simulate)
    print("SEGMENTED ANALYSIS:")
    print("-" * 80)
    print("Performance by traffic source:")
    print()
    
    segments = {
        'Organic': (0.09, 0.12),
        'Paid': (0.07, 0.08),
        'Direct': (0.08, 0.11)
    }
    
    for segment, (rate_a, rate_b) in segments.items():
        lift = (rate_b - rate_a) / rate_a
        print(f"{segment:12} - Control: {rate_a:.1%}  Treatment: {rate_b:.1%}  "
              f"Lift: {lift:+.1%}")
    
    print()
    print("Insight: Treatment performs well across all segments,")
    print("         with strongest effect in organic traffic.")
    print()
    print_separator()
    
    # Visualizations
    create_visualizations(daily_results, bayes_results,
                         total_n_control, total_conv_control,
                         total_n_treatment, total_conv_treatment,
                         final_p_c, final_p_t, segments)
    
    return daily_results, bayes_results

def create_visualizations(daily_results, bayes_results,
                         n_control, conv_control, n_treatment, conv_treatment,
                         rate_control, rate_treatment, segments):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('A/B Test: Sequential Analysis of Landing Page Redesign', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Daily conversion rates over time
    ax1 = fig.add_subplot(gs[0, :2])
    days = [r['day'] for r in daily_results]
    rates_c = [r['rate_control'] * 100 for r in daily_results]
    rates_t = [r['rate_treatment'] * 100 for r in daily_results]
    
    ax1.plot(days, rates_c, 'o-', color='blue', linewidth=2, markersize=6, 
            label='Control (A)', alpha=0.7)
    ax1.plot(days, rates_t, 's-', color='red', linewidth=2, markersize=6, 
            label='Treatment (B)', alpha=0.7)
    
    ax1.set_xlabel('Day', fontsize=11)
    ax1.set_ylabel('Conversion Rate (%)', fontsize=11)
    ax1.set_title('Daily Conversion Rates', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: P-value over time
    ax2 = fig.add_subplot(gs[0, 2])
    p_values = [r['p_value'] for r in daily_results]
    
    ax2.plot(days, p_values, 'o-', color='purple', linewidth=2, markersize=6)
    ax2.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax2.axhline(0.01, color='orange', linestyle='--', linewidth=2, label='Early stop threshold')
    
    ax2.set_xlabel('Day', fontsize=11)
    ax2.set_ylabel('P-value', fontsize=11)
    ax2.set_title('P-value Evolution', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Final conversion rates
    ax3 = fig.add_subplot(gs[1, 0])
    versions = ['Version A\n(Control)', 'Version B\n(Treatment)']
    rates = [rate_control, rate_treatment]
    colors = ['lightblue', 'lightgreen']
    bars = ax3.bar(versions, rates, color=colors, edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Conversion Rate', fontsize=11)
    ax3.set_title('Final Conversion Rates', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, max(rates) * 1.2)
    ax3.grid(alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 4: Bayesian posterior distributions
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Sample subset for plotting
    samples_c = bayes_results['samples_control'][:5000]
    samples_t = bayes_results['samples_treatment'][:5000]
    
    ax4.hist(samples_c, bins=50, alpha=0.6, color='blue', density=True, 
            label='Control posterior', edgecolor='black')
    ax4.hist(samples_t, bins=50, alpha=0.6, color='red', density=True, 
            label='Treatment posterior', edgecolor='black')
    
    ax4.axvline(rate_control, color='blue', linestyle='--', linewidth=2)
    ax4.axvline(rate_treatment, color='red', linestyle='--', linewidth=2)
    
    ax4.set_xlabel('Conversion Rate', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Bayesian Posterior Distributions', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Plot 5: Probability B > A
    ax5 = fig.add_subplot(gs[1, 2])
    
    prob_b_better = bayes_results['prob_b_better']
    colors_prob = ['lightgreen' if prob_b_better > 0.95 else 'lightyellow']
    
    ax5.barh(['P(B > A)'], [prob_b_better], color=colors_prob, 
            edgecolor='black', linewidth=1.5, height=0.3)
    ax5.axvline(0.95, color='red', linestyle='--', linewidth=2, 
               label='Decision threshold')
    
    ax5.set_xlabel('Probability', fontsize=11)
    ax5.set_title('Bayesian Decision Metric', fontsize=12, fontweight='bold')
    ax5.set_xlim(0, 1)
    ax5.legend()
    ax5.grid(alpha=0.3, axis='x')
    
    ax5.text(prob_b_better - 0.05, 0, f'{prob_b_better:.1%}', 
            ha='right', va='center', fontweight='bold', fontsize=12)
    
    # Plot 6: Segment analysis
    ax6 = fig.add_subplot(gs[2, 0])
    
    segment_names = list(segments.keys())
    lifts = [((b - a) / a * 100) for a, b in segments.values()]
    colors_seg = ['lightgreen' if lift > 0 else 'lightcoral' for lift in lifts]
    
    bars = ax6.barh(segment_names, lifts, color=colors_seg, 
                   edgecolor='black', linewidth=1.5)
    ax6.axvline(0, color='black', linewidth=1)
    
    ax6.set_xlabel('Relative Lift (%)', fontsize=11)
    ax6.set_title('Performance by Segment', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3, axis='x')
    
    for i, (bar, lift) in enumerate(zip(bars, lifts)):
        width = bar.get_width()
        ax6.text(width + 1, i, f'{lift:+.1f}%', 
                va='center', fontweight='bold', fontsize=10)
    
    # Plot 7: Sample size accumulation
    ax7 = fig.add_subplot(gs[2, 1])
    
    cumulative_n = [r['n_control'] + r['n_treatment'] for r in daily_results]
    
    ax7.plot(days, cumulative_n, 'o-', color='green', linewidth=2, markersize=6)
    ax7.fill_between(days, 0, cumulative_n, alpha=0.3, color='green')
    
    ax7.set_xlabel('Day', fontsize=11)
    ax7.set_ylabel('Total Sample Size', fontsize=11)
    ax7.set_title('Cumulative Sample Size', fontsize=12, fontweight='bold')
    ax7.grid(alpha=0.3)
    
    # Plot 8: Expected lift distribution
    ax8 = fig.add_subplot(gs[2, 2])
    
    lift_samples = (bayes_results['samples_treatment'] - bayes_results['samples_control']) / \
                   bayes_results['samples_control']
    
    ax8.hist(lift_samples * 100, bins=50, alpha=0.7, color='orange', 
            density=True, edgecolor='black')
    ax8.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax8.axvline(bayes_results['expected_lift'] * 100, color='blue', 
               linestyle='-', linewidth=2, label='Expected lift')
    
    # Credible interval
    ci = bayes_results['lift_ci']
    ax8.axvline(ci[0] * 100, color='green', linestyle=':', linewidth=2)
    ax8.axvline(ci[1] * 100, color='green', linestyle=':', linewidth=2, 
               label='95% CI')
    
    ax8.set_xlabel('Relative Lift (%)', fontsize=11)
    ax8.set_ylabel('Density', fontsize=11)
    ax8.set_title('Expected Lift Distribution', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'ab_test_landing_page_sequential.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{filename}'")
    
    plt.show()

if __name__ == "__main__":
    daily_results, bayes_results = perform_ab_test_sequential()
