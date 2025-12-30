# Statistical Testing Learning Project

A comprehensive collection of Python examples demonstrating statistical hypothesis testing, confidence intervals, and A/B testing techniques. This project is designed for learning and understanding the practical application of statistics in data analysis and business decision-making.

## 📚 Overview

This repository contains practical, runnable examples of common statistical tests used in applied settings such as A/B testing, scientific research, and data analysis. Each script generates detailed statistical output and visualizations to help understand the concepts.

## 🎯 Learning Objectives

- Understand different types of statistical tests and when to use them
- Learn how to formulate null and alternative hypotheses
- Interpret p-values, confidence intervals, and effect sizes
- Apply statistical methods to real-world business problems
- Understand the difference between statistical and practical significance
- Learn bootstrap resampling techniques
- Master A/B testing methodology

## 📁 Project Structure

```
statistics/
├── README.md                              # This file
├── height_mean_t_test.py                  # One-sample t-test
├── two_sample_t_test.py                   # Independent two-sample t-test
├── paired_t_test.py                       # Paired (dependent) t-test
├── bootstrap_confidence_intervals.py      # Bootstrap resampling methods
├── proportion_z_tests.py                  # Z-tests for proportions
├── ab_test_1.py                          # A/B test: Email campaign
└── ab_test_2.py                          # A/B test: Sequential testing
```

## 🔬 Examples Included

### 1. One-Sample T-Test (`height_mean_t_test.py`)

**Use Case:** Testing if the average height differs from a hypothesized value (170 cm)

**Key Concepts:**
- One-sample t-test methodology
- Null and alternative hypothesis formulation
- Confidence intervals for the mean
- Effect size (Cohen's d)
- When to use t-test vs. z-test

**Output:**
- Statistical test results with p-values
- 95% confidence interval
- Comprehensive visualizations including distribution plots and t-distribution

**Run:**
```bash
python height_mean_t_test.py
```

---

### 2. Two-Sample T-Test (`two_sample_t_test.py`)

**Use Case:** Comparing test scores between traditional and new teaching methods

**Key Concepts:**
- Independent two-sample t-test
- Assumption of equal variances (Levene's test)
- Welch's t-test for unequal variances
- Pooled standard deviation
- Effect size for group comparisons

**Output:**
- Variance equality test results
- Statistical comparison of two groups
- Mean comparison with confidence intervals
- Box plots and distribution visualizations

**Run:**
```bash
python two_sample_t_test.py
```

---

### 3. Paired T-Test (`paired_t_test.py`)

**Use Case:** Evaluating weight loss from a fitness program (before-after design)

**Key Concepts:**
- Paired (dependent) samples t-test
- When to use paired vs. independent tests
- Analysis of differences
- Normality assumption checking (Shapiro-Wilk test)
- One-tailed vs. two-tailed tests

**Output:**
- Before-after comparison statistics
- Statistical significance of changes
- Individual trajectory plots
- Distribution of differences

**Run:**
```bash
python paired_t_test.py
```

---

### 4. Bootstrap Confidence Intervals (`bootstrap_confidence_intervals.py`)

**Use Case:** Estimating confidence intervals when data don't meet parametric assumptions

**Three Examples:**
1. **Bootstrap CI for mean** - Handling skewed data
2. **Bootstrap CI for median** - Robust to outliers
3. **Bootstrap CI for difference** - Comparing two groups

**Key Concepts:**
- Bootstrap resampling with replacement
- Percentile method for confidence intervals
- When to use bootstrap vs. parametric methods
- No distributional assumptions required
- Comparison with traditional methods

**Output:**
- Multiple bootstrap examples
- Comparison of parametric vs. bootstrap CIs
- Bootstrap distribution visualizations
- Comprehensive multi-panel plots

**Run:**
```bash
python bootstrap_confidence_intervals.py
```

---

### 5. Proportion Z-Tests (`proportion_z_tests.py`)

**Use Case:** Testing conversion rates and proportions

**Two Examples:**
1. **One-sample proportion test** - Compare website conversion rate to industry standard
2. **Two-sample proportion test** - A/B test for landing page designs

**Key Concepts:**
- Z-test for proportions
- Sample size requirements (np ≥ 10 rule)
- Pooled proportion for two-sample tests
- Relative risk and odds ratios
- Confidence intervals for proportions

**Output:**
- Proportion comparisons with statistical tests
- Effect size measures (relative risk, odds ratio)
- Comprehensive visualizations including funnel charts
- z-distribution with rejection regions

**Run:**
```bash
python proportion_z_tests.py
```

---

### 6. A/B Test: Email Campaign (`ab_test_1.py`)

**Use Case:** Testing email subject lines to improve click-through rates

**Key Concepts:**
- Complete A/B test workflow
- Sample size calculation (power analysis)
- Statistical significance vs. practical significance
- Business impact assessment
- Revenue lift calculations
- Statistical power evaluation

**Output:**
- Detailed A/B test results
- Sample size recommendations
- Business impact metrics
- Revenue projections
- Multi-metric visualization dashboard

**Run:**
```bash
python ab_test_1.py
```

---

### 7. A/B Test: Sequential Testing (`ab_test_2.py`)

**Use Case:** Landing page redesign with early stopping capability

**Key Concepts:**
- Sequential A/B testing
- Early stopping rules
- Bayesian analysis (Beta-Binomial model)
- Posterior distributions and credible intervals
- Probability of superiority
- Segmented analysis
- Daily monitoring and decision-making

**Output:**
- Day-by-day test evolution
- Frequentist and Bayesian results
- Probability that treatment is better
- Expected lift with credible intervals
- Segment-level performance
- Comprehensive sequential analysis dashboard

**Run:**
```bash
python ab_test_2.py
```

## 🛠️ Requirements

### Python Version
- Python 3.7 or higher

### Dependencies

Install required packages:

```bash
pip install numpy scipy matplotlib seaborn
```

Or using requirements file:

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Numerical computing and random sampling
- `scipy` - Statistical functions and tests
- `matplotlib` - Plotting and visualization
- `seaborn` - Enhanced statistical visualizations

## 🚀 Getting Started

1. **Clone or download** this repository to your local machine

2. **Install dependencies:**
   ```bash
   pip install numpy scipy matplotlib seaborn
   ```

3. **Run any example:**
   ```bash
   python height_mean_t_test.py
   ```

4. **Explore the outputs:**
   - Console output with detailed statistical results
   - PNG image files with visualizations
   - Interactive plots (if using interactive backend)

## 📊 Understanding the Output

Each script provides:

### Console Output
- **Scenario description**: Business context and research question
- **Hypotheses**: Clearly stated null (H₀) and alternative (H₁) hypotheses
- **Sample statistics**: Descriptive statistics of the data
- **Test results**: Test statistics, p-values, degrees of freedom
- **Confidence intervals**: 95% CIs with interpretations
- **Decision**: Clear reject/fail-to-reject statement
- **Effect size**: Practical significance measures (Cohen's d, etc.)
- **Conclusion**: Plain English interpretation

### Visualizations
- Distribution plots (histograms, density curves)
- Box plots and comparison charts
- Confidence interval visualizations
- Statistical distribution plots (t-distribution, z-distribution)
- Multi-panel comprehensive dashboards

All visualizations are saved as high-resolution PNG files (300 dpi) in the same directory.

## 📖 Statistical Concepts Reference

### Hypothesis Testing Framework

1. **Null Hypothesis (H₀)**: The default assumption (no effect, no difference)
2. **Alternative Hypothesis (H₁)**: What we're testing for (there is an effect)
3. **Significance Level (α)**: Probability of Type I error (typically 0.05)
4. **P-value**: Probability of observing data this extreme if H₀ is true
5. **Decision Rule**: Reject H₀ if p-value < α

### Test Selection Guide

| Scenario | Test Type | Script |
|----------|-----------|--------|
| Compare sample mean to known value | One-sample t-test | `height_mean_t_test.py` |
| Compare means of two independent groups | Two-sample t-test | `two_sample_t_test.py` |
| Compare before-after measurements | Paired t-test | `paired_t_test.py` |
| Compare proportion to known value | One-sample z-test | `proportion_z_tests.py` |
| Compare proportions between two groups | Two-sample z-test | `proportion_z_tests.py` |
| Non-normal data or small samples | Bootstrap methods | `bootstrap_confidence_intervals.py` |
| A/B test for proportions | Z-test with power analysis | `ab_test_1.py`, `ab_test_2.py` |

### Assumptions

**T-tests:**
- Data are approximately normally distributed (or large sample size)
- Independent observations
- For two-sample: Equal variances (or use Welch's correction)

**Z-tests for proportions:**
- np ≥ 10 and n(1-p) ≥ 10 (sample size requirement)
- Independent observations
- Random sampling

**Bootstrap:**
- Independent observations
- Random sampling
- No parametric distribution assumptions needed

### Effect Sizes

- **Cohen's d**: Standardized mean difference
  - Small: |d| = 0.2
  - Medium: |d| = 0.5
  - Large: |d| = 0.8

- **Relative Risk (RR)**: Ratio of proportions
  - RR = 1: No effect
  - RR > 1: Increased risk/rate
  - RR < 1: Decreased risk/rate

## 💡 Best Practices

### When Planning a Test

1. **Define hypotheses before collecting data**
2. **Calculate required sample size** (power analysis)
3. **Choose appropriate significance level** (typically α = 0.05)
4. **Consider practical significance**, not just statistical significance
5. **Plan for multiple comparisons** if doing several tests

### When Analyzing Results

1. **Check assumptions** (normality, equal variances, etc.)
2. **Report both p-values and effect sizes**
3. **Include confidence intervals**
4. **Consider practical/business significance**
5. **Visualize your data** before and after testing

### A/B Testing Specific

1. **Run tests long enough** to capture day-of-week effects
2. **Ensure random assignment** to groups
3. **Monitor for novelty effects**
4. **Consider segmented analysis**
5. **Calculate business impact**, not just statistical significance
6. **Use sequential testing carefully** (adjust for multiple looks)

## 🎓 Learning Path

**Suggested order for learning:**

1. **Start with one-sample t-test** (`height_mean_t_test.py`)
   - Understand basic hypothesis testing
   - Learn to interpret p-values and confidence intervals

2. **Move to two-sample tests** (`two_sample_t_test.py`)
   - Compare two independent groups
   - Understand pooled variance and effect sizes

3. **Learn paired tests** (`paired_t_test.py`)
   - Understand dependent samples
   - Before-after experimental designs

4. **Explore bootstrap methods** (`bootstrap_confidence_intervals.py`)
   - Non-parametric approaches
   - Handling non-normal data

5. **Master proportion tests** (`proportion_z_tests.py`)
   - Categorical data analysis
   - Foundation for A/B testing

6. **Apply to A/B testing** (`ab_test_1.py`, `ab_test_2.py`)
   - Real-world business applications
   - Complete experimental workflow
   - Advanced techniques (sequential testing, Bayesian analysis)

## 📚 Additional Resources

### Books
- "Statistics" by David Freedman, Robert Pisani, Roger Purves
- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
- "Trustworthy Online Controlled Experiments" by Kohavi, Tang, Xu

### Online Resources
- Khan Academy: Statistics and Probability
- StatQuest YouTube channel
- Coursera: Statistics with Python specialization

### Documentation
- [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## 🤝 Contributing

This is a learning project. Feel free to:
- Add more examples
- Improve documentation
- Suggest additional statistical tests
- Report issues or bugs

## 📝 Notes

- All examples use **simulated data** for reproducibility
- Random seed is set (42) for consistent results across runs
- Visualizations are automatically saved as PNG files
- Scripts are self-contained and can be run independently

## ⚠️ Disclaimer

These examples are for **educational purposes**. When applying statistical methods to real-world data:
- Carefully check all assumptions
- Consider consulting a statistician for critical decisions
- Understand the limitations of each method
- Consider multiple sources of evidence

## 📄 License

This project is open source and available for educational use.

---

**Happy Learning! 📊✨**

For questions or suggestions, feel free to open an issue or contribute to the project.
