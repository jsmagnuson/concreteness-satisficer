"""Run statistical tests (t-test, Mann-Whitney, Cohen's d, Shapiro-Wilk) on
length and Percent_known across High/Low concreteness groups in a candidate
CSV. Usage: python3 statistical_tests.py [path_to_candidate.csv]"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / 'candidate_01.csv')
print(f"File: {path}")
df = pd.read_csv(path)

# Separate the two groups
high_conc = df[df['Conc_Category'] == 'High']
low_conc = df[df['Conc_Category'] == 'Low']

print("=" * 70)
print("STATISTICAL TESTS: High vs Low Concreteness Nouns")
print("=" * 70)

# ============ WORD LENGTH TESTS ============
print("\n1. WORD LENGTH COMPARISON")
print("-" * 70)
print(f"High Concreteness: M={high_conc['Length (letters)'].mean():.2f}, SD={high_conc['Length (letters)'].std():.2f}")
print(f"Low Concreteness:  M={low_conc['Length (letters)'].mean():.2f}, SD={low_conc['Length (letters)'].std():.2f}")

# Check normality with Shapiro-Wilk test
stat_high_len, p_high_len = stats.shapiro(high_conc['Length (letters)'])
stat_low_len, p_low_len = stats.shapiro(low_conc['Length (letters)'])

print(f"\nNormality tests (Shapiro-Wilk):")
print(f"  High: W={stat_high_len:.4f}, p={p_high_len:.4f} {'(normal)' if p_high_len > 0.05 else '(not normal)'}")
print(f"  Low:  W={stat_low_len:.4f}, p={p_low_len:.4f} {'(normal)' if p_low_len > 0.05 else '(not normal)'}")

# Independent samples t-test
t_stat_len, p_t_len = stats.ttest_ind(high_conc['Length (letters)'], low_conc['Length (letters)'])
print(f"\nIndependent samples t-test:")
print(f"  t({len(high_conc) + len(low_conc) - 2}) = {t_stat_len:.4f}, p = {p_t_len:.4f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat_len, p_u_len = stats.mannwhitneyu(high_conc['Length (letters)'], low_conc['Length (letters)'], alternative='two-sided')
print(f"\nMann-Whitney U test (non-parametric):")
print(f"  U = {u_stat_len:.1f}, p = {p_u_len:.4f}")

# Effect size (Cohen's d)
pooled_sd = np.sqrt(((len(high_conc)-1)*high_conc['Length (letters)'].std()**2 + (len(low_conc)-1)*low_conc['Length (letters)'].std()**2) / (len(high_conc) + len(low_conc) - 2))
cohens_d_len = (high_conc['Length (letters)'].mean() - low_conc['Length (letters)'].mean()) / pooled_sd
print(f"\nEffect size (Cohen's d) = {cohens_d_len:.4f} ", end="")
if abs(cohens_d_len) < 0.2:
    print("(negligible)")
elif abs(cohens_d_len) < 0.5:
    print("(small)")
elif abs(cohens_d_len) < 0.8:
    print("(medium)")
else:
    print("(large)")

# ============ PERCENT_KNOWN TESTS ============
print("\n\n2. PERCENT_KNOWN COMPARISON")
print("-" * 70)
print(f"High Concreteness: M={high_conc['Percent_known'].mean():.4f}, SD={high_conc['Percent_known'].std():.4f}")
print(f"Low Concreteness:  M={low_conc['Percent_known'].mean():.4f}, SD={low_conc['Percent_known'].std():.4f}")

# Check normality
stat_high_pk, p_high_pk = stats.shapiro(high_conc['Percent_known'])
stat_low_pk, p_low_pk = stats.shapiro(low_conc['Percent_known'])

print(f"\nNormality tests (Shapiro-Wilk):")
print(f"  High: W={stat_high_pk:.4f}, p={p_high_pk:.4f} {'(normal)' if p_high_pk > 0.05 else '(not normal)'}")
print(f"  Low:  W={stat_low_pk:.4f}, p={p_low_pk:.4f} {'(normal)' if p_low_pk > 0.05 else '(not normal)'}")

# Independent samples t-test
t_stat_pk, p_t_pk = stats.ttest_ind(high_conc['Percent_known'], low_conc['Percent_known'])
print(f"\nIndependent samples t-test:")
print(f"  t({len(high_conc) + len(low_conc) - 2}) = {t_stat_pk:.4f}, p = {p_t_pk:.4f}")

# Mann-Whitney U test
u_stat_pk, p_u_pk = stats.mannwhitneyu(high_conc['Percent_known'], low_conc['Percent_known'], alternative='two-sided')
print(f"\nMann-Whitney U test (non-parametric):")
print(f"  U = {u_stat_pk:.1f}, p = {p_u_pk:.4f}")

# Effect size (Cohen's d)
pooled_sd_pk = np.sqrt(((len(high_conc)-1)*high_conc['Percent_known'].std()**2 + (len(low_conc)-1)*low_conc['Percent_known'].std()**2) / (len(high_conc) + len(low_conc) - 2))
cohens_d_pk = (high_conc['Percent_known'].mean() - low_conc['Percent_known'].mean()) / pooled_sd_pk
print(f"\nEffect size (Cohen's d) = {cohens_d_pk:.4f} ", end="")
if abs(cohens_d_pk) < 0.2:
    print("(negligible)")
elif abs(cohens_d_pk) < 0.5:
    print("(small)")
elif abs(cohens_d_pk) < 0.8:
    print("(medium)")
else:
    print("(large)")

# ============ SUMMARY ============
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nWord Length:")
print(f"  Difference in means: {abs(high_conc['Length (letters)'].mean() - low_conc['Length (letters)'].mean()):.2f}")
print(f"  p-value (t-test): {p_t_len:.4f}")
print(f"  p-value (Mann-Whitney): {p_u_len:.4f}")
if p_t_len > 0.05 and p_u_len > 0.05:
    print(f"  → NO significant difference (α = 0.05)")
else:
    print(f"  → SIGNIFICANT difference (α = 0.05)")

print(f"\nPercent Known:")
print(f"  Difference in means: {abs(high_conc['Percent_known'].mean() - low_conc['Percent_known'].mean()):.4f}")
print(f"  p-value (t-test): {p_t_pk:.4f}")
print(f"  p-value (Mann-Whitney): {p_u_pk:.4f}")
if p_t_pk > 0.05 and p_u_pk > 0.05:
    print(f"  → NO significant difference (α = 0.05)")
else:
    print(f"  → SIGNIFICANT difference (α = 0.05)")
