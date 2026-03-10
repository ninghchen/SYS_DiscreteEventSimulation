# SYS 3062: Project 1 
# Client: Bygg & Bo

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

np.random.seed(42)
N = 10000


print("\n" + "=" * 80)
print("Key Deliverables & Metrics")
print("=" * 55)

# General Notes: 
# The key metrics use the correlated total cost not independent. 
# The samples represent costs 
# VaR and CVaR percentile cutoff = 1 - .95 
# Calculate Variance Contribution method: covariance method (other methods work too)
# Sensitivity analysis uses Spearman rank
# What is Normality sample size? write in report

# For Skew and Kurotsis Bias = FALSE. Depends on assumptions


# =============================================================================
# Phase 1: Base Task Estimation (Standard variability in tasks)
# =============================================================================

A = np.random.uniform(40, 80, N)             # Site Survey    ~ Uniform(40, 80)
B = np.random.triangular(100, 150, 300, N)   # Demolition     ~ Triangular(100, 150, 300)
C_ind = np.random.normal(300, 40, N)		 # Concrete		  ~ Normal(300, 40)
D = np.random.triangular(80, 110, 160, N)    # Electrical     ~ Triangular(80, 110, 160)
E = np.random.uniform(90, 210, N)            # Plumbing       ~ Uniform(90, 210)
F_ind = np.random.normal(200, 30, N)		 # HVAC			  ~ Normal(200, 30)
G = np.random.triangular(70, 90, 120, N)     # Interior       ~ Triangular(70, 90, 120)


# =============================================================================
# Phase 2: Risk Register (Specific disasters like strikes)
# =============================================================================

R1_prob = np.random.rand(N) < 0.30 # R1: Asbestos
R2_prob = np.random.rand(N) < 0.10 # R2: Strike
R3_prob = np.random.rand(N) < 0.50 # R3: Penalty

R1_cost = np.where(R1_prob, np.random.normal(200, 50, N), 0)
R2_cost = np.where(R2_prob, np.random.uniform(100, 500, N), 0)
R3_cost = np.where(R3_prob, 50, 0)

risk_cost = R1_cost + R2_cost + R3_cost


# =============================================================================
# Phase 3: Modeling Market Correlation (Commodity prices affecting tasks C and F)
# =============================================================================

rho = 0.7
M = np.random.normal(0, 1, N) # shared market factor

# cost for Task C and F:
Z_C = np.random.normal(0, 1, N)
Z_F = np.random.normal(0, 1, N)

Corr_C = 300 + 40 * (rho * M + np.sqrt(1 - rho**2) * Z_C) # Concrete: C
Corr_F = 200 + 30 * (rho * M + np.sqrt(1 - rho**2) * Z_F) # HVAC: F


# =============================================================================
# Correlated Model
total_cost = A + B + Corr_C + D + E + Corr_F + G + risk_cost

# Independent Model 
total_ind_cost = A + B + C_ind + D + E + F_ind + G + risk_cost


# =============================================================================
# 1. "The Perfect Storm” Probability 
# =============================================================================
# chance that Asbestos occurs AND Total Cost > $1,800k
# P(R1_prob AND total_cost > 1800)

perfect_storm_prob = (R1_prob) & (total_cost > 1800) # array
print(f"P(Asbestos AND Total Cost > 1800k) = {np.mean(perfect_storm_prob)*100:.3f}%") # converged value 


# =============================================================================
# 2. Impact of Correlation
# =============================================================================
# low sd: values are close to the mean; high sd: values are spread out
# Hypothesis: Correlation increases the spread (risk).

sigma_total_corr = np.std(total_cost)
sigma_total_ind = np.std(total_ind_cost)
print(f"Correlated Model Sigma:  {sigma_total_corr:.2f}")
print(f"Independent Model Sigma: {sigma_total_ind:.2f}")
print(f"Difference in sigma:     {sigma_total_corr - sigma_total_ind:.2f}")


# =============================================================================
# 3. Tail Risk Metrics: VaR & CVaR
# =============================================================================
# https://www.pyquantnews.com/free-python-resources/risk-metrics-in-python-var-and-cvar-guide

confidence_level = 0.95

VaR = np.percentile(total_cost, (1 - confidence_level) * 100) # VaR
CVaR = total_cost[total_cost > VaR].mean() # CVaR

print(f"VaR (95%):   {VaR:.2f} kUSD")
print(f"CVaR (95%):  {CVaR:.2f} kUSD")



# =============================================================================
# Print Outputs Function
# =============================================================================
def print_outputs(sample, label=""):
    CL = .95
    mean   = np.mean(sample)
    std    = np.std(sample)
    VaR95  = np.percentile(sample, (1 - CL) * 100)
    CVaR95 = sample[sample >= VaR95].mean()
    skew   = stats.skew(sample)
    kurt   = stats.kurtosis(sample)

    print(f"{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Mean:               {mean:.1f} kUSD")
    print(f"  Std Dev:            {std:.1f} kUSD")
    print(f"  Skewness:           {skew:.4f}")
    print(f"  Kurtosis:           {kurt:.4f}")
    print(f"  VaR  (95%):         {VaR95:.1f} kUSD")
    print(f"  CVaR (95%):         {CVaR95:.1f} kUSD")

print_outputs(total_cost,     label="Scenario A — Correlated")
print_outputs(total_ind_cost, label="Scenario B — Independent")



# =============================================================================
# Visualization 1. Overlay Histogram
# =============================================================================
# Visualizes how correlation ”flattens the curve” and extends the tail

fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle("Bygg & Bo Cost Simulation (N=10,000)", fontsize=13, fontweight='bold')

ax.hist(total_ind_cost, bins=80, density=True, alpha=0.55, color='steelblue', label='Scenario B — Independent (ρ=0)')
ax.hist(total_cost, bins=80, density=True, alpha=0.55, color='tomato',    label='Scenario A — Correlated (ρ=0.7)')
ax.axvline(VaR, color='darkred', lw=2, linestyle='--', label=f'VaR 95% = {VaR:.0f} kUSD')
ax.axvline(CVaR, color='black', lw=2, linestyle=':', label=f'CVaR 95% = {CVaR:.0f} kUSD')
ax.set_xlabel("Total Cost (kUSD)")
ax.set_ylabel("Density")
ax.set_title("Overlay Histogram")
ax.legend(fontsize=8)


# =============================================================================
# Variance Contribution
# =============================================================================
# Contribution_i = Cov(component_i, total_cost) / Var(total_cost)

components = {
    "A: Site Survey": A,
    "B: Demolition": B,
    "C: Concrete": Corr_C,
    "D: Electrical": D,
    "E: Plumbing": E,
    "F: HVAC": Corr_F,
    "G: Interior": G,
    "R1: Asbestos": R1_cost,
    "R2: Strike": R2_cost,
    "R3: Penalty": R3_cost
}

total_var = np.var(total_cost, ddof=0)

var_contrib = {}
for name, arr in components.items():
    contrib = np.cov(arr, total_cost, ddof=0)[0, 1] / total_var
    var_contrib[name] = contrib

# Sort from largest to smallest contribution
var_contrib = dict(sorted(var_contrib.items(), key=lambda x: x[1], reverse=True))

print("\n" + "=" * 55)
print("Variance Contribution to total cost (covariance method)")
print("=" * 55)

for name, contrib in var_contrib.items():
    print(f"{name:<18} {contrib*100:8.2f}%")

print("-" * 55)
print(f"{'Total':<18} {sum(var_contrib.values())*100:8.2f}%")


# =============================================================================
# Sensitivity Analysis
# =============================================================================

# Step 1: Spearman rank correlations
corr_rows = []
for name, arr in components.items():
    spearman_rho, pval = stats.spearmanr(arr, total_cost)
    corr_rows.append([name, spearman_rho, abs(spearman_rho), pval])

corr_df = pd.DataFrame(
    corr_rows,
    columns=["Input", "Spearman_rho", "Abs_rho", "p_value"]
)

# Sort largest to smallest
corr_df = corr_df.sort_values("Abs_rho", ascending=False).reset_index(drop=True)

# Step 2: square them
corr_df["rho_squared"] = corr_df["Spearman_rho"] ** 2

# Step 3: normalize to 100%
total_squared = corr_df["rho_squared"].sum()
corr_df["Contribution_to_Variance_pct"] = (
    corr_df["rho_squared"] / total_squared * 100
)

print("\n" + "=" * 55)
print("\nRank inputs by correlation with Total Cost (Spearman rank correlations)")
print("=" * 55)
for i, row in corr_df.iterrows():
    print(f"{i+1:>2}. {row['Input']:<18} rho = {row['Spearman_rho']:+.4f}")


# =============================================================================
# Visualization 2. Tornado Chart (Sensitivity by Spearman correlation)
# =============================================================================

components = {
    "A: Site Survey": A,
    "B: Demolition": B,
    "C: Concrete": Corr_C,
    "D: Electrical": D,
    "E: Plumbing": E,
    "F: HVAC": Corr_F,
    "G: Interior": G,
    "R1: Asbestos": R1_cost,
    "R2: Strike": R2_cost,
    "R3: Penalty": R3_cost
}

# Step 1: Spearman rank correlations
corr_rows = []
for name, arr in components.items():
    rho, pval = stats.spearmanr(arr, total_cost)
    corr_rows.append([name, rho, abs(rho), pval])

corr_df = pd.DataFrame(
    corr_rows,
    columns=["Input", "Spearman_rho", "Abs_rho", "p_value"]
)

# Sort largest to smallest by absolute sensitivity
corr_df = corr_df.sort_values("Abs_rho", ascending=False).reset_index(drop=True)

# plot
fig, ax = plt.subplots(figsize=(12, 6))

y_pos = np.arange(len(corr_df))
values = corr_df["Spearman_rho"].values
labels = corr_df["Input"].values

colors = ["tomato" if v > 0 else "steelblue" for v in values]

ax.barh(y_pos, values, color=colors, edgecolor="black")
ax.axvline(0, color="black", linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # biggest on top

ax.set_xlabel("Spearman Correlation with Total Cost")
ax.set_title("Tornado Chart")

# optional value labels
for i, v in enumerate(values):
    if v >= 0:
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
    else:
        ax.text(v - 0.01, i, f"{v:.3f}", va="center", ha="right")

plt.tight_layout()
plt.show()
