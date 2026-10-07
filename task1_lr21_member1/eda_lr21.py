"""
HIT140 Assessment 3 -- LR 2.1 Exploratory Data Analysis
Goal Difference prediction, 2026 FIFA World Cup (104 matches)

Author: Jahid Hasan (Member 1 -- LR 2.1 Data + EDA)

Input : lr21_dataset_104.csv  (target + 8 pre-match features, verified against
        official FBref standings and cross-checked news sources -- see the
        'LR2.1 Feature Documentation' sheet of the companion Excel workbook
        for exact sources and documented assumptions for every variable)
Output: eda_outputs/*.png  (one chart per check)
        eda_outputs/eda_summary.txt  (printable numeric summary for the report)

Run with:  python eda_lr21.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.family"] = "DejaVu Sans"

OUT = "eda_outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Load data
df = pd.read_csv("lr21_dataset_104.csv")

TARGET = "Goal Difference (Target)"
FEATURES = [
    "X1 FIFA Rank Diff (Away-Home)",
    "X2 Squad Value Diff EURm (Home-Away)",
    "X3 Squad Age Diff (Home-Away)",
    "X4 Prior WC Best Result Diff (Home-Away)",
    "X5 Qualifying GD Diff (Home-Away)",
    "X6 Host Advantage Diff (Home-Away)",
    "X7 Confederation Strength Diff (Home-Away)",
    "X8 Rest Days Diff (Home-Away)",
]
SHORT = {  # short labels for compact chart axes
    "X1 FIFA Rank Diff (Away-Home)": "X1 FIFA Rank",
    "X2 Squad Value Diff EURm (Home-Away)": "X2 Squad Value",
    "X3 Squad Age Diff (Home-Away)": "X3 Squad Age",
    "X4 Prior WC Best Result Diff (Home-Away)": "X4 Prior WC Best",
    "X5 Qualifying GD Diff (Home-Away)": "X5 Qualifying GD",
    "X6 Host Advantage Diff (Home-Away)": "X6 Host Advantage",
    "X7 Confederation Strength Diff (Home-Away)": "X7 Confederation",
    "X8 Rest Days Diff (Home-Away)": "X8 Rest Days",
    "Goal Difference (Target)": "Goal Difference",
}

cols = [TARGET] + FEATURES
summary_lines = []

def log(msg=""):
    print(msg)
    summary_lines.append(msg)

log("=" * 70)
log("LR 2.1 -- Exploratory Data Analysis")
log(f"Rows: {len(df)}   Columns analysed: {len(cols)} (1 target + 8 features)")
log("=" * 70)

# ---------------------------------------------------------------- 1. Missing values
log("\n--- 1. Missing-value check ---")
na_counts = df[cols].isna().sum()
if na_counts.sum() == 0:
    log("No missing values in target or any of the 8 features (0/104 in every column).")
else:
    log(na_counts[na_counts > 0].to_string())

# ---------------------------------------------------------------- 2. Descriptive stats
log("\n--- 2. Descriptive statistics ---")
desc = df[cols].describe().T
desc["skew"] = df[cols].skew()
desc = desc.round(2)
log(desc.to_string())
desc.to_csv(f"{OUT}/descriptive_statistics.csv")

# ---------------------------------------------------------------- 3. Target distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df[TARGET], bins=range(int(df[TARGET].min()) - 1, int(df[TARGET].max()) + 2),
             kde=False, color="#2E5C8A", ax=axes[0])
axes[0].set_title("Distribution of Goal Difference (target)")
axes[0].set_xlabel("Home Score − Away Score")
axes[0].set_ylabel("Number of matches")

stats.probplot(df[TARGET], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot vs Normal distribution")
plt.tight_layout()
plt.savefig(f"{OUT}/01_target_distribution.png")
plt.close()

sh_w, sh_p = stats.shapiro(df[TARGET])
log(f"\n--- 3. Target normality (Shapiro-Wilk) ---")
log(f"W = {sh_w:.4f}, p = {sh_p:.4f}  -> "
    f"{'target looks roughly normal (p>0.05)' if sh_p > 0.05 else 'target deviates from normal (p<0.05) -- expected for a bounded integer count like goal difference'}")

# ---------------------------------------------------------------- 4. Feature distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, feat in zip(axes.flat, FEATURES):
    sns.histplot(df[feat], bins=15, color="#4C8C6B", ax=ax)
    ax.set_title(SHORT[feat], fontsize=10)
    ax.set_xlabel("")
plt.suptitle("Distributions of the 8 pre-match explanatory variables", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/02_feature_distributions.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- 5. Correlation matrix (incl. target)
corr = df[cols].corr(numeric_only=True)
corr_display = corr.rename(columns=SHORT, index=SHORT)
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr_display, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation matrix -- Target + 8 features", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/03_correlation_matrix.png")
plt.close()
corr.to_csv(f"{OUT}/correlation_matrix.csv")

log("\n--- 4. Correlation of each feature with the target (Goal Difference) ---")
target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
for feat, r in target_corr.items():
    log(f"  {SHORT[feat]:<20s} r = {r:+.3f}")

# ---------------------------------------------------------------- 6. Multicollinearity check among features (VIF-style via pairwise corr)
log("\n--- 5. Multicollinearity check (pairwise correlation among the 8 features) ---")
feat_corr = corr.loc[FEATURES, FEATURES]
high_pairs = []
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        r = feat_corr.iloc[i, j]
        if abs(r) >= 0.6:
            high_pairs.append((SHORT[FEATURES[i]], SHORT[FEATURES[j]], r))
if high_pairs:
    for a, b, r in high_pairs:
        log(f"  WARNING: {a} <-> {b}  r = {r:+.3f}  (|r| >= 0.6, worth a VIF check before final model)")
else:
    log("  No feature pair exceeds |r| = 0.6 -- no strong multicollinearity red flag from pairwise correlation alone.")

# Variance Inflation Factor (proper multicollinearity diagnostic)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    X = add_constant(df[FEATURES])
    vif = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])],
        index=FEATURES, name="VIF"
    ).round(2)
    log("\n--- 6. Variance Inflation Factor (VIF) per feature ---")
    log("  (VIF < 5 generally fine; VIF > 10 signals serious multicollinearity)")
    for feat, v in vif.sort_values(ascending=False).items():
        flag = "  <-- investigate" if v > 5 else ""
        log(f"  {SHORT[feat]:<20s} VIF = {v:6.2f}{flag}")
    vif.to_csv(f"{OUT}/vif.csv")
except ImportError:
    log("\n(statsmodels not available -- VIF step skipped; pairwise correlation above still covers the basics)")

# ---------------------------------------------------------------- 7. Scatter: each feature vs target with regression line
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
for ax, feat in zip(axes.flat, FEATURES):
    sns.regplot(x=df[feat], y=df[TARGET], ax=ax, scatter_kws={"s": 14, "alpha": 0.6, "color": "#2E5C8A"},
                line_kws={"color": "#C0392B", "linewidth": 1.5})
    r = target_corr[feat]
    ax.set_title(f"{SHORT[feat]}  (r={r:+.2f})", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Goal Diff" if ax in axes[:, 0] else "")
plt.suptitle("Each feature vs. Goal Difference (with linear fit)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/04_feature_vs_target_scatter.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- 8. Boxplots by Stage (sanity check, not a model feature)
fig, ax = plt.subplots(figsize=(8, 4))
order = ["Group stage", "Round of 32", "Round of 16", "Quarter-final", "Semi-final",
         "Third-place match", "Final"]
order = [s for s in order if s in df["Stage"].unique()]
sns.boxplot(data=df, x="Stage", y=TARGET, order=order, palette="Blues", ax=ax)
ax.set_title("Goal Difference by tournament stage (context, not a model input)")
ax.set_xlabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/05_goal_diff_by_stage.png")
plt.close()

# ---------------------------------------------------------------- Save summary
with open(f"{OUT}/eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

log(f"\nAll outputs written to ./{OUT}/")
