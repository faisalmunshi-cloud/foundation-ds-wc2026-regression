# HIT140 Assessment 3 - LR 2.2 EDA
# descriptive stats, distributions, skewness, log(goals + 1), correlations, VIF

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

df = pd.read_csv("lr22_dataset_208.csv")
os.makedirs("eda_outputs_lr22", exist_ok=True)

target = "Goals Scored (Target)"
features = [c for c in df.columns if c.startswith("X")]
short = ["X1 Team rank", "X2 Opp rank", "X3 Squad value", "X4 Host",
         "X5 Qual GF/match", "X6 Opp qual GA/match", "X7 Knockout", "X8 WC apps"]

out = open("eda_outputs_lr22/eda_summary.txt", "w")

def write(text):
    print(text)
    out.write(str(text) + "\n")

write("LR 2.2 EDA")
write("rows: " + str(len(df)) + "  features: " + str(len(features)))

# 1. missing values
write("\n1. Missing values")
write(df[[target] + features].isnull().sum())

# 2. descriptive statistics
write("\n2. Descriptive statistics")
desc = df[[target] + features].describe().T
desc["skew"] = df[[target] + features].skew()
desc = desc.round(3)
desc.to_csv("eda_outputs_lr22/descriptive_statistics.csv")
pd.set_option("display.width", 200)
write(desc.to_string())

# 3. target distribution
write("\n3. Target (goals scored)")
write("value counts:")
write(df[target].value_counts().sort_index())
write("mean = " + str(round(df[target].mean(), 3)) + ", variance = " + str(round(df[target].var(), 3)))

# log(goals + 1) test
df["log_goals"] = np.log(df[target] + 1)
skew_raw = df[target].skew()
skew_log = df["log_goals"].skew()
sw_raw = stats.shapiro(df[target])
sw_log = stats.shapiro(df["log_goals"])
write("skew raw goals = " + str(round(skew_raw, 3)) + ", skew log(goals+1) = " + str(round(skew_log, 3)))
write("Shapiro-Wilk raw: W = %.3f, p = %.5f" % (sw_raw.statistic, sw_raw.pvalue))
write("Shapiro-Wilk log: W = %.3f, p = %.5f" % (sw_log.statistic, sw_log.pvalue))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
counts = df[target].value_counts().sort_index()
ax[0].bar(counts.index, counts.values, color="steelblue")
ax[0].set_title("Goals scored per team per match")
ax[0].set_xlabel("Goals")
ax[0].set_ylabel("Number of rows")
log_counts = df["log_goals"].value_counts().sort_index()
ax[1].bar(log_counts.index, log_counts.values, width=0.15, color="darkorange")
ax[1].set_title("log(goals + 1)")
ax[1].set_xlabel("log(goals + 1)")
ax[1].set_ylabel("Number of rows")
ax[0].text(0.6, 0.85, "skew = %.2f" % skew_raw, transform=ax[0].transAxes)
ax[1].text(0.6, 0.85, "skew = %.2f" % skew_log, transform=ax[1].transAxes)
plt.tight_layout()
plt.savefig("eda_outputs_lr22/01_target_distribution.png", dpi=150)
plt.close()

# 4. feature distributions
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
axes = axes.flatten()
for i in range(8):
    axes[i].hist(df[features[i]], bins=15, color="steelblue", edgecolor="white")
    axes[i].set_title(short[i] + "\nskew = %.2f" % df[features[i]].skew(), fontsize=10)
plt.tight_layout()
plt.savefig("eda_outputs_lr22/02_feature_distributions.png", dpi=150)
plt.close()

# 5. correlation with target
write("\n5. Correlation with target")
corr_rows = []
for i in range(8):
    r, p = stats.pearsonr(df[features[i]], df[target])
    rho, p2 = stats.spearmanr(df[features[i]], df[target])
    corr_rows.append([short[i], round(r, 3), round(p, 5), round(rho, 3), round(p2, 5)])
corr_table = pd.DataFrame(corr_rows, columns=["Feature", "Pearson r", "p (Pearson)", "Spearman rho", "p (Spearman)"])
corr_table.to_csv("eda_outputs_lr22/correlation_with_target.csv", index=False)
write(corr_table.to_string(index=False))

# correlation heatmap
corr = df[features + [target]].corr()
labels = short + ["Goals"]
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
for a in range(len(labels)):
    for b in range(len(labels)):
        ax.text(b, a, "%.2f" % corr.iloc[a, b], ha="center", va="center", fontsize=8)
plt.colorbar(im)
ax.set_title("Correlation matrix (LR 2.2)")
plt.tight_layout()
plt.savefig("eda_outputs_lr22/03_correlation_matrix.png", dpi=150)
plt.close()
corr.round(3).to_csv("eda_outputs_lr22/correlation_matrix.csv")

# strong pairs between features
write("\nFeature pairs with |r| >= 0.6")
for a in range(8):
    for b in range(a + 1, 8):
        if abs(corr.iloc[a, b]) >= 0.6:
            write(short[a] + " <-> " + short[b] + "  r = %.3f" % corr.iloc[a, b])

# 6. scatter plots feature vs target
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
axes = axes.flatten()
for i in range(8):
    # small jitter so the points do not sit on top of each other
    y = df[target] + np.random.RandomState(1).uniform(-0.15, 0.15, len(df))
    axes[i].scatter(df[features[i]], y, s=12, alpha=0.5, color="steelblue")
    line = np.polyfit(df[features[i]], df[target], 1)
    xs = np.linspace(df[features[i]].min(), df[features[i]].max(), 50)
    axes[i].plot(xs, line[0] * xs + line[1], color="red")
    axes[i].set_title(short[i] + "  (r = %.2f)" % corr_rows[i][1], fontsize=10)
    axes[i].set_ylabel("Goals")
plt.tight_layout()
plt.savefig("eda_outputs_lr22/04_feature_vs_target_scatter.png", dpi=150)
plt.close()

# 7. VIF
write("\n7. VIF")
X = add_constant(df[features])
vif = []
for i in range(1, X.shape[1]):
    vif.append([short[i - 1], round(variance_inflation_factor(X.values, i), 2)])
vif = pd.DataFrame(vif, columns=["Feature", "VIF"]).sort_values("VIF", ascending=False)
vif.to_csv("eda_outputs_lr22/vif.csv", index=False)
write(vif.to_string(index=False))

out.close()
print("\nsaved outputs in eda_outputs_lr22/")
