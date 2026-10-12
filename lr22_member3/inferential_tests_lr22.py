# HIT140 Assessment 3 - LR 2.2 inferential tests
# main test: goals scored against higher ranked vs lower ranked opponents

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("lr22_dataset_208.csv")
target = "Goals Scored (Target)"

out = open("eda_outputs_lr22/inferential_tests.txt", "w")

def write(text):
    print(text)
    out.write(str(text) + "\n")

# mean with 95% confidence interval
def mean_ci(x):
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return m, m - t * se, m + t * se

# ---- Test 1: higher vs lower ranked opponent ----
# lower rank number = better team
stronger_opp = df[df["X2 Opponent FIFA Rank"] < df["X1 Team FIFA Rank"]][target]
weaker_opp = df[df["X2 Opponent FIFA Rank"] > df["X1 Team FIFA Rank"]][target]

write("Test 1: goals vs higher ranked and lower ranked opponents")
write("H0: mean goals are the same in both groups")
write("H1: mean goals are different")
for name, g in [("opponent ranked higher", stronger_opp), ("opponent ranked lower", weaker_opp)]:
    m, lo, hi = mean_ci(g)
    write("%s: n = %d, mean = %.3f, 95%% CI [%.3f, %.3f]" % (name, len(g), m, lo, hi))

t1 = stats.ttest_ind(weaker_opp, stronger_opp, equal_var=False)
diff = weaker_opp.mean() - stronger_opp.mean()
se_diff = np.sqrt(weaker_opp.var() / len(weaker_opp) + stronger_opp.var() / len(stronger_opp))
write("difference in means = %.3f goals" % diff)
write("95%% CI of difference approx [%.3f, %.3f]" % (diff - 1.96 * se_diff, diff + 1.96 * se_diff))
write("Welch t-test: t = %.3f, p = %.6f" % (t1.statistic, t1.pvalue))
u1 = stats.mannwhitneyu(weaker_opp, stronger_opp)
write("Mann-Whitney U (check, goals are counts): U = %.1f, p = %.6f" % (u1.statistic, u1.pvalue))
# every match gives one row to each group, so paired test by match too
better = df[df["X1 Team FIFA Rank"] < df["X2 Opponent FIFA Rank"]].sort_values("Match #")
worse = df[df["X1 Team FIFA Rank"] > df["X2 Opponent FIFA Rank"]].sort_values("Match #")
tp = stats.ttest_rel(better[target].values, worse[target].values)
write("paired t-test by match (better ranked team vs worse ranked team): t = %.3f, p = %.6f" % (tp.statistic, tp.pvalue))
if t1.pvalue < 0.05:
    write("result: reject H0 at 5% level")
else:
    write("result: fail to reject H0 at 5% level")

# boxplot for this test
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.boxplot([stronger_opp, weaker_opp], showmeans=True)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Opponent ranked higher\n(n=%d)" % len(stronger_opp),
                    "Opponent ranked lower\n(n=%d)" % len(weaker_opp)])
ax.set_ylabel("Goals scored")
if t1.pvalue < 0.001:
    ax.set_title("Goals scored by opponent strength (p < 0.001)")
else:
    ax.set_title("Goals scored by opponent strength (p = %.3f)" % t1.pvalue)
plt.tight_layout()
plt.savefig("eda_outputs_lr22/05_goals_vs_opponent_rank.png", dpi=150)
plt.close()

# ---- Test 2: host vs non-host ----
write("\nTest 2: hosts vs non-hosts")
host = df[df["X4 Team Is Host"] == 1][target]
non_host = df[df["X4 Team Is Host"] == 0][target]
for name, g in [("host", host), ("non-host", non_host)]:
    m, lo, hi = mean_ci(g)
    write("%s: n = %d, mean = %.3f, 95%% CI [%.3f, %.3f]" % (name, len(g), m, lo, hi))
t2 = stats.ttest_ind(host, non_host, equal_var=False)
write("Welch t-test: t = %.3f, p = %.4f" % (t2.statistic, t2.pvalue))

# ---- Test 3: group stage vs knockout ----
write("\nTest 3: group stage vs knockout matches")
group = df[df["X7 Knockout Match"] == 0][target]
ko = df[df["X7 Knockout Match"] == 1][target]
for name, g in [("group stage", group), ("knockout", ko)]:
    m, lo, hi = mean_ci(g)
    write("%s: n = %d, mean = %.3f, 95%% CI [%.3f, %.3f]" % (name, len(g), m, lo, hi))
t3 = stats.ttest_ind(group, ko, equal_var=False)
write("Welch t-test: t = %.3f, p = %.4f" % (t3.statistic, t3.pvalue))

# ---- overall mean goals ----
write("\nOverall mean goals per team per match")
m, lo, hi = mean_ci(df[target])
write("mean = %.3f, 95%% CI [%.3f, %.3f]" % (m, lo, hi))

write("\nnote: the two rows of a match are not fully independent, so p-values are a bit optimistic")
out.close()
