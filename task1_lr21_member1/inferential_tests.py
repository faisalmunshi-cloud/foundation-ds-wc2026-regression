import pandas as pd, numpy as np
from scipy import stats
df = pd.read_csv("lr21_dataset_104.csv")
T = "Goal Difference (Target)"
F = [c for c in df.columns if c.startswith("X") and c[1].isdigit() and " " in c and c.split()[0] in [f"X{i}" for i in range(1,9)]]
F = F[:8]
rows=[]
n=len(df)
for f in F:
    r,p = stats.pearsonr(df[f], df[T])
    z=np.arctanh(r); se=1/np.sqrt(n-3)
    lo,hi=np.tanh(z-1.96*se),np.tanh(z+1.96*se)
    rho,ps = stats.spearmanr(df[f], df[T])
    rows.append([f.split(" ")[0], f, round(r,3), round(lo,3), round(hi,3), round(p,5), round(rho,3), round(ps,5)])
corr = pd.DataFrame(rows, columns=["Var","Feature","Pearson r","CI95 low","CI95 high","p (Pearson)","Spearman rho","p (Spearman)"])
print(corr.to_string(index=False))
corr.to_csv("eda_outputs/correlation_significance.csv", index=False)

# Host t-test: matches where a host plays (X6 != 0) vs not. Use host-perspective GD.
h = df[df["X6 Host Advantage Diff (Home-Away)"]!=0].copy()
h["host_gd"] = h[T]*h["X6 Host Advantage Diff (Home-Away)"]  # + = host better
print("\nHost matches:", len(h), " mean host GD:", round(h.host_gd.mean(),3))
t1 = stats.ttest_1samp(h.host_gd, 0)
print("One-sample t-test (host GD vs 0): t=%.3f p=%.4f"%(t1.statistic,t1.pvalue))
# Host vs non-host: compare |relative|? Use Welch on raw GD from home perspective: matches with host as home vs non-host matches
nonhost = df[df["X6 Host Advantage Diff (Home-Away)"]==0][T]
hostmatches = h["host_gd"]
w = stats.ttest_ind(hostmatches, nonhost, equal_var=False)
print("Welch t-test host-perspective GD (n=%d, mean %.2f) vs non-host home-perspective GD (n=%d, mean %.2f): t=%.3f p=%.4f"%(len(hostmatches),hostmatches.mean(),len(nonhost),nonhost.mean(),w.statistic,w.pvalue))
# Fairer control: host effect after controlling for strength = regression
import statsmodels.api as sm
X = sm.add_constant(df[["X1 FIFA Rank Diff (Away-Home)","X6 Host Advantage Diff (Home-Away)"]])
m = sm.OLS(df[T], X).fit()
print("\nOLS GD ~ FIFA rank diff + host:"); print(m.summary().tables[1])
# skewness / transformation notes
print("\nSkew:", df[[T]+F].skew().round(2).to_dict())
print("Home-perspective mean GD (all matches): %.3f  one-sample t vs 0: p=%.4f"%(df[T].mean(), stats.ttest_1samp(df[T],0).pvalue))
