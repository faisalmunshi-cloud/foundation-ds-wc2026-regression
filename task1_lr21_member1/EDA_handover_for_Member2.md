# EDA findings for LR 2.1 modelling (from Member 1)

- 104 rows, 8 features, no missing values.
- Target (goal difference): mean +0.83, SD 1.91, skew 0.00, range -4 to +6. Shapiro-Wilk p=0.0015 (not normal, as expected for integers). Positive mean because the Home slot tends to be the stronger/seeded team, so expect a positive intercept.
- Feature skewness is small (|skew| <= 0.51), so **no log or power transformation is needed**. Suggested transformations to test: standardise all features (needed for Ridge/Lasso/ElasticNet; X2 is in EUR millions with SD 553 vs X6 SD 0.38), and optionally a square-root-style signed transform of X2.
- Correlation with target: X1 +0.61, X2 +0.54, X4 +0.48, X7 +0.35, X5 +0.29, X8 +0.19, X6 +0.14, X3 -0.08.
- Significant at 5% (Pearson and Spearman): X1, X2, X4, X5, X7. Not significant: X3, X6. X8 borderline (p=0.054).
- Multicollinearity: X1-X2 r=0.67, X1-X4 r=0.72, X2-X4 r=0.80. VIFs: X4 3.8, X2 3.6, X1 3.0, others < 2. All below 5, but these three overlap (all measure team strength), so compare a reduced OLS dropping X4 and let Lasso/ElasticNet choose.
- X6 (host): only 15 of 104 matches involve a host; no significant effect. X6 and X8 are discrete (values -1..1 and -2..3).
- Files: lr21_dataset_104.csv, data_dictionary.csv, variable_justification.csv.
