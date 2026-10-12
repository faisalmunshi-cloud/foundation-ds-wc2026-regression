# LR 2.2 EDA findings for modelling

- 208 rows, 8 features, no missing values.
- Target (goals scored): mean 1.48, variance 1.94, range 0 to 7. Most rows are 0 goals (55) or 1 goal (71).
- Target is right skewed (skew 1.15). log(goals + 1) brings skew down to 0.03, so a log(goals + 1) target is worth testing against raw goals. Shapiro-Wilk is still significant for both (p < 0.001), which is normal for small integer counts.
- Correlation with goals (Pearson): X3 squad value +0.38, X1 team rank -0.37, X2 opponent rank +0.35, X8 WC appearances +0.34, X5 qualifying goals per match +0.26 (all p < 0.001). X4 host +0.10, X6 opponent qualifying goals conceded +0.09 and X7 knockout -0.01 are not significant.
- Multicollinearity: X1, X3 and X8 overlap (|r| about 0.70 to 0.72). VIF max 3.17, so below 5, but a reduced model (for example without X8) and Lasso / ElasticNet are worth comparing.
- Skewed features: X3 squad value (skew 1.13), X5 and X6 (about 1.0 to 1.1). A log transform of X3 is a good candidate for the "transformed features" model. Standardise all features for Ridge / Lasso / ElasticNet (X3 is in millions, X4 and X7 are 0/1).
- Inferential test: teams facing a lower ranked opponent scored 2.12 goals on average vs 0.85 against a higher ranked opponent (difference 1.27, 95% CI 0.93 to 1.61). Welch t = 7.37, paired t-test by match t = 7.87, both p < 0.001.
- Host vs non-host (2.00 vs 1.44, p = 0.17) and group vs knockout (1.49 vs 1.45, p = 0.84) are not significant.
- Each match gives 2 rows. Use GroupKFold (or group by "Match #") in cross-validation so both rows of a match are in the same fold.
- Linear regression can predict negative or fractional goals. Worth mentioning in limitations.
