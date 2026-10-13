# Member 4 LR 2.2 modelling results summary

This file is a factual results handover for Member 3. Rewrite the final report section in the group's own words.

## Validation design
- 10 repeats of 5-fold grouped cross-validation were used.
- Both team rows from the same match were kept in the same fold using Match # as the group.
- Ridge, Lasso and ElasticNet hyperparameters were tuned inside the training portion of each outer fold.

## Raw-target model comparison
- Best raw-target model by mean CV RMSE: Ridge.
- Mean CV MAE: 0.9215.
- Mean CV RMSE: 1.2045.
- Mean CV R2: 0.2226.

## Raw goals vs log(goals + 1)
- Selected final formulation: Ridge (raw).
- Mean CV MAE: 0.9215.
- Mean CV RMSE: 1.2045.
- Mean CV R2: 0.2226.

## Paired model comparison
- Paired fold-RMSE t-test against raw full OLS: t=2.4133, p=0.0196.
- At alpha=0.05, the difference is statistically significant.

## OLS diagnostics
- Maximum VIF: 3.167.
- Cook's-distance threshold 4/n: 0.0192.
- Observations above the Cook's-distance threshold: 12.
- See the residual, Q-Q, VIF, Cook's-distance and coefficient output files for interpretation.

## Cross-model comparison
- LR 2.1 model used: Ridge (a=50).
- LR 2.1 direct goal-difference RMSE: 1.5141.
- LR 2.2-derived goal-difference RMSE: 1.5638.
- Lower-RMSE approach: LR 2.1 direct (Ridge (a=50)).
- Paired absolute-error test: t=-0.6950, p=0.4886.

## Limitations to mention
- Ordinary linear regression can produce fractional goals and, for raw-target models, negative predictions.
- The target is a non-negative integer count, so linear regression is an approximation rather than a count-data model.
- Each match contributes two related team rows; grouped validation was therefore used to avoid putting opposite sides of the same match into train and test folds.
- The dataset contains only 104 matches (208 team rows), so coefficient estimates and validation results can be sensitive to influential matches.