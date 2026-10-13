# Member 4 peer review of Member 2 LR 2.1 modelling code

Reviewed file: `Task_2_lr21_Md_Foysal/lr21_modelling.py`

## What is already done well

- The code compares the full set of models assigned in the task division: baseline, full OLS, transformed OLS, reduced OLS, Ridge, Lasso and ElasticNet.
- Standardisation is correctly included before Ridge, Lasso and ElasticNet.
- MAE, RMSE and R² are reported.
- The script produces residual diagnostics, Q-Q analysis, VIF, Cook's distance and coefficient confidence intervals.
- The output files are clearly separated into a modelling output folder.
- The model comparison identifies Ridge with alpha 50 as the lowest-RMSE LR 2.1 model in the supplied output table.

## Items to check before final submission

1. Hyperparameter tuning is performed on the complete 104-row dataset before the repeated cross-validation comparison. This means information from later validation folds can indirectly influence the selected alpha. A stricter evaluation would tune the hyperparameters only inside each outer training fold using nested cross-validation.

2. The predicted-vs-actual and residual diagnostic plots are based on an OLS model fitted on all 104 rows. These plots are useful for model diagnostics, but they are in-sample plots and should not be described as independent predictive performance. The repeated cross-validation metrics are the stronger evidence for generalisation.

3. The LR 2.1 code uses ordinary repeated K-fold CV because there is only one row per match. Member 4's LR 2.2 code must use grouped splitting because LR 2.2 contains two rows for every match. This difference is intentional.

4. The final report should distinguish between the predictive model selected by cross-validation and the full OLS model used for coefficient inference and regression diagnostics.

These points do not remove the completed LR 2.1 work. They are checks to make the evaluation wording and methodology more defensible.
