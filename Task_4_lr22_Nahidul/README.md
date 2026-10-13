# Member 4: LR 2.2 Modelling + Cross-model Comparison

Member: Nahidul  
Assessment: HIT140 Assessment 3  
Role: LR 2.2 Modelling + Cross-model Comparison

## What this part does

`lr22_modelling_nahidul.py` completes the Member 4 modelling tasks using Member 3's final 208-row LR 2.2 dataset.

It:

1. Validates that the LR 2.2 dataset contains 208 rows, 104 matches and exactly 8 explanatory variables.
2. Uses repeated 5-fold grouped cross-validation, with both team rows from the same match kept in the same fold.
3. Compares Baseline, full OLS, transformed-feature OLS, reduced OLS, Ridge, Lasso and ElasticNet.
4. Tunes Ridge, Lasso and ElasticNet inside the training folds.
5. Tests raw goals against a `log(goals + 1)` target.
6. Runs a paired t-test comparing the selected model's fold RMSE with full OLS.
7. Produces OLS residual, Q-Q, VIF, Cook's distance and coefficient/95% CI diagnostics.
8. Generates out-of-fold LR 2.2 goal predictions.
9. Uses those predictions to calculate a predicted goal difference for each match.
10. Compares the LR 2.2-derived goal difference with Member 2's LR 2.1 direct goal-difference model.
11. Writes CSV tables, charts and a concise results handover for the report lead.

## Required existing files

Keep the repository structure like this:

```text
foundation-ds-wc2026-regression-main/
├── lr21_dataset_104.csv
├── lr22_member3/
│   └── lr22_dataset_208.csv
├── Task_2_lr21_Md_Foysal/
│   └── model_comparison.csv
├── Task_4_lr22_Nahidul/
│   ├── lr22_modelling_nahidul.py
│   ├── README.md
│   ├── member2_code_review.md
│   └── requirements_member4.txt
└── modelling_outputs_for_lr22_nahidul/
    └── generated after the script runs
```

## How to run

From anywhere, run:

```bash
python Task_4_lr22_Nahidul/lr22_modelling_nahidul.py
```

The script resolves paths relative to its own location, so it does not require the terminal to be opened in a specific folder.

## Main generated outputs

The script creates `modelling_outputs_for_lr22_nahidul/` containing:

- `model_comparison_lr22.csv`
- `target_transform_comparison.csv`
- `cv_fold_scores_lr22.csv`
- `paired_ttest_final_vs_ols.csv`
- `ols_coefficients_lr22.csv`
- `ols_full_summary_lr22.txt`
- `vif_lr22.csv`
- `influential_observations_lr22.csv`
- `lr22_oof_predictions_208.csv`
- `final_model_coefficients_lr22.csv`
- `cross_model_predictions_104.csv`
- `cross_model_summary.csv`
- `cross_model_paired_test.csv`
- `final_recommendation.txt`
- `results_summary_for_report.md`
- nine PNG charts

## Current run result

With the repository data supplied for this assessment, the selected LR 2.2 model is Ridge regression on the raw goals target. The exact metrics are stored in `final_recommendation.txt` and the CSV output files. Re-run the script after any dataset change rather than copying the old numbers.
