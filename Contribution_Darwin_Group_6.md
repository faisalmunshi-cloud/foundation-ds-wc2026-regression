# Individual Contribution Record

**Unit:** HIT140 Foundations of Data Science
**Assessment 3:** Group Project Report (Objective 2)
**Group:** HIT140 2026 Darwin Group 6
**Prepared by:** Nasid Alve (report lead), checked by all members

This record shows who did what in our group. All of our work can be checked in our GitHub repository (commit history) and in our MS Teams channel, where we held and recorded our meetings.

---

## Summary

| Member | Name | Student ID | Role | Folder in GitHub |
|---|---|---|---|---|
| 1 | Jahid Hasan | S395084 | LR 2.1 data and EDA | `task1_lr21_member1` |
| 2 | Md Foysal | S395914 | LR 2.1 modelling, GitHub repo owner | `Task_2_lr21_Md_Foysal`, `modelling_outputs_for_lr21_md_foysal` |
| 3 | Nasid Alve | S396257 | LR 2.2 data and EDA, report lead | `lr22_member3` |
| 4 | Nahidul Islam Nadim | S396122 | LR 2.2 modelling and cross-model comparison | `Task_4_lr22_Nahidul`, `modelling_outputs_for_lr22_nahidul` |

---

## Member 1: Jahid Hasan (S395084)

Jahid prepared the data for Linear Regression 2.1. He built the 104-row dataset where each row is one match and the target is the goal difference. He chose the 8 pre-match variables and wrote the variable justification and data dictionary. He also did the EDA for 2.1 (summary statistics, distributions, correlation matrix, VIF and scatter plots) and ran some inferential tests. He wrote a handover note so Foysal could start the modelling. His work was added through a pull request, which Foysal reviewed and merged.

**Main files:** `build_matches.py`, `build_lr21.py`, `team_data.py`, `eda_lr21.py`, `inferential_tests.py`, `lr21_dataset_104.csv`, `variable_justification.csv`, `data_dictionary.csv`, `EDA_handover_for_Member2.md`

---

## Member 2: Md Foysal (S395914)

Foysal created our GitHub repository, added everyone and merged Jahid's pull request. For Linear Regression 2.1 he used Jahid's dataset and compared seven models: a mean baseline, full OLS, OLS with transformed features, a reduced OLS, Ridge, Lasso and ElasticNet. He used cross-validation and compared them with MAE, RMSE and R². Ridge (alpha = 50) gave the lowest RMSE. He also made the charts for this part, including model comparison, predicted vs actual, residual checks, Cook's distance, coefficients, Lasso paths and the biggest upsets.

**Main files:** `lr21_modelling.py`, `model_comparison.csv`, `ols_coefficients.csv`, `ols_full_summary.txt`, and the charts in `modelling_outputs_for_lr21_md_foysal`

---

## Member 3: Nasid Alve (S396257)

I prepared the data for Linear Regression 2.2. I built the 208-row dataset, where each match appears twice (once for each team) and the target is goals scored. I chose 8 variables that are all known before kick-off: team FIFA rank, opponent FIFA rank, squad value, host nation, qualifying goals scored per match, opponent's qualifying goals conceded per match, knockout match and past World Cup appearances. I kept a shared-variable register to make sure only 4 variables overlap with 2.1, as the brief allows.

For EDA I did summary statistics, distributions, a correlation matrix, VIF and scatter plots. I also ran t-tests with 95% confidence intervals: teams against lower-ranked opponents vs higher-ranked opponents, hosts vs non-hosts, and group stage vs knockout. I wrote my EDA findings for Nahidul to use in the modelling.

As report lead, I divided the tasks, chaired our meetings, merged everyone's sections into the final report, and handled Turnitin and the submission.

**Main files:** `build_lr22.py`, `eda_lr22.py`, `inferential_tests_lr22.py`, `lr22_dataset_208.csv`, `variable_justification_lr22.csv`, `data_dictionary_lr22.csv`, `shared_variable_register.csv`, `EDA_findings_for_modelling.md`, `verification_log.md`

---

## Member 4: Nahidul Islam Nadim (S396122)

Nahidul did the modelling for Linear Regression 2.2 using my dataset. He compared the same seven models as Foysal, using repeated 5-fold grouped cross-validation so that both rows of the same match always stay in the same fold. He also tested a log(goals + 1) target, but the normal goals target worked better. Ridge was the best model, and a paired t-test showed it was significantly better than plain OLS (p = 0.0196). He checked the model with residual plots, a Q-Q plot, VIF, Cook's distance and confidence intervals for the coefficients. He then compared the goal difference predicted from his model with Foysal's 2.1 model. He also reviewed Foysal's modelling code.

**Main files:** `lr22_modelling_nahidul.py`, `member2_code_review.md`, `requirements_member4.txt`, and the results and charts in `modelling_outputs_for_lr22_nahidul`

---

## How we worked together

- We used our MS Teams channel for all discussion and updates, and we recorded our meetings and kept minutes (see the `meeting_minutes` folder).
- Everyone pushed their own work to GitHub from their own account, so the commit history shows each person's part.
- We reviewed each other's work: Foysal reviewed and merged Jahid's pull request, and Nahidul reviewed Foysal's code.

## Agreement

We have all read this record and agree that it shows each member's contribution correctly.

| Member | Name | Agreed |
|---|---|---|
| 1 | Jahid Hasan | |
| 2 | Md Foysal | |
| 3 | Nasid Alve | |
| 4 | Nahidul Islam Nadim | |
