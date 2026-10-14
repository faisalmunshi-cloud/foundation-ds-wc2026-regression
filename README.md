# foundation-ds-wc2026-regression

HIT140 Foundations of Data Science – Assessment 3 (Objective 2), **HIT140 2026 Darwin Group 6**.

Linear regression analysis of the FIFA World Cup 2026:

- **LR 2.1** predicts the goal difference of a match (104 rows, one per match)
- **LR 2.2** predicts the goals scored by a team in a match (208 rows, one per team per match)

Each model uses exactly 8 explanatory variables that are known before kick-off, and only 4 variables are shared between the two models.

## Team

| Member | Name | Student ID | Part | Folder |
|---|---|---|---|---|
| 1 | Jahid Hasan | S395084 | LR 2.1 data and EDA | `task1_lr21_member1` |
| 2 | Md Foysal | S395914 | LR 2.1 modelling | `Task_2_lr21_Md_Foysal`, `modelling_outputs_for_lr21_md_foysal` |
| 3 | Nasid Alve | S396257 | LR 2.2 data and EDA, report lead | `lr22_member3` |
| 4 | Nahidul Islam Nadim | S396122 | LR 2.2 modelling and cross-model comparison | `Task_4_lr22_Nahidul`, `modelling_outputs_for_lr22_nahidul` |

More detail: [Contribution_Darwin_Group_6.md](Contribution_Darwin_Group_6.md) · Meeting records: [meeting_minutes](meeting_minutes)

## Datasets

- `lr21_dataset_104.csv` (also in `task1_lr21_member1/`) – LR 2.1 dataset
- `lr22_member3/lr22_dataset_208.csv` – LR 2.2 dataset

## How to run

Install the libraries:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

Run the scripts in this order:

```bash
# 1. LR 2.1 data and EDA (run inside the folder)
cd task1_lr21_member1
python build_lr21.py
python eda_lr21.py
python inferential_tests.py
cd ..

# 2. LR 2.1 modelling (run from the repository root)
python Task_2_lr21_Md_Foysal/lr21_modelling.py

# 3. LR 2.2 data and EDA (run inside the folder)
cd lr22_member3
python build_lr22.py
python eda_lr22.py
python inferential_tests_lr22.py
cd ..

# 4. LR 2.2 modelling and cross-model comparison (can run from anywhere)
python Task_4_lr22_Nahidul/lr22_modelling_nahidul.py
```

Results, tables and charts are saved in each member's output folder.
