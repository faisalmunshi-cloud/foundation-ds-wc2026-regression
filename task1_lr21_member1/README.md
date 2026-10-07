# LR 2.1 - Data + EDA (Member 1)

Target: goal difference (Home - Away) for all 104 matches of the 2026 FIFA World Cup. 8 pre-match features.

## Run order (from this folder)
1. `python build_matches.py`   -> worldcup2026_all_104_matches.csv (72 group + 32 knockout, verified)
2. `python build_lr21.py`      -> lr21_dataset_104.csv (target + X1..X8 + raw team values; uses team_data.py)
3. `python eda_lr21.py`        -> eda_outputs/ (charts, descriptive stats, VIF)
4. `python inferential_tests.py` -> correlation significance, host t-tests

## Files
- `data_dictionary.csv`, `variable_justification.csv` (reason, expected sign, source per variable)
- `team_level_data_48.csv`: one row per team, for LR 2.2 (Member 3) to reuse
- `2026_World_Cup_Full_Dataset.xlsx`: everything in one workbook

## Verification
Group scores were checked against official group standings (GF/GA per team); 12 scores were corrected after cross-checking news reports.
FIFA rank = official 11 Jun 2026 release. Squad values: Transfermarkt via Planet Football (see limitation in the workbook).
