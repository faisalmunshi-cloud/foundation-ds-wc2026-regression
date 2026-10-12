# LR 2.2 Data and EDA

Target: goals scored by one team in one match of the 2026 FIFA World Cup.
208 rows = 104 matches x 2 teams. 8 pre-match variables (4 shared with LR 2.1, 4 new).

## How to run (in this folder)
1. python build_lr22.py            -> lr22_dataset_208.csv
2. python eda_lr22.py              -> eda_outputs_lr22/ (charts, stats, correlations, VIF)
3. python inferential_tests_lr22.py -> eda_outputs_lr22/inferential_tests.txt + boxplot

Needs: pandas, numpy, matplotlib, scipy, statsmodels

## Files
- lr22_dataset_208.csv : final dataset
- data_dictionary_lr22.csv : meaning of every column
- variable_justification_lr22.csv : reason, expected effect and source for each variable
- shared_variable_register.csv : which variables overlap with LR 2.1 (4 shared, 4 new)
- qualifying_stats_45.csv : qualifying matches, goals for, goals against per team
- wc_appearances_48.csv : World Cup appearances before 2026
- worldcup2026_all_104_matches.csv, team_level_data_48.csv : match results and team data used as input
- verification_log.md : checks against a second source
- EDA_findings_for_modelling.md : main EDA results for the LR 2.2 model
- eda_outputs_lr22/ : all EDA charts and tables

## Assumptions
1. Qualifying stats use each team's final qualifying stage plus any play-offs (UEFA group + play-offs, CONMEBOL league, CAF group + play-offs, AFC third round onward, CONCACAF third round, OFC group + final round). Early rounds against very weak teams are left out. Rates are per match because teams played different numbers of games.
2. Hosts (USA, Mexico, Canada) did not play qualifiers, so they get the median rate of the 45 qualifiers (2.167 scored, 0.667 conceded).
3. Goals in extra time count, shoot-out goals do not.
4. The two rows of one match are not independent, so both rows of a match should stay in the same CV fold.
