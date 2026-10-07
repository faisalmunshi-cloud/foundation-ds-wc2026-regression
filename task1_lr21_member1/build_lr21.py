import csv
from datetime import date
from team_data import (fifa_rank, squad_value_m, squad_age, best_result_ordinal,
                        qualifying_gd, HOSTS, confederation, conf_strength)

rows = []
with open("worldcup2026_all_104_matches.csv", newline="", encoding="utf-8") as f:
    r = csv.reader(f)
    header = next(r)
    for row in r:
        rows.append(row)

def parse_date(s):
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))

# ---- Rest days: days since each team's previous match in this tournament.
# For a team's first match of the tournament, use a fixed baseline (14 days,
# representing standard pre-tournament training-camp assembly) -- documented assumption.
BASELINE_REST = 14
last_played = {}  # team -> date of last match played
match_dates_sorted = sorted(range(len(rows)), key=lambda i: parse_date(rows[i][2]))

rest_days = {}  # (row_index, 'home'/'away') -> rest days
for i in match_dates_sorted:
    stage, group, d, home, hs, as_, away, notes = rows[i]
    dt = parse_date(d)
    for side, team in (("home", home), ("away", away)):
        if team in last_played:
            rest = (dt - last_played[team]).days
        else:
            rest = BASELINE_REST
        rest_days[(i, side)] = rest
    last_played[home] = dt
    last_played[away] = dt

out_rows = []
for i, row in enumerate(rows):
    stage, group, d, home, hs, as_, away, notes = row
    hs, as_ = int(hs), int(as_)
    target_gd = hs - as_

    f_rank_diff = fifa_rank[away] - fifa_rank[home]          # + favors home (home ranked better)
    f_value_diff = squad_value_m[home] - squad_value_m[away]  # + favors home
    f_age_diff = squad_age[home] - squad_age[away]
    f_result_diff = best_result_ordinal[home] - best_result_ordinal[away]
    f_qualgd_diff = qualifying_gd[home] - qualifying_gd[away]
    f_host_diff = (1 if home in HOSTS else 0) - (1 if away in HOSTS else 0)
    f_conf_diff = conf_strength[confederation[home]] - conf_strength[confederation[away]]
    f_rest_diff = rest_days[(i, "home")] - rest_days[(i, "away")]

    out_rows.append([
        i + 1, stage, group, d, home, away, hs, as_, target_gd,
        f_rank_diff, f_value_diff, round(f_age_diff, 2), f_result_diff,
        f_qualgd_diff, f_host_diff, f_conf_diff, f_rest_diff,
        fifa_rank[home], fifa_rank[away],
        squad_value_m[home], squad_value_m[away],
        squad_age[home], squad_age[away],
        best_result_ordinal[home], best_result_ordinal[away],
        qualifying_gd[home], qualifying_gd[away],
        confederation[home], confederation[away],
        rest_days[(i, "home")], rest_days[(i, "away")],
    ])

header_out = [
    "Match #","Stage","Group","Date","Home","Away","Home Score","Away Score","Goal Difference (Target)",
    "X1 FIFA Rank Diff (Away-Home)","X2 Squad Value Diff EURm (Home-Away)","X3 Squad Age Diff (Home-Away)",
    "X4 Prior WC Best Result Diff (Home-Away)","X5 Qualifying GD Diff (Home-Away)",
    "X6 Host Advantage Diff (Home-Away)","X7 Confederation Strength Diff (Home-Away)",
    "X8 Rest Days Diff (Home-Away)",
    "Home FIFA Rank","Away FIFA Rank","Home Squad Value EURm","Away Squad Value EURm",
    "Home Squad Age","Away Squad Age","Home Prior-WC-Best (ordinal)","Away Prior-WC-Best (ordinal)",
    "Home Qualifying GD","Away Qualifying GD","Home Confederation","Away Confederation",
    "Home Rest Days","Away Rest Days",
]

with open("lr21_dataset_104.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header_out)
    w.writerows(out_rows)

print(f"Wrote {len(out_rows)} rows, {len(header_out)} columns.")
