# HIT140 Assessment 3 - LR 2.2 dataset
# target = goals scored by one team in one match (104 matches x 2 = 208 rows)

import pandas as pd

# load files
matches = pd.read_csv("worldcup2026_all_104_matches.csv")
teams = pd.read_csv("team_level_data_48.csv")
qual = pd.read_csv("qualifying_stats_45.csv")
apps = pd.read_csv("wc_appearances_48.csv")

# team name as index for easy lookup
teams = teams.set_index("Team")
qual = qual.set_index("Team")
apps = apps.set_index("Team")

# qualifying goals per match
qual["GF_per_match"] = qual["Qual_GF"] / qual["Qual_Matches"]
qual["GA_per_match"] = qual["Qual_GA"] / qual["Qual_Matches"]

# hosts had no qualifiers so use median of other teams
median_gf = qual["GF_per_match"].median()
median_ga = qual["GA_per_match"].median()
for host in ["USA", "Mexico", "Canada"]:
    qual.loc[host, "GF_per_match"] = median_gf
    qual.loc[host, "GA_per_match"] = median_ga

print("median GF per match for hosts:", round(median_gf, 3))
print("median GA per match for hosts:", round(median_ga, 3))

# check all 48 teams are in every file
all_teams = set(matches["Home"]) | set(matches["Away"])
print("number of teams:", len(all_teams))
for t in all_teams:
    if t not in teams.index or t not in qual.index or t not in apps.index:
        print("missing team:", t)

# make 2 rows for every match
rows = []
for i in range(len(matches)):
    m = matches.iloc[i]

    if m["Stage"] == "Group stage":
        knockout = 0
    else:
        knockout = 1

    # first row = home team, second row = away team
    for side in ["Home", "Away"]:
        if side == "Home":
            team = m["Home"]
            opp = m["Away"]
            goals = m["Home Score"]
        else:
            team = m["Away"]
            opp = m["Home"]
            goals = m["Away Score"]

        row = {
            "Row ID": len(rows) + 1,
            "Match #": i + 1,
            "Stage": m["Stage"],
            "Group": m["Group"],
            "Date": m["Date"],
            "Fixture Slot": side + " slot",
            "Team": team,
            "Opponent": opp,
            "Goals Scored (Target)": goals,
            "X1 Team FIFA Rank": teams.loc[team, "FIFA_Rank_11Jun2026"],
            "X2 Opponent FIFA Rank": teams.loc[opp, "FIFA_Rank_11Jun2026"],
            "X3 Team Squad Value EURm": teams.loc[team, "Squad_Value_EURm"],
            "X4 Team Is Host": teams.loc[team, "Is_Host"],
            "X5 Team Qual Goals Scored per Match": round(qual.loc[team, "GF_per_match"], 3),
            "X6 Opponent Qual Goals Conceded per Match": round(qual.loc[opp, "GA_per_match"], 3),
            "X7 Knockout Match": knockout,
            "X8 Team WC Appearances Before 2026": apps.loc[team, "WC_Appearances_Before_2026"],
        }
        rows.append(row)

df = pd.DataFrame(rows)

# quick checks
print("rows:", len(df))
print("missing values:", df.drop(columns=["Group"]).isnull().sum().sum())

# goals of both rows should add up to the real match score
total_check = df.groupby("Match #")["Goals Scored (Target)"].sum()
real_total = matches["Home Score"] + matches["Away Score"]
print("match totals correct:", list(total_check) == list(real_total))

df.to_csv("lr22_dataset_208.csv", index=False)
print("saved lr22_dataset_208.csv")
