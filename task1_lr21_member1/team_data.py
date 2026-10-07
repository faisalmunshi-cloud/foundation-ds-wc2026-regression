# Team-level raw data for the 48 teams at the 2026 FIFA World Cup
# Sourced per agent research (see Sources sheet in final workbook)

# 1. FIFA Ranking position as of 11 June 2026 (lower = better)
fifa_rank = {
    "Argentina": 1, "Spain": 2, "France": 3, "England": 4, "Portugal": 5,
    "Brazil": 6, "Morocco": 7, "Netherlands": 8, "Belgium": 9, "Germany": 10,
    "Croatia": 11, "Colombia": 13, "Mexico": 14, "Senegal": 15, "Uruguay": 16,
    "USA": 17, "Japan": 18, "Switzerland": 19, "IR Iran": 20, "Turkiye": 22,
    "Ecuador": 23, "Austria": 24, "South Korea": 25, "Australia": 27, "Algeria": 28,
    "Egypt": 29, "Canada": 30, "Norway": 31, "Cote d'Ivoire": 33, "Panama": 34,
    "Sweden": 38, "Czechia": 40, "Paraguay": 41, "Scotland": 42, "Tunisia": 45,
    "DR Congo": 46, "Uzbekistan": 50, "Qatar": 56, "Iraq": 57, "South Africa": 60,
    "Saudi Arabia": 61, "Jordan": 63, "Bosnia and Herzegovina": 64, "Cabo Verde": 67,
    "Ghana": 73, "Curacao": 82, "Haiti": 83, "New Zealand": 85,
}

# 2. Squad market value in EUR millions (Transfermarkt-sourced, via Planet Football)
squad_value_m = {
    "France": 1520, "England": 1360, "Spain": 1220, "Portugal": 1010, "Germany": 947,
    "Brazil": 928.2, "Argentina": 807.5, "Netherlands": 754.2, "Norway": 589.9,
    "Belgium": 547.5, "Cote d'Ivoire": 522.1, "Senegal": 478.1, "Turkiye": 473.7,
    "Morocco": 447.7, "Sweden": 406.08, "Croatia": 387.3, "USA": 385.6, "Ecuador": 368.7,
    "Uruguay": 359.3, "Switzerland": 332.5, "Colombia": 302.35, "Japan": 270.85,
    "Algeria": 256.9, "Austria": 245.2, "Ghana": 234.5, "Canada": 198.65, "Mexico": 191.85,
    "Czechia": 188.18, "Scotland": 170.25, "Paraguay": 153.65, "Bosnia and Herzegovina": 146.4,
    "DR Congo": 143.9, "South Korea": 139.05, "Egypt": 116.48, "Uzbekistan": 85.33,
    "Australia": 77.45, "Tunisia": 69.95, "Haiti": 55.9, "Cabo Verde": 49.25,
    "South Africa": 49.25, "Saudi Arabia": 40.68, "Panama": 34.55, "New Zealand": 34.45,
    "IR Iran": 32.05, "Curacao": 25.78, "Iraq": 21.2, "Jordan": 20.3, "Qatar": 19.93,
}

# 3. Average squad age (years)
squad_age = {
    "France": 26.58, "England": 26.62, "Spain": 26.19, "Portugal": 27.54, "Germany": 27.54,
    "Brazil": 28.65, "Argentina": 28.62, "Netherlands": 27.27, "Norway": 26.35,
    "Belgium": 27.12, "Cote d'Ivoire": 25.35, "Senegal": 26.62, "Turkiye": 27.23,
    "Morocco": 25.92, "Sweden": 27.00, "Croatia": 27.88, "USA": 26.42, "Ecuador": 25.58,
    "Uruguay": 28.19, "Switzerland": 27.81, "Colombia": 29.58, "Japan": 27.19,
    "Algeria": 26.46, "Austria": 28.12, "Ghana": 26.42, "Canada": 26.42, "Mexico": 27.50,
    "Czechia": 27.23, "Scotland": 28.73, "Paraguay": 28.54, "Bosnia and Herzegovina": 25.92,
    "DR Congo": 28.50, "South Korea": 27.46, "Egypt": 28.69, "Uzbekistan": 27.96,
    "Australia": 26.88, "Tunisia": 26.15, "Haiti": 27.08, "Cabo Verde": 29.23,
    "South Africa": 26.35, "Saudi Arabia": 27.96, "Panama": 30.00, "New Zealand": 27.62,
    "IR Iran": 29.81, "Curacao": 27.54, "Iraq": 26.65, "Jordan": 28.08, "Qatar": 28.92,
}

# 4. Previous World Cup best-ever result, converted to ordinal score
# Winner=8, Runner-up=7, Third=6, Fourth=5, QF=4, R16=3, Group stage=2, Never qualified=1
_result_score = {
    "Winner": 8, "Runner-up": 7, "Third place": 6, "Fourth place": 5,
    "Quarter-final": 4, "Round of 16": 3, "Group stage": 2, "Never qualified": 1,
}
best_result_category = {
    "Mexico": "Quarter-final", "South Africa": "Group stage", "South Korea": "Fourth place",
    "Czechia": "Runner-up", "Canada": "Group stage", "Bosnia and Herzegovina": "Group stage",
    "Switzerland": "Quarter-final", "Qatar": "Group stage", "Brazil": "Winner",
    "Morocco": "Fourth place", "Scotland": "Group stage", "Haiti": "Group stage",
    "USA": "Third place", "Australia": "Round of 16", "Paraguay": "Round of 16",
    "Turkiye": "Third place", "Germany": "Winner", "Cote d'Ivoire": "Group stage",
    "Ecuador": "Round of 16", "Curacao": "Never qualified", "Netherlands": "Runner-up",
    "Japan": "Round of 16", "Sweden": "Runner-up", "Tunisia": "Group stage",
    "Belgium": "Third place", "Egypt": "Group stage", "IR Iran": "Group stage",
    "New Zealand": "Group stage", "Spain": "Winner", "Cabo Verde": "Never qualified",
    "Saudi Arabia": "Round of 16", "Uruguay": "Winner", "France": "Winner",
    "Norway": "Round of 16", "Senegal": "Quarter-final", "Iraq": "Group stage",
    "Argentina": "Winner", "Austria": "Third place", "Algeria": "Group stage",
    "Jordan": "Never qualified", "Portugal": "Third place", "Colombia": "Quarter-final",
    "DR Congo": "Group stage", "Uzbekistan": "Never qualified", "England": "Winner",
    "Croatia": "Runner-up", "Ghana": "Quarter-final", "Panama": "Group stage",
}
best_result_ordinal = {t: _result_score[c] for t, c in best_result_category.items()}

# 5. Qualifying campaign goal difference (hosts = no qualifying -> 0, documented assumption)
qualifying_gd = {
    "Mexico": 0, "USA": 0, "Canada": 0,  # hosts - no qualifying campaign (assumption: neutral 0)
    "Germany": 13, "Switzerland": 12, "Scotland": 6, "Turkiye": 7, "Spain": 19,
    "Portugal": 13, "Netherlands": 23, "Bosnia and Herzegovina": 10, "Norway": 32,
    "Belgium": 22, "England": 22, "Croatia": 22, "Czechia": 10, "Sweden": -5,
    "Austria": 18, "France": 12, "Argentina": 21, "Ecuador": 9, "Colombia": 10,
    "Uruguay": 10, "Brazil": 7, "Paraguay": 4, "South Africa": 6, "Morocco": 20,
    "Cote d'Ivoire": 25, "Tunisia": 22, "Egypt": 18, "Cabo Verde": 8, "Senegal": 19,
    "Algeria": 16, "Ghana": 17, "DR Congo": 7, "South Korea": 13, "Qatar": -6,
    "Australia": 9, "Japan": 27, "IR Iran": 11, "Saudi Arabia": 0, "Jordan": 8,
    "Iraq": 4, "Uzbekistan": 7, "Haiti": 3, "Curacao": 10, "Panama": 5, "New Zealand": 18,
}

# 6. Host nations (true home-soil advantage)
HOSTS = {"USA", "Mexico", "Canada"}

# 7. Confederation assignment (documented ordinal strength index: UEFA=6, CONMEBOL=5, CAF=4,
#    CONCACAF=3, AFC=2, OFC=1 -- based on historical World Cup performance tiers)
confederation = {
    # UEFA (6)
    "Czechia": "UEFA", "Bosnia and Herzegovina": "UEFA", "Switzerland": "UEFA", "Scotland": "UEFA",
    "Turkiye": "UEFA", "Germany": "UEFA", "Netherlands": "UEFA", "Sweden": "UEFA", "Belgium": "UEFA",
    "Spain": "UEFA", "France": "UEFA", "Norway": "UEFA", "Austria": "UEFA", "Portugal": "UEFA",
    "Croatia": "UEFA", "England": "UEFA",
    # CONMEBOL (5)
    "Brazil": "CONMEBOL", "Paraguay": "CONMEBOL", "Ecuador": "CONMEBOL", "Argentina": "CONMEBOL",
    "Colombia": "CONMEBOL", "Uruguay": "CONMEBOL",
    # CAF (4)
    "South Africa": "CAF", "Morocco": "CAF", "Cote d'Ivoire": "CAF", "Egypt": "CAF",
    "Cabo Verde": "CAF", "Senegal": "CAF", "Algeria": "CAF", "DR Congo": "CAF",
    "Ghana": "CAF", "Tunisia": "CAF",
    # CONCACAF (3)
    "Mexico": "CONCACAF", "Canada": "CONCACAF", "USA": "CONCACAF", "Curacao": "CONCACAF",
    "Haiti": "CONCACAF", "Panama": "CONCACAF",
    # AFC (2)
    "South Korea": "AFC", "Qatar": "AFC", "Japan": "AFC", "IR Iran": "AFC",
    "Saudi Arabia": "AFC", "Iraq": "AFC", "Uzbekistan": "AFC", "Jordan": "AFC", "Australia": "AFC",
    # OFC (1)
    "New Zealand": "OFC",
}
conf_strength = {"UEFA": 6, "CONMEBOL": 5, "CAF": 4, "CONCACAF": 3, "AFC": 2, "OFC": 1}

ALL_TEAMS = set(fifa_rank) | set(squad_value_m) | set(squad_age) | set(best_result_ordinal) | set(qualifying_gd) | set(confederation)
assert len(ALL_TEAMS) == 48, f"Expected 48 teams, got {len(ALL_TEAMS)}: missing/extra check needed"
for d, name in [(fifa_rank,"fifa_rank"), (squad_value_m,"squad_value_m"), (squad_age,"squad_age"),
                (best_result_ordinal,"best_result_ordinal"), (qualifying_gd,"qualifying_gd"),
                (confederation,"confederation")]:
    missing = ALL_TEAMS - set(d)
    assert not missing, f"{name} missing: {missing}"
print("All 48 teams present and consistent across all 6 data dicts.")
