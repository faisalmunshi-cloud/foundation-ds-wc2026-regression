# Build verified full 104-match dataset for 2026 FIFA World Cup
import csv

matches = []

def add(stage, group, date, home, hs, aways, away, notes=""):
    matches.append([stage, group, date, home, hs, aways, away, notes])

# ---------------- GROUP STAGE (72 matches) - verified against final standings ----------------
# Group A
add("Group stage","A","2026-06-11","Mexico",2,0,"South Africa")
add("Group stage","A","2026-06-12","South Korea",2,1,"Czechia")
add("Group stage","A","2026-06-18","Czechia",1,1,"South Africa")
add("Group stage","A","2026-06-18","Mexico",1,0,"South Korea")
add("Group stage","A","2026-06-24","Czechia",0,3,"Mexico")
add("Group stage","A","2026-06-24","South Africa",1,0,"South Korea")

# Group B
add("Group stage","B","2026-06-12","Canada",1,1,"Bosnia and Herzegovina")
add("Group stage","B","2026-06-13","Qatar",1,1,"Switzerland")
add("Group stage","B","2026-06-18","Switzerland",4,1,"Bosnia and Herzegovina")
add("Group stage","B","2026-06-19","Canada",6,0,"Qatar")
add("Group stage","B","2026-06-24","Switzerland",2,1,"Canada")
add("Group stage","B","2026-06-24","Bosnia and Herzegovina",3,1,"Qatar")

# Group C
add("Group stage","C","2026-06-13","Brazil",1,1,"Morocco")
add("Group stage","C","2026-06-14","Haiti",0,1,"Scotland")
add("Group stage","C","2026-06-19","Scotland",0,1,"Morocco")
add("Group stage","C","2026-06-20","Brazil",3,0,"Haiti")
add("Group stage","C","2026-06-24","Scotland",0,3,"Brazil")
add("Group stage","C","2026-06-24","Morocco",4,2,"Haiti")

# Group D
add("Group stage","D","2026-06-12","USA",4,1,"Paraguay")
add("Group stage","D","2026-06-14","Australia",2,0,"Turkiye")
add("Group stage","D","2026-06-19","USA",2,0,"Australia")
add("Group stage","D","2026-06-20","Turkiye",0,1,"Paraguay")
add("Group stage","D","2026-06-25","Turkiye",3,2,"USA")
add("Group stage","D","2026-06-25","Paraguay",0,0,"Australia")

# Group E
add("Group stage","E","2026-06-14","Germany",7,1,"Curacao")
add("Group stage","E","2026-06-14","Cote d'Ivoire",1,0,"Ecuador")
add("Group stage","E","2026-06-20","Germany",2,1,"Cote d'Ivoire")
add("Group stage","E","2026-06-20","Ecuador",0,0,"Curacao")
add("Group stage","E","2026-06-25","Curacao",0,2,"Cote d'Ivoire")
add("Group stage","E","2026-06-25","Ecuador",2,1,"Germany")

# Group F
add("Group stage","F","2026-06-14","Netherlands",2,2,"Japan")
add("Group stage","F","2026-06-15","Sweden",5,1,"Tunisia")
add("Group stage","F","2026-06-20","Netherlands",5,1,"Sweden")
add("Group stage","F","2026-06-21","Tunisia",0,4,"Japan")
add("Group stage","F","2026-06-25","Japan",1,1,"Sweden")
add("Group stage","F","2026-06-25","Tunisia",1,3,"Netherlands")

# Group G
add("Group stage","G","2026-06-15","Belgium",1,1,"Egypt")
add("Group stage","G","2026-06-15","IR Iran",2,2,"New Zealand")
add("Group stage","G","2026-06-21","Belgium",0,0,"IR Iran")
add("Group stage","G","2026-06-22","New Zealand",1,3,"Egypt")
add("Group stage","G","2026-06-26","Egypt",1,1,"IR Iran")
add("Group stage","G","2026-06-26","New Zealand",1,5,"Belgium")

# Group H
add("Group stage","H","2026-06-15","Spain",0,0,"Cabo Verde")
add("Group stage","H","2026-06-15","Saudi Arabia",1,1,"Uruguay")
add("Group stage","H","2026-06-21","Spain",4,0,"Saudi Arabia")
add("Group stage","H","2026-06-21","Uruguay",2,2,"Cabo Verde")
add("Group stage","H","2026-06-26","Cabo Verde",0,0,"Saudi Arabia")
add("Group stage","H","2026-06-26","Uruguay",0,1,"Spain")

# Group I
add("Group stage","I","2026-06-16","France",3,1,"Senegal")
add("Group stage","I","2026-06-17","Iraq",1,4,"Norway")
add("Group stage","I","2026-06-22","France",3,0,"Iraq")
add("Group stage","I","2026-06-23","Norway",3,2,"Senegal")
add("Group stage","I","2026-06-26","Norway",1,4,"France")
add("Group stage","I","2026-06-26","Senegal",5,0,"Iraq")

# Group J
add("Group stage","J","2026-06-16","Argentina",3,0,"Algeria")
add("Group stage","J","2026-06-16","Austria",3,1,"Jordan")
add("Group stage","J","2026-06-22","Argentina",2,0,"Austria")
add("Group stage","J","2026-06-22","Jordan",1,2,"Algeria")
add("Group stage","J","2026-06-27","Algeria",3,3,"Austria")
add("Group stage","J","2026-06-27","Jordan",1,3,"Argentina")

# Group K
add("Group stage","K","2026-06-17","Portugal",1,1,"DR Congo")
add("Group stage","K","2026-06-17","Uzbekistan",1,3,"Colombia")
add("Group stage","K","2026-06-23","Portugal",5,0,"Uzbekistan")
add("Group stage","K","2026-06-23","Colombia",1,0,"DR Congo")
add("Group stage","K","2026-06-27","Colombia",0,0,"Portugal")
add("Group stage","K","2026-06-27","DR Congo",3,1,"Uzbekistan")

# Group L
add("Group stage","L","2026-06-17","England",4,2,"Croatia")
add("Group stage","L","2026-06-17","Ghana",1,0,"Panama")
add("Group stage","L","2026-06-23","England",0,0,"Ghana")
add("Group stage","L","2026-06-23","Panama",0,1,"Croatia")
add("Group stage","L","2026-06-27","Panama",0,2,"England")
add("Group stage","L","2026-06-27","Croatia",2,1,"Ghana")

assert len(matches) == 72, f"Group stage count = {len(matches)}"

# ---------------- KNOCKOUT STAGE (32 matches) - from official FBref knockout bracket ----------------
# Round of 32
add("Round of 32","-","2026-06-28","Canada",1,0,"South Africa")
add("Round of 32","-","2026-06-29","Brazil",2,1,"Japan")
add("Round of 32","-","2026-06-29","Paraguay",1,1,"Germany","Paraguay won on penalties")
add("Round of 32","-","2026-06-30","Morocco",1,1,"Netherlands","Morocco won on penalties")
add("Round of 32","-","2026-06-30","Norway",2,1,"Cote d'Ivoire")
add("Round of 32","-","2026-07-01","France",3,0,"Sweden")
add("Round of 32","-","2026-07-01","Mexico",2,0,"Ecuador")
add("Round of 32","-","2026-07-01","England",2,1,"DR Congo")
add("Round of 32","-","2026-07-02","Belgium",3,2,"Senegal","Belgium won after extra time")
add("Round of 32","-","2026-07-02","USA",2,0,"Bosnia and Herzegovina")
add("Round of 32","-","2026-07-02","Spain",3,0,"Austria")
add("Round of 32","-","2026-07-02","Portugal",2,1,"Croatia")
add("Round of 32","-","2026-07-03","Switzerland",2,0,"Algeria")
add("Round of 32","-","2026-07-03","Egypt",1,1,"Australia","Egypt won on penalties")
add("Round of 32","-","2026-07-03","Argentina",3,2,"Cabo Verde","Argentina won after extra time")
add("Round of 32","-","2026-07-03","Colombia",1,0,"Ghana")

# Round of 16
add("Round of 16","-","2026-07-04","Morocco",3,0,"Canada")
add("Round of 16","-","2026-07-04","France",1,0,"Paraguay")
add("Round of 16","-","2026-07-05","Norway",2,1,"Brazil")
add("Round of 16","-","2026-07-05","England",3,2,"Mexico")
add("Round of 16","-","2026-07-06","Spain",1,0,"Portugal")
add("Round of 16","-","2026-07-06","Belgium",4,1,"USA")
add("Round of 16","-","2026-07-07","Argentina",3,2,"Egypt")
add("Round of 16","-","2026-07-07","Switzerland",0,0,"Colombia","Switzerland won on penalties after extra time")

# Quarter-finals
add("Quarter-final","-","2026-07-09","France",2,0,"Morocco")
add("Quarter-final","-","2026-07-10","Spain",2,1,"Belgium")
add("Quarter-final","-","2026-07-10","England",2,1,"Norway","England won after extra time")
add("Quarter-final","-","2026-07-11","Argentina",3,1,"Switzerland","Argentina won after extra time")

# Semi-finals
add("Semi-final","-","2026-07-14","Spain",2,0,"France")
add("Semi-final","-","2026-07-15","Argentina",2,1,"England")

# Third-place match
add("Third-place match","-","2026-07-18","England",6,4,"France")

# Final
add("Final","-","2026-07-19","Spain",1,0,"Argentina","Spain won after extra time")

assert len(matches) == 104, f"Total match count = {len(matches)}"

with open("worldcup2026_all_104_matches.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Stage","Group","Date","Home","Home Score","Away Score","Away","Notes"])
    w.writerows(matches)

print("Wrote", len(matches), "matches")
