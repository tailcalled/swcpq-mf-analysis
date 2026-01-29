from bs4 import BeautifulSoup
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open("SWCPQ-Features-Aggregated-Dataset-January2025/codebook.html", "r") as f:
    soup = BeautifulSoup(f.read(), "html.parser")
characters_table, traits_table = soup.find_all("table")

def read_table(table):
    rows = table.find_all("tr")
    header = [x.text.strip() for x in rows[0].find_all("th")]
    return [
        {k:v.text.strip() for k, v in zip(header, row.find_all("td"))}
        for row in rows[1:]
    ]

character_infos = read_table(characters_table)
character_names = {
    info["ID"]: info["Character display name"] + " - " + info["Fictional work"]
    for info in character_infos
}
print(character_names)

trait_infos = read_table(traits_table)
trait_names = {
    info["ID"]: info["low"] + " -> " + info["high"]
    for info in trait_infos
}
print(trait_names)

data = pd.read_csv("SWCPQ-Features-Aggregated-Dataset-January2025/data files/characters-aggregated-scores.csv", sep="\t", index_col=0)
data = data.rename(columns=trait_names, index=character_names)
print(data)

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

masculine = -(data["lion -> zebra"] * 0.059 + data["🏀 -> 🎨"] * 0.090 + data["thick-skinned -> sensitive"] * 0.020 + data["tailor -> blacksmith"] * -0.151 + data["gossiping -> confidential"] * -0.160 + data["feminist -> sexist"] * -0.137 + data["creepy -> disarming"] * -0.148 + data["giggling -> chortling"] * -0.145 + data["cat person -> dog person"] * -0.099 + data["delicate -> coarse"] * -0.102 + data["young -> old"] * -0.072 + data["city-slicker -> country-bumpkin"] * 0.065 + data["uptight -> easy"] * -0.064 + data["never cries -> often crying"] * 0.079 + data["🚴 -> 🏋️‍♂️"] * -0.049)
feminine = data["tailor -> blacksmith"] * -0.086 + data["giggling -> chortling"] * -0.180 + data["glamorous -> spartan"] * -0.085 + data["comedic -> dramatic"] * 0.044 + data["gamer -> non-gamer"] * 0.086 + data["🏀 -> 🎨"] * 0.125 + data["scruffy -> manicured"] * 0.113 + data["emotional -> logical"] * -0.108 + data["oppressed -> privileged"] * -0.086 + data["plastic -> wooden"] * -0.164 + data["punk rock -> preppy"] * 0.112 + data["beautiful -> ugly"] * -0.139 + data["serene -> pensive"] * -0.119 + data["cat person -> dog person"] * -0.082 + data["pronatalist -> child free"] * -0.060
sexdia = 50 * (data["feminist -> sexist"] * 0.010 + data["racist -> egalitarian"] * 0.008 + data["chic -> cheesy"] * 0.004 + data["deep -> shallow"] * -0.006 + data["tall -> short"] * -0.005 + data["🐐 -> 🦒"] * -0.009 + data["smooth -> rough"] * -0.003 + data["beautiful -> ugly"] * 0.006 + data["gamer -> non-gamer"] * -0.002 + data["cat person -> dog person"] * 0.005 + data["hugs -> handshakes"] * 0.004 + data["thick-skinned -> sensitive"] * 0.005 + data["celebrity -> boy/girl-next-door"] * -0.004 + data["bossy -> meek"] * 0.005 + data["narcissistic -> low self esteem"] * -0.004)

sex_data = data[["masculine -> feminine", "androgynous -> gendered"]]
sex_data["masculine"] = masculine
sex_data["feminine"] = feminine
sex_data["sexdia"] = sexdia
sex_model = GaussianMixture(2).fit(sex_data)
sex_imputed = sex_model.predict(sex_data)
if data["masculine -> feminine"][sex_imputed == 0].mean() < 50:
    male = sex_imputed == 0; msex = 0
else:
    male = sex_imputed == 1; msex = 1

with open("sex_override.json", "r") as f:
    sex_overrides = json.load(f)
    for character, sex in sex_overrides.items():
        male[sex_data.index.get_loc(character)] = (1 if sex == "male" else 0)
data["male"] = male

sex_pcs = PCA(2).fit_transform(sex_data)
plt.scatter(sex_pcs[male, 0], sex_pcs[male, 1])
plt.scatter(sex_pcs[~male, 0], sex_pcs[~male, 1])
plt.savefig("plots/00_sex_data_pca.png"); plt.close()

Cohen_ds = (data[male].mean(axis=0) - data[~male].mean(axis=0))/(0.5*(data[male].std(axis=0) + data[~male].std(axis=0)))
for trait in (Cohen_ds**2).sort_values().index:
    print(Cohen_ds[trait], trait)

data["sex_prediction"] = sex_model.predict_proba(sex_data)[:, msex]
plt.hist(data["sex_prediction"], bins=100)
plt.yscale("log")
plt.savefig("plots/01_sex_distribution.png"); plt.close()

plt.scatter(masculine[male], feminine[male])
plt.scatter(masculine[~male], feminine[~male])
plt.savefig("plots/02_mf_distribution.png"); plt.close()

plt.suptitle("Personality of Fictional Characters")
plt.subplot(121)
plt.scatter(masculine[male], sexdia[male])
plt.scatter(masculine[~male], sexdia[~male])
plt.xlabel("Masculinity according to\nstandards for men")
plt.ylabel("Multivariate sex-diagnostic axis")
plt.subplot(122)
plt.scatter(feminine[male], sexdia[male])
plt.scatter(feminine[~male], sexdia[~male])
plt.xlabel("Femininity according to\nstandards for women")
plt.ylabel("Multivariate sex-diagnostic axis")
plt.savefig("plots/02_mf_vs_sexdia.png"); plt.close()

for character in data["sex_prediction"].sort_values().index:
    print(data["sex_prediction"][character], character)

def stepwise_regression(dep, indeps):
    steps = []
    model = LinearRegression()
    for i in range(15):
        best = None
        best_score = 0.0
        for indep in indeps.columns:
            if indep in steps:
                continue
            sub_indeps = indeps[steps + [indep]]
            model.fit(sub_indeps, dep)
            score = model.score(sub_indeps, dep)
            if score > best_score:
                best_score = score
                best = indep
        steps.append(best)
        print(i+1, best_score)
    model.fit(indeps[steps], dep)
    return { k: v for k, v in zip(steps, model.coef_) }

res_columns = [x for x in data.columns if x not in ["masculine -> feminine", "androgynous -> gendered", "macho -> metrosexual", "straight -> queer", "🐴 -> 🦄", "sex_prediction", "male"]]
print("Male predictors:")
print(" + ".join(f"data[\"{k}\"] * {v:.3f}" for k, v in stepwise_regression(data["masculine -> feminine"][male], data[res_columns][male]).items()))
print("Female predictors:")
print(" + ".join(f"data[\"{k}\"] * {v:.3f}" for k, v in stepwise_regression(data["masculine -> feminine"][~male], data[res_columns][~male]).items()))
print("Sex predictors:")
print(" + ".join(f"data[\"{k}\"] * {v:.3f}" for k, v in stepwise_regression(50*data["male"], data[res_columns]).items()))
