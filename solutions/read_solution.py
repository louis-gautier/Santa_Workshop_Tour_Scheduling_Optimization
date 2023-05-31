import numpy as np
import pandas as pd

preferences_df = pd.read_csv("data/family_data.csv")
nb_families = len(preferences_df.index)
nb_days = 100

def get_size(f):
    return preferences_df.loc[f, "n_people"]

def get_day(f, p):
    return preferences_df.loc[f, "choice_"+str(p)]

def get_overflow_day(f, pbar):
    preferences = preferences_df.loc[f].filter(like="choice").to_numpy()
    overflow_days = np.array([[i for i in range(1, nb_days+1) if i not in preferences]])
    return overflow_days[pbar]

assignment = np.zeros(nb_families)

with open("solutions/basic_MIP_model.sol", "r") as f:
    f.readline()
    for line in f:
        coef, value = line.split("] ")
        value = int(value)
        variable, indices = coef.split("[")
        if variable != "y":
            index1, index2 = map(lambda x: int(x), indices.split(","))
            if variable == "x" and value:
                if assignment[index1] != 0:
                    raise RuntimeError("Infeasible")
                if index2 <= 9:
                    assignment[index1] = get_day(index1, index2)
            if variable == "overflow" and value:
                if assignment[index1] != 0:
                    raise RuntimeError("Infeasible")
                if index2 <= 9:
                    assignment[index1] = get_overflow_day(index1, index2)

results_dict = {"family_id": range(nb_families), "assigned_day": assignment.astype(int)}
results_df = pd.DataFrame(results_dict)
results_df.to_csv("solutions/first_submission.csv", index=False)