import numpy as np
import pandas as pd

preferences_df = pd.read_csv("data/family_data.csv")
nb_families = len(preferences_df.index)
nb_days = 100
nb_preferences = 11
nb_possible_crowds = 350 - 125 + 1

def get_size(f):
    return preferences_df.loc[f, "n_people"]

def get_day(f, p):
    return preferences_df.loc[f, "choice_"+str(p)]

def get_overflow_day(f, pbar):
    preferences = preferences_df.loc[f].filter(like="choice").to_numpy()
    overflow_days = np.array([[i for i in range(1, nb_days+1) if i not in preferences]])
    return overflow_days[pbar]

# One-hot encoded versions of the decision variables
x = np.zeros((nb_families, nb_preferences)).astype(int)
y = np.zeros((nb_days, nb_possible_crowds, nb_possible_crowds)).astype(int)
overflow = np.zeros((nb_families, nb_days - 10)).astype(int)
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
                x[index1, index2] = 1
                if index2 <= 9:
                    assignment[index1] = get_day(index1, index2)
            if variable == "overflow" and value:
                if assignment[index1] != 0:
                    raise RuntimeError("Infeasible")
                overflow[index1, index2] = 1
                if index2 <= 9:
                    assignment[index1] = get_overflow_day(index1, index2)
        else:
            if value:
                index1, index2, index3 = map(lambda x: int(x), indices.split(","))
                y[index1, index2, index3] = 1

np.save("solutions/0/x.npy", x)
np.save("solutions/0/overflow.npy", overflow)
np.save("solutions/0/y.npy", y)

results_dict = {"family_id": range(nb_families), "assigned_day": assignment.astype(int)}
results_df = pd.DataFrame(results_dict)
results_df.to_csv("solutions/second_submission.csv", index=False)