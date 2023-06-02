import numpy as np
import pandas as pd
import os

results_folder = "solutions/sa/2"

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

x_file = os.path.join(results_folder, "x.npy")
y_file = os.path.join(results_folder, "y.npy")
overflow_file = os.path.join(results_folder, "overflow.npy")
x = np.load(x_file)
y = np.load(y_file)
overflow = np.load(overflow_file)

assignment = np.zeros(nb_families)
for i, pref in enumerate(x):
    assignment[i] = preferences_df.loc[i, "choice_"+str(pref)]

results_dict = {"family_id": range(nb_families), "assigned_day": assignment.astype(int)}
results_df = pd.DataFrame(results_dict)
results_df.to_csv("solutions/second_submission.csv", index=False)