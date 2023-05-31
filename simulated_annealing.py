import numpy as np
import pandas as pd
import gurobipy as gp

# Util functions
preferences_df = pd.read_csv("data/family_data.csv")
nb_families = len(preferences_df.index)
nb_preferences = 11
nb_days = 100
nb_possible_crowd = 300 - 125 + 1
nb_people = preferences_df["n_people"].sum()
family_sizes = preferences_df["n_people"].to_numpy()
preferences = preferences_df.filter(like="choice").to_numpy()
preferences = preferences - 1
indicators_preferences = np.eye(100)[preferences]
# shape = (5000, 10, 100)
overflow_days = np.array([[i for i in range(nb_days) if i not in preferences[j,:]] for j in range(nb_families)])
indicators_overflow_days = np.eye(100)[overflow_days]
# shape = (5000, 90, 100)

def preference_cost(p, s):
    # p=10 if the assigned preference is 10 or larger
    fixed_costs = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
    marginal_costs = [0, 0, 9, 9, 9, 18, 18, 36, 36, 235, 434]
    return fixed_costs[p] + s*marginal_costs[p]

def accounting_cost(n_d, n_dplus1):
    return ((n_d-125)/400)*n_d**(1/2+abs(n_d-n_dplus1)/50)

def get_size(f):
    return preferences_df.loc[f, "n_people"]

# We load our first solution
x_onehot = np.load("solutions/0/x.npy")
x = np.argwhere(x_onehot, axis=1)
# Shape (100, 176, 176)
y_onehot = np.load("solutions/0/y.npy")
y = np.argwhere(y_onehot).reshape((-1,2))
overflow_onehot = np.load("solutions/0/overflow.npy")
overflow = np.argwhere(overflow_onehot, axis=1)

N = 10000

def T(n):
    # Temperature decay principle
    pass

for n in range(N):
    # Find a swap to try to perform between two preferences
    family1, family2 = np.random.choice(0, 5000, size=2)
    family1_size = get_size(family1)
    family2_size = get_size(family2)
    pref1 = x[family1]
    if x[family1] == 10:
        day1 = overflow_days[overflow[family1]]
    else:
        day1 = preferences[pref1]
    pref2 = x[family2]
    if x[family2] == 10:
        day2 = overflow_days[overflow[family2]]
    else:
        day2 = preferences[pref2]

    nb_people_day1 = 125 + y[day1][0]
    nb_people_day2 = 125 + y[day2][0]
    # Case where the two picked individuals have been assigned preferences
    feasible = nb_people_day1 - family1_size + family2_size >= 125 and nb_people_day1 - family1_size + family2_size <= 350
    deltaE_preference = preference_cost(pref1, family2_size) + preference_cost
    deltaE_accounting =
    deltaE = deltaE_preference + deltaE_accounting
    if not feasible:
        continue
    if deltaE >= 0:
        acceptance_prob = np.exp(-deltaE/T(n))
        if np.random.rand() > acceptance_prob:
            continue
    # Accept the move: update x, y and overflow
    