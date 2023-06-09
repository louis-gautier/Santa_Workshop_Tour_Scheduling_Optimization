import gurobipy as gp
import pandas as pd
import numpy as np

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

model = gp.Model()
# Define the decision variables: x[f,p]: matching of family with preference number
# Shape (5000, 10)
x = model.addVars(nb_families, nb_preferences, vtype=gp.GRB.BINARY, name="x")
# overflow[f, \bar{p}] is between 0 and 89: the day chosen among all non-preference listed days if no preference was allocated to family f
overflow = model.addVars(nb_families, nb_days-10, vtype=gp.GRB.BINARY, name="overflow")
# y[d, i, j]: number of people allocated do day d and to day d+1 (previous day), one-hot encoded in [125,350]
# Shape (100, 176, 176)
y = model.addVars(nb_days, nb_possible_crowd, nb_possible_crowd, vtype=gp.GRB.BINARY, name="y")

# Define the objective function
# Preference cost
pref_cost = gp.quicksum(x[f, p]*preference_cost(p, get_size(f)) for f in range(nb_families) for p in range(nb_preferences))
# Accounting cost
acc_cost = gp.quicksum(y[d, i, j]*accounting_cost(i+125,j+125) for d in range(nb_days) for i in range(nb_possible_crowd) for j in range(nb_possible_crowd))
model.setObjective(pref_cost+acc_cost, gp.GRB.MINIMIZE)

# Ensure that each family is matched to only one preference and one day if assigned to no preference
model.addConstrs(gp.quicksum(x[f,p] for p in range(nb_preferences)) == 1 for f in range(nb_families))
model.addConstrs(gp.quicksum(overflow[f, pbar] for pbar in range(nb_days - 10)) == x[f,10] for f in range(nb_families))
# Ensure that there is only one coefficient of y equal to 1 for each day (only a single number of people allocated to day d and day d+1)
model.addConstrs(gp.quicksum(y[d,i,j] for j in range(nb_possible_crowd) for i in range(nb_possible_crowd))==1 for d in range(nb_days))
# Ensure that the values in y are consistent with each other for each pair of two contiguous days
model.addConstrs(gp.quicksum(y[d,i,n] for i in range(nb_possible_crowd)) == gp.quicksum(y[d+1,n,j] for j in range(nb_possible_crowd)) for d in range(nb_days-1) for n in range(nb_possible_crowd))
model.addConstrs(y[99,i,j] == 0 for i in range(nb_possible_crowd) for j in range(nb_possible_crowd) if i != j) # Initial condition on the number of people on day 100
# Ensure that the number of people assigned on each day based on preferences doesn't exceed the value contained in y (it's an inequality because of the possibility of assigning to the last priority and thus to any day)
model.addConstrs(gp.quicksum(family_sizes[f]*indicators_preferences[f,p,d]*x[f,p] for f in range(nb_families) for p in range(nb_preferences-1)) + gp.quicksum(family_sizes[f]*overflow[f, pbar]*indicators_overflow_days[f, pbar, d] for f in range(nb_families) for pbar in range(nb_days - 10)) == gp.quicksum((i+125)*y[d,i,j] for i in range(nb_possible_crowd) for j in range(nb_possible_crowd)) for d in range(nb_days))
# Ensure that the sum of y matches the total number of people
model.addConstr(gp.quicksum((i+125)*y[d,i,j] for d in range(nb_days) for i in range(nb_possible_crowd) for j in range(nb_possible_crowd)) == nb_people)

# Save intermediary solutions
model.setParam('OutputFlag', 1)
model.setParam('ResultFile', 'solutions/basic_MIP_model.sol')
model.setParam('TimeLimit', 12*3600)

model.optimize()