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
indicators_preferences = preferences_df.filter(like="choice").to_numpy()
indicators_preferences = indicators_preferences - 1
indicators_preferences = np.eye(100)[indicators_preferences]
# shape = (5000, 10, 100)

def preference_cost(p, s):
    # p=10 if the assigned preference is 10 or larger
    fixed_costs = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
    marginal_costs = [0, 0, 9, 9, 9, 18, 18, 36, 36, 235, 434]
    return fixed_costs[p] + s*marginal_costs[p]

def accounting_cost(n_d, n_dplus1):
    return ((n_d-125)/400)*n_d**(1/2+abs(n_d-n_dplus1)/50)

def get_size(i):
    return preferences_df.loc[i, "n_people"]

model = gp.Model()
# Define the decision variables: x[f,p]: matching of family with preference number
x = model.addVars(nb_families, nb_preferences, vtype=gp.GRB.BINARY, name="x")
# y[d, i, j]: number of people allocated do day d and to day d+1, one-hot encoded in [125,350]
y = model.addVars(nb_days, nb_possible_crowd, nb_possible_crowd, vtype=gp.GRB.BINARY, name="y")

# Define the objective function
pref_cost = gp.quicksum(x[f, p]*preference_cost(p, get_size(f)) for f in range(nb_families) for p in range(nb_preferences))
acc_cost = gp.quicksum(y[d, i, j]*accounting_cost(i+125,j+125) for d in range(nb_days) for i in range(nb_possible_crowd) for j in range(nb_possible_crowd))
model.setObjective(pref_cost+acc_cost, gp.GRB.MINIMIZE)

model.addConstrs(gp.quicksum(x[f,p] for p in range(nb_preferences))==1 for f in range(nb_families))
model.addConstrs(gp.quicksum(y[d,i,j] for j in range(nb_possible_crowd) for i in range(nb_possible_crowd))==1 for d in range(nb_days))
model.addConstrs(gp.quicksum(y[d,i,n] for i in range(nb_possible_crowd)) == gp.quicksum(y[d+1,n,j] for j in range(nb_possible_crowd)) for d in range(nb_days-1) for n in range(nb_possible_crowd))
model.addConstrs(gp.quicksum(family_sizes[f]*indicators_preferences[f,p,d]*x[f,p] for f in range(nb_families) for p in range(nb_preferences-1)) <= gp.quicksum((i+125)*y[d,i,j] for i in range(nb_possible_crowd) for j in range(nb_possible_crowd)) for d in range(nb_days))
model.addConstr(gp.quicksum((i+125)*y[d,i,j] for d in range(nb_days) for i in range(nb_possible_crowd) for j in range(nb_possible_crowd)) == nb_people)

model.optimize()