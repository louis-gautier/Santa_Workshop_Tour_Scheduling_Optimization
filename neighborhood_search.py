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

def get_day(f, p):
    return preferences_df.loc[f, "choice_"+str(p)]

def get_overflow_day(f, pbar):
    preferences = preferences_df.loc[f].filter(like="choice").to_numpy()
    overflow_days = np.array([[i for i in range(1, nb_days+1) if i not in preferences]])
    return overflow_days[pbar]
  
def read_solution(file):
    # One-hot encoded versions of the decision variables
    x = np.zeros((nb_families, nb_preferences)).astype(int)
    y = np.zeros((nb_days, nb_possible_crowd, nb_possible_crowd)).astype(int)
    overflow = np.zeros((nb_families, nb_days - 10)).astype(int)
    assignment = np.zeros(nb_families)

    with open(file, "r") as f:
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
    return x,y,overflow,assignment

ac_thresh = 27226
TR = 70
x,y,overflow,assignment = read_solution("solutions/basic_MIP_model.sol")
n = np.matrix(preferences_df['n_people'].values)
l = np.array([n @ (assignment == d+1) for d in range(nb_days)]).flatten()
y_feas = [(d, i, j) for d in range(nb_days-1) for i in range(max(int(l[d])-TR-125, 0), min(int(l[d])+TR-125+1, nb_possible_crowd)) for j in range(max(int(l[d+1])-TR-125, 0), min(int(l[d+1])+TR-125+1, nb_possible_crowd)) if accounting_cost(i+125,j+125)<ac_thresh]
y_feas += [(99, i, i) for i in range(max(int(l[99])-TR-125, 0), min(int(l[99])+TR-125+1, nb_possible_crowd)) if accounting_cost(i+125,i+125)<ac_thresh]

model = gp.Model()
# Define the decision variables: x[f,p]: matching of family with preference number
# Shape (5000, 10)
x = model.addVars(nb_families, nb_preferences, vtype=gp.GRB.BINARY, name="x")
# z[d]: number of people allocated to day d
z = model.addVars(nb_days, vtype=gp.GRB.CONTINUOUS, name="z")
# y[d, i, j]: number of people allocated do day d and to day d+1 (previous day), one-hot encoded in [125,350]
# Shape (100, 176, 176)
y = {}
for (d,i,j) in y_feas:
    y[d,i,j] = model.addVar(1, vtype=gp.GRB.BINARY, name=f"y[{d,i,j}]")

# Define the objective function
# Preference cost
pref_cost = gp.quicksum(x[f, p]*preference_cost(p, get_size(f)) for f in range(nb_families) for p in range(nb_preferences))
# Accounting cost
acc_cost = gp.quicksum(y[d,i,j]*accounting_cost(i+125,j+125) for (d,i,j) in y_feas)
model.setObjective(pref_cost + acc_cost, gp.GRB.MINIMIZE)

# Ensure that each family is matched to only one preference and one day if assigned to no preference
model.addConstrs(gp.quicksum(x[f,p] for p in range(nb_preferences)) == 1 for f in range(nb_families)) #2
# model.addConstrs(gp.quicksum(overflow[f, pbar] for pbar in range(nb_days - 10)) == x[f,10] for f in range(nb_families))

# Ensure that there is only one coefficient of y equal to 1 for each day (only a single number of people allocated to day d and day d+1)
model.addConstrs(gp.quicksum(y[d,i,j] for (d,i,j) in y_feas if d == day)==1 for day in range(nb_days)) #3

# Ensure that the values in y are consistent with each other for each pair of two contiguous days
model.addConstrs(gp.quicksum(y[d,i,j] for (d,i,j) in y_feas if d == day and j == n) == gp.quicksum(y[d,i,j] for (d,i,j) in y_feas if d == day+1 and i == n) for day in range(nb_days-1) for n in range(nb_possible_crowd)) #4

# Ensure that the sum of y matches the number of people assigned to that day
model.addConstrs(gp.quicksum((i+125)*y[d,i,j] for (d,i,j) in y_feas if d == day) == z[day] for day in range(nb_days)) #5

# Ensure that the number of people assigned on each day based on preferences doesn't exceed the value contained in y (it's an inequality because of the possibility of assigning to the last priority and thus to any day)
model.addConstrs(gp.quicksum(family_sizes[f]*indicators_preferences[f,p,d]*x[f,p] for f in range(nb_families) for p in range(nb_preferences-1)) <= z[d] for d in range(nb_days)) #6

# Ensure that the sum of y matches the total number of people
model.addConstr(gp.quicksum(z[d] for d in range(nb_days)) == nb_people) #7

# Limit the accounting cost that can be incurred
model.addConstr(acc_cost <= ac_thresh) #9

# Save intermediary solutions
model.setParam('OutputFlag', 1)
model.setParam('ResultFile', 'solutions/neighborhood_search_model.sol')

model.optimize()