import gurobipy as gp
import pandas as pd
import numpy as np
from utils import preference_cost, accounting_cost, get_day, get_overflow_day, read_solution

print("loading data")

family_data = pd.read_csv("data/family_data.csv")
n_fam = len(family_data.index)
n_pref = 11
n_days = 100
max_vis = 300
min_vis = 125
n_vis = max_vis - min_vis + 1
n_ppl = family_data["n_people"].sum()
fam_size = family_data["n_people"].to_numpy()
pref = family_data.filter(like="choice").to_numpy() - 1

# Neighborhood search parameters
AC_thresh = 27226
TR = 70
x_init,y_init,z_init,overflow_init,assignment_init = read_solution("solutions/basic_MIP_model.sol")

P = np.ones((n_fam,n_days)) # initialize matrix
for f in range(n_fam):
    P[f,:] *= preference_cost(10,fam_size[f]) # scale row to worst case cost
    for p in range(n_pref-1):
        d = pref[f, p] # get day preference p
        P[f,d] = preference_cost(p,fam_size[f]) # update cost matrix

PC = np.array([[preference_cost(p, fam_size[f]) for p in range(n_pref)] for f in range(n_fam)])

AC = np.ones((n_vis,n_vis)) # initialize matrix
for i in range(n_vis):
    for j in range(n_vis):
        AC[i,j] = accounting_cost(i+125,j+125) # update cost matrix

print("constructing neighborhood")

# Construct neighborhood
l = np.array([fam_size @ (assignment_init == d+1) for d in range(n_days)]).flatten()

d_feas = {}
y_feas = []
for d in range(n_days):
    d_feas[d] = []
    lb_i = max(int(l[d])-TR-125, 0)
    ub_i = min(int(l[d])+TR-125+1, n_vis)
    for i in range(lb_i, ub_i):
        if d < n_days-1:
            lb_j = max(int(l[d+1])-TR-125, 0)
            ub_j = min(int(l[d+1])+TR-125+1, n_vis)
            for j in range(lb_j, ub_j):
                if AC[i,j] <= AC_thresh:
                    y_feas.append((d,i,j))
        else:
            if AC[i,j] <= AC_thresh:
                y_feas.append((d,i,i))
        d_feas[d].append(i)
# y_feas = [(d, i, j) for d in range(nb_days-1) for i in range(max(int(l[d])-TR-125, 0), min(int(l[d])+TR-125+1, nb_possible_crowd)) for j in range(max(int(l[d+1])-TR-125, 0), min(int(l[d+1])+TR-125+1, nb_possible_crowd)) if accounting_cost(i+125,j+125)<ac_thresh]
# y_feas += [(99, i, i) for i in range(max(int(l[99])-TR-125, 0), min(int(l[99])+TR-125+1, nb_possible_crowd)) if accounting_cost(i+125,i+125)<ac_thresh]

print("building model")

model = gp.Model()

# Define the decision variables: x[f,p]: matching of family with preference number
x = model.addVars(n_fam, n_pref, vtype=gp.GRB.BINARY, name="x")

# z[d]: number of people allocated to day d
z = model.addVars(n_days, lb=125, ub=300, vtype=gp.GRB.CONTINUOUS, name="z")

# y[d, i, j]: number of people allocated do day d and to day d+1 (previous day), one-hot encoded in [125,350]
y = {}
for (d,i,j) in y_feas:
    y[d,i,j] = model.addVar(vtype=gp.GRB.BINARY, name=f"y[{d},{i},{j}]")

# Define the objective function
# Preference cost
pref_cost = gp.quicksum(PC[f, p] * x[f, p] for p in range(n_pref) for f in range(n_fam))

# Accounting cost
acc_cost = gp.quicksum(AC[i,j] * y[d,i,j] for (d,i,j) in y_feas)

model.setObjective(pref_cost + acc_cost, gp.GRB.MINIMIZE)

# Ensure that each family is matched to only one preference and one day if assigned to no preference
model.addConstrs(gp.quicksum(x[f,p] for p in range(n_pref)) == 1 for f in range(n_fam)) #2

# Ensure that there is only one coefficient of y equal to 1 for each day (only a single number of people allocated to day d and day d+1)
model.addConstrs(gp.quicksum(y[d,i,j] for (d,i,j) in y_feas if d == day)==1 for day in range(n_days)) #3

# Ensure that the values in y are consistent with each other for each pair of two contiguous days
model.addConstrs(gp.quicksum(y[d,i,j] for (d,i,j) in y_feas if d == day and j == n) == gp.quicksum(y[d,i,j] for (d,i,j) in y_feas if d == day+1 and i == n) for day in range(n_days-1) for n in d_feas[day+1]) #4

# Ensure that the sum of y matches the number of people assigned to that day
model.addConstrs(gp.quicksum((i+125)*y[d,i,j] for (d,i,j) in y_feas if d == day) == z[day] for day in range(n_days)) #5

# Ensure that the number of people assigned on each day based on preferences doesn't exceed the value contained in y (it's an inequality because of the possibility of assigning to the last priority and thus to any day)
model.addConstrs(gp.quicksum(fam_size[f]*x[f,p] for f in range(n_fam) for p in np.where(pref[f,:] == d)[0]) <= z[d] for d in range(n_days)) #6

# Ensure that the sum of z matches the total number of people
model.addConstr(gp.quicksum(z[d] for d in range(n_days)) == n_ppl) #7

# Limit the accounting cost that can be incurred
model.addConstr(acc_cost <= AC_thresh) #9

# Save intermediary solutions
model.setParam('OutputFlag', 1)
model.setParam('ResultFile', f'solutions/neighborhood_search_{TR}_thres.sol')
model.optimize()