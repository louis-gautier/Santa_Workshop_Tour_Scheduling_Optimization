import gurobipy as gp
import pandas as pd
import numpy as np

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

def preference_cost(p, s):
    # p=10 if the assigned preference is 10 or larger
    fixed_costs = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
    marginal_costs = [0, 0, 9, 9, 9, 18, 18, 36, 36, 235, 434]
    return fixed_costs[p] + s*marginal_costs[p]

P = np.ones((n_fam,n_days)) # initialize matrix
for f in range(n_fam):
    P[f,:] *= preference_cost(10,fam_size[f]) # scale row to worst case cost
    for p in range(n_pref-1):
        d = pref[f, p] # get day preference p
        P[f,d] = preference_cost(p,fam_size[f]) # update cost matrix

PC = np.array([[preference_cost(p, fam_size[f]) for p in range(n_pref)] for f in range(n_fam)])

def accounting_cost(n_d, n_dplus1):
    return ((n_d-125)/400)*n_d**(1/2+abs(n_d-n_dplus1)/50)

AC = np.ones((n_vis,n_vis)) # initialize matrix
for i in range(n_vis):
    for j in range(n_vis):
        AC[i,j] = accounting_cost(i+125,j+125) # update cost matrix

def get_day(f, p):
    return family_data.loc[f, "choice_"+str(p)]

def get_overflow_day(f, pbar):
    preferences = family_data.loc[f].filter(like="choice").to_numpy()
    overflow_days = np.array([[i for i in range(1, n_days+1) if i not in preferences]])
    return overflow_days[pbar]
  
def read_solution(file):
    # One-hot encoded versions of the decision variables
    x = np.zeros((n_fam, n_pref)).astype(int)
    z = np.zeros(n_days).astype(int)
    y = np.zeros((n_days, n_vis, n_vis)).astype(int)
    overflow = np.zeros((n_fam, n_days - 10)).astype(int)
    assignment = np.zeros(n_fam)

    with open(file, "r") as f:
        f.readline()
        for line in f:
            coef, value = line.split("] ")
            value = int(np.round(float(value)))
            # try:
            #     value = int(value.strip())
            # except:
            #     print(line)
            #     raise Exception
            variable, indices = coef.split("[")
            if variable != "z":
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
            else:
                index = int(indices)
                z[index] = value
    return x,y,z,overflow,assignment
