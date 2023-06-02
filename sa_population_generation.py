import numpy as np
import pandas as pd
import os

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
x = np.nonzero(x_onehot)[1]
# Shape (100, 176, 176)
y_onehot = np.load("solutions/0/y.npy")
y = np.argwhere(y_onehot)[:,1:]
overflow_onehot = np.load("solutions/0/overflow.npy")
overflow = np.nonzero(overflow_onehot)[1]
population_size = 100
save_freq = 25
save_folder = "solutions/ga_initialization/"

def T(n, T0=5, Tf=0.001, n_decay=save_freq*population_size*4):
    # Linear temperature decay
    return T0 + (n/n_decay)*(Tf-T0)
n = 0
objective_value = 6.9341431520841768e+04
while True:
    if n == save_freq*population_size + 1:
        break
    # Find a swap to try to perform between two preferences
    family1, family2 = np.random.choice(5000, 2, replace=False)
    family1_size = get_size(family1)
    family2_size = get_size(family2)
    pref1 = x[family1]
    pref2 = x[family2]
    modified_nb_people = {}
    past_y_coefficients = {}
    new_y_coefficients = {}
    
    if pref1 == pref2:
        # If the two preferences are equal, we modify one of the two preferences (at random), by doing + or - 1 (at random)
        which_family, which_sign = np.random.binomial(1, 0.5, size=2)
        chosen_family = [family1, family2][which_family]
        old_pref = x[chosen_family]
        family_size = [family1_size, family2_size][which_family]
        which_sign = 2*which_sign - 1
        if x[chosen_family] == 0:
            which_sign = 1
        if x[chosen_family] == 10:
            which_sign = - 1
            original_day = overflow_days[overflow[chosen_family]]
        else:
            original_day = preferences[chosen_family, x[chosen_family]]
        new_pref = x[chosen_family] + which_sign
        if new_pref < 10:
            new_day = preferences[chosen_family, new_pref]
        else:
            new_day = np.random.choice(overflow_days[chosen_family,:])
        modified_nb_people[original_day] = 125 + y[original_day, 0] - family_size
        modified_nb_people[new_day] = 125 + y[new_day, 0] + family_size
        if original_day not in past_y_coefficients.keys():
            past_y_coefficients[original_day] = [y[original_day, 0], y[original_day, 1]]
            new_y_coefficients[original_day] = [y[original_day, 0] - family_size, y[original_day, 1]]
        else:
            new_y_coefficients[original_day][0] -= family_size
        if original_day > 0 and original_day - 1 not in past_y_coefficients.keys():
            past_y_coefficients[original_day-1] = [y[original_day-1, 0], y[original_day-1, 1]]
            new_y_coefficients[original_day-1] = [y[original_day-1, 0], y[original_day-1, 1] - family_size]
        elif original_day > 0:
            new_y_coefficients[original_day-1][1] -= family_size
        if new_day not in past_y_coefficients.keys():
            past_y_coefficients[new_day] = [y[new_day, 0], y[new_day, 1]]
            new_y_coefficients[new_day] = [y[new_day, 0] + family_size, y[new_day, 1]]
        else:
            new_y_coefficients[new_day][0] += family_size
        if new_day > 0 and new_day - 1 not in past_y_coefficients.keys():
            past_y_coefficients[new_day-1] = [y[new_day-1, 0], y[new_day-1, 1]]
            new_y_coefficients[new_day-1] = [y[new_day-1, 0], y[new_day-1, 1] + family_size]
        elif new_day > 0:
            new_y_coefficients[new_day-1][1] += family_size
        deltaE_preference = preference_cost(new_pref, family_size) - preference_cost(old_pref, family_size) 
    else:    
        # Swap preferences
        if pref1 == 10:
            print("Preference 10 chosen")
            original_day1 = overflow_days[overflow[family1]]
            new_day2 = np.random.choice(overflow_days[family2,:])
        else:
            original_day1 = preferences[family1, pref1]
            new_day2 = preferences[family2, pref1]
        
        if pref2 == 10:
            print("Preference 10 chosen")
            original_day2 = overflow_days[overflow[family2]]
            new_day1 = np.random.choice(overflow_days[family1,:])
        else:
            original_day2 = preferences[family2, pref2]
            new_day1 = preferences[family1, pref2]
        
        deltaE_preference = preference_cost(pref1, family2_size) + preference_cost(pref2, family1_size) - preference_cost(pref1, family1_size) - preference_cost(pref2, family2_size)

        
        # original_day1: no more family1
        if original_day1 not in modified_nb_people.keys():
            modified_nb_people[original_day1] = 125 + y[original_day1, 0] - family1_size
        else:
            modified_nb_people[original_day1] -= family1_size
        if original_day1 not in past_y_coefficients.keys():
            past_y_coefficients[original_day1] = [y[original_day1, 0], y[original_day1, 1]]
        if original_day1 > 0 and original_day1 - 1 not in past_y_coefficients.keys():
            past_y_coefficients[original_day1-1] = [y[original_day1-1, 0], y[original_day1-1, 1]]
        if original_day1 not in new_y_coefficients.keys():
            new_y_coefficients[original_day1] = [y[original_day1, 0] - family1_size, y[original_day1, 1]]
        else:
            new_y_coefficients[original_day1][0] -= family1_size
        if original_day1 > 0 and original_day1 - 1 not in new_y_coefficients.keys():
            new_y_coefficients[original_day1-1] = [y[original_day1-1, 0], y[original_day1-1, 1] - family1_size]
        elif original_day1 > 0:
            new_y_coefficients[original_day1-1][1] -= family1_size
        
        # original_day2: no more family2
        if original_day2 not in modified_nb_people.keys():
            modified_nb_people[original_day2] = 125 + y[original_day2, 0] - family2_size
        else:
            modified_nb_people[original_day2] -= family2_size
        if original_day2 not in past_y_coefficients.keys():
            past_y_coefficients[original_day2] = [y[original_day2, 0], y[original_day2, 1]]
        if original_day2 > 0 and original_day2 - 1 not in past_y_coefficients.keys():
            past_y_coefficients[original_day2-1] = [y[original_day2-1, 0], y[original_day2-1, 1]]
        if original_day2 not in new_y_coefficients.keys():
            new_y_coefficients[original_day2] = [y[original_day2, 0] - family2_size, y[original_day2, 1]]
        else:
            new_y_coefficients[original_day2][0] -= family2_size
        if original_day2 > 0 and original_day2 - 1 not in new_y_coefficients.keys():
            new_y_coefficients[original_day2-1] = [y[original_day2-1, 0], y[original_day2-1, 1] - family2_size]
        elif original_day2 > 0:
            new_y_coefficients[original_day2-1][1] -= family2_size

        # new_day1: new family family1
        if new_day1 not in modified_nb_people.keys():
            modified_nb_people[new_day1] = 125 + y[new_day1, 0] + family1_size
        else:
            modified_nb_people[new_day1] += family1_size
        if new_day1 not in past_y_coefficients.keys():
            past_y_coefficients[new_day1] = [y[new_day1, 0], y[new_day1, 1]]
        if new_day1 > 0 and new_day1 - 1 not in past_y_coefficients.keys():
            past_y_coefficients[new_day1-1] = [y[new_day1-1, 0], y[new_day1-1, 1]]
        if new_day1 not in new_y_coefficients.keys():
            new_y_coefficients[new_day1] = [y[new_day1, 0] + family1_size, y[new_day1, 1]]
        else:
            new_y_coefficients[new_day1][0] += family1_size
        if new_day1 > 0 and new_day1 - 1 not in new_y_coefficients.keys():
            new_y_coefficients[new_day1-1] = [y[new_day1-1, 0], y[new_day1-1, 1] + family1_size]
        elif new_day1 > 0:
            new_y_coefficients[new_day1-1][1] += family1_size

        # new_day2: new family family2
        if new_day2 not in modified_nb_people.keys():
            modified_nb_people[new_day2] = 125 + y[new_day2, 0] + family2_size
        else:
            modified_nb_people[new_day2] += family2_size
        if new_day2 not in past_y_coefficients.keys():
            past_y_coefficients[new_day2] = [y[new_day2, 0], y[new_day2, 1]]
        if new_day2 > 0 and new_day2 - 1 not in past_y_coefficients.keys():
            past_y_coefficients[new_day2-1] = [y[new_day2-1, 0], y[new_day2-1, 1]]
        if new_day2 not in new_y_coefficients.keys():
            new_y_coefficients[new_day2] = [y[new_day2, 0] + family2_size, y[new_day2, 1]]
        else:
            new_y_coefficients[new_day2][0] += family2_size
        if new_day2 > 0 and new_day2 - 1 not in new_y_coefficients.keys():
            new_y_coefficients[new_day2-1] = [y[new_day2-1, 0], y[new_day2-1, 1] + family2_size]
        elif new_day2 > 0:
            new_y_coefficients[new_day2-1][1] += family2_size

    feasible = np.all([mnp >= 125 and mnp <= 300 for mnp in modified_nb_people.values()])
    if not feasible:
        continue
    
    deltaE_accounting = np.sum([accounting_cost(nd + 125, ndplus1 + 125) for (nd, ndplus1) in new_y_coefficients.values()]) - np.sum([accounting_cost(nd + 125, ndplus1 + 125) for (nd, ndplus1) in past_y_coefficients.values()])
    deltaE = deltaE_preference + deltaE_accounting
    if deltaE >= 0:
        acceptance_prob = np.exp(-deltaE/T(n))
        if np.random.rand() > acceptance_prob:
            continue
    # Accept the move: update x, y and overflow
    if pref1 != pref2:
        x[family1] = pref2
        x[family2] = pref1
        if pref2 == 10:
            overflow[family1] = np.argwhere(overflow_days[family1,:] == new_day1)
        if pref1 == 10:
            overflow[family2] = np.argwhere(overflow_days[family2,:] == new_day2)
    else:
        x[chosen_family] = new_pref
        if x[chosen_family] == 10:
            overflow[chosen_family] = np.argwhere(overflow_days[chosen_family,:] == new_day)
        if x[chosen_family] == 9 and which_sign == -1:
            overflow[chosen_family] = 0
    for idx, tab in new_y_coefficients.items():
        y[idx,:] = tab
    
    
    objective_value += deltaE
    if n % save_freq == 0:
        save_idx = n // save_freq
        print("Iteration "+str(n)+": current objective: "+str(objective_value))
        current_saving_folder = os.path.join(save_folder, str(save_idx))
        os.makedirs(current_saving_folder)
        np.save(os.path.join(current_saving_folder, "x.npy"), x)
        np.save(os.path.join(current_saving_folder, "overflow.npy"), overflow)
        np.save(os.path.join(current_saving_folder, "y.npy"), y)
        with open(os.path.join(current_saving_folder, "objective.txt"), "w") as f:
            f.write(str(objective_value))
    n += 1