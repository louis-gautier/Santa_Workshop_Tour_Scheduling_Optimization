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
N = 5000

population_size = 100
def initialize_population():
    solution_population = []
    data_folder = "solutions/ga_initialization/"
    for subfolder_name in range(100):
        subfolder_path = os.path.join(data_folder, str(subfolder_name))
        x_file = os.path.join(subfolder_path, "x.npy")
        y_file = os.path.join(subfolder_path, "y.npy")
        overflow_file = os.path.join(subfolder_path, "overflow.npy")
        objective_file = os.path.join(subfolder_path, "objective.txt")
        x = np.load(x_file)
        y = np.load(y_file)
        overflow = np.load(overflow_file)
        with open(objective_file, "r") as f:
            objective = float(f.readline())
        data_dict = {"x": x, "y": y, "overflow": overflow, "objective": objective}
        solution_population.append(data_dict)
    return solution_population

def custom_sort_key(dictionary):
    return dictionary["objective"]

def select(solution_population):
    solution_population = sorted(solution_population, key=custom_sort_key)
    return solution_population[:population_size]

def get_objective(x, y):
    pref_cost = np.sum([preference_cost(x[i], get_size(i)) for i in range(nb_families)])
    acc_cost = np.sum(accounting_cost(y[:, 0] + 125, y[:, 1] + 125))
    return pref_cost + acc_cost


def crossover(solution_population, parent1, parent2):
    attempt_idx = 1
    max_attempt = 500
    parent1_x = solution_population[parent1]["x"]
    parent1_overflow = solution_population[parent1]["overflow"]
    parent2_x = solution_population[parent2]["x"]
    parent2_overflow = solution_population[parent2]["overflow"]
    parents_x = np.vstack((parent1_x, parent2_x))
    parents_overflow = np.vstack((parent1_overflow, parent2_overflow))
    while attempt_idx < max_attempt:
        selected_preferences = np.random.binomial(1, 0.5, size=nb_families)
        child_x = parents_x[selected_preferences, np.arange(nb_families)]
        child_overflow = parents_overflow[selected_preferences, :]
        child_assignment = np.zeros(nb_days)
        for family_idx, pref in enumerate(child_x):
            if pref < 10:
                child_assignment[preferences[family_idx, pref]] += get_size(family_idx)
            else:
                child_assignment[overflow_days[child_overflow[family_idx]]] += get_size(family_idx)
        feasible = np.all((child_assignment >= 125) & (child_assignment <= 300))
        child_assignment = (child_assignment - 125).astype(int)
        if feasible:
            child_y = np.zeros((nb_days, 2))
            child_y[:, 0] = child_assignment
            child_y[:-1, 1] = child_assignment[1:]
            child_y[-1, 1] = child_assignment[-1]
            child_objective = get_objective(child_x, child_y)
            child = {"x": child_x, "overflow": child_overflow, "y": child_y, "objective": child_objective}
            solution_population.append(child)
            return True, solution_population
        else:
            attempt_idx += 1
    return False, solution_population

def mutate(solution_population, individual):
    family_idx = np.random.randint(0, nb_families)
    which_sign = 2*np.random.binomial(1, 0.5) - 1
    x = solution_population[individual]["x"]
    overflow = solution_population[individual]["overflow"]
    y = solution_population[individual]["y"]
    old_pref = x[family_idx]
    family_size = get_size(family_idx)
    if old_pref == 0:
        which_sign = 1
    if old_pref == 10:
        which_sign = - 1
        original_day = overflow_days[overflow[family_idx]]
    else:
        original_day = preferences[family_idx, x[family_idx]]
    new_pref = x[family_idx] + which_sign
    if new_pref < 10:
        new_day = preferences[family_idx, new_pref]
    else:
        new_day = np.random.choice(overflow_days[family_idx,:])
    modified_nb_people = {}
    modified_nb_people[original_day] = 125 + y[original_day, 0] - family_size
    modified_nb_people[new_day] = 125 + y[new_day, 0] + family_size
    feasible = np.all([mnp >= 125 and mnp <=300 for mnp in modified_nb_people.values()])
    if feasible:
        past_y_coefficients = {}
        new_y_coefficients = {}
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
        deltaE_accounting = np.sum([accounting_cost(nd + 125, ndplus1 + 125) for (nd, ndplus1) in new_y_coefficients.values()]) - np.sum([accounting_cost(nd + 125, ndplus1 + 125) for (nd, ndplus1) in past_y_coefficients.values()])
        deltaE = deltaE_preference + deltaE_accounting
        for idx, tab in new_y_coefficients.items():
            y[idx,:] = tab
        x[family_idx] = new_pref
        if x[family_idx] == 10:
            overflow[family_idx] = np.argwhere(overflow_days[family_idx,:] == new_day)
        if x[family_idx] == 9 and which_sign == -1:
            overflow[family_idx] = 0
        solution_population[individual]["x"] = x
        solution_population[individual]["y"] = y
        solution_population[individual]["overflow"] = overflow
        solution_population[individual]["objective"] += deltaE
        return True, solution_population
    else:
        return False, solution_population

def save_best_solution(solution_population, iteration, best_so_far):
    best_solution = min(solution_population, key=custom_sort_key)
    mean_pop = np.mean(np.array(list(map(custom_sort_key, solution_population))))
    std_pop = np.std(np.array(list(map(custom_sort_key, solution_population))))
    if best_solution["objective"] < best_so_far:
        np.save("solutions/ga/0/x.npy", best_solution["x"])
        np.save("solutions/ga/0/overflow.npy", best_solution["overflow"])
        np.save("solutions/ga/0/y.npy", best_solution["y"])
        best_so_far = best_solution["objective"]
    print("Mean objective in the population at iteration "+str(iteration)+": "+str(mean_pop))
    print("Std objective in the population at iteration "+str(iteration)+": "+str(std_pop))
    print("Best objective in the population at iteration "+str(iteration)+": "+str(best_solution["objective"]))
    return best_so_far

solution_population = initialize_population()
previous_solution_population = None
n = 0
best_objective = min(solution_population, key=custom_sort_key)["objective"]
nb_trials = 100
while True:
    solution_population = select(solution_population)
    best_objective = save_best_solution(solution_population, n, best_objective)
    if previous_solution_population is not None and sorted(solution_population, key=custom_sort_key) == sorted(previous_solution_population, key=custom_sort_key):
        break
    previous_solution_population = solution_population
    for crossover_idx in range(population_size):
        crossover_success = False
        parent1, parent2 = np.random.choice(range(population_size), size=2, replace=False)
        trial_idx = 0
        while trial_idx <= nb_trials and not crossover_success:
            crossover_success, solution_population = crossover(solution_population, parent1, parent2)
            trial_idx += 1
        if trial_idx >= nb_trials:
            print("crossover failure between "+str(parent1)+" and "+str(parent2))
    
    for mutate_idx in range(int(0.3*population_size)):
        mutate_success = False
        individual = np.random.randint(0, len(solution_population))
        trial_idx = 0
        while trial_idx <= nb_trials and not mutate_success:
            mutate_success, solution_population = mutate(solution_population, individual)
            trial_idx += 1
        if trial_idx >= nb_trials:
            print("mutate failure for "+str(individual))
    best_objective = save_best_solution(solution_population, n, best_objective)
    
    n += 1