from matplotlib import pyplot as plt
import numpy as np

objectives = []
iteration_numbers = []

with open("../solutions/objective_sequences/simulated_annealing_1.txt", "r") as f:
    for line in f:
        line = line.split(":")
        if len(line) == 2:
            continue
        else:
            iteration_number = int(line[0].strip("Iteration "))
            iteration_numbers.append(iteration_number)
            objective = float(line[2].strip())
            objectives.append(objective)

plt.plot(iteration_numbers, objectives, label='Objective')

# Set labels and title
plt.xlabel('Iteration Number')
plt.ylabel('Objective')
plt.title('Simulated Annealing search')
plt.axhline(y=objectives[0], color='red', linestyle='--', label='Starting Objective')
# Add legend
plt.legend()

# Show the plot
plt.show()