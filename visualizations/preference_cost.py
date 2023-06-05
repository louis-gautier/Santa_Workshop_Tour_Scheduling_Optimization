import matplotlib.pyplot as plt

def preference_cost(p, s):
    fixed_costs = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
    marginal_costs = [0, 0, 9, 9, 9, 18, 18, 36, 36, 235, 434]
    return fixed_costs[p] + s*marginal_costs[p]

s_values = range(2, 9)  # Values of s from 2 to 8
p_values = range(11)  # Values of p from 0 to 10

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the line plots for different values of s
for s in s_values:
    costs = [preference_cost(p, s) for p in p_values]
    ax.plot(p_values, costs, label=s)

# Set labels and title
ax.set_xlabel('Assigned preference')
ax.set_ylabel('Preference Cost')
ax.set_title('Preference cost structure')

# Add legend
ax.legend(title="Family size")

# Show the plot
plt.show()