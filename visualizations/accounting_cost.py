import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def accounting_cost(n_d, n_dplus1):
    return ((n_d-125)/400)*n_d**(1/2+abs(n_d-n_dplus1)/50)
# Define the range of values for n_d and n_dplus1

n_d_values = np.arange(125, 301)
n_dplus1_values = np.arange(125, 301)

# Create a grid of values for n_d and n_dplus1
n_d_grid, n_dplus1_grid = np.meshgrid(n_d_values, n_dplus1_values)

# Calculate the accounting cost for each combination of n_d and n_dplus1
costs = accounting_cost(n_d_grid, n_dplus1_grid)

# Create a figure and axes
fig, ax = plt.subplots()

# Create the heatmap
norm = colors.LogNorm(vmin=0.01, vmax=25073)
heatmap = ax.imshow(costs, cmap='viridis', interpolation='nearest', norm=norm)

# Remove the axis ticks and labels
ax.set_xticks([])
ax.set_yticks([])

# Add a colorbar
cbar = fig.colorbar(heatmap)
ticks = cbar.get_ticks().astype(int).astype(str)
ticks[-1] = "> 25073"
cbar.set_ticklabels(ticks)

# Set the title
ax.set_title('Accounting cost structure')
ax.set_ylabel('Number of people assigned on one day')
ax.set_xlabel('Number of people assigned on the following day')

# Show the plot
plt.show()