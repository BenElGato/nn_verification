import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Load the CSV file
file_path = '/home/benedikt/PycharmProjects/nn_verification/ACC/cora/results.csv'
df = pd.read_csv(file_path, header=None)

def validate_values(df):
    unique_values = np.unique(df.values)
    valid_values = {-1, 0, 1}
    if not all(value in valid_values for value in unique_values):
        raise ValueError("Data contains values other than -1, 0, 1")

validate_values(df)

cmap = ListedColormap(['red', 'orange', 'green'])

# Plotting the grid
plt.figure(figsize=(12, 8))
plt.imshow(df, cmap=cmap, interpolation='none', vmin=-1, vmax=1,aspect='equal')
plt.xticks(np.arange(-0.5, len(df.columns), 1), [])
plt.yticks(np.arange(-0.5, len(df), 1), [])
plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
plt.tick_params(axis='both', which='both', length=0)

plt.savefig('/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/results30a.png' )

