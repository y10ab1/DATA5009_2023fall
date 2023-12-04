import pandas as pd
import numpy as np
import random

# Load the distance table
distances_table = pd.read_csv('distances_table.csv', index_col=0)

# Define the random walk function
def random_walk(distances, num_iterations=100):
    cities = distances.columns.tolist()
    best_path = None
    best_distance = float('inf')

    for _ in range(num_iterations):
        path = ['Incheon'] + random.sample(cities, len(cities))
        path.append('Incheon')  # Return to starting city
        total_distance = sum(distances[path[i]][path[i+1]] for i in range(len(path)-1))

        if total_distance < best_distance:
            best_distance = total_distance
            best_path = path

    return best_path, best_distance

# Run the random walk algorithm
best_path, best_distance = random_walk(distances_table)
print(f'Best path: {best_path}')
print(f'Total distance: {best_distance}')
