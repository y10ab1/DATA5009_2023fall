import pandas as pd
import numpy as np
import random
import math

# Load the distance table
distances_df = pd.read_csv('distances_table.csv', index_col=0)

# Define the Simulated Annealing function
def simulated_annealing(distances, initial_temp=1000, cooling_rate=0.99, iterations=1000):
    cities = distances.columns.tolist()
    current_path = ['Incheon'] + random.sample(cities, len(cities))
    current_path.append('Incheon')  # Return to the starting city
    current_distance = calculate_total_distance(distances, current_path)
    best_path, best_distance = current_path, current_distance
    iter_log = []

    temperature = initial_temp

    for i in range(iterations):
        new_path = generate_neighbor(current_path)
        new_distance = calculate_total_distance(distances, new_path)

        if acceptance_probability(current_distance, new_distance, temperature) > random.random():
            current_path, current_distance = new_path, new_distance

            if new_distance < best_distance:
                best_path, best_distance = new_path, new_distance

        iter_log.append(best_distance)
        temperature *= cooling_rate

    return best_path, best_distance, iter_log

# Calculate acceptance probability
def acceptance_probability(old_distance, new_distance, temperature):
    if new_distance < old_distance:
        return 1.0
    else:
        return math.exp((old_distance - new_distance) / temperature)

# Generate a neighbor solution
def generate_neighbor(path):
    a, b = np.random.randint(1, len(path) - 1, size=2)
    path[a], path[b] = path[b], path[a]
    return path

# Calculate the total distance of a path
def calculate_total_distance(distances, path):
    return sum(distances.loc[path[i], path[i+1]] for i in range(len(path)-1))

# Run the Simulated Annealing algorithm
best_path, best_distance, _ = simulated_annealing(distances_df)
print(f'Best path found: {best_path}')
print(f'Total distance: {best_distance}')
