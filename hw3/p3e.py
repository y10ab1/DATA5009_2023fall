import pandas as pd
import numpy as np
import random

# Load the distance table
distances_df = pd.read_csv('distances_table.csv', index_col=0)

# Define the Tabu Search function
def tabu_search(distances, iterations=100, tabu_list_length=10):
    cities = distances.columns.tolist()
    current_path = ['Incheon'] + random.sample(cities, len(cities))
    current_path.append('Incheon')  # Return to the starting city
    best_path = current_path
    best_distance = calculate_total_distance(distances, current_path)
    tabu_list = []
    iter_log = []

    for _ in range(iterations):
        neighbors = generate_neighbors(current_path)
        current_path, current_distance = best_neighbor(distances, neighbors, tabu_list, best_distance)

        # Update Tabu list
        if len(tabu_list) >= tabu_list_length:
            tabu_list.pop(0)
        tabu_list.append((current_path[-3], current_path[-2]))

        # Check for improvement
        if current_distance < best_distance:
            best_distance = current_distance
            best_path = current_path
        
        iter_log.append(best_distance)

    return best_path, best_distance, iter_log

# Function to generate neighbors by swapping two cities
def generate_neighbors(path):
    neighbors = []
    for i in range(1, len(path) - 2):
        for j in range(i+1, len(path) - 1):
            new_path = path[:]
            new_path[i], new_path[j] = new_path[j], new_path[i]
            neighbors.append(new_path)
    return neighbors

# Choose the best neighbor not in the tabu list
def best_neighbor(distances, neighbors, tabu_list, best_distance):
    best_path = None
    min_distance = float('inf')
    for neighbor in neighbors:
        if (neighbor[-3], neighbor[-2]) not in tabu_list:
            distance = calculate_total_distance(distances, neighbor)
            if distance < min_distance:
                best_path = neighbor
                min_distance = distance
        elif calculate_total_distance(distances, neighbor) < best_distance:  # Aspiration criterion
            return neighbor, calculate_total_distance(distances, neighbor)
    return best_path, min_distance

# Function to calculate the total distance of a path
def calculate_total_distance(distances, path):
    return sum(distances.loc[path[i], path[i+1]] for i in range(len(path)-1))

# Run the Tabu Search algorithm
best_path, best_distance, _ = tabu_search(distances_df)
print(f'Best path found: {best_path}')
print(f'Total distance: {best_distance}')
