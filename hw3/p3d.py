import pandas as pd
import random

# Load the distance table
distances_df = pd.read_csv('distances_table.csv', index_col=0)


# Define the hill climbing function
def hill_climbing(distances, iterations=100):
    cities = distances.columns.tolist()
    current_path = ['Incheon'] + random.sample(cities, len(cities))
    current_path.append('Incheon')  # Return to the starting city
    best_distance = calculate_total_distance(distances, current_path)
    best_path = current_path
    iter_log = []

    for _ in range(iterations):
        # Generate a neighbor path by swapping two cities
        new_path = current_path[:]
        a, b = random.sample(range(1, len(cities)), 2)  # Exclude 'Incheon' from swapping
        new_path[a], new_path[b] = new_path[b], new_path[a]
        new_distance = calculate_total_distance(distances, new_path)
        
        # If the new path has a shorter distance, accept it as the current best path
        if new_distance < best_distance:
            current_path = new_path
            best_distance = new_distance
            best_path = new_path

        iter_log.append(best_distance)
    return best_path, best_distance, iter_log

# Function to calculate the total distance of a path
def calculate_total_distance(distances, path):
    return sum(distances.loc[path[i], path[i+1]] for i in range(len(path)-1))

# Run the hill climbing algorithm
best_path, best_distance, _ = hill_climbing(distances_df)
print(f'Best path found: {best_path}')
print(f'Total distance: {best_distance}')
