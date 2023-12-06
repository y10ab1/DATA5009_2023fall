import pandas as pd
import numpy as np
import random

# Load the distance table
distances = pd.read_csv('distances_table.csv', index_col=0)
    
def ACO(distances, num_ants=10, num_iterations=100, decay=0.5, alpha=1, beta=2):
    # Initialize pheromones
    pheromones = pd.DataFrame(1.0, index=distances.index, columns=distances.columns)

    def select_next_city(current_city, unvisited, pheromones, distances):
        pheromones = np.power(pheromones.loc[current_city, list(unvisited)], alpha)
        distances = np.power(1 / distances.loc[current_city, list(unvisited)], beta)
        probabilities = pheromones * distances
        probabilities /= probabilities.sum()
        next_city = np.random.choice(list(unvisited), p=probabilities)
        return next_city

    def update_pheromones(pheromones, ants_paths, distances):
        for path in ants_paths:
            total_distance = sum(distances.loc[path[i], path[i + 1]] for i in range(len(path) - 1))
            pheromone_deposit = 1 / total_distance
            for i in range(len(path) - 1):
                pheromones.loc[path[i], path[i + 1]] += pheromone_deposit
        return pheromones

    # Ant Colony Optimization
    best_path = None
    best_distance = float('inf')
    iter_log = []
    for iteration in range(num_iterations):
        ants_paths = []
        for ant in range(num_ants):
            path = ['Incheon']
            unvisited = set(distances.columns.drop('Incheon'))
            
            while unvisited:
                next_city = select_next_city(path[-1], unvisited, pheromones, distances)
                path.append(next_city)
                unvisited.remove(next_city)

            path.append('Incheon')
            ants_paths.append(path)
            path_distance = sum(distances.loc[path[i], path[i + 1]] for i in range(len(path) - 1))

            if path_distance < best_distance:
                best_path = path
                best_distance = path_distance
            
        iter_log.append(best_distance)

        pheromones *= decay  # Pheromone evaporation
        pheromones = update_pheromones(pheromones, ants_paths, distances)
    
    return best_path, best_distance, iter_log
best_path, best_distance, _ = ACO(distances)
print(f'Best path found: {best_path}')
print(f'Total distance: {best_distance}')
