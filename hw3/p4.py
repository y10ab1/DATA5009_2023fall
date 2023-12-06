import pandas as pd
import numpy as np
import random

# Load the distance table
distances = pd.read_csv('distances_table.csv', index_col=0)

def calculate_total_distance(path):
    """ Calculate the total distance of a path """
    return sum(distances.loc[path[i], path[i+1]] for i in range(len(path) - 1))

# def generate_initial_velocity(size):
#     """ Generate an initial velocity for a particle """

def update_velocity(current_velocity, pBest, gBest, current_position, w=0.5, c1=1, c2=2, r1=0.5, r2=0.5, v_max=4, v_min=-4):
    """ Update the velocity of a particle """
    dist_pBest = calculate_total_distance(pBest)
    dist_gBest = calculate_total_distance(gBest)
    dist_current_position = calculate_total_distance(current_position)
    new_velocity = w * current_velocity + c1 * r1 * (dist_pBest - dist_current_position) + c2 * r2 * (dist_gBest - dist_current_position)
    if new_velocity > v_max:
        new_velocity = v_max
    elif new_velocity < v_min:
        new_velocity = v_min
    return new_velocity


def apply_velocity_to_path(path, velocity):
    """ Apply the velocity (sequence of swaps) to the path """
    path = path.copy()
    if velocity == 0:
        return path
    else:
        # For a high velocity, we want to swap more cities in the current path
        for _ in range(abs(int(velocity))):
            i = random.randint(1, len(path) - 2)
            j = random.randint(1, len(path) - 2)
            path[i], path[j] = path[j], path[i]

    return path


def pso_tsp(distances, num_particles=50, num_iterations=100):
    cities = distances.columns.tolist()
    # Initialize particles and velocities
    particles = [['Incheon'] + random.sample(cities[1:-1], len(cities) - 2) + ['Incheon'] for _ in range(num_particles)]
    velocities = [0 for _ in range(num_particles)]
    personal_best = particles.copy()
    global_best = min(particles, key=calculate_total_distance)
    iter_log = []
    for _ in range(num_iterations):
        for i in range(num_particles):
            # Update velocity
            
            velocities[i] = update_velocity(velocities[i], personal_best[i], global_best, particles[i])
            # Apply velocity to particle
            particles[i] = apply_velocity_to_path(particles[i], velocities[i])

            # Update personal and global bests
            current_distance = calculate_total_distance(particles[i])
            if current_distance < calculate_total_distance(personal_best[i]):
                personal_best[i] = particles[i]
            if current_distance < calculate_total_distance(global_best):
                global_best = particles[i]
                
        iter_log.append(calculate_total_distance(global_best))

    best_path = global_best
    best_distance = calculate_total_distance(best_path)
    return best_path, best_distance, iter_log

# Run PSO for TSP
best_path, best_distance, _ = pso_tsp(distances)
print(f"Best path: {best_path}")
print(f"Total distance: {best_distance}")
