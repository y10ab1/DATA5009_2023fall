import random
import numpy as np
from p2cd import *

# Define mutation operator for hill climbing and random walk
def mutate(chromosome):
    # Select a random index to mutate
    index = random.randrange(len(chromosome))
    # Flip the bit at the selected index
    chromosome[index] = 1 - chromosome[index]
    return chromosome

# Hill Climbing Algorithm
def hill_climbing():
    current_solution = initialize_population(1, NUM_ITEMS)[0]
    current_fitness = fitness(current_solution)
    
    for _ in range(200):  # Maximum number of iterations
        neighbor = mutate(current_solution[:])
        neighbor_fitness = fitness(neighbor)
        
        # If the neighboring solution is better, move to it
        if neighbor_fitness > current_fitness:
            current_solution, current_fitness = neighbor, neighbor_fitness
    
    return current_solution, current_fitness

# Random Walk Algorithm
def random_walk():
    current_solution = initialize_population(1, NUM_ITEMS)[0]
    current_fitness = fitness(current_solution)
    
    for _ in range(200):  # Maximum number of iterations
        # Unlike hill climbing, always move to the neighbor regardless of its fitness
        current_solution = mutate(current_solution[:])
        current_fitness = fitness(current_solution)
    return current_solution, current_fitness

# Run the Hill Climbing Algorithm
hc_solution, hc_fitness = hill_climbing()
print(f'Hill Climbing Best Solution: {hc_solution}')
print(f'Hill Climbing Best Solution Fitness: {hc_fitness}')

# Run the Random Walk Algorithm
rw_solution, rw_fitness = random_walk()
print(f'Random Walk Best Solution: {rw_solution}')
print(f'Random Walk Best Solution Fitness: {rw_fitness}')
