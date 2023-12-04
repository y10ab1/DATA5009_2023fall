import matplotlib.pyplot as plt
from p2e import *
from p2cd import *

# We need to modify the hill climbing and random walk algorithms to keep track of the best fitness over time.

def hill_climbing_tracking():
    current_solution = initialize_population(1, NUM_ITEMS)[0]
    current_fitness = fitness(current_solution)
    best_fitness_over_time = [current_fitness]
    
    for _ in range(200):  # Maximum number of iterations
        neighbor = mutate(current_solution[:])
        neighbor_fitness = fitness(neighbor)
        
        # If the neighboring solution is better, move to it and record the fitness
        if neighbor_fitness > current_fitness:
            current_solution, current_fitness = neighbor, neighbor_fitness
        best_fitness_over_time.append(current_fitness)
    
    return best_fitness_over_time

def random_walk_tracking():
    current_solution = initialize_population(1, NUM_ITEMS)[0]
    current_fitness = fitness(current_solution)
    fitness_over_time = [current_fitness]
    
    for _ in range(200):  # Maximum number of iterations
        current_solution = mutate(current_solution[:])
        current_fitness = fitness(current_solution)
        fitness_over_time.append(current_fitness)
    
    return fitness_over_time

# Modify genetic_algorithm to track best fitness over time
def genetic_algorithm_tracking():
    population = initialize_population(POPULATION_SIZE, NUM_ITEMS)
    best_fitness_over_time = []
    
    for _ in range(NUM_GENERATIONS):
        fitnesses = np.array([fitness(ind) for ind in population])
        best_fitness_over_time.append(max(fitnesses))
        
        parents = selection(population, fitnesses)
        
        # Create the next generation using crossover
        next_generation = []
        for i in range(0, len(parents), 2):
            child1 = crossover(parents[i], parents[i+1])
            child2 = crossover(parents[i], parents[i+1])
            next_generation.extend([child1, child2])
        
        # Apply mutation to the new generation
        next_generation = [mutation(child) for child in next_generation[:POPULATION_SIZE]]
        
        # Replace the old population with the new generation
        population = np.array(next_generation)
    
    return best_fitness_over_time

# Collect data
ga_progress = genetic_algorithm_tracking()
hc_progress = hill_climbing_tracking()
rw_progress = random_walk_tracking()

# Plot progress diagrams
plt.figure(figsize=(12, 6))
plt.plot(ga_progress, label='Genetic Algorithm')
plt.plot(hc_progress, label='Hill Climbing')
plt.plot(rw_progress, label='Random Walk')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Best Objective Function Value')
plt.title('Progress Diagrams of GA, Hill Climbing, and Random Walk')
plt.legend()
plt.savefig('p2f.png')