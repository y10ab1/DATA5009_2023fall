import random
import numpy as np

# Define the GA parameters
POPULATION_SIZE = 10
NUM_GENERATIONS = 200
CROSSOVER_PROB = 0.1
MUTATION_PROB = 0.1  # Adjust this as needed for mutation
NUM_ITEMS = 15  # Number of items based on the problem

# Item weights and survival points (extracted from the problem statement)
weights = [3.3, 3.4, 6.0, 26.1, 37.6, 62.5, 100.2, 141.1, 119.2, 122.4, 247.6, 352.0, 24.2, 32.1, 42.5]
survival_points = [7, 8, 13, 29, 48, 99, 177, 213, 202, 210, 380, 485, 9, 12, 15]
weight_limit = 529

# Define bonus points
bonus_combinations = {
    'daggers_magnum': ([0, 5], 5),
    'compact_primary': ([3, 9, 10], 15),
    'shotgun_rifle_magnum_shield': ([7, 10, 5, 14], 25),
    'all_equipment': ([12, 13, 14], 70)
}

# Initialize population
def initialize_population(pop_size, num_items):
    return np.random.randint(2, size=(pop_size, num_items))

# Fitness function
def fitness(chromosome):
    total_weight = np.dot(chromosome, weights)
    total_survival_points = np.dot(chromosome, survival_points)
    
    # Apply bonus points
    for key, value in bonus_combinations.items():
        items, bonus = value
        if all(chromosome[i] == 1 for i in items):
            total_survival_points += bonus
    
    # Penalize if weight exceeds the limit
    if total_weight > weight_limit:
        total_survival_points = 0
    # Penalize if the combination does not contain at least 1 knife, 1 pistol, and 1 equipment
    if not all([np.sum(chromosome[0:3]) >= 1, np.sum(chromosome[3:6]) >= 1, np.sum(chromosome[12:15]) >= 1]):
        total_survival_points = 0
    return total_survival_points

# Roulette-wheel selection
def selection(population, fitnesses):
    # Make sure the total fitness is not 0
    if np.sum(fitnesses) == 0:
        fitnesses = np.ones(len(fitnesses))
    # Make sure we select an even number of parents for crossover
    selected_parents = random.choices(population, weights=fitnesses, k=len(population))
    if len(selected_parents) % 2 != 0:
        selected_parents.append(random.choices(population, weights=fitnesses, k=1)[0])
    return selected_parents

# Uniform crossover
def crossover(parent1, parent2):
    child = [p1 if random.random() > CROSSOVER_PROB else p2 for p1, p2 in zip(parent1, parent2)]
    return child

# Mutation function
def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_PROB:
            chromosome[i] = 1 if chromosome[i] == 0 else 0
    return chromosome

# Main GA function
def genetic_algorithm():
    # Initialize the population
    population = initialize_population(POPULATION_SIZE, NUM_ITEMS)
    
    # Run the GA for the specified number of generations
    for _ in range(NUM_GENERATIONS):
        # Calculate fitness for each individual
        fitnesses = np.array([fitness(ind) for ind in population])
        
        # Select parents based on fitness
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
    
    # Find the best solution
    final_fitnesses = np.array([fitness(ind) for ind in population])
    best_index = np.argmax(final_fitnesses)
    best_individual = population[best_index]
    return best_individual, final_fitnesses[best_index]

# Run the GA and get the best solution
best_solution, best_solution_fitness = genetic_algorithm()
print(f'Best solution: {best_solution}')
print(f'Best solution fitness: {best_solution_fitness}')
