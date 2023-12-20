import numpy as np
import random
import pandas as pd
import logging
import argparse
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Load task and transfer tables
task_table = pd.read_csv("task_table.csv", index_col="Oij")
transfer_table = pd.read_csv("transfer_table.csv", index_col="From/To")
logger.info(task_table)
logger.info(transfer_table)



def schedule(chromosome, idx, gantt, job_status_machine, job_status_time):
    # Function to schedule a job step on the best available machine
    jobid = chromosome[idx]
    step = np.sum(np.array(chromosome[:idx+1]) == jobid)

    time_cost_list = []
    for machine in range(5):
        try:
            process_time, setup_time = task_table.loc[f"O{jobid}{step}", f"M{machine+1}"].split("/")
            delivery_time = transfer_table.loc[f"M{job_status_machine[jobid-1]}", f"M{machine+1}"]
        except KeyError:
            logger.error("Invalid task or transfer table data")
            return -1
        setup_time = 0 if machine + 1 == job_status_machine[jobid - 1] else int(setup_time) # if on same machine, setup time = 0
        setup_time = 100000 if (job_status_machine[jobid - 1] != 0 and setup_time > delivery_time) else 0 # if not first step and setup time > delivery time, setup time = 100000
        
        time_cost = int(process_time) + setup_time + int(delivery_time)
        time_cost = 100000 if gantt[machine, job_status_time[jobid - 1]] != 0 else time_cost
        time_cost_list.append((time_cost, machine + 1))

    logging.debug(f"Job {jobid} step {step} time cost list: {time_cost_list}")
    min_time_cost = min(time_cost_list, key=lambda x: x[0])[0]
    candidate_machines = [machine for time_cost, machine in time_cost_list if time_cost == min_time_cost]
    selected_machine = random.choice(candidate_machines) if candidate_machines else -1
    update_gantt(jobid, step, selected_machine, min_time_cost, gantt, job_status_machine, job_status_time)

    return selected_machine

def update_gantt(jobid, step, machine, time, gantt, job_status_machine, job_status_time):
    # Update Gantt chart and job status
    current_time = job_status_time[jobid - 1]
    logging.debug(f"End time of job {jobid} step {step}: {current_time + time}")
    if current_time + time <= gantt.shape[1]:
        gantt[machine - 1, current_time:current_time + time] = jobid
        job_status_machine[jobid - 1] = machine
        job_status_time[jobid - 1] = current_time + time
    else:
        logger.warning(f"Unable to schedule job {jobid} at step {step} due to time constraints")

def evaluate(gantt):
    # Evaluate the Gantt chart and return the makespan
    return np.max(np.where(np.sum(gantt, axis=0) != 0, 1, 0).cumsum(axis=0))


def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        chromosome = chromosome = [1, 1, 1, 2, 2, 2, 3, 3, 4, 3, 4, 3]
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def schedule_chromosome(chromosome, gantt, job_status_machine, job_status_time):
    # Scheduling process
    for idx in range(len(chromosome)):
        schedule(chromosome, idx, gantt, job_status_machine, job_status_time)
    return gantt

def calculate_fitness(chromosome):
    # You should integrate this with your existing scheduling and evaluation logic
    # Returns the inverse of makespan
    
    # Initialize Gantt chart and job status
    gantt = np.zeros((5, 50), dtype=int)  # 5 machines, 40 time slots
    job_status_machine = np.zeros(4, dtype=int)  # 0: not started, > 0: machine ID
    job_status_time = np.zeros(4, dtype=int)  # 0: not started, >= 0: time slot
    new_gantt = schedule_chromosome(chromosome, gantt, job_status_machine, job_status_time)
    return 1.0 / evaluate(new_gantt), new_gantt


def tournament_selection(population, fitness, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament = np.random.choice(len(population), size=tournament_size, replace=False)
        best_in_tournament = tournament[np.argmax(fitness[tournament])]
        selected_parents.append(population[best_in_tournament])
    return np.array(selected_parents)

def mutate(chromosome, mutation_rate):
    for _ in range(len(chromosome)):
        if random.random() < mutation_rate:
            idx1, idx2 = np.random.randint(len(chromosome), size=2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def order_crossover(parent1, parent2):
    size = len(parent1)
    # Initialize children with empty values
    child1, child2 = np.empty(size, dtype=parent1.dtype), np.empty(size, dtype=parent2.dtype)
    child1.fill(-1)
    child2.fill(-1)

    # Randomly select a subset of the first parent
    start, end = sorted(random.sample(range(size), 2))
    child1[start:end], child2[start:end] = parent1[start:end], parent2[start:end]

    # Fill the remaining positions
    fill_remaining(child1, parent2, start, end)
    fill_remaining(child2, parent1, start, end)
    return child1, child2

def fill_remaining(child, parent, start, end):
    size = len(child)
    tofill = parent[:start].tolist() + parent[end:].tolist()
    
    for i in range(size):
        if child[i] == -1:
            child[i] = tofill.pop(0)
            


def genetic_algorithm(pop_size, max_generations, mutation_rate):
    population = initialize_population(pop_size)
    best_solution = None
    best_fitness = 0
    best_gantt = None
    gantt_list = []
    fitness_list = []

    for generation in range(max_generations):
        # fitness = np.array([calculate_fitness(chromosome) for chromosome in population])
        
        for chromosome in population:
            f, g = calculate_fitness(chromosome)
            fitness_list.append(f)
            gantt_list.append(g)
        fitness = np.array(fitness_list)
        parents = tournament_selection(population, fitness, tournament_size=3)

        # Generate next generation
        next_population = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1, child2 = order_crossover(parent1, parent2)
            next_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = next_population
        

        # Check for new best solution
        current_best = np.argmax(fitness)
        print(fitness)
        if fitness[current_best] > best_fitness:
            best_fitness = fitness[current_best]
            best_solution = population[current_best]
            best_gantt = gantt_list[current_best]

    return best_solution, best_fitness, best_gantt

# Main function
def main(args):
    # Genetic Algorithm parameters
    pop_size = 10
    max_generations = 1
    mutation_rate = 0.1

    best_solution, best_fitness, best_gantt = genetic_algorithm(pop_size, max_generations, mutation_rate)
    logger.info(f"Best solution found: {best_solution}, makespan: {1.0 / best_fitness}")
    logger.info(f"Best Gantt chart:\n{best_gantt}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug messages")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    main(args)
