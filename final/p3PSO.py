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
    gantt = np.zeros((5, 35), dtype=int)  # 5 machines, 40 time slots
    job_status_machine = np.zeros(4, dtype=int)  # 0: not started, > 0: machine ID
    job_status_time = np.zeros(4, dtype=int)  # 0: not started, >= 0: time slot
    new_gantt = schedule_chromosome(chromosome, gantt, job_status_machine, job_status_time)
    return 1.0 / evaluate(new_gantt), new_gantt
##############################################################################################################
def update_velocity(velocity, pbest, gbest, chromosome, w=0.5, c1=1, c2=1):
    """Update the velocity based on pbest, gbest, and current position."""
    new_velocity = []
    for v, p, pb, gb in zip(velocity, chromosome, pbest, gbest):
        new_v = w * v + c1 * random.random() * (pb - p) + c2 * random.random() * (gb - p)
        new_velocity.append(new_v)
    return new_velocity

def update_position(chromosome, velocity):
    """Update the position based on the velocity."""
    new_position = chromosome.copy()
    swap_indices = [i for i, v in enumerate(velocity) if v > random.random()]
    for i in swap_indices:
        swap_with = (i + 1) % len(chromosome)
        new_position[i], new_position[swap_with] = new_position[swap_with], new_position[i]
    return new_position

def particle_swarm_optimization(pop_size, iterations):
    # Initialize particles
    particles = initialize_population(pop_size)
    velocities = [np.zeros(len(p)) for p in particles]
    pbest = particles.copy()
    gbest, best_gantt, best_fitness = None, None, None
    for p in particles:
        fitness, gantt = calculate_fitness(p)
        if gbest is None or fitness > calculate_fitness(gbest)[0]:
            gbest = p
            best_gantt = gantt
            best_fitness = fitness
            
    for _ in range(iterations):
        for i, particle in enumerate(particles):
            fitness, gantt = calculate_fitness(particle)
            if fitness > calculate_fitness(pbest[i])[0]:
                pbest[i] = particle
                
            if fitness > calculate_fitness(gbest)[0]:
                gbest = particle
                best_gantt = gantt
                best_fitness = fitness
                
            velocities[i] = update_velocity(velocities[i], pbest[i], gbest, particle)
            particles[i] = update_position(particle, velocities[i])

    return gbest, best_fitness, best_gantt

# Main function
def PSO(args, pop_size=2, iterations=1):

    best_solution, best_fitness, best_gantt = particle_swarm_optimization(pop_size, iterations)
    logger.info(f"Best solution found: {best_solution}, makespan: {1.0 / best_fitness}")
    logger.info(f"Best Gantt chart:\n{best_gantt}")
    return best_solution, best_fitness, best_gantt
    

if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    # Load task and transfer tables
    task_table = pd.read_csv("task_table.csv", index_col="Oij")
    transfer_table = pd.read_csv("transfer_table.csv", index_col="From/To")
    logger.info(task_table)
    logger.info(transfer_table)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug messages")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    PSO(args)
