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

# Initialize Gantt chart and job status
gantt = np.zeros((5, 40), dtype=int)  # 5 machines, 40 time slots
job_status_machine = np.zeros(4, dtype=int)  # 0: not started, > 0: machine ID
job_status_time = np.zeros(4, dtype=int)  # 0: not started, >= 0: time slot

# Shuffle the chromosome
chromosome = [1, 1, 1, 2, 2, 2, 3, 3, 4, 3, 4, 3]
random.shuffle(chromosome)
logger.info(f"Shuffled Chromosome: {chromosome}")

def schedule(chromosome, idx, gantt):
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

        setup_time = 0 if machine + 1 == job_status_machine[jobid - 1] else int(setup_time)
        setup_time = 100000 if (job_status_machine[jobid - 1] != 0 and setup_time > delivery_time) else 0 # if not first step and setup time > delivery time, setup time = 100000
        
        time_cost = int(process_time) + setup_time + int(delivery_time)
        time_cost = 100000 if gantt[machine, job_status_time[jobid - 1]] != 0 else time_cost
        time_cost_list.append((time_cost, machine + 1))

    min_time_cost = min(time_cost_list, key=lambda x: x[0])[0]
    candidate_machines = [machine for time_cost, machine in time_cost_list if time_cost == min_time_cost]
    selected_machine = random.choice(candidate_machines) if candidate_machines else -1
    update_gantt(jobid, step, selected_machine, min_time_cost)

    return selected_machine

def update_gantt(jobid, step, machine, time):
    # Update Gantt chart and job status
    current_time = job_status_time[jobid - 1]
    if current_time + time <= gantt.shape[1]:
        gantt[machine - 1, current_time:current_time + time] = jobid
        job_status_machine[jobid - 1] = machine
        job_status_time[jobid - 1] = current_time + time
    else:
        logger.warning(f"Unable to schedule job {jobid} at step {step} due to time constraints")

def evaluate(gantt):
    # Evaluate the Gantt chart and return the makespan
    return np.max(np.where(np.sum(gantt, axis=0) != 0, 1, 0).cumsum(axis=0))

def main(args):
    # Scheduling process
    for idx in range(len(chromosome)):
        selected_machine = schedule(chromosome, idx, gantt)
        logger.info(f"Selected Machine for Job {chromosome[idx]} Step {idx}: {selected_machine}")
        logger.debug(f"Gantt Chart:\n{gantt}")
        logger.debug(f"Job Status Machine: {job_status_machine}")
        logger.debug("---")

    logger.info(f"The makespan is {evaluate(gantt)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug messages")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    main(args)