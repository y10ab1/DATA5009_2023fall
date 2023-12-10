from itertools import permutations

# Job processing times
job_times = [4, 6, 10, 9]

# Generate all permutations of the four jobs
all_permutations = list(permutations(job_times))

# Function to calculate the time for each machine in a given schedule
def calculate_times(schedule):
    return max(sum(schedule[:2]), sum(schedule[2:]))

# Find the optimal schedule
optimal_schedule = min(all_permutations, key=calculate_times)
optimal_time = calculate_times(optimal_schedule)

print(optimal_schedule, optimal_time)