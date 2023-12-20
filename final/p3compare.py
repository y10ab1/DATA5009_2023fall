from p3GA import *
from p3TS import TS, generate_neighbor, tabu_search
from p3PSO import PSO, update_position, update_velocity, particle_swarm_optimization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Load task and transfer tables
task_table = pd.read_csv("task_table.csv", index_col="Oij")
transfer_table = pd.read_csv("transfer_table.csv", index_col="From/To")
logger.info(task_table)
logger.info(transfer_table)

def get_custom_cmap():
    # Define a new set of colors for specific values
    colors = [
        (0.9, 0.9, 0.9),  # Very light gray for 0 (idle time)
        (0.0, 0.8, 0.4),  # Light green for 1
        (0.4, 0.4, 0.8),  # Light blue for 2
        (0.8, 0.2, 0.2),  # Light red for 3
        (0.8, 0.8, 0.4),  # Light yellow for 4
    ]

    # Create the colormap with specific colors
    custom_cmap = mcolors.ListedColormap(colors)

    # Create a BoundaryNorm to map values to colors
    bounds = [0, 1, 2, 3, 4, 5]  # Define boundaries (including one extra for the upper bound)
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

    return custom_cmap, norm

def compare(args):
    
    start_time = time.perf_counter()
    TSbest_solution, TSbest_fitness, TSbest_gantt = TS(args, pop_size=args.pop_size, iterations=args.iterations)
    TS_time_end = time.perf_counter()
    GAbest_solution, GAbest_fitness, GAbest_gantt = GA(args, pop_size=args.pop_size, max_generations=args.iterations, mutation_rate=0.1)
    GA_time_end = time.perf_counter()
    PSObest_solution, PSObest_fitness, PSObest_gantt = PSO(args, pop_size=args.pop_size, iterations=args.iterations)
    PSO_time_end = time.perf_counter()
    
    print(f"TS time: {TS_time_end - start_time}")
    print(f"GA time: {GA_time_end - TS_time_end}")
    print(f"PSO time: {PSO_time_end - GA_time_end}")
    
    aspect_ratio = 3  # 'auto' adjusts the plot to fill the space, or use a numeric value as needed

    # Plot Gantt chart
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, sharey=True)
    cmap, norm = get_custom_cmap()
    ax[0].matshow(TSbest_gantt, cmap=cmap, aspect=aspect_ratio, norm=norm)
    ax[0].set_title(f"TS makespan: {1.0 / TSbest_fitness}")
    ax[1].matshow(GAbest_gantt, cmap=cmap, aspect=aspect_ratio, norm=norm)
    ax[1].set_title(f"GA makespan: {1.0 / GAbest_fitness}")
    ax[2].matshow(PSObest_gantt, cmap=cmap, aspect=aspect_ratio, norm=norm)
    ax[2].set_title(f"PSO makespan: {1.0 / PSObest_fitness}")
    # state the color of each number
    for i in range(3):
        for (j, k), z in np.ndenumerate(eval(f"{['TSbest_gantt', 'GAbest_gantt', 'PSObest_gantt'][i]}")):
            if z != 0:
                ax[i].text(k, j, f"{z}", ha='center', va='center', fontsize=10)
    plt.savefig("gantt.png")
    
    # Statistics of 3 algorithms
    exp_num = args.exp_num
    TS_fitness_list = []
    GA_fitness_list = []
    PSO_fitness_list = []
    TS_time_list = []
    GA_time_list = []
    PSO_time_list = []
    for i in range(exp_num):
        TSbest_solution, TSbest_fitness, TSbest_gantt = TS(args, pop_size=args.pop_size, iterations=args.iterations)
        TS_fitness_list.append(1.0 / TSbest_fitness)
        TS_time_list.append(time.perf_counter() - start_time)
        GAbest_solution, GAbest_fitness, GAbest_gantt = GA(args, pop_size=args.pop_size, max_generations=args.iterations, mutation_rate=0.1)
        GA_fitness_list.append(1.0 / GAbest_fitness)
        GA_time_list.append(time.perf_counter() - TS_time_end)
        PSObest_solution, PSObest_fitness, PSObest_gantt = PSO(args, pop_size=args.pop_size, iterations=args.iterations)
        PSO_fitness_list.append(1.0 / PSObest_fitness)
        PSO_time_list.append(time.perf_counter() - GA_time_end)
        
    # Box plot of makespan
    plt.figure(figsize=(10, 5))
    plt.boxplot([TS_fitness_list, GA_fitness_list, PSO_fitness_list], labels=['TS', 'GA', 'PSO'])
    plt.ylabel('makespan')
    plt.savefig("boxplot.png")
    
    # Box plot of time
    plt.figure(figsize=(10, 5))
    plt.boxplot([TS_time_list, GA_time_list, PSO_time_list], labels=['TS', 'GA', 'PSO'])
    plt.ylabel('time')
    plt.savefig("boxplot_time.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug messages")
    parser.add_argument("--iterations", "-i", type=int, default=8, help="Number of iterations")
    parser.add_argument("--pop_size", "-p", type=int, default=4, help="Population size")
    parser.add_argument("--exp_num", "-e", type=int, default=50, help="Number of experiments")
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG) 
    
    compare(args)