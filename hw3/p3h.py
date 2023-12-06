from p3c import *
from p3d import *
from p3e import *
from p3f import *
from p3g import *
from p4 import *

import matplotlib.pyplot as plt
import time
from tqdm import tqdm


distances_table = pd.read_csv('distances_table.csv', index_col=0)

iter = 1000
start = time.time()
RW = random_walk(distances_table, num_iterations=iter)
RW_time = time.time()
HC = hill_climbing(distances_table, iterations=iter)
HC_time = time.time()
SA = simulated_annealing(distances_table, iterations=iter)
SA_time = time.time()
TAS = tabu_search(distances_table, iterations=iter)
TAS_time = time.time()
PSO = pso_tsp(distances_table, num_iterations=iter)
PSO_time = time.time()
ACO = ACO(distances_table, num_iterations=iter)
ACO_time = time.time()

print(f'Results for Random Walk: {RW[1]}')
print(f'Results for Hill Climbing: {HC[1]}')
print(f'Results for Simulated Annealing: {SA[1]}')
print(f'Results for Ant Colony Optimization: {ACO[1]}')
print(f'Results for Tabu Search: {TAS[1]}')
print(f'Results for Particle Swarm Optimization: {PSO[1]}')

# Deal with NaN values
RW = np.nan_to_num(RW[-1], nan=np.inf)
HC = np.nan_to_num(HC[-1], nan=np.inf)
SA = np.nan_to_num(SA[-1], nan=np.inf)
ACO = np.nan_to_num(ACO[-1], nan=np.inf)
TAS = np.nan_to_num(TAS[-1], nan=np.inf)
PSO = np.nan_to_num(PSO[-1], nan=np.inf)

plt.figure(figsize=(10, 10))
plt.title('Best distance over time')
plt.xlabel('Number of iterations')
plt.ylabel('Best distance')
plt.plot(RW, label=f'Random Walk, iter/sec={iter/(RW_time-start):.2f}')
plt.plot(HC, label=f'Hill Climbing, iter/sec={iter/(HC_time-RW_time):.2f}')
plt.plot(SA, label=f'Simulated Annealing, iter/sec={iter/(SA_time-HC_time):.2f}')
plt.plot(TAS, label=f'Tabu Search, iter/sec={iter/(TAS_time-SA_time):.2f}')
plt.plot(PSO, label=f'Particle Swarm Optimization, iter/sec={iter/(PSO_time-TAS_time):.2f}')
plt.plot(ACO, label=f'Ant Colony Optimization, iter/sec={iter/(ACO_time-PSO_time):.2f}')
plt.legend()
plt.savefig('p3h.png')