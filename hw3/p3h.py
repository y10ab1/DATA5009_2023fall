from p3c import *
from p3d import *
from p3e import *
from p3f import *
from p3g import *

import matplotlib.pyplot as plt
import time
from tqdm import tqdm


distances_table = pd.read_csv('distances_table.csv', index_col=0)

iter = 1000
start = time.time()
RW = [random_walk(distances_table)[1] for _ in tqdm(range(iter))]
RW_time = time.time()
HC = [hill_climbing(distances_table)[1] for _ in tqdm(range(iter))]
HC_time = time.time()
SA = [simulated_annealing(distances_table)[1] for _ in tqdm(range(iter))]
SA_time = time.time()
TAS = [tabu_search(distances_table)[1] for _ in tqdm(range(iter))]
TAS_time = time.time()
ACO = [ACO(distances_table)[1] for _ in tqdm(range(iter))]
ACO_time = time.time()

print(f'Results for Random Walk: {RW}')
print(f'Results for Hill Climbing: {HC}')
print(f'Results for Simulated Annealing: {SA}')
print(f'Results for Ant Colony Optimization: {ACO}')
print(f'Results for Tabu Search: {TAS}')

# Deal with NaN values
RW = np.nan_to_num(RW, nan=np.inf)
HC = np.nan_to_num(HC, nan=np.inf)
SA = np.nan_to_num(SA, nan=np.inf)
ACO = np.nan_to_num(ACO, nan=np.inf)
TAS = np.nan_to_num(TAS, nan=np.inf)

plt.figure(figsize=(10, 10))
plt.title('Best distance over time')
plt.xlabel('Number of iterations')
plt.ylabel('Best distance')
plt.plot(RW, label=f'Random Walk, iter/sec={iter/(RW_time-start):.2f}')
plt.plot(HC, label=f'Hill Climbing, iter/sec={iter/(HC_time-RW_time):.2f}')
plt.plot(SA, label=f'Simulated Annealing, iter/sec={iter/(SA_time-HC_time):.2f}')
plt.plot(TAS, label=f'Tabu Search, iter/sec={iter/(TAS_time-SA_time):.2f}')
plt.plot(ACO, label=f'Ant Colony Optimization, iter/sec={iter/(ACO_time-TAS_time):.2f}')
plt.legend()
plt.savefig('p3h.png')