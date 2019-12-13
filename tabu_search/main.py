import numpy as np

from load_data_and_constants import tree, data_array
from metrics import computes_total_costs, computes_occ_acc_costs


MEMORY_SIZE = 200
N_ITER = 5000

population = np.load('data/population_nsga.npy', allow_pickle=True)
costs = computes_total_costs(population)
best_solution = population[np.argmin(costs)]

for i in range(N_ITER):
    occ_costs, acc_costs = computes_occ_acc_costs([best_solution])
    total_costs = (occ_costs + acc_costs)[0]
    sort_idx = np.argsort(total_costs)


