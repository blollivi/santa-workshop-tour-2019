from matplotlib import pyplot as plt
import numpy as np

from initializations import initialise_population
from genetic_algoritms.selections import tournament

from genetic_algoritms.crossovers import gap_crossovers
from genetic_algoritms.mutations import gap_mutations
from metrics import computes_occ_acc_costs

N = 500
N_GENERATIONS = 5000
UNIFORM_CROSSOVER_RATE = 0.5
SELECTION_RATE = 0.5
ELITISM_RATE = 1
MUTATION_RATE = 1
RANDOM_FAMILY_RATE = 1
RANDOM_CHOICE_RATE = 0.5
STEP_MUTATION_RATE = 0.5
N_OPPONENTS = 2

# population = np.load('data/population_nsga.npy', allow_pickle=True)
 
offsprings = []

for n in range(N_GENERATIONS):
    combined_population = np.array(population.tolist() + offsprings)
    occ_costs, acc_costs = computes_occ_acc_costs(combined_population)
    costs_array = np.vstack([occ_costs.sum(axis=1), acc_costs.sum(axis=1)]).T
    total_costs = costs_array.sum(axis=1)
    print(
        f'Generation {n}: \n min_occ_cost: {np.min(costs_array[:, 0])}' +
        f' - min_acc_cost: {np.min(costs_array[:, 1])}' +
        f' - min_total_cost: {np.min(total_costs)}' +
        f' - median_total_cost: {np.median(total_costs)}'
    )
    if n % 20 == 0:
        print('Saving population')
        np.save('data/population_nsga.npy',
                population, allow_pickle=True)
    # Elitism
    population = uniform_selection(combined_population, total_costs, N)

    # Genetic operations
    parents = tournament(population, total_costs, SELECTION_RATE, N_OPPONENTS)
    offsprings = gap_crossovers(parents, UNIFORM_CROSSOVER_RATE)
    del offsprings[-1]
    offsprings = gap_mutations(offsprings, MUTATION_RATE,
                               RANDOM_FAMILY_RATE,
                               RANDOM_CHOICE_RATE,
                               STEP_MUTATION_RATE,
                               occ_costs, acc_costs)
