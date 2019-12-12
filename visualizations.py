from matplotlib import pyplot as plt
import numpy as np

from metrics import computes_occ_acc_costs
from .evaluation import dominates, fast_non_dominated_sort


def domination_plot(population, index, log=True):
    occ_costs, acc_costs = computes_occ_acc_costs(population)
    dominations = []
    for individual in population:
        dominations.append(dominates(
            population[index],
            individual
        ))
    if log:
        occ_costs = np.log(occ_costs)
        acc_costs = np.log(acc_costs)
    plt.figure()
    plt.scatter(occ_costs, acc_costs,
                c=dominations)
    plt.scatter(occ_costs[index], acc_costs[index], c='red')
    plt.colorbar()
    plt.xlabel('Occupation Cost')
    plt.ylabel('Accounting Cost')


def domination_fronts_plot(population, log=True):
    front_groups = fast_non_dominated_sort(population)
    occ_costs, acc_costs = computes_occ_acc_costs(population)
    if log:
        occ_costs = np.log(occ_costs)
        acc_costs = np.log(acc_costs)

    plt.figure()
    for group in front_groups:
        plt.scatter(
            np.asarray(occ_costs)[group],
            np.asarray(acc_costs)[group]
        )
