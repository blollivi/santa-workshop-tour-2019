import numpy as np
from .evaluation import fast_non_dominated_sort, crowding_distance


def environment_selection(population, costs_array, N, selection_rate):
    '''
    environmental selection in NSGA-II, with the specification that
    only the points near the minimum of occ_cost + acc_cost are selected.
    :param costs_array: costs_array of the current population
    :param N: number of selected individuals
    :param selection_rate: params that control the selection on the total cost
                           occ_cost + acc_cost
    :return: next population
    '''
    total_costs = costs_array.sum(axis=1)
    sorted_idx = np.argsort(total_costs)
    selection_idx = sorted_idx[:int(selection_rate * len(sorted_idx))]
    costs_array = costs_array[selection_idx, :]
    population = population[selection_idx]
    front_groups, ranks = fast_non_dominated_sort(costs_array)

    crowd_dis = crowding_distance(costs_array, ranks)

    return (population, ranks,
            crowd_dis)