from numba import njit
import numpy as np


@njit
def dominates(cost1: np.array, cost2: np.array) -> bool:
    """Returns True if individual1 dominates individual2,
       False otherwise.
    """
    # Computes the costs
    occ_cost1, acc_cost1 = cost1
    occ_cost2, acc_cost2 = cost2

    # Define the dominant
    return (occ_cost1 <= occ_cost2 and acc_cost1 <= acc_cost2) and \
           (occ_cost1 < occ_cost2 or acc_cost1 < acc_cost2)


@njit
def fast_non_dominated_sort(costs_array):
    front_groups = []
    ranks = np.zeros(len(costs_array))
    dominated_lists = []
    domination_counters = []
    # First Front
    first_front = []
    for idx1 in range(len(costs_array)):
        cost1 = costs_array[idx1]
        dominated_list = []
        domination_counter = 0
        for idx2 in range(len(costs_array)):
            cost2 = costs_array[idx2]
            if dominates(cost1, cost2):
                dominated_list.append(idx2)
            elif dominates(cost2, cost1):
                domination_counter += 1
        dominated_lists.append(dominated_list)
        domination_counters.append(domination_counter)
        if domination_counters[idx1] == 0:
            ranks[idx1] = 0
            first_front.append(idx1)
    front_groups.append(first_front)

    # Other fronts
    i = 0
    while len(front_groups[i]) > 0:
        next_front = []
        for idx1 in front_groups[i]:
            for idx2 in dominated_lists[idx1]:
                domination_counters[idx2] += -1
                if domination_counters[idx2] == 0:
                    ranks[idx2] = i + 1
                    next_front.append(idx2)
        i += 1
        front_groups.append(next_front)

    if len(front_groups[-1]) == 0:
        del front_groups[-1]

    return front_groups, ranks


def crowding_distance(costs_array: np.array, front_no: np.array) -> np.array:
    """
    The crowding distance of each Pareto front
    :param costs_array: 2d array, with:
           first column: occupancy cost
           second column: accounting cost
    :param front_no: front numbers
    :return: crowding distance
    """
    n, M = np.shape(costs_array)
    crowd_dis = np.zeros(n)
    front = np.unique(front_no)
    Fronts = front[front != np.inf]
    for f in range(len(Fronts)):
        Front = np.array([k for k in range(len(front_no))
                          if front_no[k] == Fronts[f]])
        Fmax = costs_array[Front, :].max(0)
        Fmin = costs_array[Front, :].min(0)
        min_cost_idx = np.argmin(costs_array[Front, :].sum(axis=1))
        for i in range(M):
            rank = np.argsort(costs_array[Front, i])
            crowd_dis[Front[rank[0]]] = np.inf
            crowd_dis[Front[rank[-1]]] = np.inf
            crowd_dis[Front[min_cost_idx]] = np.inf
            for j in range(1, len(Front) - 1):
                crowd_dis[Front[rank[j]]] = crowd_dis[Front[rank[j]]] + (costs_array[(Front[rank[j + 1]], i)] - costs_array[
                    (Front[rank[j]], i)]) / (Fmax[i] - Fmin[i])
    return crowd_dis
