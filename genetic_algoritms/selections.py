from numba import njit
import numpy as np

from utils import proportional_random_choice

@njit
def sort_population(population, scores):
    sort_idx = np.argsort(scores)
    return population[sort_idx]


def uniform_selection(population, scores, selection_rate):
    population = sort_population(population, scores)
    return population[:int(selection_rate * len(population))]


def stochastic_selection(population, scores, selection_rate):
    N = int(selection_rate * len(population))
    selection_idx = proportional_random_choice(scores, N)
    return population[selection_idx]


@njit
def tournament(N, n_opponents, fit):
    '''
    Binary tournament selection
    :param N: number of solutions to be selected
    :param fit: fitness vectors.
                1st column: front_no,
                2nd column: crowding distance
    :return: index of selected solutions
    '''
    n = len(fit)
    winners = []
    for i in range(N):
        a = np.random.randint(n)
        for j in range(n_opponents):
            b = np.random.randint(n)
            for r in range(fit[0, :].size):
                if fit[(b, r)] < fit[(a, r)]:
                    a = b
        winners.append(a)

    return winners

