from sklearn.neighbors import BallTree
from numba import njit, jit
import numpy as np

from metrics import (computes_family_occ_costs, computes_total_costs,
                     family_distance)
from load_data_and_constants import cost_matrix, data_array
from utils import proportional_random_choice


tree = BallTree(
    data_array,
    metric=family_distance
)


class DynamicNeighborhood():
    def __init__(self, tabu_search, neighborhood_size=2000,
                 family_neighbor_size=100):
        self.ts = tabu_search
        self.neighborhood_size = neighborhood_size
        self.family_neighbor_size = family_neighbor_size

    def get_neighbor(self, solution: np.array) -> np.array:
        """Because it is inefficient to explore all the neighbors of a
        given solution. This function samples only a fixed number of neighbors
        and returns the best one.

        :param solution: current solution
        """

        family_costs = computes_family_occ_costs(solution)
        list_families = list(range(len(solution)))

        n_explored_neighbors = 0
        best_neighbor_score = np.inf
        while n_explored_neighbors <= self.neighborhood_size:
            random_idx = proportional_random_choice(family_costs, 1)[0]
            family_idx = list_families[random_idx]
            family_costs = np.delete(family_costs, random_idx)
            del list_families[random_idx]

            best_move, best_move_score, n = \
                find_best_new_move(solution, family_idx,
                                   self.family_neighbor_size,
                                   self.ts.tabu_list, self.ts.iter)

            n_explored_neighbors += n

            if best_move_score < best_neighbor_score:
                best_neighbor_score = best_move_score
                best_neighbor = best_move
            if best_neighbor_score < self.ts.best_score:
                break

        return best_neighbor, best_neighbor_score


# @njit
def is_tabu_active(solution: np.array,
                   tabu_list: np.array) -> bool:
    """A feasible move is a shift or a swap that is not tabu active.
    """
    return (solution.reshape((-1, 1)) == tabu_list).all(axis=0).any()


# @njit
def get_shifts(solution: np.array, family_idx: int,
               tabu_list: np.array,
               iter: int) -> tuple([list, list]):
    """
    Finds new moves of family_idx resulting from switching its current
    choice with other preferred choices.

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    """
    current_choice = solution[family_idx]
    list_choices = data_array[family_idx]
    shifts = []
    for new_choice in list_choices:
        if (new_choice != current_choice):
            shift = solution.copy()
            shift[family_idx] = new_choice
            if is_tabu_active(shift, tabu_list):
                continue
            shifts.append(shift)
    return shifts


def get_swaps(solution: np.array, family_idx: int, k: int,
              tabu_list: np.array,
              iter: int) -> tuple([list, list]):
    """
    Finds all new moves of family_idx resulting from swapping its choices
    with the choices of families within a neighborhood defined by a custom
    distance metric between families (see metrics.family_distance).

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    :param k: number of closest families to consider
    """
    dist, ind = tree.query(
        data_array[family_idx:family_idx+1],
        k=k
    )
    swaps = []
    current_choice = solution[family_idx]
    for swap_idx in ind[0][1:]:
        if swap_idx != family_idx:
            swap = solution.copy()
            new_choice = swap[swap_idx]
            swap[family_idx] = new_choice
            swap[swap_idx] = current_choice

            if is_tabu_active(swap, tabu_list):
                continue

    return swaps


# @njit
def find_best_new_move(solution: np.array, family_idx: int, k: int,
                             tabu_list, iter):
    """
    Lists all new feasible moves for a given family_idx. Evaluate them
    and return the best one.

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    :param k: see get_swaps definition
    """
    shifts = get_shifts(
        solution, family_idx,
        tabu_list, iter
    )
    swaps = get_swaps(
        solution, family_idx, k,
        tabu_list, iter
    )
    moves = shifts + swaps

    moves_scores = computes_total_costs(moves)
    best_move_arg = np.argmin(moves_scores)
    best_move = moves[best_move_arg]
    best_move_score = moves_scores[best_move_arg]

    return best_move, best_move_score, len(moves)
