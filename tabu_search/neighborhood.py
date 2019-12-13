from numba import njit, jit
import numpy as np

from metrics import computes_family_occ_costs, computes_total_costs
from load_data_and_constants import tree, data_array, cost_matrix, N_DAYS
from utils import proportional_random_choice


@njit
def get_shifts(solution: np.array, family_idx: int) -> list:
    """
    Finds all new solutions resulting from switching the choice
    of family_idx by all other days.

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    """
    current_choice = solution[family_idx]
    shifts = []
    for new_choice in range(N_DAYS):
        if new_choice != current_choice:
            shift = solution.copy()
            shift[family_idx] = new_choice
            shifts.append(shift)
    return shifts


@jit
def get_swaps(solution: np.array, family_idx: int, k: int) -> list:
    """
    Finds all new solutions resulting from swapping the choices
    of family_idx with the choices of the k "closest" families.

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    :param k: number of closest families to consider.
    """
    dist, nearest_families_idx = tree.query(
        data_array[family_idx:family_idx+1], k=k
    )
    swaps = []
    current_choice = solution[family_idx]
    for swap_idx in nearest_families_idx[0]:
        swap = solution.copy()
        current_choice
        swap[family_idx] = swap[swap_idx]
        swap[swap_idx] = current_choice
        swaps.append(swap)
    return swaps


def evaluate_neighbors(solution: np.array, family_idx: int, k: int):
    """
    Builds the sub-neighborhood of a solution for a given family_idx,
    defined by all possible shifts of the family's choice, and choice swaps
    with the k "closest" families. The notion of distance between families is
    defined in the module metrics.
    Returns the costs of all these new solutions.

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    :param k: see get_swaps function.
    """
    shifts = get_shifts(solution, family_idx)
    swaps = get_swaps(solution, family_idx, 100)
    neighbors = shifts + swaps

    return computes_total_costs(neighbors)


def build_neighborhood(solution: np.array, k: int)