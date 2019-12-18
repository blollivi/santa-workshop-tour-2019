from numba import njit, jit
from tqdm import tqdm
import numpy as np

from metrics import computes_family_occ_costs, computes_total_costs
from load_data_and_constants import N_DAYS, cost_matrix


@njit
def is_feasible(assignment_ref: tuple([int, int]),
                tabu_matrix: np.array,
                iter: int) -> bool:
    """A feasible assignment is a shift or a swap that is not tabu active.
    """
    return tabu_matrix[assignment_ref[0], assignment_ref[1] - 1] < iter


@njit
def get_shifts(solution: np.array, family_idx: int,
               tabu_matrix: np.array,
               iter: int) -> tuple([list, list]):
    """
    Finds new assignments of family_idx resulting from switching its choice
    with all other days.

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    """
    current_choice = solution[family_idx]
    shifts = []
    assignments_ref = []
    for new_choice in range(1, N_DAYS + 1):
        if (new_choice != current_choice) & \
                is_feasible((family_idx, new_choice), tabu_matrix, iter):
            shift = solution.copy()
            shift[family_idx] = new_choice
            shifts.append(shift)
            # Keeps in memory the assignment.
            assignments_ref.append((family_idx, new_choice))
    return shifts, assignments_ref


@njit
def get_swaps(solution: np.array, family_idx: int,
              tabu_matrix: np.array,
              iter: int) -> tuple([list, list]):
    """
    Finds all new assignments of family_idx resulting from swapping its choices
    with the choices of all other families.

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    """
    swaps = []
    assignments_ref = []
    current_choice = solution[family_idx]
    for swap_idx in range(len(solution)):
        if swap_idx != family_idx: 
            swap = solution.copy()
            new_choice = swap[swap_idx]
            swap[family_idx] = new_choice
            swap[swap_idx] = current_choice

            # Keeps in memory only the assignment with the highest cost
            if cost_matrix[family_idx, new_choice - 1] > \
                    cost_matrix[swap_idx, current_choice - 1]:
                if is_feasible((family_idx, new_choice), tabu_matrix, iter):
                    swaps.append(swap)
                    assignments_ref.append((family_idx, new_choice))
            else:
                if is_feasible((swap_idx, current_choice), tabu_matrix, iter):
                    swaps.append(swap)
                    assignments_ref.append((swap_idx, current_choice))

    return swaps, assignments_ref


@njit
def find_best_new_assignment(solution: np.array, family_idx: int,
                             tabu_matrix, iter):
    """
    Lists all new feasible assignments for a given family_idx. Evaluate them
    and return the best one.

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    """
    shifts, shift_assignments_ref = get_shifts(
        solution, family_idx,
        tabu_matrix, iter
    )
    swaps, swap_assignments_ref = get_swaps(
        solution, family_idx,
        tabu_matrix, iter
    )
    assignments = shifts + swaps
    assignments_ref = shift_assignments_ref + swap_assignments_ref

    assignment_scores = computes_total_costs(assignments)
    best_assignment_arg = np.argmin(assignment_scores)
    best_assignment = assignments[best_assignment_arg]
    best_assignment_score = assignment_scores[best_assignment_arg]
    best_assignments_ref = assignments_ref[best_assignment_arg]

    return best_assignment, best_assignment_score, best_assignments_ref


def get_neighbor(solution: np.array, fixed_assignments: list,
                 best_score: float,
                 tabu_matrix, iter) -> np.array:
    """Neighbor generation mechanism

    :param solution: current solution
    :param fixed_assignments: list of family indices that can't be mutated
                              during the neighbor research.
    :param best_score: best score so far.
    :return: a new solution array
    """

    family_costs = computes_family_occ_costs(solution)
    sort_idx = np.argsort(-family_costs)
    sort_idx = [idx for idx in sort_idx if idx not in fixed_assignments]
    if len(sort_idx) == 0:
        return solution, best_score, []
    best_neighbor_score = np.inf
    best_neighbor = None
    best_neighbor_assignment = None
    n = min(500, len(sort_idx))
    for family_idx in sort_idx[:n]:
        best_assignment, best_assignment_score, best_assignments_ref = \
            find_best_new_assignment(solution, family_idx,
                                     tabu_matrix, iter)

        if best_assignment_score < best_neighbor_score:
            best_neighbor_score = best_assignment_score
            best_neighbor = best_assignment
            best_neighbor_assignment = best_assignments_ref
        if best_neighbor_score < best_score:
            break

    return best_neighbor, best_neighbor_score, best_neighbor_assignment
