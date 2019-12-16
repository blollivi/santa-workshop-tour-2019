from numba import njit, jit
import numpy as np

from metrics import computes_family_occ_costs, computes_total_costs
from load_data_and_constants import tree, data_array, N_DAYS, cost_matrix


@njit
def get_shifts(solution: np.array, family_idx: int) -> tuple([list, list]):
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
        if new_choice != current_choice:
            shift = solution.copy()
            shift[family_idx] = new_choice
            shifts.append(shift)
            # Keeps in memory the assignment.
            assignments_ref.append((family_idx, new_choice))
    return shifts, assignments_ref


@jit
def get_swaps(solution: np.array,
              family_idx: int, k: int) -> tuple([list, list]):
    """
    Finds all new assignments of family_idx resulting from swapping its choices
    with the choices of the k "closest" families. The notion of distance
    between families is defined in the module metrics.

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    :param k: number of closest families to consider.
    """
    dist, nearest_families_idx = tree.query(
        data_array[family_idx:family_idx+1], k=k
    )
    swaps = []
    assignments_ref = []
    current_choice = solution[family_idx]
    for swap_idx in nearest_families_idx[0][1:]:
        swap = solution.copy()
        new_choice = swap[swap_idx]
        swap[family_idx] = new_choice
        swap[swap_idx] = current_choice
        swaps.append(swap)
        # Keeps in memory the assignment with the highest cost
        if cost_matrix[family_idx, new_choice - 1] > \
                cost_matrix[swap_idx, current_choice - 1]:
            assignments_ref.append((family_idx, new_choice))
        else:
            assignments_ref.append((swap_idx, current_choice))

    return swaps, assignments_ref


@njit
def filter_feasible_assignment(assignments_ref, tabu_matrix, iter):
    """A feasible assignment is a shift or a swap that is not tabu active.
    """
    feasible_assignment_idx = []
    for idx, ref in enumerate(assignments_ref):
        if tabu_matrix[ref[0], ref[1] - 1] < iter:
            feasible_assignment_idx.append(idx)
    return feasible_assignment_idx


def find_best_new_assignment(solution: np.array, family_idx: int, k: int,
                             tabu_matrix, iter):
    """
    Lists all new feasible assignments for a given family_idx. Evaluate them
    and return the best one.

    :param solution: 1D solution array
    :param family_idx: index mapping a family choice in solution
    :param k: see get_swaps function.
    """
    shifts, shift_assignments_ref = get_shifts(solution, family_idx)
    swaps, swap_assignments_ref = get_swaps(solution, family_idx, k)
    assignments = shifts + swaps
    assignments_ref = shift_assignments_ref + swap_assignments_ref
    feasible_assignment_idx = filter_feasible_assignment(
        assignments_ref, tabu_matrix, iter
    )
    assignments = [assignments[i] for i in feasible_assignment_idx]
    assignments_ref = [assignments_ref[i] for i in feasible_assignment_idx]

    assignment_scores = computes_total_costs(assignments)
    best_assignment_arg = np.argmin(assignment_scores)
    best_assignment = assignments[best_assignment_arg]
    best_assignment_score = assignment_scores[best_assignment_arg]
    best_assignments_ref = assignments_ref[best_assignment_arg]

    return best_assignment, best_assignment_score, best_assignments_ref


def get_neighbor(solution: np.array, fixed_assignments: list,
                 best_score: float, k: int,
                 tabu_matrix, iter) -> np.array:
    """Neighbor generation mechanism

    :param solution: current solution
    :param fixed_assignments: list of family indices that can't be mutated
                              during the neighbor research.
    :param best_score: best score so far.
    :param k: size of the neighborhood for the swap function.
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
    for family_idx in sort_idx:
        best_assignment, best_assignment_score, best_assignments_ref = \
            find_best_new_assignment(solution, family_idx, k,
                                     tabu_matrix, iter)

        if best_assignment_score < best_neighbor_score:
            best_neighbor_score = best_assignment_score
            best_neighbor = best_assignment
            best_neighbor_assignment = best_assignments_ref
        if best_neighbor_score < best_score:
            break

    return best_neighbor, best_neighbor_score, best_neighbor_assignment
