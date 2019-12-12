from numba import njit
import numpy as np

from load_data_and_constants import (N_DAYS, MAX_OCCUPANCY,
                                     MIN_OCCUPANCY, cost_matrix,
                                     family_size, penalties)


@njit(fastmath=True)
def cost_function(
    individual: np.array
) -> tuple([float, int, int]):

    cost = 0
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    for i, (pred, n) in enumerate(zip(individual, family_size)):
        daily_occupancy[pred - 1] += n
        cost += cost_matrix[i, pred - 1]

    accounting_cost = 0
    violation_score = 0
    n_violations = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_next = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        violation_score += max(0, n - MAX_OCCUPANCY)
        violation_score += max(0, MIN_OCCUPANCY - n)
        n_violations += (n > MAX_OCCUPANCY)
        n_violations += (n < MIN_OCCUPANCY)
        diff = abs(n - n_next)
        accounting_cost += max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))

    cost += accounting_cost

    return cost, n_violations, violation_score


@njit
def apply_cost_function(
    population: np.array
) -> tuple([np.array, np.array, np.array]):

    scores = np.zeros(len(population))
    n_violations = np.zeros(len(population))
    violation_scores = np.zeros(len(population))

    for i in range(len(population)):
        score, n_viol, viol_score = cost_function(population[i])
        scores[i] = score
        n_violations[i] = n_viol
        violation_scores[i] = viol_score

    return scores, n_violations, violation_scores


@njit(fastmath=True)
def occupancy_cost(individual: np.array) -> int:
    daiy_cost = np.zeros(N_DAYS)
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    for i, (pred, n) in enumerate(zip(individual, family_size)):
        daily_occupancy[pred - 1] += n
        daiy_cost[pred - 1] += cost_matrix[i, pred - 1]
    return daiy_cost, daily_occupancy


@njit(fastmath=True)
def accounting_cost(individual: np.array,
                    daily_occupancy: np.array) -> int:
    daiy_cost = np.zeros(N_DAYS)
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_next = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        diff = abs(n - n_next)
        cost = max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))
        daiy_cost[day] = cost
    return daiy_cost


@njit
def computes_occ_acc_costs(population):
    occ_costs = np.zeros((len(population), N_DAYS))
    acc_costs = np.zeros((len(population), N_DAYS))
    for i in range(len(population)):
        occ_cost, daily_occ = occupancy_cost(population[i])
        acc_cost = accounting_cost(population[i], daily_occ)
        # Penalization
        violation_score = compute_violation_score(daily_occ)
        occ_cost = occ_cost * np.exp(violation_score)
        acc_cost = acc_cost * np.exp(violation_score)

        occ_costs[i, :] = occ_cost
        acc_costs[i, :] = acc_cost
    return occ_costs, acc_costs


@njit
def compute_daily_occupancy(individual):
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    for i, (pred, n) in enumerate(zip(individual, family_size)):
        daily_occupancy[pred] += n
    daily_occupancy = daily_occupancy[1:]
    return daily_occupancy


@njit
def compute_violation_score(daily_occupancy):
    violation_score = np.zeros(N_DAYS, dtype=np.int64)
    for day in range(len(daily_occupancy)):
        n = daily_occupancy[day]
        violation_score[day] += max(0, n - MAX_OCCUPANCY)
        violation_score[day] += max(0, MIN_OCCUPANCY - n)
    return violation_score


@njit
def violation_score_variation(family_idx, current_choice, new_choice,
                              daily_occupancy):
    occ_current_day = daily_occupancy[current_choice - 1]
    occ_next_day = daily_occupancy[new_choice - 1]
    current_violation_score = compute_violation_score(
        [occ_current_day, occ_next_day]
    )

    n = family_size[family_idx]
    anticipated_occ_current_day = occ_current_day - n
    anticipated_occ_next_day = occ_next_day + n

    next_violation_score = compute_violation_score(
        [anticipated_occ_current_day, anticipated_occ_next_day]
    )

    return current_violation_score.sum(), next_violation_score.sum()
