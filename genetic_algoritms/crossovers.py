from numba import njit
import numpy as np

from metrics import (violation_score_variation,
                     compute_daily_occupancy)
from load_data_and_constants import cost_matrix, family_size


@njit
def update_daily_occupancy(idx, choice_parent, choice_offspring, daily_occ):
    n = family_size[idx]
    daily_occ[choice_parent - 1] -= n
    daily_occ[choice_offspring - 1] += n
    return daily_occ


@njit
def gap_crossovers(population, uniform_rate):
    parents1 = np.arange(len(population))
    parents2 = np.arange(len(population))
    np.random.shuffle(parents1)
    np.random.shuffle(parents2)

    output = [population[0]]

    for p1, p2 in zip(parents1, parents2):
        offspring1 = population[p1]
        offspring2 = population[p2]

        daily_occ1 = compute_daily_occupancy(offspring1)
        daily_occ2 = compute_daily_occupancy(offspring2)

        n_crossovers = np.random.randint(population.shape[1] + 1)
        crossover_positions = np.arange(population.shape[1])
        np.random.shuffle(crossover_positions)
        crossover_positions = crossover_positions[:n_crossovers]

        p = np.random.random()
        for idx in crossover_positions:
            choice_parent1 = offspring1[idx]
            choice_parent2 = offspring2[idx]
            choice_offspring1 = choice_parent2
            choice_offspring2 = choice_parent1
            if p < uniform_rate:
                offspring1[idx] = choice_offspring1
                offspring2[idx] = choice_offspring2
            else:
                viol_score_parent1, violation_score1 = violation_score_variation(
                    idx, choice_parent1, choice_offspring1, daily_occ1
                )
                viol_score_parent2, violation_score2 = violation_score_variation(
                    idx, choice_parent2, choice_offspring2, daily_occ2
                )

                occupancy_cost1 = cost_matrix[idx, choice_offspring1-1]
                occupancy_cost2 = cost_matrix[idx, choice_offspring2-1]

                if (violation_score1 == 0 & violation_score2 == 0) & \
                   (occupancy_cost1 < occupancy_cost2):
                    offspring1[idx] = choice_offspring1
                    offspring2[idx] = choice_offspring2
                elif (violation_score1 <= viol_score_parent1) | \
                     (violation_score2 <= viol_score_parent2):
                    offspring1[idx] = choice_offspring1
                    offspring2[idx] = choice_offspring2
                else:
                    choice_offspring1 = choice_parent1
                    choice_offspring2 = choice_parent2

            daily_occ1 = update_daily_occupancy(
                idx, choice_parent1, choice_offspring1, daily_occ1
            )
            daily_occ2 = update_daily_occupancy(
                idx, choice_parent2, choice_offspring2, daily_occ2
            )

        output += [offspring1, offspring2]

    return output