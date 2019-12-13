from numba import njit
import numpy as np

from metrics import compute_daily_occupancy
from load_data_and_constants import (cost_matrix, data_array, family_size,
                                     days_array, MIN_OCCUPANCY, MAX_OCCUPANCY)
from utils import proportional_random_choice


@njit
def gap_mutations(population, mutation_rate, random_family_rate,
                  random_choice_rate, step_mutation_rate,
                  occ_costs, acc_costs):

    # total_costs = occ_costs + acc_costs
    output = population

    for i in range(len(population)):
        individual = population[i]
        # costs = total_costs[i]
        p = np.random.random()
        if p < mutation_rate:
            p = np.random.random()
            if p < random_family_rate:
                family_idx = np.random.randint(len(individual))
                day = individual[family_idx]
            else:
                # daily_occupancy = compute_daily_occupancy(individual)
                # day = proportional_random_choice(costs / daily_occupancy, 1)[0]
                # family_indexes = np.where(individual == day + 1)[0]
                family_costs = cost_matrix[:, day]
                family_idx = proportional_random_choice(family_costs, 1)[0]
                # family_idx = family_indexes[random_family]

            p = np.random.random()
            if p <= random_choice_rate:
                new_day = np.random.randint(1, 101)
                individual[family_idx] = new_day
            else:
                family_choices = data_array[family_idx]
                p = np.random.random()
                new_day = day
                while(new_day == day):
                    if p < step_mutation_rate:
                        day_idx = np.where(family_choices == day)[0][0]
                        step = np.random.choice(np.arange(-1, 2, 2))
                        new_day_idx = min(9, max(0, day_idx + step))
                        new_day = family_choices[new_day_idx]
                    else:
                        new_day = np.random.choice(family_choices)

            individual[family_idx] = new_day

        output[i] = individual

    return output


def assure_feasibility(individual, family_number):
    daily_occupancy = compute_daily_occupancy(individual)
    current_day = individual[family_number]
    n = family_size[family_number]
    anticipated_occupancy = daily_occupancy[current_day - 1] - n
    while anticipated_occupancy < MIN_OCCUPANCY:
        family_number = np.random.randint(len(family_size))
        current_day = individual[family_number]
        anticipated_occupancy = daily_occupancy[current_day - 1] - n

    available_days = days_array[
        (daily_occupancy <= MAX_OCCUPANCY - n) &
        (days_array != current_day)
    ]

    return current_day, available_days
