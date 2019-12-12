from numba import njit, jit
from tqdm import tqdm
import numpy as np

from load_data_and_constants import (family_size,
                                     MIN_OCCUPANCY,
                                     MAX_OCCUPANCY,
                                     days_array, data)

data_array = data.loc[:, 'choice_0': 'choice_9'].to_numpy()
days_list = days_array.tolist()


@njit
def generate_random_individual():
    """Complete randomness"""

    daily_occupancy = np.zeros((len(days_array)+1))
    while (daily_occupancy[1:] < MIN_OCCUPANCY).any():
        daily_occupancy = np.zeros((len(days_array)+1))
        prediction = np.zeros(len(family_size))

        for i, n in enumerate(family_size):
            anticipated_occupation = 301
            while anticipated_occupation > MAX_OCCUPANCY:
                random_day = np.random.choice(days_array)
                anticipated_occupation = daily_occupancy[random_day] + n

            daily_occupancy[random_day] = anticipated_occupation
            prediction[i] = random_day

    return prediction


@jit
def generate_naive_random_optimum():
    """Generate a random individual, with consideration of
    family choices.
    """
    daily_occupancy = np.zeros((len(days_array)+1))
    prediction = np.zeros(len(family_size))
    available_days = list(days_array)
    for i, n in enumerate(family_size):
        prefered_days = data_array[i, :].tolist()
        prefered_days = [d for d in prefered_days if d in available_days]
        anticipated_occupation = 301
        while anticipated_occupation > MAX_OCCUPANCY:
            if len(prefered_days) == 0:
                choice = np.random.choice(available_days)
            else:
                choice = np.random.choice(prefered_days)

            anticipated_occupation = daily_occupancy[choice] + n
            if anticipated_occupation > MAX_OCCUPANCY:
                available_days.remove(choice)
                if len(prefered_days) > 0:
                    prefered_days.remove(choice)

        prediction[i] = choice
        daily_occupancy[choice] += n

    return prediction


def initialise_population(n_individuals):
    population = []
    for i in tqdm(range(n_individuals),
                  desc=f'Generating {n_individuals} individuals:'):
        population.append(
            generate_random_individual().astype(int)
        )
    return np.array(population)