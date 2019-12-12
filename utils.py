from numba import njit
import numpy as np
import pickle


def save_assets(best_population, mean_scores,
                best_scores, min_scores):

    pickle.dump(best_population,
                open('data/evolution_results/best_population.pkl', 'wb')
                )
    pickle.dump(mean_scores,
                open('data/evolution_results/mean_scores.pkl', 'wb')
                )
    pickle.dump(best_scores,
                open('data/evolution_results/best_scores.pkl', 'wb')
                )
    pickle.dump(min_scores,
                open('data/evolution_results/min_scores.pkl', 'wb')
                )


def restore_assets(path: str):
    population = pickle.load(
        open(path, 'rb'))
    return np.array(population)


@njit
def proportional_random_choice(arr, size):
    """
    :param arr: A 1D numpy array of values to sample from.
                The probability of a sample is proportional to its value.
    :return: Indexes of the samples.
    """
    # transforms the values into probabilities
    proba = arr/np.sum(arr)

    cumulative_distribution = np.cumsum(proba)
    cumulative_distribution /= cumulative_distribution[-1]
    uniform_samples = np.random.rand(size)
    index = np.searchsorted(cumulative_distribution, uniform_samples, side="right")
    return index
