from sklearn.neighbors import BallTree
from numba import njit
import pandas as pd
import numpy as np


data = pd.read_csv('data/family_data.csv', index_col='family_id')
data_array = data.loc[:, 'choice_0': 'choice_9'].to_numpy()
submission = pd.read_csv('data/sample_submission.csv', index_col='family_id')


# Constants
N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

family_size = data.n_people.values
days_array = np.arange(N_DAYS, 0, -1)
choice_dict = data.loc[:, 'choice_0': 'choice_9'].T.to_dict()

penalties = np.asarray([
    [
        0,
        50,
        50 + 9 * n,
        100 + 9 * n,
        200 + 9 * n,
        200 + 18 * n,
        300 + 18 * n,
        300 + 36 * n,
        400 + 36 * n,
        500 + 36 * n + 199 * n,
        500 + 36 * n + 398 * n
    ] for n in range(family_size.max() + 1)
])


def compute_cost_matrix():
    cost_matrix = np.concatenate(data.n_people.apply(
        lambda n: np.repeat(penalties[n, 10], 100).reshape(1, 100)))
    for fam in data.index:
        for choice_order, day in enumerate(data.loc[fam].drop("n_people")):
            cost_matrix[fam, day -
                        1] = penalties[data.loc[fam, "n_people"], choice_order]
    return cost_matrix


cost_matrix = compute_cost_matrix()
av_penalties = penalties[int(np.mean(family_size))]
weights = 1 + av_penalties / av_penalties.max()
max_similarity_distance = 20 * weights.sum()


@njit
def family_distance(family1, family2):
    """Distance between two family defined as the cost
    for Santa Claus to replace family1'choices by
    family2's choices."""

    similarity_distance = 0
    for idx1 in range(len(family1)):
        choice = family1[idx1]
        idx_choice_family2 = -1
        for idx2 in range(len(family2)):
            if family2[idx2] == choice:
                idx_choice_family2 = idx2
        if idx_choice_family2 >= 0:
            similarity_distance += weights[idx1] * np.abs(idx_choice_family2 - idx1)
        else:
            similarity_distance += weights[idx1] * 10

    similarity_distance /= max_similarity_distance

    # mae = np.abs(family1 - family2).mean() / 90

    return similarity_distance


tree = BallTree(
    data_array,
    metric=family_distance
)
