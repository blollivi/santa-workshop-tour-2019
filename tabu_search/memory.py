import numpy as np


class ShortTermMemory():
    def __init__(self, n_families: int, n_days: int):
        """The assignment of a family to a given day, is kept into memory
        with a (n_families, n_days) array. Each element (f, d) of this array
        records the last iteration number for which the attribution of family
        f to the day d+1 will remain Tabu-active.
        """
        self.tabu_matrix = np.zeros((n_families, n_days))
        self.tabu_duration = 50

    def update(self, family_idx, day, iter):
        self.tabu_matrix[family_idx, day] = iter + self.tabu_duration

    