import numpy as np

from dynamic_tabu_search.dynamic_neighborhood import DynamicNeighborhood

from metrics import computes_total_costs

# From:
# A dynamic tabu search for large-scale generalised assignment problems
# A.J. Higgins


class DynamicTabuSearch():
    def __init__(self, initial_solution):
        self.n_families = initial_solution.shape[0]
        self.n_days = initial_solution.max() + 1
        self.best_score = computes_total_costs([initial_solution])[0]
        self.best_solution = initial_solution.copy()
        self.solution = initial_solution.copy()
        self.iter = 0
        self.initialize_tabu_list()

    def local_search(self):
        i = 0
        self.solution = self.best_solution
        while i <= self.n_iter_without_improvement:
            neighbor, neighbor_score = self.dn.get_neighbor(
                self.solution
            )

            self.solution = neighbor.copy()
            self.update_tabu_list(neighbor)

            print(f'Iteration {self.iter} - Score: {neighbor_score}')
            if self.iter % 50 == 0:
                self.save_solution()
            self.iter += 1
            i += 1

            if neighbor_score < self.best_score:
                self.best_solution = neighbor.copy()
                self.best_score = neighbor_score
                break

    def update_tabu_list(self, new_solution):
        self.tabu_list = np.hstack([self.tabu_list, new_solution])
        l = self.tabu_list.shape[1]
        self.tabu_list = self.tabu_list[:, min(l, self.tabu_duration)]

    def initialize_tabu_list(self):
        self.tabu_list = self.best_solution.reshape((-1, 1))

    def run(self, iter_max, neighborhood_size, family_neighbor_size,
            tabu_duration, n_iter_without_improvement):
        self.tabu_duration = tabu_duration
        self.n_iter_without_improvement = n_iter_without_improvement
        self.dn = DynamicNeighborhood(
            self, neighborhood_size,
            family_neighbor_size
        )
        print(f'Initial score: {self.best_score}')
        for i in range(iter_max):
            self.local_search()

    def save_solution(self):
        np.save('data/best_solution_ts.npy', self.best_solution)
