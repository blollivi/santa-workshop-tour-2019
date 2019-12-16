import numpy as np

from load_data_and_constants import tree, data_array
from metrics import computes_total_costs, computes_occ_acc_costs
from tabu_search.neighborhood import get_neighbor

initial_solution = np.load('data/best_solution_ts.npy', allow_pickle=True)


N_ITER = 5000


class TabuSearch():
    def __init__(self, initial_solution, neighborhood_size,
                 tabu_duration, n_iter_without_improvement):
        self.n_families = initial_solution.shape[0]
        self.n_days = initial_solution.max() + 1
        self.best_score = computes_total_costs([initial_solution])[0]
        self.best_solution = initial_solution.copy()
        self.solution = initial_solution.copy()
        self.neighborhood_size = neighborhood_size
        self.iter = 0
        self.tabu_duration = tabu_duration
        self.n_iter_without_improvement = n_iter_without_improvement
        self.initialize_tabu_matrix()
        self.initialize_frequency_matrix()
        self.unfix_all_assignments()

    def short_term_memory_phase(self):
        i = 0
        while i <= self.n_iter_without_improvement:
            neighbor, neighbor_score, assignment = get_neighbor(
                self.solution,
                self.fixed_assignments,
                self.best_score,
                self.neighborhood_size,
                self.tabu_matrix,
                self.iter
            )
            
            self.solution = neighbor.copy()
            self.update_tabu_matrix(assignment)
            self.update_frequency_matrix(neighbor)
            print(f'Iteration {self.iter} - Score: {neighbor_score}')
            self.iter += 1
            i += 1

            if neighbor_score < self.best_score:
                self.best_solution = neighbor.copy()
                self.best_score = neighbor_score
                break
    
    def intensification_phase(self):
        print('Intensification phase')
        self.solution = self.best_solution.copy()
        self.fix_most_frequent_assignments(self.solution)
        self.short_term_memory_phase()
    
    def diversification_phase(self):
        print('Diversification phase')
        self.unfix_all_assignments()
        self.short_term_memory_phase()

    def fix_most_frequent_assignments(self, solution):
        """Fix assignments of a given solution based on their frequency.
        In this context, fixing means making sure that the selected
        assignments are immutable during future research operations.
        """
        fixed_assignments = []
        for family_idx, day in enumerate(solution):
            if self.frequency_matrix[family_idx, day - 1] > 0.85*self.iter:
                fixed_assignments.append(family_idx)
        self.fixed_assignments = fixed_assignments
    
    def unfix_all_assignments(self):
        self.fixed_assignments = []

    def update_tabu_matrix(self, assignment):
        if len(assignment) > 0:
            self.tabu_matrix[assignment[0], assignment[1] - 1] += self.iter + self.tabu_duration

    def update_frequency_matrix(self, solution):
        for family_idx, day in enumerate(solution):
            self.frequency_matrix[family_idx, day - 1] += 1

    def initialize_tabu_matrix(self):
        self.tabu_matrix = - np.ones((self.n_families, self.n_days))

    def initialize_frequency_matrix(self):
        self.frequency_matrix = np.ones((self.n_families, self.n_days))

    def run(self, iter_max):
        print(f'Initial score: {self.best_score}')
        self.short_term_memory_phase()
        for i in range(iter_max):
            self.intensification_phase()
            self.diversification_phase()
    
    def save_solution(self):
        np.save('data/best_solution_ts.npy', self.best_solution)



