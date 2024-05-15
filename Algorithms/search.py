from random import choice, uniform
from math import exp
from copy import deepcopy
import time
import matplotlib.pyplot as plt


class Hill_climbing:
    def __init__(self, individual):
        assert individual.board is not None, "The board has to be initialized"
        self.individual = deepcopy(individual)
        self.history = {}
        self.params = {}

    def run(self, max_iterations = 10000, num_neighbours = 1, swap_number = 1, plateau_threshold = 100, verbose = 0, stop_fitness = 0, smart = False):
        """Hill climbs a given search space.
        Args:
            individual (Individual): Individual object to hill climb.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 10000.
            num_neighbours (int, optional): Number of neighbours to generate. Defaults to 1.
            swap_number (int, optional): Number of swaps to make in the board. Defaults to 1.
            plateau_threshold (int, optional): Threshold to stop if stuck in a plateau. Defaults to 100.
            verbose (int, optional): Verbosity level. Defaults to 0.
            stop_fitness (int, optional): Fitness value to stop at. Defaults to 0.
            smart (bool, optional): Whether to use smart swapping. Defaults to False.
        Returns:
            Individual: Local optima Individual found in the search.
        """
        # Check if history is empty
        if self.history != {}:
            print("History is not empty, clearing it") if verbose >= 4 else None
            self.history = {}
            self.params = {}
        self.params = {'max_iterations': max_iterations, 'num_neighbours': num_neighbours, 'swap_number': swap_number, 'plateau_threshold': plateau_threshold}
        
        # current solution is i-start
        position = deepcopy(self.individual)
        iter_plateau = 0
        iteration_count = 0

        start_time = time.time()

        while position.fitness != stop_fitness:
            if iteration_count == max_iterations:
                print(f"Reached max iterations, returned {position}") if verbose >= 1 else None
                return position

            if iter_plateau > plateau_threshold:
                print(f"Stuck at a plateau, returned {position}") if verbose >= 2 else None
                return position

            # generate solution from neighbours

            neighbours = position.get_neighbours(num_neighbours, swap_number, smart=smart)
            n_fit = [i.fitness for i in neighbours]

            best_n = neighbours[n_fit.index(min(n_fit))]
            # if neighbour is better than current solution
            if best_n.fitness < position.fitness:
                print(f"Iteration {iteration_count} : Found a better solution with fitness: {best_n.fitness}") if verbose >= 3 else None
                # neighbour is the new solution
                position = best_n
                iter_plateau = 0
            elif best_n.fitness == position.fitness:
                print(f"Iteration {iteration_count} : Found a solution with the same fitness: {best_n.fitness}") if verbose >= 4 else None
                # neighbour is the new solution
                position = best_n
                iter_plateau += 1

            # Update history
            time_elapsed = time.time() - start_time
            self.history[iteration_count] = time_elapsed, position.fitness

            iteration_count += 1
        
        print(f"Hill climbing found {position}") if verbose >= 1 else None
        return position
    

    def plot_history(self, time = True, grid = False, info = False):
        """Function to plot the history of the hill climbing algorithm
        Args:
            time (bool, optional): Whether to plot time, instead of iterations. Defaults to True.
            grid (bool, optional): Whether to show the grid. Defaults to False.
            info (bool, optional): Whether to show the parameters of the run. Defaults to False.
        """
        if self.history == {}:
            print("No history to plot")
            return
        
        to_plot = {v[0]: v[1] for k, v in self.history.items()} if time else {k: v[1] for k, v in self.history.items()}
        plt.plot(list(to_plot.keys()), list(to_plot.values()))
        plt.xlabel("Time (s)") if time else plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        if info:
            plt.title(f"Hill climbing with parameters:\n {self.params}", fontsize=8)
        else:
            plt.title("Hill climbing")
        plt.grid() if grid else None
        plt.show()



class Sim_annealing:
    def __init__(self, individual): 
        assert individual.board is not None, "The board has to be initialized"
        self.individual = deepcopy(individual)
        self.history = {}
        self.params = {}

    def run(self, L=20, c=10, alpha=0.95, threshold=0.05, verbose=0, num_neighbours=1, swap_number=1, smart=False):
        """Simulated annealing implementation.

        Args:
            individual (Individual): Individual object to optimize.
            L (int, optional): Internal loop parameter. Defaults to 20.
            c (int, optional): Temperature parameter. Defaults to 10.
            alpha (float, optional): Alpha to decrease the temperature. Defaults to 0.95.
            threshold (float, optional): Threshold to stop the search. Defaults to 0.05.
            verbose (int, optional): Verbosity level. Defaults to 0.
            num_neighbours (int, optional): Number of neighbours to generate. Defaults to 1.
            swap_number (int, optional): Number of swaps to make in the board. Defaults to 1.
            smart (bool, optional): Whether to use smart swapping. Defaults to False.
        Returns:
            Individual: an Individual object - the best found by SA.
        """
        if self.history != {}:
            print("History is not empty, clearing it") if verbose >= 4 else None
            self.history = {}
            self.params = {}
        self.params = {'L': L, 'c': c, 'alpha': alpha, 'threshold': threshold, 'num_neighbours': num_neighbours, 'swap_number': swap_number}

        # 1. random init
        elite = deepcopy(self.individual)
        position = deepcopy(self.individual)
        # 2. L and c init as inputs
        # 3. repeat until termination condition
        time_start = time.time()
        iteration_count = 0
        while c > threshold:
            # 3.1 repeat L times
            for iteration in range(L):
                # 3.1.1 get random neighbour
                neighbour = choice(position.get_neighbours(num_neighbours, swap_number, smart=smart))

                # 3.1.2 if neighbour fitness is better or equal, accept
                if neighbour.fitness <= position.fitness:
                    position = neighbour
                    print(f"{iteration}: Found better solution with fitness {position.fitness}") if verbose >= 2 else None
                    if position.fitness <= elite.fitness:
                        elite = deepcopy(position)
                # accept with a probability
                else:
                    # p: probability of accepting a worse solution
                    p = exp(-abs(neighbour.fitness-position.fitness)/c)
                    x = uniform(0, 1)
                    if p > x:
                        position = neighbour
                        print(f"{iteration}: Accepted a worse solution with fitness {position.fitness}") if verbose >= 3 else None
                
                # 3.1.3 update history
                time_elapsed = time.time() - time_start
                self.history[iteration_count] = time_elapsed, position.fitness, c
                iteration_count += 1
            # 3.3 decrement c
            c = c * alpha
            print(f"Temperature is now {c}") if verbose >= 3 else None

        # 4. return the best solution
        print(f"SA found with fitness {position.fitness}") if verbose >= 1 else None
        return elite
    

    def plot_history(self, time = True, grid = False, info = False, temperature = False):
        """Function to plot the history of the hill climbing algorithm
        Args:
            time (bool, optional): Whether to plot time, instead of iterations. Defaults to True.
            grid (bool, optional): Whether to show the grid. Defaults to False.
            info (bool, optional): Whether to show the parameters of the run. Defaults to False.
            temperature (bool, optional): Whether to show the temperature. Defaults to False.
        """
        if self.history == {}:
            print("No history to plot")
            return
        
        to_plot = {v[0]: v[1] for k, v in self.history.items()} if time else {k: v[1] for k, v in self.history.items()}
        plt.plot(list(to_plot.keys()), list(to_plot.values()))

        if temperature:
            # Change axis 
            plt.twinx()
            to_plot = {v[0]: v[2] for k, v in self.history.items()} if time else {k: v[2] for k, v in self.history.items()}
            plt.plot(list(to_plot.keys()), list(to_plot.values()), 'r')
            plt.ylabel("Temperature")
        
        plt.xlabel("Time (s)") if time else plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        if info:
            plt.title(f"Sim. annealing with parameters:\n {self.params}", fontsize=8)
        else:
            plt.title("Simulated annealing")
        plt.grid() if grid else None
        plt.show()