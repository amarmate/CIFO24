from random import choice, uniform
from math import exp


def hill_climbing(individual, max_iterations = 10000, num_neighbours = 1, swap_number = 1, plateau_threshold = 100, verbose = 0, stop_fitness = 0):
    """Hill climbs a given search space.

    Args:
        individual (Individual): Individual object to hill climb.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 10000.
        num_neighbours (int, optional): Number of neighbours to generate. Defaults to 1.
        swap_number (int, optional): Number of swaps to make in the board. Defaults to 1.
        plateau_threshold (int, optional): Threshold to stop if stuck in a plateau. Defaults to 100.
        verbose (int, optional): Verbosity level. Defaults to 0.
        stop_fitness (int, optional): Fitness value to stop at. Defaults to 0.
    Returns:
        Individual: Local optima Individual found in the search.
    """

    # current solution is i-start
    position = individual
    iter_plateau = 0
    iteration_count = 0

    while position.fitness != stop_fitness:
        if iteration_count == max_iterations:
            print(f"Reached max iterations, returned {position}") if verbose >= 1 else None
            return position

        if iter_plateau > plateau_threshold:
            print(f"Stuck at a plateau, returned {position}") if verbose >= 2 else None
            return position

        # generate solution from neighbours
        neighbours = position.get_neighbours(num_neighbours, swap_number)
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

        iteration_count += 1
    
    print(f"Hill climbing found {position}") if verbose >= 1 else None
    return position
    


def sim_annealing(individual, L=20, c=10, alpha=0.95, threshold=0.05, verbose=0, num_neighbours=1, swap_number=1):
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
    Returns:
        Individual: an Individual object - the best found by SA.
    """
    # 1. random init
    elite = individual
    position = individual
    # 2. L and c init as inputs
    # 3. repeat until termination condition
    while c > threshold:
        # 3.1 repeat L times
        for iteration in range(L):
            # 3.1.1 get random neighbour
            neighbour = choice(position.get_neighbours(num_neighbours, swap_number))

            # 3.1.2 if neighbour fitness is better or equal, accept
            if neighbour.fitness <= position.fitness:
                position = neighbour
                print(f"{iteration}: Found better solution with fitness {position.fitness}") if verbose >= 2 else None
                if position.fitness <= elite.fitness:
                    elite = position
            # accept with a probability
            else:
                # p: probability of accepting a worse solution
                p = exp(-abs(neighbour.fitness-position.fitness)/c)
                x = uniform(0, 1)
                if p > x:
                    position = neighbour
                    print(f"{iteration}: Accepted a worse solution with fitness {position.fitness}") if verbose >= 3 else None

        # 3.3 decrement c
        c = c * alpha
    # 4. return the best solution
    print(f"SA found with fitness {position.fitness}") if verbose >= 1 else None
    return elite


