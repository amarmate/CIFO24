import numpy as np 
from Objects.sudoku import Sudoku
from Operators.variators import single_crossover
from copy import deepcopy
from Operators.fitness import fitness as fitness_function

class Population:
    """
    A class to represent a population of sudoku puzzles
    """
    def __init__(self, size : int = 100, initial_board : np.ndarray = None, sudoku_arguments = {}):
        """
        Constructor for the population class
        :param size: an int representing the size of the population
        :param initial_board: an NxN numpy array representing the initial state of the board, where N has to be a perfect square. 0s represent empty cells
        :param sudoku_arguments: a dictionary with the arguments to pass to the sudoku class 
            available: fill_board, size, difficulty, fitness_function, hill_climbing_args, diff_function
        """
        self.size = size
        self.individuals = []

        if initial_board is None:
            first_individual = Sudoku()
            initial_board = first_individual.initial_board

        # appending the population with individuals
        for _ in range(size):
            self.individuals.append(Sudoku(initial_board, **sudoku_arguments))


    def evolve(self, gens, xo_prob, mut_prob, select_type, xo, 
               elite_size : int = 1, 
               mutation='change', 
               swap_number : int = 1, 
               keep_distribution=False):
        assert elite_size <= self.size, "The number of elite individuals has to be less than the population size"
        assert elite_size >= 0, "The number of elite individuals has to be greater than 0"

        for i in range(gens):
            if elite_size > 0:
                best_individuals = self.get_best_individuals(elite_size)

            self.selection(select_type)
            self.crossover(type=xo, prob=xo_prob)
            for j in range(len(self.individuals)):
                self[j] = self[j].mutate(mut_prob, swap_number, mutation=mutation)

            if keep_distribution:
                self.keep_distribution()

            if elite_size > 0:
                for j in range(elite_size):
                    self[j] = best_individuals[j]

            mean_fitness = np.mean([ind.fitness for ind in self.individuals])
            print(f"Best individual of gen #{i + 1}: {min([ind.fitness for ind in self.individuals])}. Mean fitness: {mean_fitness}")

            if self.get_best_individuals(1)[0].fitness == 0:
                print(f"Solution found in generation {i + 1}!")
                self.get_best_individuals(1)[0].display()
                break

    # -------------------------------------------------------------------------------------------------------
    
    def get_best_individuals(self, n : int = 1):
        """
        Function to get the best individual of the population
        :param n: an int representing the number of best individuals to get
        """
        best_individuals = sorted(self.individuals, key=lambda x: x.fitness)[:n]
        return best_individuals

    def __getitem__(self, position):
        return self.individuals[position]
    
    def __setitem__(self, position, value):
        self.individuals[position] = deepcopy(value)

    
    def keep_distribution(self):
        """
        Function to keep the correct distribution of numbers inside each individual 
        """

        # TODO check why it doesnt really preserve distribution in some cases
        # print("Perfect distribution", self[0].distribution)

        perfect_distribution = np.tile(self[0].distribution, (len(self), 1))
        real_distribution = np.apply_along_axis(lambda row: np.bincount(row, minlength=self[0].N + 1), axis=1, arr=[self[i].swappable for i in range(len(self))])
        difference = perfect_distribution - real_distribution

        numbers = np.tile(np.indices(difference[0].shape)[0], (len(self) + 1,1))
        add = np.where(difference < 0, 0, difference)
        remove = np.where(difference > 0, 0, -difference)

        for i in range(len(self)):
            if np.sum(remove[i]) > 0:

                values_add = np.repeat(numbers[i], add[i], axis=0)
                np.random.shuffle(values_add)
                values_remove = np.repeat(numbers[i], remove[i], axis=0)

                counts = {val: np.sum(values_remove == val) for val in np.unique(values_remove)}
                mask = np.isin(self[i].swappable, values_remove)
                
                # Get the random indices of elements to mask that match values remove in offspring
                indices_to_mask = np.concatenate([np.random.choice(np.flatnonzero(mask & (self[i].swappable == val)), 
                                                                size=counts[val], 
                                                                replace=False) for val in counts.keys()])
                
                # Put new values on these positions
                self[i].swappable[indices_to_mask] = values_add
            else:
                pass
        

    def diversity(self):
        """
        Function to calculate the diversity of the population
        """
        pass



    # ------------------------------------ Crossover ------------------------------------------------
    def crossover(self, type : str = 'single_point', prob : float = 0.5, elitism : bool = False):
        """
        Function to crossover the individuals in the population
        :param type: a string representing the type of crossover to apply
        """

        if type == 'single_point':
            self.single_point(prob)

        if type == 'single_point_2':
            return self.single_point_2()

        elif type == 'pmxc':
            self.pmx_crossover()

        elif type == 'uniform':
            self.uniform()

    def single_point(self, prob : float = 0.5, keep_distribution : bool = False):
        """
        Function to apply single point crossover
        """
        #VERY BAD CODING PRACTICE
        # TODO: Fix this
        population_array = np.array([i.swappable for i in self.individuals])
            
        # Get the number of parents and the shape of each parent
        num_parents = len(self)
        rows, cols = population_array.shape

        # Sample how many crossovers to do
        num_crossovers = sum(np.random.choice([0,1], size = int(num_parents/2), replace=True, p=[1-prob, prob]))

        # Select two random parents for each offspring without replacement
        parent_indices = np.random.choice(num_parents, size=(num_crossovers,2), replace=False)

        # Choose random crossover points for each pair of parents
        crossover_points = np.random.randint(cols-1, size=num_crossovers)

        # Matrices with parents from both sides
        parent1 = population_array[parent_indices[:, 0]]
        parent2 = population_array[parent_indices[:, 1]]

        # Mask all the values that are above/below crossover line with zeros
        parent1_masked = np.ma.masked_array(parent1, np.arange(cols) > crossover_points[:, np.newaxis]).filled(0)
        parent2_masked = np.ma.masked_array(parent2, np.arange(cols) <= crossover_points[:, np.newaxis]).filled(0)

        # offsprings are just sum of parents
        offspring1 = parent1_masked + parent2_masked
        offspring2 = (parent1 - parent1_masked) + (parent2 - parent2_masked)

        offspring = np.concatenate((offspring1, offspring2))

        # Put parents that didn't get crossovered back into the population
        if num_crossovers < num_parents:
            left_ind = np.setdiff1d(np.arange(num_parents), parent_indices.flatten())
            offspring = np.concatenate((offspring, population_array[left_ind]))


        # Append random offsprings from parents until we reach population size
        if offspring.shape[0] < num_parents:
            while offspring.shape[0] < num_parents:
                offspring = np.append(offspring, 
                                    np.expand_dims(population_array[np.random.choice(len(population_array))], axis=0), axis = 0)
                
        if keep_distribution:

            perfect_distribution = np.tile(self.individuals[0].distribution, len(offspring))
            real_distribution = np.apply_along_axis(lambda row: np.bincount(row, minlength=10), axis=1, arr=offspring)
            difference = perfect_distribution - real_distribution

        
        # Put the offsprings into the population
        for i in range(len(self)):
            swappable = offspring[i]
            board = deepcopy(self.individuals[i].board)
            np.putmask(board, self.individuals[i].swap_points, swappable)
            self.individuals[i] = Sudoku(initial_board=self.individuals[i].initial_board, board=board)


    def single_point_2(self, elitism : bool = False):
        """
        Function to apply single point crossover
        """
        elite = deepcopy(self.get_best_individual())
        print('Single Point:', elite.fitness)
        # Enumerate all the parents two by two 

        offspring = []

        # CHANGE THIS TODO
        for i in range(0, len(self), 2):
            parent1 = self[i]
            parent2 = self[i+1]
            child1, child2 = single_crossover(parent1, parent2)
            offspring.append(Sudoku(initial_board=parent1.initial_board, board=child1))
            offspring.append(Sudoku(initial_board=parent1.initial_board, board=child2))

        if elitism:
            print('elite', elite)
            offspring[0] = elite
            print('self ind', offspring[0])

        return offspring
    
    
    def pmx_crossover(self):
        """
        Function to apply partially mapped crossover
        """

    def multi_point(self, n_points : int = 2):
        """
        Function to apply multi point crossover
        :param n_points: an int representing the number of points to crossover
        """
        pass

    def uniform(self):
        """
        Function to apply uniform crossover
        """
        pass

    # ------------------------------------ Selection ------------------------------------------------
    def selection(self, type : str = 'roulette', diversify : str = None):
        """
        Function to select the individuals in the population
        :param type: a string representing the type of selection to apply
        :param diversify: a string representing the type of diversification to apply, choose between 'fitness-sharing' and 'restricted-mating'
        """
        if type == 'roulette':
            self.roulette(diversify=diversify)

        if type == 'tournament':
            self.tournament()

        if type == 'rank':
            self.rank()
    
    def roulette(self, diversify : str = None):
        """
        Function to apply roulette selection
        :param diversify: a string representing the type of diversification to apply, choose between 'fitness-sharing' and 'restricted-mating'
        """

        # Calculate the fitness of each individual, total fitness and probs
        if diversify is None:
            fitnesses = [1/(individual.fitness+0.000001) for individual in self.individuals]
        elif diversify == 'fitness-sharing':
            fitnesses = []
        elif diversify == 'restricted-mating':
            fitnesses = []
        
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        probabilities_std = [prob / sum(probabilities) for prob in probabilities]
        
        # Select the individuals using roulette wheel
        self.individuals = np.random.choice(self.individuals, size=self.size, p=probabilities_std, replace=True)
    
    def tournament(self, tournament_size : int = 3):
        """
        Function to apply tournament selection
        """

        # Define tournament
        tournament = np.random.choice(self.individuals, size=(self.size, tournament_size), replace=True)

        # Keep all fitnesses
        fitnesses = np.array([np.array([ind.fitness for ind in subarray]) for subarray in tournament])

        # Find best in each of the tournaments
        min_indices = np.argmin(fitnesses, axis=1)
        row_indices = np.arange(min_indices.size)[:, None]
        col_indices = min_indices[:, None]

        # Reassign individuals
        self.individuals = tournament[row_indices, col_indices].flatten().tolist()


    # ------------------------------------ Dunder methods ------------------------------------------------
    def __len__(self):
        return len(self.individuals)


# ------------------------- Main --------------------------------------------------
if __name__ == '__main__':
    test_board = np.array([[9, 4, 7, 3, 2, 6, 5, 8, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 7, 3, 5, 9, 2, 1, 6, 8],
        [1, 2, 9, 8, 6, 4, 7, 3, 5],
        [5, 6, 8, 7, 1, 3, 4, 9, 2],
        [7, 9, 2, 4, 5, 8, 3, 1, 6],
        [6, 1, 5, 2, 3, 9, 8, 7, 4],
        [3, 8, 4, 6, 7, 1, 2, 5, 9]])

    population = Population(
    size=10,
    initial_board=test_board,
    )
    population.evolve(gens = 10000,
                    xo_prob = 0.9, 
                    mut_prob=0.25, 
                    select_type='roulette', 
                    xo = 'single_point', 
                    mutate = True, #should be smth else
                    elitism = True,
                    keep_distribution=True)

