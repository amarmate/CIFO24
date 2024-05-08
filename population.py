import numpy as np 
from sudoku import Individual

class Population:
    """
    A class to represent a population of sudoku puzzles
    """
    def __init__(self, size : int = 100, initial_board : np.ndarray = np.zeros((9,9), dtype=int),  **kwargs):
        """
        Constructor for the population class
        :param size: an int representing the size of the population
        :param initial_board: an NxN numpy array representing the initial state of the board, where N has to be a perfect square. 0s represent empty cells
        :param kwargs: additional keyword arguments to pass to the Individual class
        """

        # population size
        self.size = size
        self.individuals = []

        # appending the population with individuals
        for _ in range(size):
            self.individuals.append(Individual(initial_board, **kwargs))


    # Lets do recursive ! TODO : Fix this
    def evolve(self, gens, xo_prob, mut_prob, select_type, xo, mutate, elitism):
        # gens = 100
        for i in range(gens):
            # selection
            self.selection(select_type)
            # crossover
            self.crossover(xo, xo_prob, elitism)

            self.individuals = [i.mutate(mut_prob) for i in self.individuals]
            print(f"Best individual of gen #{i + 1}: {max([i.fitness for i in self.individuals])}")



    # ------------------------- Dunder methods --------------------------------------------------
    def __len__(self):
        return len(self.individuals)
    
    def __getitem__(self, position):
        return self.individuals[position]
    
    # ------------------------- Helper methods --------------------------------------------------
    
    def get_best_individual(self):
        """
        Function to get the best individual of the population
        """
        fitnesses = [individual.fitness for individual in self.individuals]
        best_fitness_index = fitnesses.index(min(fitnesses))  
        return self.individuals[best_fitness_index]

    def selection(self, type : str = 'roulette'):
        """
        Function to select the individuals in the population
        :param type: a string representing the type of selection to apply
        """
        if type == 'roulette':
            self.roulette()


    def roulette(self, type='fitness_proportionate'):
        """
        Function to apply roulette selection
        :param type: a string representing the type of selection to apply, available 'fitness_proportionate' or 'rank_based'
        """
        if type == 'fitness_proportionate':
            # Calculate the fitness of each individual, total fitness and probs
            fitnesses = [1/(individual.fitness+0.000001) for individual in self.individuals]
            total_fitness = sum(fitnesses)
            probabilities = [fitness / total_fitness for fitness in fitnesses]

        elif type == 'rank_based':
            # Calculate the fitness of each individual, total fitness and probs
            fitnesses = {i: individual.fitness for i, individual in enumerate(self.individuals)}
            sorted_fitnesses = sorted(fitnesses.items(), key=lambda x: x[1])
            probabilities = {i: (2 - 2 * (rank / (self.size - 1))) / self.size for rank, (i, _) in enumerate(sorted_fitnesses)}
            probabilities = list(probabilities.values())

        # Select the individuals based on the probabilities        
        self.individuals = np.random.choice(self.individuals, size=self.size, p=list(probabilities.values()), replace=True)

    def crossover(self, type : str = 'single_point', prob : float = 0.5, elitism : bool = False):
        """
        Function to crossover the individuals in the population
        :param type: a string representing the type of crossover to apply, available 'single_point', 'multi_point', or 'uniform'
        """
        # Add elitism later
        if type == 'single_point':
            self.single_point(prob, elitism=elitism)

        elif type == 'multi_point':
            self.multi_point()

        elif type == 'uniform':
            self.uniform()
            

    def single_point(self, prob : float = 0.5, elitism : bool = False):
        """
        Function to apply single point crossover
        """
        # TODO: Add elitism

        #VERY BAD CODING PRACTICE
        # TODO: Fix this
        population_array = np.array([i.flatten() for i in self.individuals])
            
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

        #VERY BAD CODING PRACTICE
        # TODO: Fix this
        offspring_individuals = [self.individuals[0].__class__(self.individuals[0].initial_board, np.reshape(ind, self.individuals[0].board.shape)) for ind in offspring]

        # TODO understand why elitism doesn't work perfectly
        if elitism:
            self.individuals = np.concatenate((offspring_individuals[:-1], [self.get_best_individual()]))
        else:
            self.individuals = offspring_individuals

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

    def xo_sudoku(self): 
        """
        Function to do cross-over on a sudoku board
        """

        # For crossover, we need to ensure that the offspring is a valid sudoku board
        # We can only take 





# ------------------------- Main --------------------------------------------------
test_board = np.array([[9, 4, 7, 3, 2, 6, 5, 8, 1],
       [8, 0, 0, 0, 0, 7, 0, 0, 0],
       [2, 0, 0, 0, 0, 5, 0, 0, 0],
       [4, 7, 3, 5, 9, 2, 1, 6, 8],
       [1, 2, 9, 8, 6, 4, 7, 3, 5],
       [5, 6, 8, 7, 1, 3, 4, 9, 2],
       [7, 9, 2, 4, 5, 8, 3, 1, 6],
       [6, 1, 5, 2, 3, 9, 8, 7, 4],
       [3, 8, 4, 6, 7, 1, 2, 5, 9]])

population = Population(
size=100,
# initial_board=test_board,
)
population.evolve(gens = 10000,
                  xo_prob = 0.9, 
                  mut_prob=0.25, 
                  select_type='roulette', 
                  xo = 'single_point', 
                  mutate = True, #should be smth else
                  elitism = True)

