import numpy as np
from operator import attrgetter

# The purpose of this class is to generate a random sudoku puzzle
class sudoku_game:
    def __init__(self, difficulty : int, seed : int = None, dimension : int = 9):
        """
        Constructor for the sudoku_game class
        :param difficulty: an integer representing the difficulty of the puzzle to generate
        :param seed: an integer representing the seed for the random number generator
        :param dimension: an int representing the dimension of the board (default is 9). Has to be a perfect square
        """ 
        # Check if the dimension is a perfect square
        assert np.sqrt(dimension) % 1 == 0, "The dimension has to be a perfect square"

        # Set the attributes
        self.difficulty = difficulty
        self.N = dimension
        self.seed = seed
        self.board = np.zeros((self.N, self.N), dtype=int)
        np.random.seed = seed

    def generate(self):
        """
        Function to construct the board
        """
        pass

    def __getitem__(self, position):
        assert len(position) == 2, "The position has to be a tuple of length 2"
        return self.board[position]
    
    def __setitem__(self, position, value):
        assert len(position) == 2, "The position has to be a tuple of length 2"
        self.board[position] = value

    def __repr__(self):
        return str(self.board)

    # -------------------------- Board construction functions -----------------------------------

    # !!! [TODO] -> be careful with impossible boards
    def fill_diagonals(self):
        """
        Function to fill the diagonal blocks of the board, which are the easiest to fill
        """
        numbers = np.arange(1, self.N + 1)
        box_size = int(np.sqrt(self.N))
        box_order = box_size * np.arange(1, box_size + 1) - 1
        np.random.shuffle(box_order)

        for i in range(box_size):
            box = box_order[i]
            box_coordinates = self.get_box_coordinates((box, box))
            np.random.shuffle(numbers)
            for j in range(self.N):
                self.board[box_coordinates[j]] = numbers[j]

    



    # ------------------------- Check functions --------------------------------------------------

    def check_is_safe(self, position, number, verbose=False):
        """
        Function to check if a number can be placed in a certain position
        :param position: a tuple representing the position of the box
        :param number: an int representing the number to check
        :param verbose: a boolean to print the checks
        """
        assert len(position) == 2, "The position has to be a tuple of length 2"
        assert number > 0 and number <= self.N, "The number has to be between 1 and N"

        # Get the box 
        box_coordinates = self.get_box_coordinates(position)
        rows, cols = zip(*box_coordinates)
        box = self.board[np.array(rows), np.array(cols)]

        inBox = number not in box
        inRow = number not in self.board[position[0]]
        inColumn = number not in self.board[:, position[1]]

        if verbose:
            print("Number already in box: ", not inBox) if not inBox else None
            print("Number already in row: ", not inRow) if not inRow else None
            print("Number already in column: ", not inColumn) if not inColumn else None

        return inBox and inRow and inColumn    

    # ------------------------ Other functions ---------------------------------------------------

    def get_box_coordinates(self, position) -> list:
        """
        Function to get the coordinates of the box that the position is in. Returns a list of tuples
        :param position: a tuple representing the position of the box
        """
        assert len(position) == 2, "The position has to be a tuple of length 2"
        assert position[0] < self.N and position[1] < self.N, "The position has to be within the board"
        box_size = int(np.sqrt(self.N))
        row, column = position
        box_row = row // box_size
        box_column = column // box_size
        return [(box_row * box_size + i, box_column * box_size + j) for i in range(box_size) for j in range(box_size)]
    

    # [TO DO] -------------------------------------------------------------------------------------

    def fill_remaining(self):
        """
        Function to fill the remaining cells of the board
        """
        pass
    

    def remove_numbers(self):
        """
        Function to remove numbers from the board to create a puzzle
        """
        pass




# The purpose of this class is to create an individual sudoku puzzle, to which the genetic algorithm will be applied
class Individual:
    """
    A class to represent a sudoku puzzle. This class provides methods to initialize a sudoku board, fill it randomly, remove numbers, check for uniqueness, and calculate the fitness of the board.
    
    The `Individual` class has the following methods:
    
    - `__init__(self, initial_board: np.ndarray = np.zeros((9,9), dtype=int), board: np.ndarray = None)`: Constructor for the sudoku class. Initializes the board with the given initial board or a randomly filled board.
    - `random_fill_board(self)`: Fills the empty cells in the board randomly while ensuring the board is still valid.
    - `remove_numbers(self, removed_numbers: int = 20)`: Removes a specified number of numbers from the board.
    - `is_unique(self)`: Checks if the solution to the sudoku puzzle is unique.
    - `get_neighbours(self, number_of_neighbours: int = 10, swap_number: int = 2)`: Generates a specified number of neighbor boards by swapping numbers in the board.
    - `get_fitness(self)`: Calculates the fitness of the board based on the number of conflicts in the rows, columns, and boxes.
    - `get_row_conflicts(self, position)`, `get_column_conflicts(self, position)`, and `get_box_conflicts(self, position)`: Helper functions to calculate the number of conflicts in a row, column, and box, respectively.
    - `get_box_coordinates(self, position)`: Helper function to get the coordinates of the box that a given position is in.
    - `__getitem__(self, position)` and `__setitem__(self, position, value)`: Dunder methods to access and set the value of a cell in the board.
    - `__repr__(self)`: Dunder method to represent the board as a string.
    - `flatten(self)` and `unflatten(self)`: Helper methods to flatten and unflatten the board.
    - `mutate(self, mutation_rate: float = 0.1)`: A TODO method to mutate the individual, i.e., change the board randomly.
    """
    """
    A class to represent a sudoku puzzle. The `Individual` class encapsulates the state of a sudoku board, including the initial board that cannot be changed, and the current board state. It provides methods to randomly fill in the empty cells, remove numbers, check if a solution is unique, and calculate the fitness of the board based on the number of conflicts.
    
    The class also provides methods to get the neighbors of the current board by swapping numbers, and to calculate the number of conflicts in each row, column, and box of the board.
    
    The class has various dunder methods to allow indexing and representation of the board.
    """
    """
    A class to represent a sudoku puzzle
    """
    def __init__(self, initial_board : np.ndarray = np.zeros((9,9), dtype=int), board : np.ndarray = None):
        """
        Constructor for the sudoku class
        :param initial_board: an NxN numpy array representing the initial state of the board, where N has to be a perfect square. 0s represent empty cells
        :param board: an NxN numpy array representing the current state of the board. If None, the board will be filled randomly
        """
        assert np.sqrt(initial_board.shape[0]) % 1 == 0, "The dimension has to be a perfect square"
        assert np.all(np.isin(initial_board, np.arange(0, initial_board.shape[0] + 1))), "The board can only contain integers between 0 and N"
        
        # The initial board is the board that cannot be changed
        self.initial_board = initial_board.copy()
        self.N = initial_board.shape[0]
        self.swappable_positions = list(zip(*np.where(self.initial_board == 0)))

        if board is None:
            self.random_fill_board()
        else:
            self.board = board.copy()

        self.fitness = self.get_fitness()


    def random_fill_board(self):
        """
        Function to fill in the empty cells (0s) in the board randomly and ensuring that the board is still valid, i.e. no more that 9 of the same number in the board
        """
        # Flatten the board
        flat_board = self.initial_board.copy().flatten()

        # Get how many of each number is still missing from the board
        number_counts = np.bincount(flat_board, minlength=self.N + 1)
        fill_number_count = self.N - number_counts[1:]

        # Create an array of the numbers to fill in the board and shuffle
        fill_numbers = np.repeat(np.arange(1, self.N + 1), fill_number_count)
        np.random.shuffle(fill_numbers)

        # Fill in the board
        np.putmask(flat_board, flat_board == 0, fill_numbers)
        self.board = flat_board.reshape(self.N, self.N)

    def remove_numbers(self, removed_numbers : int = 20):
        # Pick two numbers from 0 to self.N to remove

        # Check if we can still do an iteration
        if sum(np.where(self.board == 0)) < removed_numbers - 1:
            removed_numbers(self, removed_numbers)


    def is_unique(self):
        """
        Function used to check if a solution is unique 
        """

        pass
        



    # ------------------------- Get functions --------------------------------------------------

    def get_neighbours(self, number_of_neighbours : int = 10, swap_number : int = 2):
        """
        Function to get the neighbours of the individual
        :param number_of_neighbours: an int representing the number of neighbours to generate
        :param swap_number: an int representing the number of swaps to make in the board
        """
        # Neighbours will be generated by swapping numbers in the board
        neighbours = []
        for _ in range(number_of_neighbours):
            neighbour = self.board.copy()
            np.random.shuffle(self.swappable_positions)
            for i in range(swap_number):
                # Randomly select two swappeable positions 
                neighbour[self.swappable_positions[2*i]], neighbour[self.swappable_positions[2*i+1]] = neighbour[self.swappable_positions[2*i+1]], neighbour[self.swappable_positions[2*i]]
            neighbour_individual = Individual(self.initial_board, neighbour)
            neighbours.append(neighbour_individual)
        return neighbours

    def get_fitness(self):
        """
        Function to get the fitness of the individual
        The fitness will be gauged by the number of conflicts in the board
        """
        row_conflicts = np.sum([self.get_row_conflicts(i) for i in range(self.N)])
        column_conflicts = np.sum([self.get_column_conflicts(i) for i in range(self.N)])
        box_conflicts = np.sum([self.get_box_conflicts(i) for i in range(self.N)])
        return row_conflicts + column_conflicts + box_conflicts

    def get_row_conflicts(self, position):
        """
        Function to get the number of conflicts in a row
        :param position: a tuple representing the coordinates of a number inside the row, or int representing the row number
        """
        assert type(position) == int or (type(position) == tuple and len(position) == 2), "The position has to be an int or a tuple of length 2"
        row = position if type(position) == int else position[0]
        row_values = self.board[row]
        return self.N - len(np.unique(row_values))
    
    
    def get_column_conflicts(self, position):
        """
        Function to get the number of conflicts in a column
        :param position: a tuple representing the coordinates of a number inside the column, or int representing the column number
        """
        assert type(position) == int or (type(position) == tuple and len(position) == 2), "The position has to be an int or a tuple of length 2"
        column = position if type(position) == int else position[1]
        column_values = self.board[:, column]
        return self.N - len(np.unique(column_values))
    
    
    def get_box_conflicts(self, position):
        """
        Function to get the number of conflicts in a box
        :param position: a tuple representing the coordinates of a number inside the box or an int representing the box to check, from left to right and top to bottom
        """
        assert type(position) == int or (type(position) == tuple and len(position) == 2), "The position has to be an int or a tuple of length 2"
        box_coordinates = self.get_box_coordinates(position)
        rows, cols = zip(*box_coordinates)
        box = self.board[np.array(rows), np.array(cols)]
        return self.N - len(np.unique(box))


    def get_box_coordinates(self, position) -> list:
        """
        Function to get the coordinates of the box that the position is in. Returns a list of tuples
        :param position: a tuple representing the coordinates of a number inside the box or a box number
        """
        # Check if it is a tuple or a box number
        if type(position) == tuple:
            assert len(position) == 2, "The position has to be a tuple of length 2"
            assert position[0] < self.N and position[1] < self.N, "The position has to be within the board"    
            box_size = int(np.sqrt(self.N))
            row, column = position
            box_row = row // box_size
            box_column = column // box_size
            return [(box_row * box_size + i, box_column * box_size + j) for i in range(box_size) for j in range(box_size)]
    
        elif type(position) == int:
            box = position
            assert box < self.N, "The box has to be within the board"
            box_size = int(np.sqrt(self.N))
            box_row = box // box_size
            box_column = box % box_size
            return [(box_row * box_size + i, box_column * box_size + j) for i in range(box_size) for j in range(box_size)]
        
        else:
            raise ValueError("The position has to be a tuple or an int")




    # ------------------------- Dunder methods --------------------------------------------------

    def __getitem__(self, position):
        assert len(position) == 2, "The position has to be a tuple of length 2"
        return self.board[position]

    def __setitem__(self, position, value):
        # Check if the position is a tuple of length 2
        assert len(position) == 2, "The position has to be a tuple of length 2"
        self.representation[position] = value

    def __repr__(self):
        return str(self.board)
    
    def flatten(self):
        return self.board.flatten()
    
    def unflatten(self):
        return self.board.reshape(self.N, self.N)


    # [TO DO] -------------------------------------------------------------------------------------

    def mutate(self, mutation_rate : float = 0.1, swap_number : int = 1):
        """
        Function to mutate the individual, i.e. change the board randomly
        """
        # TODO: Do not mutate elite
        # TODO: Check if it all works correctly
        if np.random.rand() < mutation_rate:
            mutated_board = self.board.copy()
            np.random.shuffle(self.swappable_positions)
            for i in range(swap_number):
                # Randomly select two swappable positions
                mutated_board[self.swappable_positions[2 * i]], mutated_board[self.swappable_positions[2 * i + 1]] = (
                    mutated_board[self.swappable_positions[2 * i + 1]],
                    mutated_board[self.swappable_positions[2 * i]],
                )
            return self.__class__(self.initial_board, mutated_board)
        else:
            return self



# Handle this TODO !!!!
class Population:
    """
    A class to represent a population of sudoku puzzles
    """
    def __init__(self, size, initial_board : np.ndarray = np.zeros((9,9), dtype=int),  **kwargs):
        # population size
        self.size = size
        self.individuals = []

        # appending the population with individuals
        for _ in range(size):
            self.individuals.append(Individual(initial_board, **kwargs))

    def evolve(self, gens, xo_prob, mut_prob, select_type, xo, mutate, elitism):
        # gens = 100
        for i in range(gens):
            # selection
            self.selection(select_type)
            # crossover
            self.crossover(xo, xo_prob, elitism)

            self.individuals = [i.mutate(mut_prob) for i in self.individuals]

            print(f"Best individual of gen #{i + 1}: {max([i.fitness for i in self.individuals])}")


    def __len__(self):
        return len(self.individuals)
    
    def get_best_individual(self):
        """
        Function to get the best individual of the population
        """
        fitnesses = [individual.fitness for individual in self.individuals]
        best_fitness_index = fitnesses.index(min(fitnesses))  
        return self.individuals[best_fitness_index]

    def __getitem__(self, position):
        return self.individuals[position]
    
    def selection(self, type : str = 'roulette'):
        """
        Function to select the individuals in the population
        :param type: a string representing the type of selection to apply
        """
        if type == 'roulette':
            self.roulette()

    def roulette(self):
        """
        Function to apply roulette selection
        """

        # Calculate the fitness of each individual, total fitness and probs
        fitnesses = [1/(individual.fitness+0.000001) for individual in self.individuals]
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        
        # Select the individuals using roulette wheel
        self.individuals = np.random.choice(self.individuals, size=self.size, p=probabilities, replace=True)

    def crossover(self, type : str = 'single_point', prob : float = 0.5, elitism : bool = False):
        """
        Function to crossover the individuals in the population
        :param type: a string representing the type of crossover to apply
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

