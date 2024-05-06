import numpy as np

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


    # [TO DO] -------------------------------------------------------------------------------------

    def mutate(self, mutation_rate : float = 0.1):
        """
        Function to mutate the individual, i.e. change the board randomly
        """
        pass



# Handle this TODO !!!!
class Population:
    """
    A class to represent a population of sudoku puzzles
    """
    def __init__(self, size, **kwargs):
        # population size
        self.size = size
        self.individuals = []

        # appending the population with individuals
        for _ in range(size):
            self.individuals.append(Individual())

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

    def crossover(self, type : str = 'single_point'):
        """
        Function to crossover the individuals in the population
        :param type: a string representing the type of crossover to apply
        """
        if type == 'single_point':
            self.single_point()

        elif type == 'multi_point':
            self.multi_point()

        elif type == 'uniform':
            self.uniform()
            

    def single_point(self):
        """
        Function to apply single point crossover
        """
        pass

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
        



        
        

        

