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



    # [TO DO] -------------------------------------------------------------------------------------

    def fill_diagonals(self):
        """
        Function to fill the diagonal blocks of the board, which are the easiest to fill
        """
        pass

    def fill_remaining(self):
        """
        Function to fill the remaining cells of the board
        """
        pass

    def check_unused_box(self, position, number):
        """
        Function to check if a number can be placed in a certain box
        :param position: a tuple representing the position of the box
        :param number: an int representing the number to check
        """
        assert len(position) == 2, "The position has to be a tuple of length 2"
        pass

    def check_unused_row(self, row, number):
        """
        Function to check if a number can be placed in a certain row
        :param row: an int representing the row to check, from left to right
        :param number: an int representing the number to check
        """
        pass

    def check_unused_column(self, column, number):
        """
        Function to check if a number can be placed in a certain column
        :param column: an int representing the column to check, from top to bottom
        :param number: an int representing the number to check
        """
        pass

    def check_is_safe(self, position, number):
        """
        Function to check if a number can be placed in a certain position
        :param position: a tuple representing the position of the box
        :param number: an int representing the number to check
        """
        assert len(position) == 2, "The position has to be a tuple of length 2"
        return self.check_unused_box(position, number) and self.check_unused_row(position[0], number) and self.check_unused_column(position[1], number)
    
    def generate_board(self):
        """
        Function to generate a random sudoku board
        """
        pass




# The purpose of this class is to create an individual sudoku puzzle, to which the genetic algorithm will be applied
class individual:
    """
    A class to represent a sudoku puzzle
    """
    def __init__(self, initial_board : np.ndarray):
        """
        Constructor for the sudoku class
        :param initial_board: an NxN numpy array representing the initial state of the board. 0s represent empty cells
        """
        # The initial board is the board that cannot be changed
        self.initial_board = initial_board.copy()
        self.N = initial_board.shape[0]

        # The current board is the board that will be changed and solved 
        self.current_board = self.random_fill_board()

    def random_fill_board(self):
        """
        Function to fill in the empty cells (0s) in the board randomly and ensuring that the board is still valid, i.e. no more that 9 of the same number in the board
        """
        # Flatten the board
        flat_board = self.initial_board.copy().flatten()

        # Get how many of each number is still missing from the board
        number_counts = np.bincount(flat_board, minlength=self.dimension + 1)
        fill_number_count = self.dimension - number_counts[1:]

        # Create an array of the numbers to fill in the board and shuffle
        fill_numbers = np.repeat(np.arange(1, self.dimension + 1), fill_number_count)
        np.random.shuffle(fill_numbers)

        # Fill in the board
        np.putmask(flat_board, flat_board == 0, fill_numbers)










    def __getitem__(self, position):
        # Check if the position is a tuple of length 2
        assert len(position) == 2, "The position has to be a tuple of length 2"
        return self.representation[position]

    def __setitem__(self, position, value):
        # Check if the position is a tuple of length 2
        assert len(position) == 2, "The position has to be a tuple of length 2"
        self.representation[position] = value

    def __repr__(self):
        return f" Fitness: {self.fitness}"


    # [TO DO] -------------------------------------------------------------------------------------

    def get_fitness(self):
        """
        Function to get the fitness of the individual
        """
        pass

    def mutate(self):
        """
        Function to mutate the individual
        """
        pass

    def get_neighbours(self):
        """
        Function to get the neighbours of the individual
        """
        pass




        
        

        

