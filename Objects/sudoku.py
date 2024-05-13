import numpy as np
from operator import attrgetter
from Algorithms.search import hill_climbing 
from Operators.fitness import fitness
from Operators.board_generator import difficulty_function
from copy import deepcopy

class Sudoku:
    def __init__(self, initial_board : np.ndarray = None, board : np.ndarray = None, fill_board : str = 'random', size : int = 9, difficulty : int = 50, fitness_function = fitness,
                 hill_climbing_args = {'max_iterations':100000, 'plateau_threshold':10000, 'num_neighbours': 10}, diff_function = difficulty_function):
        """
        Constructor for the sudoku class
        :param initial_board: an NxN numpy array representing the initial state of the board, where N has to be a perfect square. 0s represent empty cells
        :param board: an NxN numpy array representing the current state of the board. If None, the board will be filled randomly
        :param fill_board: a string representing how to fill the board. Can be 'random', 'logic-random' and None
        :param size: an int representing the size of the board
        :param difficulty: an int representing the number of numbers to remove, from 0 to 100, where 0 is the easiest and 100 is the hardest
        :param fitness_function: a function to calculate the fitness of the individual
        :param hill_climbing_args: a dictionary with the arguments to pass to the hill climbing algorithm : {'max_iterations': 1000, 'num_neighbours': 1, 'swap_number': 1, 'plateau_threshold': 100, 'verbose': 0}
        :param diff_function: a function to calculate the difficulty of the board
        """
        
        # The initial board is the board that cannot be changed
        if initial_board is None:
            assert np.sqrt(size) % 1 == 0, "The dimension has to be a perfect square"
            print('Warning: Board given, but no initial board given. Generating a new board, and overwriting the given board') if board is not None else None
            self.N = size
            self.generate_board(diff_function, size, difficulty, hill_climbing_args)
        else:
            assert initial_board.shape[0] == initial_board.shape[1], "The board has to be a square"
            assert np.sqrt(initial_board.shape[0]) % 1 == 0, "The dimension has to be a perfect square"
            assert np.all(np.unique(initial_board) <= initial_board.shape[0]), "The board has to have numbers from 0 to N"
            self.initial_board = initial_board.copy()
            self.N = initial_board.shape[0]

        # If no board is given, fill the board
        if board is None:
            self.fill_board(mode=fill_board)
        else:
            self.board = board.copy()
        
        self.swappable_positions = list(zip(*np.where(self.initial_board == 0)))
        self.fitness = fitness_function(self.board)

        # Other attributes for crossover
        self.swap_points = self.initial_board == 0
        self.swappable = self.board[self.swap_points]


    
    def generate_board(self, diff_function, size, difficulty, hill_climbing_args):
        """
        Function to generate a new initial
        :param size: an int representing the size of the board
        :param difficulty: an int representing the number of numbers to remove, from 0 to 100, where 0 is the easiest and 100 is the hardest
        :param hill_climbing_args: a dictionary with the arguments to pass to the hill climbing algorithm
        """
        assert 0 <= difficulty <= 100, "The difficulty has to be between 0 and 100"

        # Generate a solved board 
        sudoku = Sudoku(initial_board=np.zeros((size,size), dtype=int), fill_board='random')
        filled_sudoku = hill_climbing(sudoku, **hill_climbing_args)
        assert filled_sudoku.fitness == 0, "The board could not be solved"

        # Remove numbers from the board
        # filled_sudoku.remove_numbers(diff_function, difficulty)
        
        self.initial_board = filled_sudoku.board


    def remove_numbers(self, diff_function, difficulty : int = 50):
        """
        Recursive function to remove numbers from the board
        :param difficulty: an int representing the number of numbers to remove, from 0 to 100, where 0 is the easiest and 100 is the hardest
        :param difficulty_function: a function to calculate the difficulty of the board
        """
        if diff_function(self.initial_board) >= difficulty:
            print("The board is already at the desired difficulty")
            return
        
        # Remove a number from the board 
        
        

        # Recursively call the function to remove numbers
        self.remove_numbers(diff_function, difficulty)



            
    def fill_board(self, mode : str = 'random'):
        """
        Function to fill in the empty cells (0s) in the board, ensuring that the board is still valid, i.e. no more that 9 of the same number in the board
        :param mode: a string representing how to fill the board. Can be 'random', 'logic-random' and None
        """
        assert mode in ['random', 'logic-random', None], "The mode has to be 'random', 'logic-random' or None"

        if mode == 'logic-random':
            self.fill_board_logic()
        
        if mode == None:
            return
        
        # Randomly fill-in the rest of the board
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



    # ------------------------- Algorithmic methods --------------------------------------------------

    def get_neighbours(self, number_of_neighbours : int = 1, swap_number : int = 1):
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
            neighbour_individual = Sudoku(self.initial_board, neighbour)
            neighbours.append(neighbour_individual)
        return neighbours

    def mutate(self, mut_prob : float,  swap_number : int = 1):
        """
        Function to mutate the individual, i.e. change the board randomly
        :param mut_prob: a float representing the probability of mutation
        :param swap_number: an int representing the number of swaps to make in the board
        """
        return self.get_neighbours(number_of_neighbours=1, swap_number=swap_number)[0] if np.random.rand() < mut_prob else self


    # ------------------------- Logic methods --------------------------------------------------
        

    # ------------------------- Other methods --------------------------------------------------

    def display(self):
        """
        Function to print the board in a readable format
        """
        to_print = ""
        # Row iteration
        for row in range(self.N):
            if row % 3 == 0 and row != 0 and row != self.N - 1:
                to_print += "\n----------|-----------|----------\n"
            else:
                to_print += "\n"

            for column in range(self.N):
                if column % 3 == 0 and column != 0 and column != self.N - 1:
                    to_print += " | "
                to_print += " " + str(self.board[row, column]) + " "
        print(to_print)

    def __getitem__(self, position):
        assert len(position) == 2, "The position has to be a tuple of length 2"
        return self.board[position]

    def __setitem__(self, position, value):
        # Check if the position is a tuple of length 2
        assert len(position) == 2, "The position has to be a tuple of length 2"
        self.board[position] = deepcopy(value)

    def __repr__(self):
        return str(self.board)

