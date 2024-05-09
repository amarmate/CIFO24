import numpy as np
from operator import attrgetter


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
        self.swappable_positions = self.initial_board == 0

        if board is None:
            self.random_fill_board()
        else:
            self.board = board.copy()

        self.representation = self.board[self.swappable_positions]
        self.distribution = np.bincount(self.representation, minlength=self.N + 1)

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

    def mutate(self, mutation_rate : float = 0.1, swap_number : int = 5):
        """
        Function to mutate the individual, i.e. change the board randomly
        """
        # TODO: Do not mutate elite

        if np.random.rand() < mutation_rate:
            new_representation = self.representation.copy()
            new_board = self.board.copy()

            for i in range(swap_number):
                x, y = np.random.choice(len(new_representation), 2, replace=False)
                new_representation[x], new_representation[y] = new_representation[y], new_representation[x]

            np.putmask(new_board, self.swappable_positions, new_representation)
            return self.__class__(self.initial_board, new_board)
        else:
            return self
