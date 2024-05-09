from Algorithms.sudoku import Sudoku
import numpy as np

class Individual(Sudoku):
    def __init__(self, initial_board : np.ndarray = np.zeros((9,9), dtype=int), board : np.ndarray = None, fill_board : str = 'logic-random', size : int = 9, difficulty : int = 50):
        """
        Constructor for the individual class
        :param initial_board: an NxN numpy array representing the initial state of the board, where N has to be a perfect square. 0s represent empty cells
        :param board: an NxN numpy array representing the current state of the board. If None, the board will be filled randomly
        :param fill_board: a string representing how to fill the board. Can be 'random', 'logic-random' and None
        :param size: an int representing the size of the board
        :param difficulty: an int representing the number of numbers to remove, from 0 to 100, where 0 is the easiest and 100 is the hardest
        """

        super().__init__(initial_board, board, fill_board, size, difficulty)
        self.representation = self.board[self.swappable_positions]
        self.distribution = np.bincount(self.representation, minlength=self.N + 1)    
        self.swappable_positions = self.initial_board == 0
        
    def mutate(self, mutation_rate : float = 0.1, swap_number : int = 5):
        """
        Function to mutate the individual, i.e. change the board randomly
        :param mutation_rate: a float representing the probability of mutation
        :param swap_number: an int representing the number of swaps to make in the board
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