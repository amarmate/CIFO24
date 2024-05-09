import numpy as np 


def crossover(board1, board2): 
    """Function to do crossover between two boards
    Args:
        board1: np.array
        board2: np.array
    Returns:
        tuple with 2 np.arrays with the new boards
    """
    pass


def mutate(board, swappable_positions, mutation_rate : float = 0.1, swap_number : int = 1):
    """
    Function to mutate a board, i.e. change the board randomly
    :param board: the board to mutate as a numpy array
    :param mutation_rate: a float representing the probability of mutation
    :param swap_number: an int representing the number of swaps to make in the board

    Returns:
        A new board with the mutation applied, in the form of an numpy array
    """

    if np.random.rand() < mutation_rate:
        mutated_board = board.copy()
        np.random.shuffle(swappable_positions)
        for i in range(swap_number):
            # Randomly select two swappable positions
            mutated_board[swappable_positions[2 * i]], mutated_board[swappable_positions[2 * i + 1]] = (
                mutated_board[swappable_positions[2 * i + 1]],
                mutated_board[swappable_positions[2 * i]],
                )
        return mutated_board
    else:
        return board
