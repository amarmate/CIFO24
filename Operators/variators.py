import numpy as np 


def single_crossover(sudoku1, sudoku2): 
    """Function to do crossover between two sudokus
    Args:
        sudoku1: Sudoku object
        sudoku2: Sudoku object
    Returns:
        tuple with 2 boards with the new sudokus
    """
    # Assert that sudoku1.swappable exists
    assert hasattr(sudoku1, 'swappable') and hasattr(sudoku2, 'swappable'), "The sudoku objects have to have a swappable attribute"
    parent1, parent2 = sudoku1.swappable, sudoku2.swappable
    initial_board = sudoku1.initial_board
    point = np.random.randint(1, parent1.shape[0] - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))

    # Add the child1 numbers on top of the initial board's 0s 
    board1 = np.putmask(initial_board.copy(), initial_board == 0, child1)
    board2 = np.putmask(initial_board.copy(), initial_board == 0, child2)

    return board1, board2
