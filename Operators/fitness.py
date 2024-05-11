import numpy as np
from Operators.conflicts import row_conflicts, col_conflicts, box_conflicts

def fitness(board : np.ndarray):
    """
    Function to get the fitness of the individual
    The fitness will be gauged by the number of conflicts in the board
    :param board: the board to get the fitness of
    """
    N = board.shape[0]
    
    row_sum = np.sum([row_conflicts(board, i) for i in range(N)])
    column_sum = np.sum([col_conflicts(board, i) for i in range(N)])
    box_sum = np.sum([box_conflicts(board, i) for i in range(N)])
    return row_sum + column_sum + box_sum