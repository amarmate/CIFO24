import numpy as np 

def row_conflicts(board, position):
    """
    Function to get the number of conflicts in a row
    :param position: a tuple representing the coordinates of a number inside the row, or int representing the row number
    """
    assert type(position) == int or (type(position) == tuple and len(position) == 2), "The position has to be an int or a tuple of length 2"
    N = board.shape[0]
    row = position if type(position) == int else position[0]
    row_values = board[row]
    return N - len(np.unique(row_values))


def col_conflicts(board, position):
    """
    Function to get the number of conflicts in a column
    :param position: a tuple representing the coordinates of a number inside the column, or int representing the column number
    """
    assert type(position) == int or (type(position) == tuple and len(position) == 2), "The position has to be an int or a tuple of length 2"
    N = board.shape[0]
    column = position if type(position) == int else position[1]
    column_values = board[:, column]
    return N - len(np.unique(column_values))


def box_conflicts(board, position):
    """
    Function to get the number of conflicts in a box
    :param position: a tuple representing the coordinates of a number inside the box or an int representing the box to check, from left to right and top to bottom
    """
    assert type(position) == int or (type(position) == tuple and len(position) == 2), "The position has to be an int or a tuple of length 2"
    N = board.shape[0]

    if type(position) == tuple:
        assert len(position) == 2, "The position has to be a tuple of length 2"
        assert position[0] < N and position[1] < N, "The position has to be within the board"    
        box_size = int(np.sqrt(N))
        row, column = position
        box_row = row // box_size
        box_column = column // box_size
        box_coordinates = [(box_row * box_size + i, box_column * box_size + j) for i in range(box_size) for j in range(box_size)]

    elif type(position) == int:
        box = position
        assert box < N, "The box has to be within the board"
        box_size = int(np.sqrt(N))
        box_row = box // box_size
        box_column = box % box_size
        box_coordinates = [(box_row * box_size + i, box_column * box_size + j) for i in range(box_size) for j in range(box_size)]

    rows, cols = zip(*box_coordinates)
    box = board[np.array(rows), np.array(cols)]
    return N - len(np.unique(box))
    
