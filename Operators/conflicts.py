import numpy as np 


def get_infrequent_numbers(board, position):
    """
    Function to get the numbers that are infrequent in the row, column and box of a given position
    :param position: a tuple representing the coordinates of the number
    Returns
        list: a list of numbers that are infrequent in the row, column and box of the given position, e.g.: [((1, 0), (5, 1)), ((5, 1), (2, 0))]
    """
    size = board.shape[0]
    assert type(position) == tuple and len(position) == 2, "The position has to be a tuple of length 2"
    assert 0 <= position[0] < size and 0 <= position[1] < size, "The position has to be within the board"
    row = board[position[0]]
    column = board[:, position[1]]
    box_size = int(np.sqrt(size))
    box = board[(position[0]//box_size)*box_size:(position[0]//box_size)*box_size+box_size, (position[1]//box_size)*box_size:(position[1]//box_size)*box_size+box_size]
    all_numbers = set(range(1, size+1))
    row_numbers = set(row)
    column_numbers = set(column)
    box_numbers = set(box.flatten())
    return list(all_numbers - row_numbers - column_numbers - box_numbers)


def get_swappable_smart(board, swappable):
    """
    Function that finds the numbers that can be swapped in a smart way, from the all possible swaps
    :param board: a numpy array representing the board
    :param swappable: a list of tuples representing the positions of the numbers that can be swapped
    Returns
        list: a list of pairs of tuples representing the positions of the numbers that can be swapped in a smart way
    """
    # First lets get the numbers that can be swapped at all, as in their position there has to be an infrequent number
    dict_swappable = {}
    for position in swappable:
        number = board[position]
        if len(get_infrequent_numbers(board, position)) > 0:
            dict_swappable[position] = number, get_infrequent_numbers(board, position)

    # Now lets match the numbers that can be swapped in a smart way
    smart_swappable = []
    for position, (number, infrequent_numbers) in dict_swappable.items():
        for other_position, (other_number, other_infrequent_numbers) in dict_swappable.items():
            if position != other_position and number in other_infrequent_numbers and other_number in infrequent_numbers:
                smart_swappable.append((position, other_position))
    return smart_swappable
            


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
    
