import numpy as np
from operator import attrgetter
from Algorithms.search import Hill_climbing
from Operators.fitness import fitness
from Operators.board_generator import difficulty_function
from Operators.conflicts import get_swappable_smart, get_infrequent_numbers
from copy import deepcopy

class Sudoku:
    def __init__(self, initial_board : np.ndarray = None, board : np.ndarray = None, fill_board : str = 'random', size : int = 9, 
                 mode_generate : str = 'random', difficulty : int = 50, diff_function : callable = difficulty_function,
                 fitness_function = fitness,
                 hill_climbing_args = {'max_iterations':100000, 'plateau_threshold':10000, 'num_neighbours': 10, 'smart': False}):
        """
        Constructor for the sudoku class
        :param initial_board: an NxN numpy array representing the initial state of the board, where N has to be a perfect square. 0s represent empty cells
        :param board: an NxN numpy array representing the current state of the board. If None, the board will be filled randomly
        :param fill_board: a string representing how to fill the board. Can be 'random', 'logic-random' and None
        :param size: an int representing the size of the board
        :param mode_generate: a string representing how to generate the board. Can be 'random' or 'smart'
        :param difficulty: an int representing the number of numbers to remove, from 0 to 100, where 0 is the easiest and 100 is the hardest
        :param diff_function: a function to calculate the difficulty of the board
        :param fitness_function: a function to calculate the fitness of the individual
        :param hill_climbing_args: a dictionary with the arguments to pass to the hill climbing algorithm : {'max_iterations': 1000, 'num_neighbours': 1, 'swap_number': 1, 'plateau_threshold': 100, 'verbose': 0}
        """
        
        # The initial board is the board that cannot be changed
        if initial_board is None:
            assert np.sqrt(size) % 1 == 0, "The dimension has to be a perfect square"
            print('Warning: Board given, but no initial board given. Generating a new board, and overwriting the given board') if board is not None else None
            self.N = size
            self.generate_board(mode_generate, diff_function, size, difficulty, hill_climbing_args)
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

        # To compare later with initial distribution
        self.distribution = np.bincount(self.swappable, minlength=self.N + 1)


    def generate_board(self, mode_generate : str = 'random', diff_function : callable = None, size : int = 9, difficulty : int = 50, hill_climbing_args : dict = {}):
        """
        Function to generate a new initial
        :param size: an int representing the size of the board
        :param difficulty: an int representing the number of numbers to remove, from 0 to 100, where 0 is the easiest and 100 is the hardest
        :param hill_climbing_args: a dictionary with the arguments to pass to the hill climbing algorithm
        """
        assert 0 <= difficulty <= 100, "The difficulty has to be between 0 and 100"

        # Generate a solved board 
        sudoku = Sudoku(initial_board=np.zeros((size,size), dtype=int), fill_board='random')
        filled_sudoku = Hill_climbing(sudoku).run(**hill_climbing_args)
        assert filled_sudoku.fitness == 0, "The board could not be solved"

        # Remove numbers from the board
        filled_sudoku.remove_numbers(mode=mode_generate, diff_function=diff_function, difficulty=difficulty)
        
        self.initial_board = filled_sudoku.board


    def remove_numbers(self, mode : str = 'random', difficulty : int = 50 , diff_function : callable = None):
        """
        Recursive function to remove numbers from the board
        :param mode: a string representing how to remove the numbers. Can be 'random' or 'smart'. In case of 'smart' we need to input the difficulty function
        :param difficulty: an int representing the number of numbers to remove, from 0 to 100, where 0 is the easiest and 100 is the hardest
        :param difficulty_function: a function to calculate the difficulty of the board
        """
        assert mode in ['random', 'smart'], "The mode has to be 'random' or 'smart'"
        assert diff_function is not None or mode == 'random', "The difficulty function has to be given if the mode is 'smart'"
        assert 0 <= difficulty <= 100, "The difficulty has to be between 0 and 100"

        # Remove a number from the board 
        if mode == 'random':
            zeros = np.sum(self.board == 0)
            current_difficulty = zeros / (self.N ** 2) * 100
            if current_difficulty >= difficulty:
                return
            
            # Get a random position to remove a number from 
            non_empty_positions = np.where(self.board != 0)
            random_position = np.random.choice(np.arange(len(non_empty_positions[0])))
            position = (non_empty_positions[0][random_position], non_empty_positions[1][random_position])
            self.board[position] = 0

        elif mode == 'smart':
            if diff_function(self.initial_board) >= difficulty:
                return
            
            pass

        # Recursively call the function to remove numbers
        self.remove_numbers(mode=mode, difficulty=difficulty, diff_function=diff_function)


    def fill_board(self, mode : str = 'random'):
        """
        Function to fill in the empty cells (0s) in the board, ensuring that the board is still valid, i.e. no more that 9 of the same number in the board
        :param mode: a string representing how to fill the board. Can be 'random' or 'logic-random'
        """
        assert mode in ['random', 'logic-random', 'row_wise'], "The mode has to be 'random', 'logic-random' or 'row_wise'" 

        if mode == 'logic-random':
            self.fill_board_logic()


        if mode == 'row_wise':
            self.fill_board_row_wise()

        if mode == 'random':
            # Randomly fill-in the rest of the board
            flat_board = self.initial_board.copy().flatten()

            # Get how many of each number is still missing from the board
            number_counts = np.bincount(flat_board, minlength=self.N + 1)
            fill_number_count = self.N - number_counts[1:]

            # Create an array of the numbers to fill in the board and shuffle
            fill_numbers = np.repeat(np.arange(1, self.N + 1), fill_number_count)
            np.random.shuffle(fill_numbers)

            # Fill in the board
            # np.putmask(flat_board, flat_board == 0, fill_numbers)
            flat_board[np.where(flat_board == 0)] = fill_numbers
            self.board = flat_board.reshape(self.N, self.N)
    
    def fill_board_row_wise(self):
        """
        Method to fill the board by filling each row with missing numbers
        """

        board = deepcopy(self.initial_board)
        for i in range(self.N): # go row wise
            fill_numbers = list(set(np.arange(1,10)) - set(self.initial_board[i]))
            np.random.shuffle(fill_numbers)
            board[i][np.where(board[i] == 0)] = fill_numbers

        self.board = board

    def fill_board_logic(self):
        """
        Method to fill the board using logic, i.e. by solving the board with logic 
        """
        pass



    # ------------------------- Algorithmic methods --------------------------------------------------

    def get_neighbours(self, number_of_neighbours : int = 1, swap_number : int = 1, smart : bool = False):
        """
        Function to get the neighbours of the individual
        :param number_of_neighbours: an int representing the number of neighbours to generate
        :param swap_number: an int representing the number of swaps to make in the board
        :param smart: a boolean representing if the neighbours should be generated in a smart way
        """
        # Neighbours will be generated by swapping numbers in the board
        assert swap_number <= len(self.swappable_positions), "The number of swaps has to be less than the number of swappable positions"

        neighbours = []
        for _ in range(number_of_neighbours):
            neighbour = self.board.copy()
            np.random.shuffle(self.swappable_positions)
            if smart:
                swappable_smart = get_swappable_smart(neighbour, self.swappable_positions)
                np.random.shuffle(swappable_smart)

            for i in range(swap_number):
                if smart and len(swappable_smart) > 0:
                    neighbour[swappable_smart[0][0]], neighbour[swappable_smart[0][1]] = neighbour[swappable_smart[0][1]], neighbour[swappable_smart[0][0]]
                    swappable_smart.pop(0)
                else:
                    neighbour[self.swappable_positions[2*i]], neighbour[self.swappable_positions[2*i+1]] = neighbour[self.swappable_positions[2*i+1]], neighbour[self.swappable_positions[2*i]]
            neighbour_individual = Sudoku(self.initial_board, neighbour)
            neighbours.append(neighbour_individual)
        return neighbours
    
    
    def mutate(self, mut_prob : float, n_changes : int = 1, mutation : str = 'swap', verbose : int = 0): 
        """
        Function to mutate the individual, i.e. change the board randomly, by changing n_changes numbers in the board
        :param mut_prob: a float representing the probability of mutation
        :param n_changes: an int representing the number of swaps to make in the board
        :param mutation: a string representing the type of mutation. Can be 'swap', 'change', 'swap-smart', 'change-smart'
        """

        if mutation == 'swap':
            return self.get_neighbours(number_of_neighbours=1, swap_number=n_changes)[0] if np.random.rand() < mut_prob else self
        
        elif mutation == 'swap-smart':
            return self.get_neighbours(number_of_neighbours=1, swap_number=n_changes, smart=True)[0] if np.random.rand() < mut_prob else self
        
        elif mutation == 'swap-row': # Swapping elements inside one random row
            if np.random.rand() > mut_prob:
                return self
            else:
                rand_row = np.random.randint(0, self.N)
                row_swappables = self.board.copy()[rand_row][self.swap_points[rand_row]]

                if len(row_swappables) >= n_changes*2:
                    permute = np.random.choice(np.arange(len(row_swappables)), size = (n_changes,2), replace=False)
                    for i in permute:
                        row_swappables[i[0]] , row_swappables[i[1]] = row_swappables[i[1]], row_swappables[i[0]]
                    
                    self.board[rand_row][self.swap_points[rand_row]] = row_swappables
                else:
                    if verbose > 0:
                        print(f"Amount of swappables: {len(row_swappables)}, n_changes: {n_changes}. Skipping mutation, as the is not enough swappables")
                
                return self
        elif mutation == 'change':
            if np.random.rand() > mut_prob:
                return self
            neighbour = self.board.copy()
            np.random.shuffle(self.swappable_positions)
            for i in range(n_changes):
                new_number = np.random.randint(1, self.N + 1)
                # Randomly select two swappable positions 
                neighbour[self.swappable_positions[i]] = new_number                 
            return Sudoku(self.initial_board, neighbour)
        
        elif mutation == 'change-smart':
            if np.random.rand() > mut_prob:
                return self
            neighbour = self.board.copy()
            np.random.shuffle(self.swappable_positions)
            i, changes = 0, 0
            while changes < n_changes:
                # Get the infrequent numbers for the position
                try: 
                    infrequent_numbers = get_infrequent_numbers(neighbour, self.swappable_positions[i])
                except:
                    if verbose > 0:
                        print(f'Warning: Couldnt finish the smart mutation (only {changes}/{n_changes} done). Returning the individual with simple change mutation.')
                    new_individual = Sudoku(self.initial_board, neighbour)
                    return new_individual.mutate(mut_prob, n_changes - changes, mutation='change')
                
                # If there are infrequent numbers, change the number
                if len(infrequent_numbers) > 0:
                    new_number = np.random.choice(infrequent_numbers)
                    neighbour[self.swappable_positions[i]] = new_number
                    changes += 1
                i += 1
                
            return Sudoku(self.initial_board, neighbour)
        else: 
            print('Warning ! Mutation not recognized. Returning the individual without mutation')
            return self

    # ------------------------- Other methods --------------------------------------------------

    def display(self):
        """
        Function to print the board in a readable format
        """
        to_print = ""
        sq_size = int(np.sqrt(self.N))
        # Row iteration
        for row in range(self.N):
            if row % np.sqrt(self.N) == 0 and row != 0 and row != self.N - 1:
                to_print += "\n" + (sq_size + 5)*'-' + ((sq_size - 2) * ('|' + (self.N + 1)*'-')) + '|' + (sq_size + 5)*'-' + "\n"
            else:                
                to_print += "\n"              


            for column in range(self.N):
                if column % np.sqrt(self.N) == 0 and column != 0 and column != self.N - 1:
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

