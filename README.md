# CIFO24

TODO:
- Compare fitness sharing with version from class - DONE (added fitness sharing as in class with inverted distances)
- Beautiful function for comparison of different configurations - DONE( standard way for experimentation in a separte nb)
- Clean and comment all the code, reorganize repo better - (Only cleaned and organized population for now)
- Create easily understandable documentation in README (main classes and their functionality)
- Run different comparisons (differentmut/xo/selction, different mut and xo rates, different elite size, different sudoku difficulties, population_size) - Not fully done^ but running
- Make visualisations and statistical tests

1) (REWRITE) Individual will have multiple properties:
- initial_board - the starting board state as a list of lists
- board - the current board state as a list of lists
- swappable_positions - a mask that shows swappable positions
- representation - list with only changeable elements as a long list, the one that we will use for crossovers and mutations

So every step of population evolution we will update representation and board afterwards

Report: https://www.overleaf.com/read/jrpvvdnxmjpz#b7ee7f