import numpy as np 


def crossover(individual_1, individual_2, type='single_point'): 
    """Crossover two individuals"""
    if type == 'single_point': 
        return single_point(individual_1, individual_2)
    elif type == 'multi_point':
        return multi_point(individual_1, individual_2)


def single_point(individual_1, individual_2): 
    """Single point crossover"""
    pass

def multi_point(individual_1, individual_2, n_points : int = 2): 
    """Multi point crossover
    n_points: number of points to crossover
    """
    pass

def uniform(individual_1, individual_2): 
    """Uniform crossover"""
    pass
