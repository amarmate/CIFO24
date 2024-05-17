import numpy as np 
import pandas as pd
from Objects.sudoku import Sudoku
from copy import deepcopy
from Operators.fitness import fitness as fitness_function
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

class Population:
    """
    A class to represent a population of sudoku puzzles
    """
    def __init__(self, size : int = 100, initial_board : np.ndarray = None, sudoku_arguments = {}):
        """
        Constructor for the population class
        :param size: an int representing the size of the population
        :param initial_board: an NxN numpy array representing the initial state of the board, where N has to be a perfect square. 0s represent empty cells
        :param sudoku_arguments: a dictionary with the arguments to pass to the sudoku class 
            available: fill_board, size, difficulty, fitness_function, hill_climbing_args, diff_function
        """
        self.size = size
        self.individuals = []

        # For the history of the population
        self.history = {}
        self.params = {}
        self.gen = 0

        if initial_board is None:
            first_individual = Sudoku()
            initial_board = first_individual.initial_board

        # appending the population with individuals
        for _ in range(size):
            self.individuals.append(Sudoku(initial_board, **sudoku_arguments))


    def evolve(self, gens : int = 1000, xo_prob : float = 0.7, mut_prob : float = 0.3, select_type : str = 'roulette', xo : str = 'single_point', 
               diversify : str = None,
               elite_size : int = 1, 
               mutation : str ='change', 
               swap_number : int = 1, 
               keep_distribution : bool =False, 
               verbose : int = 0):
        """
        Function to evolve the population
        :param gens: an int representing the number of generations to evolve
        :param xo_prob: a float representing the probability of crossover
        :param mut_prob: a float representing the probability of mutation
        :param select_type: a string representing the type of selection to apply, choose between 'roulette', 'tournament', 'rank'
        :param xo: a string representing the type of crossover to apply, choose between 'single_point', 'multi_point', 'pmxc', 'uniform'
        :param diversify: a string representing the type of diversification to apply, choose between 'fitness-sharing' and 'restricted-mating'
        :param elite_size: an int representing the number of elite individuals to keep
        :param mutation: a string representing the type of mutation to apply
        :param swap_number: an int representing the number of swaps to make in the board
        :param keep_distribution: a boolean representing whether to keep the distribution of numbers in the board
        :param verbose: an int representing the verbosity level of the evolution
        """
        
        assert elite_size <= self.size, "The number of elite individuals has to be less than the population size"
        assert elite_size >= 0, "The number of elite individuals has to be greater than 0"

        self.params = {'gens': gens, 'xo_prob': xo_prob, 'mut_prob': mut_prob, 'select_type': select_type, 'xo': xo, 'diversify': diversify, 'elite_size': elite_size, 'mutation': mutation, 'swap_number': swap_number, 'keep_distribution': keep_distribution}

        for i in range(gens):
            if elite_size > 0:
                best_individuals = self.get_best_individuals(elite_size)

            self.selection(select_type, diversify)
            self.crossover(type=xo, prob=xo_prob)
            for j in range(len(self.individuals)):
                self[j] = self[j].mutate(mut_prob, swap_number, mutation=mutation)

            if keep_distribution:
                self.keep_distribution()

            if elite_size > 0:
                for j in range(elite_size):
                    self[j] = best_individuals[j]

            best_individual_fitness = min([ind.fitness for ind in self.individuals])
            print(f"Best individual of gen #{i + 1}: {best_individual_fitness}") if verbose >= 1 else None

            self.history[self.gen] = best_individual_fitness, self.phenotype_diversity(type='entropy'), self.genotype_diversity()
            self.gen += 1

            if self.get_best_individuals(1)[0].fitness == 0:
                print(f"Solution found in generation {i + 1}!")
                self.get_best_individuals(1)[0].display()
                break

    # -------------------------------------------------------------------------------------------------------
    def keep_distribution(self):
        """
        Function to keep the correct distribution of numbers inside each individual 
        """

        # TODO check why it doesnt really preserve distribution in some cases
        # print("Perfect distribution", self[0].distribution)

        perfect_distribution = np.tile(self[0].distribution, (len(self), 1))
        real_distribution = np.apply_along_axis(lambda row: np.bincount(row, minlength=self[0].N + 1), axis=1, arr=[self[i].swappable for i in range(len(self))])
        difference = perfect_distribution - real_distribution

        numbers = np.tile(np.indices(difference[0].shape)[0], (len(self) + 1,1))
        add = np.where(difference < 0, 0, difference)
        remove = np.where(difference > 0, 0, -difference)

        for i in range(len(self)):
            if np.sum(remove[i]) > 0:

                values_add = np.repeat(numbers[i], add[i], axis=0)
                np.random.shuffle(values_add)
                values_remove = np.repeat(numbers[i], remove[i], axis=0)

                counts = {val: np.sum(values_remove == val) for val in np.unique(values_remove)}
                mask = np.isin(self[i].swappable, values_remove)
                
                # Get the random indices of elements to mask that match values remove in offspring
                indices_to_mask = np.concatenate([np.random.choice(np.flatnonzero(mask & (self[i].swappable == val)), 
                                                                size=counts[val], 
                                                                replace=False) for val in counts.keys()])
                
                # Put new values on these positions
                self[i].swappable[indices_to_mask] = values_add
            else:
                pass

    # TODO too harsh normalization 
    def get_distances(self, normalize : bool = True):
        """
        Function to get the distance matrix between individuals, in terms of genotypic distance
        :param normalize: a boolean to normalize the distances
        Returns:
            np.ndarray: a 2D numpy array with the sum of all distances between one individual and all the others
        """
        individuals = np.array([ind.swappable for ind in self.individuals])
        
        # distances = np.sum(cdist(individuals, individuals, 'hamming'), axis=1) 
        def get_distance(i):
            diff = individuals[i] != individuals
            diff[i, :] = False
            return np.sum(np.sum(diff, axis=1))
        
        distances = np.array([get_distance(i) for i in range(len(self))]) 

        if normalize:
            distances = 0.5+ (distances - np.min(distances))*0.5 / (np.max(distances) - np.min(distances))
        return distances
        

    def phenotype_diversity(self, type = 'entropy'):
        """
        Function to calculate the phenotypic diversity of the population 
        :param type: a string representing the type of diversity to calculate, choose between 'entropy' and 'variance'
        """
        assert type in ['entropy', 'variance'], "The type of diversity has to be either 'entropy' or 'variance'"
        fitnesses = np.array([ind.fitness+0.000001 for ind in self.individuals])
        if type == 'entropy':
            diversity = -np.sum(fitnesses * np.log(fitnesses))
        elif type == 'variance':
            diversity = np.var(fitnesses)
        return diversity
    
    def genotype_diversity(self):
        """
        Function to calculate the phenotypic diversity of the population 
        :param type: a string representing the type of diversity to calculate, choose between 'entropy' and 'variance'
        """
        get_distances = self.get_distances(normalize=False)
        diversity = np.mean(get_distances) / (self.size * len(self.individuals[0].swappable))
        return diversity


    # ------------------------------------ Crossover ------------------------------------------------
    def crossover(self, type : str = 'single_point', prob : float = 0.5, random_fill : bool = False):
        """
        Function to crossover the individuals in the population
        :param type: a string representing the type of crossover to apply
        """
        if type != 'special_xo' and random_fill:
            print("Warning! Random fill can only be applied to special crossover.")

        if type == 'single_point':
            self.single_point(prob)

        if type == 'multi_point':
            return self.multi_point(prob)

        elif type == 'pmxc':
            self.pmx_crossover()

        elif type == 'uniform':
            self.uniform()
        
        elif type == 'cycle':
            self.cycle()

        elif type == 'special_xo':
            self.special_xo(prob, random_fill=True)



    def single_point(self, prob : float = 0.5, keep_distribution : bool = False):
        """
        Function to apply single point crossover
        """
        if prob == 0:
            pass
        else:
            population_array = np.array([i.swappable for i in self.individuals])
                
            # Get the number of parents and the shape of each parent
            num_parents = len(self)
            rows, cols = population_array.shape

            # Sample how many crossovers to do
            num_crossovers = sum(np.random.choice([0,1], size = int(num_parents/2), replace=True, p=[1-prob, prob]))

            # Select two random parents for each offspring without replacement
            parent_indices = np.random.choice(num_parents, size=(num_crossovers,2), replace=False)

            # Choose random crossover points for each pair of parents
            crossover_points = np.random.randint(cols-1, size=num_crossovers)

            # Matrices with parents from both sides
            parent1 = population_array[parent_indices[:, 0]]
            parent2 = population_array[parent_indices[:, 1]]

            # Mask all the values that are above/below crossover line with zeros
            parent1_masked = np.ma.masked_array(parent1, np.arange(cols) > crossover_points[:, np.newaxis]).filled(0)
            parent2_masked = np.ma.masked_array(parent2, np.arange(cols) <= crossover_points[:, np.newaxis]).filled(0)

            # offsprings are just sum of parents
            offspring1 = parent1_masked + parent2_masked
            offspring2 = (parent1 - parent1_masked) + (parent2 - parent2_masked)

            offspring = np.concatenate((offspring1, offspring2))

            # Put parents that didn't get crossovered back into the population
            if num_crossovers < num_parents:
                left_ind = np.setdiff1d(np.arange(num_parents), parent_indices.flatten())
                offspring = np.concatenate((offspring, population_array[left_ind]))


            # Append random offsprings from parents until we reach population size
            if offspring.shape[0] < num_parents:
                while offspring.shape[0] < num_parents:
                    offspring = np.append(offspring, 
                                        np.expand_dims(population_array[np.random.choice(len(population_array))], axis=0), axis = 0)
                    
            if keep_distribution:

                perfect_distribution = np.tile(self.individuals[0].distribution, len(offspring))
                real_distribution = np.apply_along_axis(lambda row: np.bincount(row, minlength=10), axis=1, arr=offspring)
                difference = perfect_distribution - real_distribution

            
            # Put the offsprings into the population
            for i in range(len(self)):
                swappable = offspring[i]
                board = deepcopy(self.individuals[i].board)
                np.putmask(board, self.individuals[i].swap_points, swappable)
                self.individuals[i] = Sudoku(initial_board=self.individuals[i].initial_board, board=board)


    def multi_point(self, elitism : bool = False, 
                    num_points : int = 3, 
                    prob : float = 0.5, 
                    keep_distribution : bool = False):
        """
        Function to apply multi point crossover
        """
        if prob == 0:
            pass
        else:
            population_array = np.array([i.swappable for i in self.individuals])
                
            # Get the number of parents and the shape of each parent
            num_parents = len(self)
            rows, cols = population_array.shape

            # Sample how many crossovers to do
            num_crossovers = sum(np.random.choice([0,1], size = int(num_parents/2), replace=True, p=[1-prob, prob]))

            # Select two random parents for each offspring without replacement
            parent_indices = np.random.choice(num_parents, size=(num_crossovers,2), replace=False)

            # Choose random crossover points for each pair of parents
            crossover_points = np.random.randint(cols-1, size=num_crossovers)

            # Matrices with parents from both sides
            parent1 = population_array[parent_indices[:, 0]]
            parent2 = population_array[parent_indices[:, 1]]

            #Define crossoverpoints
            crossover_points = np.random.randint(cols-1, size=(num_crossovers,num_points))
            crossover_points.sort(axis=1)
            last_index = np.array([cols]*num_crossovers).reshape(-1,1)
            crossover_points = np.hstack((crossover_points, last_index))

            # Create mask
            total_mask = []
            for crossover_line in crossover_points:
                mask_line = np.zeros(cols).astype(bool)
                for j in range(int(num_points/2)):
                    mask_line[np.arange(crossover_line[j*2], crossover_line[(j+1)*2])] = True
                total_mask.append(mask_line)
            total_mask = np.array(total_mask)

            # Mask all the values that are above/below crossover line with zeros
            parent1_masked = np.ma.masked_array(parent1, total_mask).filled(0)
            parent2_masked = np.ma.masked_array(parent2, total_mask == False).filled(0)

            # offsprings are just sum of parents
            offspring1 = parent1_masked + parent2_masked
            offspring2 = (parent1 - parent1_masked) + (parent2 - parent2_masked)

            offspring = np.concatenate((offspring1, offspring2))

            # Put parents that didn't get crossovered back into the population
            if num_crossovers < num_parents:
                left_ind = np.setdiff1d(np.arange(num_parents), parent_indices.flatten())
                offspring = np.concatenate((offspring, population_array[left_ind]))


            # Append random offsprings from parents until we reach population size
            if offspring.shape[0] < num_parents:
                while offspring.shape[0] < num_parents:
                    offspring = np.append(offspring, 
                                        np.expand_dims(population_array[np.random.choice(len(population_array))], axis=0), axis = 0)
                    
            if keep_distribution:

                perfect_distribution = np.tile(self.individuals[0].distribution, len(offspring))
                real_distribution = np.apply_along_axis(lambda row: np.bincount(row, minlength=10), axis=1, arr=offspring)
                difference = perfect_distribution - real_distribution

            
            # Put the offsprings into the population
            for i in range(len(self)):
                swappable = offspring[i]
                board = deepcopy(self.individuals[i].board)
                np.putmask(board, self.individuals[i].swap_points, swappable)
                self.individuals[i] = Sudoku(initial_board=self.individuals[i].initial_board, board=board)
    
    
    def pmx_crossover(self):
        """
        Function to apply partially mapped crossover
        """

    def uniform(self):
        """
        Function to apply uniform crossover
        """
        pass

    def special_xo(self, prob : float = 0.5, random_fill : bool = False):
        """
        Function to apply a special crossover
        """
        def special(individual1, individual2, prob : float = 0.5, random_fill : bool = False):
            if np.random.rand() > prob:
                return individual1, individual2
            
            parent1 = individual1.swappable       # E.g.: [4,4,2,6,1,9,5,5,3]
            parent2 = individual2.swappable       # E.g.: [3,4,5,9,1,6,4,2,5]

            split_index = np.random.randint(1, len(parent1)-1)
            first_part_1 = parent1[:split_index]  # E.g.: [4,4,2]
            last_part_2 = parent1[split_index:]   # E.g.: [6,1,9,5,5,3]

            # Now we need to get the first_part_2 and last_part_1 from parent2
            first_part_2, last_part_1 = [], []
            offspring_1, offspring_2 = np.zeros(len(parent1)), np.zeros(len(parent1))
            
            if random_fill:
                first_part_2 = np.random.choice(np.setdiff1d(parent2, first_part_1), len(first_part_1), replace=False)
                last_part_1 = np.random.choice(np.setdiff1d(parent1, last_part_2), len(last_part_2), replace=False)
                offspring_1 = np.concatenate((first_part_1, last_part_1))
                offspring_2 = np.concatenate((first_part_2, last_part_2))

            elif len(first_part_1) <= len(last_part_2):
                # We know that in the first_part_2 we are missing [4,4,2], so we need to find them in parent2
                missing_numbers = np.bincount(first_part_1, minlength=10)
                indices = []
                for i in range(len(parent2)):
                    number = parent2[i]
                    if missing_numbers[number] > 0:
                        # Saving the indices of the numbers for the first_part_2 
                        indices.append(i)
                        missing_numbers[number] -= 1
                first_part_2 = parent2[indices]
                last_part_1 = parent2[~np.isin(np.arange(len(parent2)), indices)]
                offspring_1 = np.concatenate((first_part_1, last_part_1))
                offspring_2 = np.concatenate((first_part_2, last_part_2))
            
            else:
                # We know that in the first_part_2 we are missing [4,4,2], so we need to find them in parent2
                missing_numbers = np.bincount(last_part_2, minlength=10)
                indices = []
                for i in range(len(parent2)):
                    number = parent2[i]
                    if missing_numbers[number] > 0:
                        # Saving the indices of the numbers for the first_part_2 
                        indices.append(i)
                        missing_numbers[number] -= 1
                first_part_2 = parent2[~np.isin(np.arange(len(parent2)), indices)]
                last_part_1 = parent2[indices]
                offspring_1 = np.concatenate((first_part_1, last_part_1))
                offspring_2 = np.concatenate((first_part_2, last_part_2))

                
            # Get the individuals
            offspring_board_1, offspring_board_2 = deepcopy(individual1.initial_board), deepcopy(individual2.initial_board)
            np.putmask(offspring_board_1, individual1.swap_points, offspring_1)
            np.putmask(offspring_board_2, individual2.swap_points, offspring_2)
            offspring_1 = Sudoku(initial_board=individual1.initial_board, board=offspring_board_1)
            offspring_2 = Sudoku(initial_board=individual2.initial_board, board=offspring_board_2)
            return offspring_1, offspring_2

        parents = np.random.choice(self.individuals, size=(int(self.size/2), 2), replace=False)
        offspring = [special(parents[i, 0], parents[i, 1], prob, random_fill) for i in range(int(self.size/2))]
        self.individuals = offspring



    def cycle(self, prob : float = 0.5,  keep_distribution : bool = False):
        """
        Function to apply cycle crossover
        """

        if prob == 0:
            pass
        else:
            population_array = np.array([i.swappable for i in self.individuals])
                
            # Get the number of parents and the shape of each parent
            num_parents = len(self)
            rows, cols = population_array.shape

            # Sample how many crossovers to do
            num_crossovers = sum(np.random.choice([0,1], size = int(num_parents/2), replace=True, p=[1-prob, prob]))

            # Select two random parents for each offspring without replacement
            parent_indices = np.random.choice(num_parents, size=(num_crossovers,2), replace=False)

            # Matrices with parents from both sides
            parent1 = population_array[parent_indices[:, 0]]
            parent2 = population_array[parent_indices[:, 1]]

            offspring = []

            for i in range(len(parent1)):
                indicator = True
                current_position = 0
                available_indicies = list(np.arange(len(parent1[i])))
                while indicator:
                    available_indicies.remove(current_position)
                    mask = np.zeros(len(parent1[i]))
                    mask[available_indicies] = True
                    possible_next_indicies = np.where(mask*parent1[i] == parent2[i][current_position])[0]
                    if len(possible_next_indicies) == 0:
                        indicator = False
                    else:
                        current_position = np.random.choice(possible_next_indicies)
                all_indicies = list(np.arange(len(parent1[i])))
                used_indicies = list(set(all_indicies) - set(available_indicies))
                offspring1 = np.zeros(len(parent1[i]), dtype=int)
                offspring2 = np.zeros(len(parent1[i]), dtype=int)

                offspring1[available_indicies] = parent2[i][available_indicies]
                offspring1[used_indicies] = parent1[i][used_indicies]
                offspring2[available_indicies] = parent1[i][available_indicies]
                offspring2[used_indicies] = parent2[i][used_indicies]
                offspring.append(offspring1)
                offspring.append(offspring2)
            
            offspring = np.array(offspring)

            # Put parents that didn't get crossovered back into the population
            if num_crossovers < num_parents:
                left_ind = np.setdiff1d(np.arange(num_parents), parent_indices.flatten())
                offspring = np.concatenate((offspring, population_array[left_ind]))


            # Append random offsprings from parents until we reach population size
            if offspring.shape[0] < num_parents:
                while offspring.shape[0] < num_parents:
                    offspring = np.append(offspring, 
                                        np.expand_dims(population_array[np.random.choice(len(population_array))], axis=0), axis = 0)
                    
            if keep_distribution:
                perfect_distribution = np.tile(self.individuals[0].distribution, len(offspring))
                real_distribution = np.apply_along_axis(lambda row: np.bincount(row, minlength=10), axis=1, arr=offspring)
                difference = perfect_distribution - real_distribution

            
            # Put the offsprings into the population
            for i in range(len(self)):
                swappable = offspring[i]
                board = deepcopy(self.individuals[i].board)
                np.putmask(board, self.individuals[i].swap_points, swappable)
                self.individuals[i] = Sudoku(initial_board=self.individuals[i].initial_board, board=board)

    # ------------------------------------ Selection ------------------------------------------------
    def selection(self, type : str = 'roulette', diversify : str = None):
        """
        Function to select the individuals in the population
        :param type: a string representing the type of selection to apply
        :param diversify: a string representing the type of diversification to apply, choose between 'fitness-sharing' and 'restricted-mating'
        """
        if type == 'roulette':
            self.roulette(diversify=diversify)

        if type == 'tournament':
            self.tournament(diversify=diversify)

        if type == 'rank':
            self.rank()
    
    def roulette(self, diversify : str = None):
        """
        Function to apply roulette selection
        :param diversify: a string representing the type of diversification to apply, choose between 'fitness-sharing' and 'restricted-mating'
        """

        # Calculate the fitness of each individual, total fitness and probs
        if diversify is None:
            fitnesses = [1/(individual.fitness+0.000001) for individual in self.individuals]
        elif diversify == 'fitness-sharing':
            # Get the distances between individuals
            distances = self.get_distances(normalize=True)
            # The larger the distance, the better the fitness
            fitnesses = [(1/(individual.fitness+0.000001)) * distances[i] for i, individual in enumerate(self.individuals)]
        elif diversify == 'restricted-mating':
            fitnesses = []
        
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        probabilities_std = [prob / sum(probabilities) for prob in probabilities]
        
        # Select the individuals using roulette wheel
        self.individuals = np.random.choice(self.individuals, size=self.size, p=probabilities_std, replace=True)
    

    def tournament(self, tournament_size : int = 3, diversify : str = None):
        """
        Function to apply tournament selection
        :param tournament_size: an int representing the size of the tournament
        :param diversify: a string representing the type of diversification to apply, choose between 'fitness-sharing' and 'restricted-mating'
        """

        # Define tournament
        tournament = np.random.choice(self.individuals, size=(self.size, tournament_size), replace=True)

        # Keep all fitnesses
        fitnesses = np.array([np.array([ind.fitness for ind in subarray]) for subarray in tournament])

        if diversify == 'fitness-sharing':
            # Get the distances between individuals
            distances = self.get_distances(normalize=True)
            # The larger the distance, the better the fitness
            fitnesses = fitnesses * distances[:, None]

        # Find best in each of the tournaments
        min_indices = np.argmin(fitnesses, axis=1)
        row_indices = np.arange(min_indices.size)[:, None]
        col_indices = min_indices[:, None]

        # Reassign individuals
        self.individuals = tournament[row_indices, col_indices].flatten().tolist()


    # ------------------------------------ Dunder and get methods ------------------------------------------------
    def __len__(self):
        return len(self.individuals)
    
    def get_best_individuals(self, n : int = 1):
        """
        Function to get the best individual of the population
        :param n: an int representing the number of best individuals to get
        """
        best_individuals = sorted(self.individuals, key=lambda x: x.fitness)[:n]
        return best_individuals

    def __getitem__(self, position):
        return self.individuals[position]
    
    def __setitem__(self, position, value):
        self.individuals[position] = deepcopy(value)

    def plot_history(self, genodiv = False, phenodiv = False, grid = False, info = False, ma_smooth_ratio = 0.05):
        """
        Function to plot the history of the population
        Args:
            genodiv (bool, optional): Whether to plot genotype diversity. Defaults to False.
            phenodiv (bool, optional): Whether to plot phenotype diversity. Defaults to False.
            grid (bool, optional): Whether to show a grid on the plot. Defaults to False.
            info (bool, optional): Whether to show the parameters used in the evolution. Defaults to False.
            ma_smooth_ratio (float, optional): Ratio of the moving average smoothing to the number of iterations. Defaults to 0.05.
        """
        assert ma_smooth_ratio > 0 and ma_smooth_ratio <= 1, "The moving average smoothing ratio has to be between 0 and 1"

        if genodiv and phenodiv:
            # Create two subplots
            print("Warning! Both genotypic and phenotypic diversity cannot be plotted at the same time. Phenotypic will be ploted.")

        fitness_plot = {iteration : v[0] for iteration, v in self.history.items()}
        plt.plot(list(fitness_plot.keys()), list(fitness_plot.values()), label='Fitness')

        # The pheno and geno have to be smoothed
        diversity_df = pd.DataFrame(self.history).T
        diversity_df = diversity_df.rolling(window=int(len(self.history) * ma_smooth_ratio)).mean()
        if phenodiv:
            ax2 = plt.twinx()
            ax2.plot(list(diversity_df.index), diversity_df[1], label='Phenotypic diversity', color='red')
        elif genodiv:
            ax2 = plt.twinx()
            ax2.plot(list(diversity_df.index), diversity_df[2], label='Genotypic diversity', color='red')
    
        if info:
            plt.title(f"Evolution with parameters: {self.params}", fontsize=8)
        else:
            plt.title("Evolution")
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.legend()
        plt.show()


# ------------------------- Main --------------------------------------------------
if __name__ == '__main__':
    test_board = np.array([[9, 4, 7, 3, 2, 6, 5, 8, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 7, 3, 5, 9, 2, 1, 6, 8],
        [1, 2, 9, 8, 6, 4, 7, 3, 5],
        [5, 6, 8, 7, 1, 3, 4, 9, 2],
        [7, 9, 2, 4, 5, 8, 3, 1, 6],
        [6, 1, 5, 2, 3, 9, 8, 7, 4],
        [3, 8, 4, 6, 7, 1, 2, 5, 9]])

    population = Population(
    size=10,
    initial_board=test_board,
    )
    population.evolve(gens = 10000,
                    xo_prob = 0.9, 
                    mut_prob=0.25, 
                    select_type='roulette', 
                    xo = 'single_point', 
                    mutate = True, #should be smth else
                    elitism = True,
                    keep_distribution=True)

