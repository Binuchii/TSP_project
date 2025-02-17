import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import random
import tsplib95

class TSPGeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 100,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8,
                 elite_size: int = 2,
                 generations: int = 500):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.generations = generations
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def initialize_population(self, cities: List[int]) -> List[List[int]]:
        """Generate initial random population."""
        return [random.sample(cities, len(cities)) for _ in range(self.population_size)]
    
    def calculate_distance(self, city1: Tuple[float, float], 
                         city2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two cities."""
        return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
    
    def calculate_route_length(self, route: List[int], 
                             city_coords: Dict[int, Tuple[float, float]]) -> float:
        """Calculate total route length."""
        total_distance = 0
        for i in range(len(route)):
            city1 = city_coords[route[i]]
            city2 = city_coords[route[(i + 1) % len(route)]]
            total_distance += self.calculate_distance(city1, city2)
        return total_distance
    
    def fitness(self, route: List[int], 
               city_coords: Dict[int, Tuple[float, float]]) -> float:
        """Calculate fitness (inverse of route length)."""
        return 1 / self.calculate_route_length(route, city_coords)
    
    def tournament_selection(self, population: List[List[int]], 
                           fitness_values: List[float], 
                           tournament_size: int = 2) -> List[int]:
        """Select parent using tournament selection."""
        tournament = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament]
        return population[tournament[tournament_fitness.index(max(tournament_fitness))]]
    
    def ordered_crossover(self, parent1: List[int], 
                         parent2: List[int]) -> List[int]:
        """Perform ordered crossover (OX1)."""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Get slice from parent1
        child = [-1] * size
        for i in range(start, end + 1):
            child[i] = parent1[i]
            
        # Fill remaining positions with cities from parent2
        parent2_idx = 0
        child_idx = (end + 1) % size
        while -1 in child:
            if parent2[parent2_idx] not in child:
                child[child_idx] = parent2[parent2_idx]
                child_idx = (child_idx + 1) % size
            parent2_idx = (parent2_idx + 1) % size
            
        return child
    
    def cycle_crossover(self, parent1: List[int], 
                       parent2: List[int]) -> List[int]:
        """Perform cycle crossover (CX)."""
        size = len(parent1)
        child = [-1] * size
        
        # Start with first position
        pos = 0
        while True:
            child[pos] = parent1[pos]
            pos = parent2.index(parent1[pos])
            if child[pos] != -1:
                break
                
        # Fill remaining positions from parent2
        for i in range(size):
            if child[i] == -1:
                child[i] = parent2[i]
                
        return child
    
    def swap_mutation(self, route: List[int]) -> List[int]:
        """Perform swap mutation."""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route
    
    def inversion_mutation(self, route: List[int]) -> List[int]:
        """Perform inversion mutation."""
        if random.random() < self.mutation_rate:
            i, j = sorted(random.sample(range(len(route)), 2))
            route[i:j+1] = reversed(route[i:j+1])
        return route
    
    def evolve_population(self, population: List[List[int]], 
                         city_coords: Dict[int, Tuple[float, float]]) -> List[List[int]]:
        """Create new generation through selection, crossover, and mutation."""
        fitness_values = [self.fitness(route, city_coords) for route in population]
        
        # Elitism
        elite = [x for _, x in sorted(zip(fitness_values, population), 
                                    reverse=True)][:self.elite_size]
        
        # Create new population
        new_population = elite.copy()
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population, fitness_values)
            parent2 = self.tournament_selection(population, fitness_values)
            
            # Crossover
            if random.random() < self.crossover_rate:
                if random.random() < 0.5:
                    child = self.ordered_crossover(parent1, parent2)
                else:
                    child = self.cycle_crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if random.random() < 0.5:
                child = self.swap_mutation(child)
            else:
                child = self.inversion_mutation(child)
                
            new_population.append(child)
            
        return new_population
    
    def solve(self, problem_file: str) -> Tuple[List[int], float, float]:
        """Solve TSP instance using genetic algorithm."""
        # Load problem
        problem = tsplib95.load(problem_file)
        cities = list(problem.get_nodes())
        city_coords = {i: problem.node_coords[i] for i in cities}
        
        # Initialize population
        start_time = time.time()
        population = self.initialize_population(cities)
        
        # Evolution loop
        for generation in range(self.generations):
            population = self.evolve_population(population, city_coords)
            
            # Track statistics
            route_distances = [self.calculate_route_length(route, city_coords) for route in population]
            self.best_fitness_history.append(min(route_distances))  # Use min since lower distance is better
            self.avg_fitness_history.append(sum(route_distances) / len(route_distances))
            
        # Get best solution
        best_route = population[np.argmax([self.fitness(route, city_coords) 
                                         for route in population])]
        best_distance = self.calculate_route_length(best_route, city_coords)
        computation_time = time.time() - start_time
        
        return best_route, best_distance, computation_time
    
    def plot_fitness_history(self):
        """Plot fitness history over generations."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='Best Distance')
        plt.plot(self.avg_fitness_history, label='Average Distance')
        plt.xlabel('Generation')
        plt.ylabel('Distance')
        plt.title('Distance Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()