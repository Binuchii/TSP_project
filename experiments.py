from TSPGeneticAlgorithm import TSPGeneticAlgorithm

def run_experiment(problem_file, pop_size, mut_rate, cross_rate, elite_size, generations):
    """
    Run experiment with given parameters.
    
    Args:
        problem_file (str): Path to the TSP problem file
        pop_size (int): Size of the population
        mut_rate (float): Mutation rate
        cross_rate (float): Crossover rate
        elite_size (int): Number of elite individuals to preserve
        generations (int): Number of generations to run
    """
    ga = TSPGeneticAlgorithm(
        population_size=pop_size,
        mutation_rate=mut_rate,
        crossover_rate=cross_rate,
        elite_size=elite_size,
        generations=generations
    )
    
    # Solve TSP instance
    best_route, best_distance, computation_time = ga.solve(problem_file)
    
    # Print results
    print(f"\nResults for {problem_file}")
    print(f"Population Size: {pop_size}")
    print(f"Mutation Rate: {mut_rate}")
    print(f"Crossover Rate: {cross_rate}")
    print(f"Elite Size: {elite_size}")
    print(f"Best Distance: {best_distance:.2f}")
    print(f"Computation Time: {computation_time:.2f} seconds")
    
    # Plot fitness history
    ga.plot_fitness_history()
    
    return best_distance, computation_time

# Parameter configurations
configurations = [
    {"pop_size": 150, "mut_rate": 0.02, "cross_rate": 0.9, "elite_size": 10},
]
# Test different configurations on berlin52
for config in configurations:
    run_experiment(
        problem_file="pr1002.tsp",
        pop_size=config["pop_size"],
        mut_rate=config["mut_rate"],
        cross_rate=config["cross_rate"],
        elite_size=config["elite_size"],
        generations=1000
    )