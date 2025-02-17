import os
from TSPGeneticAlgorithm import TSPGeneticAlgorithm
import pandas as pd
import itertools
import tsplib95

def grid_search(problem_file, param_grid):
    if not os.path.exists(problem_file):
        raise FileNotFoundError(f"Could not find {problem_file}")
    
    try:
        problem = tsplib95.load(problem_file, special=None)
    except Exception as e:
        raise Exception(f"Error loading {problem_file}: {str(e)}")
    
    results = []
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                        for v in itertools.product(*param_grid.values())]
    total_combinations = len(param_combinations)
    
    for idx, params in enumerate(param_combinations, 1):
        print(f"\nTesting combination {idx}/{total_combinations}")
        print(f"Parameters: {params}")
        
        try:
            ga = TSPGeneticAlgorithm(
                population_size=params['population_size'],
                mutation_rate=params['mutation_rate'],
                crossover_rate=params['crossover_rate'],
                elite_size=params['elite_size'],
                generations=params['generations']
            )
            
            best_route, best_distance, computation_time = ga.solve(problem_file)
            
            results.append({
                'population_size': params['population_size'],
                'mutation_rate': params['mutation_rate'],
                'crossover_rate': params['crossover_rate'],
                'elite_size': params['elite_size'],
                'generations': params['generations'],
                'best_distance': best_distance,
                'computation_time': computation_time
            })
            
            print(f"Distance: {best_distance:.2f}, Time: {computation_time:.2f}s")
            
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue
    
    return pd.DataFrame(results).sort_values('best_distance')

def run_grid_search_for_problem(problem_file, param_grid):
    print(f"\nGrid Search Results for {problem_file}")
    try:
        results = grid_search(problem_file, param_grid)
        output_file = f'grid_search_{problem_file.split(".")[0]}.csv'
        results.to_csv(output_file, index=False)
        print(f"\nTop 5 Parameter Combinations for {problem_file}:")
        print(results.head().to_string(index=False))
        return results
    except Exception as e:
        print(f"Error processing {problem_file}: {str(e)}")
        return None

param_grid = {
    'population_size': [150, 200, 300],
    'mutation_rate': [0.01, 0.02, 0.03],
    'crossover_rate': [0.9, 0.95, 0.98],
    'elite_size': [10],
    'generations': [1000]
}

problems = ['kroA100.tsp']

for problem in problems:
    run_grid_search_for_problem(problem, param_grid)