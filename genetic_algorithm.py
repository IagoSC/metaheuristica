import numpy as np
import time
from read_entries import read_cvrp_file
import sys
import math

VISUAL = False

def phenotype_execution(chromosome, demands, capacity, distance_matrix):
    n = len(chromosome)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    predecessors = [-1] * (n + 1)
    
    for i in range(1, n + 1):
        total_demand = 0
        for j in range(i, 0, -1):
            customer = chromosome[j-1]
            total_demand += demands[customer]
            if total_demand > capacity:
                break
            cost = distance_matrix[0][chromosome[j-1]]  # Depot to first customer
            for k in range(j, i):
                cost += distance_matrix[chromosome[k-1]][chromosome[k]]
            cost += distance_matrix[chromosome[i-1]][0]  # Last customer to depot
            if dp[j-1] + cost < dp[i]:
                dp[i] = dp[j-1] + cost
                predecessors[i] = j-1
    
    splits = []
    current = n
    while current > 0:
        prev = predecessors[current]
        splits.append(chromosome[prev:current])
        current = prev
    splits.reverse()
    return dp[n], splits

def tournament_selection(population, fitness, tournament_size):
    selected = []
    for _ in range(len(population)):
        candidates = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = candidates[np.argmin([fitness[c] for c in candidates])]
        selected.append(population[best_idx])
    return selected

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(np.random.choice(size, 2, replace=False))
    child = [-1] * size
    child[start:end+1] = parent1[start:end+1]
    ptr = 0
    for gene in parent2:
        if gene not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = gene
    return child

def mutate(chromosome, mutation_rate=0.1):
    for i in range(math.floor(len(chromosome) * mutation_rate)):
        i, j = np.random.choice(len(chromosome), 2, replace=False)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

def genetic_algorithm(cvrp_instance, pop_size=100, tournament_size=3, elitism_type='fixed', elitism_factor=0.1):
    demands = cvrp_instance['demands']
    capacity = cvrp_instance['vehicle_capacity']
    distance_matrix = cvrp_instance['distance_matrix']

    population = init_population(pop_size, len(demands) - 1)

    best_fitness = float('inf')
    best_solution = None

    time_limit = 300  # 5 minutes
    start_time = time.time()

    # Generation loop
    for _ in range(int('inf')):
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            break
        
        fitness = []
        routes = []

        for ind in population:
            dist, splits = phenotype_execution(ind, demands, capacity, distance_matrix)
            fitness.append(dist)
            routes.append(splits)
            if dist < best_fitness:
                best_fitness = dist
                best_solution = splits

        # Selection
        selected = tournament_selection(population, fitness, tournament_size)

        # Crossover
        # We take two consecutive positioned individuals and apply ordered crossover
        offspring = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected[i], selected[i+1]
            child1 = ordered_crossover(parent1, parent2)
            child2 = ordered_crossover(parent2, parent1)
            offspring.extend([child1, child2])

        # Mutation
        offspring = [mutate(ind) for ind in offspring]

        # Elitism
        if elitism_type == 'fixed':
            elite_size = int(elitism_factor * pop_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            new_population = [population[i] for i in elite_indices]
            new_population += offspring[:pop_size - elite_size]
            population = new_population
        elif elitism_type == 'slope':
            print("Not implemented")
            # TODO

    return best_solution, best_fitness

def init_population(pop_size, num_customers):
    population = []
    for _ in range(pop_size):
        population.append(np.random.permutation(num_customers))
    return population

if __name__ == "__main__":
    args = sys.argv[1:]
    population_size = 100
    tournament_size = 3
    elitism_type = 'fixed'
    elitism_factor = 0.1

    if args[0] == '--visual':
        VISUAL = True

    if len(args) == 4:
        population_size = int(args[1])
        tournament_size = int(args[2])
        elitism_type = args[3]
        elitism_factor = float(args[4])


    cvrp_instance = read_cvrp_file('A/A-n80-k10.vrp')
    best_solution, best_fitness = genetic_algorithm(cvrp_instance, )
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)