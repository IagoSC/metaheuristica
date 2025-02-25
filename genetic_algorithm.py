import numpy as np
import time
from read_entries import read_cvrp_file
import sys
import math
from database import insert_entry, init_database

VISUAL = False

def phenotype_execution(chromosome, demands, capacity, distance_matrix):
    chromosome = remove_numpy(chromosome)

    n = len(chromosome)
    acc_cost = [float('inf')] * (n + 1)
    acc_cost[0] = 0
    predecessors = [-1] * (n + 1)

    for i in range(0, n):
        total_demand = 0
        cost = 0
        for j in range(i, n):
            customer = chromosome[j]
            total_demand += demands[customer]
            if j == i:
                cost += distance_matrix[0][customer]  # Depot to first customer
            else:
                cost += distance_matrix[chromosome[j-1]][customer]
            if total_demand > capacity:
                break
            route_cost = cost + distance_matrix[customer][0]  # Last customer to depot
            if acc_cost[i] + route_cost < acc_cost[j+1]:
                predecessors[j+1] = i
                acc_cost[j+1] = acc_cost[i] + route_cost
    
    # Backtrack to find the splits
    splits = []
    current = n
    while current > 0:
        prev = predecessors[current]
        splits.append(chromosome[prev:current])
        current = prev
    splits.reverse()
    return acc_cost[n], splits

def tournament_selection(population, fitness, tournament_size):
    selected = []
    for _ in range(len(population)):
        candidates = np.random.choice(len(population), tournament_size, replace=False)
        candidates_fitness = [fitness[c] for c in candidates]
        best_idx = candidates[np.argmin(candidates_fitness)]
        selected.append(population[best_idx])
    return selected

def ordered_crossover(parent1, parent2, type, crossover_ux_rate):
    size = len(parent1)
    child = [-1] * size
    if type == '2x':
        start, end = sorted(np.random.choice(size, 2, replace=False))
        child[start:end+1] = parent1[start:end+1]
    elif type == 'ux':
        child = [np.random.choice([-1, parent1[idx]], 1, p=[1-crossover_ux_rate, crossover_ux_rate])[0] for idx in range(size)]

    ptr = 0
    for gene in parent2:
        if gene not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = gene
    return child

def mutate(chromosome, mutation_rate):
    for i in range(math.floor(len(chromosome) * mutation_rate)):
        i, j = np.random.choice(len(chromosome), 2, replace=False)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

def remove_numpy(array):
    return  [int(el) for el in array] 

def genetic_algorithm(cvrp_instance, parameters):
    pop_size = parameters['population_size']
    tournament_size = parameters['tournament_size']
    elitism_factor = parameters['elitism_factor']
    crossover_type = parameters['crossover_type']
    mutation_rate = parameters['mutation_rate']
    crossover_ux_rate = parameters['crossover_ux_rate']
        
    demands = cvrp_instance['demands']
    capacity = cvrp_instance['vehicle_capacity']
    distance_matrix = cvrp_instance['distance_matrix']

    population = init_population(pop_size, len(demands) - 1) # exclude depot

    best_fitness = float('inf')
    best_solution = None
    best_fitness_time = 0
    best_fitness_timestamp = float('inf')

    time_limit = 300  # 5 minutes
    start_time = time.time()

    # Generation loop
    avg_gen_time = 0
    gen_num = 1
    last_gen_time = start_time

    while True:
        total_elapsed_time = time.time() - start_time
        if total_elapsed_time > time_limit:
            break
    
        elapsed_time_since_improved_fitness = time.time() - best_fitness_timestamp
        # elif elapsed_time_since_improved_fitness > 45:
        #     break

        fitness = []
        routes = []

        for ind in population:
            dist, ind_routes = phenotype_execution(ind, demands, capacity, distance_matrix)
            fitness.append(dist)
            routes.append(ind_routes)
            if dist < best_fitness:
                print("New best fitness:", dist)
                print("Elapsed time:", time.time() - start_time)
                best_fitness = dist
                best_solution = [remove_numpy(sbl) for sbl in ind_routes]
                best_fitness_timestamp = time.time()
                best_fitness_time = time.time() - start_time


        # Selection
        selected = tournament_selection(population, fitness, tournament_size)

        # Crossover
        offspring = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected[i], selected[i+1]
            child1 = ordered_crossover(parent1, parent2, crossover_type, crossover_ux_rate)
            child2 = ordered_crossover(parent2, parent1, crossover_type, crossover_ux_rate)
            offspring.extend([child1, child2])

        # Mutation
        offspring = [mutate(ind, mutation_rate) for ind in offspring]

        # Elitism
        elite_size = int(elitism_factor * pop_size)
        elite_indices = np.argsort(fitness)[:elite_size]
        new_population = [population[i] for i in elite_indices]
        new_population += offspring[:pop_size - elite_size]
        population = new_population

        now = time.time()
        avg_gen_time = (now - last_gen_time) / gen_num
        last_gen_time = now

    return best_solution, best_fitness, best_fitness_time, avg_gen_time

def init_population(pop_size, num_customers):
    return [np.random.permutation(np.arange(1, num_customers + 1)) for _ in range(pop_size)]

if __name__ == "__main__":
    print(sys.argv)

    table_name = sys.argv[1]

    init_database(table_name)

    instance = sys.argv[2]
    args = sys.argv[3:]

    if args[0] == '--visual':
        VISUAL = True

    parameters = {
        "population_size": int(args[0]),
        "tournament_size": int(args[1]),
        "elitism_factor": float(args[2]),
        "mutation_rate": float(args[3]),
        "crossover_type": args[4],
        "crossover_ux_rate": float(args[5]),
    }
    

    cvrp_instance = read_cvrp_file(instance)
    best_solution, best_fitness, best_fitness_time, avg_gen_time = genetic_algorithm(cvrp_instance, parameters)

    insert_entry(instance, args, best_solution, best_fitness, best_fitness_time, parameters, avg_gen_time)

    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)