import numpy as np
from read_entries import parse_vrp_file

def split_algorithm(permutation, demands, capacity, distance_matrix):
    n = len(permutation)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    predecessors = [-1] * (n + 1)
    
    for i in range(1, n + 1):
        total_demand = 0
        for j in range(i, 0, -1):
            customer = permutation[j-1]
            total_demand += demands[customer]
            if total_demand > capacity:
                break
            cost = distance_matrix[0][permutation[j-1]]  # Depot to first customer
            for k in range(j, i):
                cost += distance_matrix[permutation[k-1]][permutation[k]]
            cost += distance_matrix[permutation[i-1]][0]  # Last customer to depot
            if dp[j-1] + cost < dp[i]:
                dp[i] = dp[j-1] + cost
                predecessors[i] = j-1
    
    # Backtrack to find the splits
    splits = []
    current = n
    while current > 0:
        prev = predecessors[current]
        splits.append(permutation[prev:current])
        current = prev
    splits.reverse()
    return dp[n], splits

def initialize_population(num_individuals, num_customers):
    return [np.random.permutation(num_customers) + 1 for _ in range(num_individuals)]

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

def swap_mutation(individual):
    i, j = np.random.choice(len(individual), 2, replace=False)
    individual[i], individual[j] = individual[j], individual[i]
    return individual

def genetic_algorithm(cvrp_instance, generations=10000, pop_size=50):
    demands = cvrp_instance['demands']
    capacity = cvrp_instance['vehicle_capacity']
    distance_matrix = cvrp_instance['distance_matrix']
    population = initialize_population(pop_size, len(demands)-1)  # Exclude depot
    best_fitness = float('inf')
    best_solution = None
    
    for gen in range(generations):
        fitness = []
        routes = []
        for ind in population:
            dist, splits = split_algorithm(ind, demands, capacity, distance_matrix)
            fitness.append(dist)
            routes.append(splits)
            if dist < best_fitness:
                best_fitness = dist
                best_solution = splits
        
        # Selection
        selected = tournament_selection(population, fitness, tournament_size=3)
        
        # Crossover
        offspring = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected[i], selected[i+1]
            child1 = ordered_crossover(parent1, parent2)
            child2 = ordered_crossover(parent2, parent1)
            offspring.extend([child1, child2])
        
        # Mutation
        offspring = [swap_mutation(ind) for ind in offspring]
        
        # Elitism: Keep top 10% of previous population
        elite_size = int(0.1 * pop_size)
        elite_indices = np.argsort(fitness)[:elite_size]
        new_population = [population[i] for i in elite_indices]
        new_population += offspring[:pop_size - elite_size]
        population = new_population
    
    return best_solution, best_fitness

# Example usage
if __name__ == "__main__":
    cvrp_instance = parse_vrp_file('A/A-n32-k5.vrp')
    best_solution, best_fitness = genetic_algorithm(cvrp_instance)
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)