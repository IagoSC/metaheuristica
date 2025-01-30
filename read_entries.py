import math

def parse_vrp_file(file_path):
    """
    Parses CVRP instances in the .vrp format used on CVRPLIB
    Returns a dictionary with problem data:
    - depot: Depot index (0-based)
    - customer_coords: List of (x, y) tuples (including depot at index 0)
    - demands: List of demands (depot demand = 0)
    - distance_matrix: Precomputed Euclidean distances
    - vehicle_capacity: Vehicle capacity
    """
    data = {
        'customer_coords': [],
        'demands': [],
        'distance_matrix': [],
        'vehicle_capacity': 0,
        'depot': 0
    }

    with open(file_path, 'r') as f:
        content = f.readlines()

    # Parse metadata and find section starts
    node_coord_section = False
    demand_section = False
    depot_section = False
    dimension = 0
    capacity = 0

    for line in content:
        line = line.strip()
        if not line:
            continue

        # Parse metadata
        if line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            data['vehicle_capacity'] = int(line.split(':')[1].strip())
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            edge_weight_type = line.split(':')[1].strip()
            if edge_weight_type != 'EUC_2D':
                raise ValueError("Only EUC_2D distance type is supported")

        # Detect section starts
        if line.startswith('NODE_COORD_SECTION'):
            node_coord_section = True
            demand_section = False
            depot_section = False
            continue
        elif line.startswith('DEMAND_SECTION'):
            demand_section = True
            node_coord_section = False
            depot_section = False
            continue
        elif line.startswith('DEPOT_SECTION'):
            depot_section = True
            node_coord_section = False
            demand_section = False
            continue
        elif line.startswith('EOF'):
            break

        # Parse sections
        if node_coord_section:
            parts = line.split()
            node_id = int(parts[0]) - 1  # Convert to 0-based index
            x = int(float(parts[1]))
            y = int(float(parts[2]))
            data['customer_coords'].append((x, y))

        elif demand_section:
            parts = line.split()
            node_id = int(parts[0]) - 1  # Convert to 0-based index
            demand = int(parts[1])
            data['demands'].append(demand)

        elif depot_section:
            if line.strip() == '-1':
                break
            depot_id = int(line.strip()) - 1  # Convert to 0-based index
            data['depot'] = depot_id

    # Validate data
    if len(data['customer_coords']) != dimension:
        raise ValueError("Mismatch in dimension and number of coordinates")
    
    if len(data['demands']) != dimension:
        raise ValueError("Mismatch in dimension and number of demands")

    # Create distance matrix
    data['distance_matrix'] = compute_distance_matrix(data['customer_coords'])
    
    return data

def compute_distance_matrix(coordinates):
    """Compute Euclidean distance matrix with integer rounding"""
    n = len(coordinates)
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coordinates[i][0] - coordinates[j][0]
                dy = coordinates[i][1] - coordinates[j][1]
                matrix[i][j] = int(round(math.sqrt(dx**2 + dy**2)))
    
    return matrix

# Example usage
if __name__ == "__main__":
    # Download a test instance from CVRPLIB first, e.g., A-n32-k5.vrp
    problem_data = parse_vrp_file('A-n32-k5.vrp')
    
    print(f"Depot index: {problem_data['depot']}")
    print(f"Vehicle capacity: {problem_data['vehicle_capacity']}")
    print(f"Number of customers: {len(problem_data['customer_coords']) - 1}")
    print(f"Sample demand: {problem_data['demands'][1]}")  # First customer demand
    print(f"Sample distance: {problem_data['distance_matrix'][0][1]}")  # Depot to first customer