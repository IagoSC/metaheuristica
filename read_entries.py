import math

def read_cvrp_file(file_path):
    data = {
        'customer_coords': [],
        'demands': [],
        'distance_matrix': [],
        'vehicle_capacity': 0,
        'depot': 0
    }

    with open(file_path, 'r') as f:
        content = f.readlines()

    node_coord_section = False
    demand_section = False
    depot_section = False
    dimension = 0

    for line in content:
        line = line.strip()
        if not line:
            continue

        if line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            data['vehicle_capacity'] = int(line.split(':')[1].strip())
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            edge_weight_type = line.split(':')[1].strip()
            if edge_weight_type != 'EUC_2D':
                raise ValueError("Distance type not supported")

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

        if node_coord_section:
            parts = line.split()
            x = int(float(parts[1]))
            y = int(float(parts[2]))
            data['customer_coords'].append((x, y))

        elif demand_section:
            parts = line.split()
            demand = int(parts[1])
            data['demands'].append(demand)

        elif depot_section:
            if line.strip() == '-1':
                break
            depot_id = int(line.strip()) -1
            data['depot'] = depot_id

    if len(data['customer_coords']) != dimension:
        raise ValueError("Mismatch in dimension and number of coordinates")
    
    if len(data['demands']) != dimension:
        raise ValueError("Mismatch in dimension and number of demands")

    data['distance_matrix'] = compute_distance_matrix(data['customer_coords'])
    
    return data

def compute_distance_matrix(coordinates):
    n = len(coordinates)
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coordinates[i][0] - coordinates[j][0]
                dy = coordinates[i][1] - coordinates[j][1]
                matrix[i][j] = int(round(math.sqrt(dx**2 + dy**2)))
    
    return matrix