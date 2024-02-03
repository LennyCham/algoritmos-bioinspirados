import numpy as np

def ant_colony(distances, n_ants, n_best, n_best_update, decay, alpha=1, beta=2, num_iterations=100):
    pheromones = np.ones(distances.shape) / len(distances)
    all_inds = range(len(distances))
    best_path = None
    best_path_length = np.inf

    for _ in range(num_iterations):
        ants = generate_ant_tours(pheromones, distances, n_ants, alpha, beta)
        update_pheromones(pheromones, ants, n_best, decay)
        current_best_path = min(ants, key=lambda x: path_length(x, distances))

        if path_length(current_best_path, distances) < best_path_length:
            best_path = current_best_path
            best_path_length = path_length(current_best_path, distances)

    return best_path

def generate_ant_tours(pheromones, distances, n_ants, alpha, beta):
    ants = []
    for _ in range(n_ants):
        ant_path = build_ant_path(pheromones, distances, alpha, beta)
        ants.append(ant_path)
    return ants

def build_ant_path(pheromones, distances, alpha, beta):
    num_cities = len(pheromones)
    start_city = np.random.randint(num_cities)
    ant_path = [start_city]
    visited_cities = set([start_city])

    while len(visited_cities) < num_cities:
        next_city = select_next_city(pheromones, distances, ant_path[-1], visited_cities, alpha, beta)
        ant_path.append(next_city)
        visited_cities.add(next_city)

    return ant_path

def select_next_city(pheromones, distances, current_city, visited_cities, alpha, beta):
    unvisited_cities = [city for city in range(len(pheromones)) if city not in visited_cities]
    probabilities = calculate_probabilities(pheromones, distances, current_city, unvisited_cities, alpha, beta)
    next_city = np.random.choice(unvisited_cities, p=probabilities)
    return next_city

def calculate_probabilities(pheromones, distances, current_city, unvisited_cities, alpha, beta):
    tau = pheromones[current_city, unvisited_cities]
    eta = 1 / distances[current_city, unvisited_cities]
    probabilities = (tau**alpha) * (eta**beta) / np.sum((tau**alpha) * (eta**beta))
    return probabilities

def update_pheromones(pheromones, ants, n_best, decay):
    pheromones *= (1.0 - decay)
    for ant in ants[:n_best]:
        pheromones[ant[:-1], ant[1:]] += 1.0 / path_length(ant, distances)

def path_length(path, distances):
    return sum(distances[path[i], path[i+1]] for i in range(len(path)-1))

# Ejemplo de uso
distances = np.array([
    [np.inf, 10, 15],
    [10, np.inf, 20],
    [15, 20, np.inf]
])

best_path = ant_colony(distances, n_ants=5, n_best=2, n_best_update=2, decay=0.95, alpha=1, beta=2, num_iterations=100)
print(f"Mejor ruta encontrada: {best_path}")
