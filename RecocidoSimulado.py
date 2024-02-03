import numpy as np
import random
import math

def simulated_annealing(distances, temperature, cooling_rate, num_iterations):
    num_cities = len(distances)
    current_solution = list(range(num_cities))
    best_solution = current_solution.copy()
    best_distance = sum(distances[i][j] for i, j in zip(best_solution, best_solution[1:] + [best_solution[0]]))

    for _ in range(num_iterations):
        new_solution = current_solution.copy()
        i, j = sorted(random.sample(range(num_cities), 2))
        new_solution[i:j+1] = reversed(new_solution[i:j+1])

        current_distance = sum(distances[i][j] for i, j in zip(current_solution, current_solution[1:] + [current_solution[0]]))
        new_distance = sum(distances[i][j] for i, j in zip(new_solution, new_solution[1:] + [new_solution[0]]))

        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temperature):
            current_solution = new_solution.copy()

        if new_distance < best_distance:
            best_solution = new_solution.copy()
            best_distance = new_distance

        temperature *= 1 - cooling_rate

    return best_solution, best_distance

# Ejemplo de uso
np.random.seed(42)
num_cities = 10
distances = np.random.randint(1, 100, size=(num_cities, num_cities)).astype(float)
np.fill_diagonal(distances, np.inf)

initial_temperature = 1000.0
cooling_rate = 0.003
num_iterations = 1000

best_solution, best_distance = simulated_annealing(distances, initial_temperature, cooling_rate, num_iterations)

print(f"Mejor ruta encontrada: {best_solution}")
print(f"Distancia total: {best_distance}")