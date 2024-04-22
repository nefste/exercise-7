import tsplib95
import numpy as np


class Environment:
    def __init__(self, tsp_file, rho):
        self.rho = rho
        self.problem = tsplib95.load(tsp_file)
        self.num_cities = self.problem.dimension
        self.distance_matrix = self._create_distance_matrix()
        self.pheromone_map = None
        self.initialize_pheromone_map()
        print(f"Loading environment from {tsp_file} with evaporation rate {rho}")
        print(f"Environment contains {self.num_cities} cities.")

    def _create_distance_matrix(self):
        print("Creating distance matrix...")
        distance_matrix = np.zeros((self.num_cities, self.num_cities), dtype=int)
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    distance_matrix[i][j] = self.problem.get_weight(i + 1, j + 1)
        print("Distance matrix created.")
        return distance_matrix

    def initialize_pheromone_map(self, initial_pheromone=1.0):
        print("Initializing pheromone map...")
        initial_pheromone = 1 / (self.num_cities ** 2)
        self.pheromone_map = np.full((self.num_cities, self.num_cities), initial_pheromone)
        print("Pheromone map initialized.")

    def get_pheromone_map(self):
        return self.pheromone_map

    def get_distance_matrix(self):
        return self.distance_matrix
    
    
    def update_pheromone_map(self, ants):
        print("Updating pheromone map...")
        # Apply the evaporation to the pheromone trails
        self.pheromone_map *= (1 - self.rho)
    
        # Add new pheromone to the trails based on the quality of each ant's tour
        for ant in ants:
            if ant.travelled_distance > 0:
                pheromone_contribution = 1 / ant.travelled_distance  # Correct attribute used
            else:
                pheromone_contribution = 0
            tour = ant.tour
            for i in range(len(tour) - 1):
                self.pheromone_map[tour[i], tour[i+1]] += pheromone_contribution
                self.pheromone_map[tour[i+1], tour[i]] += pheromone_contribution

        print("Pheromone map updated.")


    
