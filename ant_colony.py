import numpy as np
from environment import Environment
from ant import Ant 
from plot_utils import PlotUtils

import streamlit as st

class AntColony:
    def __init__(self, ant_population: int, iterations: int, alpha: float, beta: float, rho: float, tsp_file):
        print(20*'-')
        print(f"Initializing Ant Colony with {ant_population} ants, alpha={alpha}, beta={beta}, rho={rho}")
        print(20*'-')
        self.ant_population = ant_population
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho 

        # Initialize the environment of the ant colony with the TSP file and pheromone evaporation rate
        self.environment = Environment(tsp_file, self.rho)


        self.ants = [Ant(self.alpha, self.beta, np.random.randint(self.environment.num_cities)) for _ in range(ant_population)]
        print("Ants initialized and positioned in the environment.")

        for ant in self.ants:
            ant.join(self.environment)
        

    def solve(self):
        shortest_distance = np.inf
        best_tour = None
        print("Starting the solution process...")
        for _ in range(self.iterations): 
            print(f"--- Iteration {_} ---")

            for ant in self.ants:
                ant.current_location = np.random.randint(self.environment.num_cities)
                ant.tour = [ant.current_location]
                ant.travelled_distance = 0

            # Each ant makes a complete tour
            for ant in self.ants:
                ant.run()
        
            # Update pheromones based on the tours made by ants
            self.environment.update_pheromone_map(self.ants)
            print("Pheromone map updated.")
        
            # Check for the best solution found in this iteration
            for ant in self.ants:
                if ant.travelled_distance < shortest_distance:
                    best_tour = ant.tour
                    shortest_distance = ant.travelled_distance
            # st.success(f"New shortest distance: {shortest_distance} on tour: {best_tour}")
            print(f"*** Current best distance: {shortest_distance} ***")
        print("Optimization complete.")
        return best_tour, shortest_distance

# def main():
#     tsp_file = 'att48-specs/att48.tsp'
#     ant_colony = AntColony(ant_population=48, iterations=100, alpha=1.0, beta=1.0, rho=0.5, tsp_file=tsp_file)
#     plotter = PlotUtils(ant_colony.environment)

#     for _ in range(ant_colony.iterations):
#         for ant in ant_colony.ants:
#             ant.run()
#             plotter.update_tour(ant.tour, color='blue', opacity=0.2)
        
#         ant_colony.environment.update_pheromone_map(ant_colony.ants)

#     solution, distance = ant_colony.solve()
#     plotter.highlight_best_tour(solution)
#     plotter.show_plot()

#     print("Best tour: ", solution)
#     print("Shortest distance: ", distance)
    

# if __name__ == '__main__':
#     main()
