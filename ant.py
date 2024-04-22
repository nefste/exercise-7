import numpy as np
import streamlit as st
# Class representing an artificial ant of the ant colony
"""
    alpha: a parameter controlling the influence of the amount of pheromone during ants' path selection process
    beta: a parameter controlling the influence of the distance to the next node during ants' path selection process
"""

class Ant:
    def __init__(self, alpha: float, beta: float, initial_location):
        print(10*"-")
        print(f"New Ant created with alpha={alpha}, beta={beta} at location {initial_location}")
        print(10*"-")
        self.alpha = alpha
        self.beta = beta
        self.current_location = initial_location
        self.travelled_distance = 0
        self.tour = [initial_location]  
        self.environment = None  

    def run(self):
        """
        Visit all cities exactly once in the environment, then return to the start city.
        """
        print("Ant starts running through the cities...")
        unvisited = set(range(self.environment.num_cities)) - set(self.tour)
        while unvisited:
            next_city = self.select_path(unvisited)
            self.travel_to(next_city)
            unvisited.remove(next_city)


        start_city = self.tour[0]
        self.travel_back_to_start(start_city)
        print(f"Ant completed a tour: {self.tour} and returned to start city {start_city}")

    def travel_back_to_start(self, start_city):
        """
        Calculate and add the distance from the last city in the tour back to the start city.
        """
        last_city = self.tour[-1]
        distance_back_to_start = self.environment.get_distance_matrix()[last_city][start_city]
        self.travelled_distance += distance_back_to_start
        print(f"Ant returned to start city {start_city}, total distance now: {self.travelled_distance}")

    

    def select_path(self, unvisited):
        """
        Select the next city to visit based on a probabilistic decision rule taking into account
        pheromone levels and distance.
        """
        print("Calculating path probabilities...")
        probabilities = []
        pheromones = self.environment.get_pheromone_map()
        distances = self.environment._create_distance_matrix()

        for next_city in unvisited:
            pheromone_level = pheromones[self.current_location][next_city] ** self.alpha
            distance_influence = (1 / distances[self.current_location][next_city]) ** self.beta
            probabilities.append(pheromone_level * distance_influence)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        return np.random.choice(list(unvisited), p=probabilities)

    def travel_to(self, next_city):
        """
        Move the ant to the next city, updating the traveled distance.
        """
        print(f"Ant moves to city {next_city}")
        distance = self.environment.get_distance_matrix()[self.current_location][next_city]
        self.travelled_distance += distance
        self.tour.append(next_city)
        self.current_location = next_city
        print(f"Ant traveled to {next_city}, total distance: {self.travelled_distance}")


    def join(self, environment):
        """
        Position the ant within an environment.
        """
        self.environment = environment
        self.current_location = np.random.randint(environment.num_cities)  # Random initial location
        self.tour = [self.current_location]  # Reset the tour to start at the new location
        print(f"Ant joined environment, ready at city {self.current_location}.")