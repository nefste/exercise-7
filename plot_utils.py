# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:49:46 2024

@author: StephanNef
"""

import plotly.graph_objects as go
import numpy as np

class PlotUtils:
    def __init__(self, environment):
        self.environment = environment
        self.fig = go.Figure()
        self.coords = [(self.environment.problem.node_coords[i+1][0], self.environment.problem.node_coords[i+1][1]) for i in range(self.environment.num_cities)]
        self.distance_fig = go.Figure()



    def create_figure(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[coord[0] for coord in self.coords], 
                                 y=[coord[1] for coord in self.coords], 
                                 mode='markers', marker=dict(color='black', size=6), name='Nodes/Cities'))
        return fig
    

    def update_pheromone_map(self, opacity_scale=1.0):
        self.fig = self.create_figure()  
        pheromone_map = self.environment.get_pheromone_map()
        max_pheromone = np.max(pheromone_map)
        for i in range(len(self.coords)):
            for j in range(len(self.coords)):
                if i != j:
                    opacity = (pheromone_map[i][j] / max_pheromone) * opacity_scale
                    color = f'rgba(0, 0, 255, {opacity})'
                    self.fig.add_trace(go.Scatter(
                        x=[self.coords[i][0], self.coords[j][0]],
                        y=[self.coords[i][1], self.coords[j][1]],
                        mode='lines',
                        line=dict(color=color, width=0.1)
                    ))

    def update_tour(self, tour, color='red', opacity=0.6):
        rgba_color = f'rgba(255, 0, 0, {opacity})'
        for i in range(len(tour) - 1):
            self.fig.add_trace(go.Scatter(
                x=[self.coords[tour[i]][0], self.coords[tour[i+1]][0]],
                y=[self.coords[tour[i]][1], self.coords[tour[i+1]][1]],
                mode='lines',
                line=dict(color=rgba_color, width=2)
            ))
            
         
        self.fig.add_trace(go.Scatter(
            x=[self.coords[tour[-1]][0], self.coords[tour[0]][0]],
            y=[self.coords[tour[-1]][1], self.coords[tour[0]][1]],
            mode='lines',
            line=dict(color=rgba_color, width=2)
        ))
        
        self.fig.update_layout(
            title="Best Tour",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            showlegend=True
        )


    def plot_distance_convergence(self, distances):
        self.distance_fig.data = []
        self.distance_fig.add_trace(go.Scatter(
            x=list(range(len(distances))),
            y=distances,
            mode='lines+markers',
            name='Distance Convergence'
        ))
        
        self.distance_fig.add_trace(go.Scatter(
            x=[0, len(distances)-1],
            y=[10628, 10628],  
            mode='lines',
            name='Target Lmin',
            line=dict(color='red', dash='dash')
        ))

        self.distance_fig.update_layout(
            title="Distance Convergence Over Iterations",
            xaxis_title="Iteration",
            yaxis_title="Distance",
            showlegend=True
        )


    def get_figures(self):
        return self.fig, self.distance_fig