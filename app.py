# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:07:18 2024

@author: StephanNef
"""


import streamlit as st
from plot_utils import PlotUtils
from environment import Environment
from ant_colony import AntColony
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import streamlit.components.v1 as components


st.set_page_config(
     page_title="Smart Colony Optimization",
     page_icon="🐜",
     layout="wide",
)


st.sidebar.header('⚙️ Colony Parameters')
ant_population = st.sidebar.number_input('Ant Population', min_value=1, value=48, step=1)
iterations = st.sidebar.number_input('Iterations', min_value=1, value=10, step=1)
alpha = st.sidebar.slider('Alpha', min_value=0.0, max_value=10.0, value=4.0, step=0.1)
beta = st.sidebar.slider('Beta', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
rho = st.sidebar.slider('Rho', min_value=0.0, max_value=1.0, value=0.3, step=0.05)
scale = (st.sidebar.slider('Lines Plot Scale',min_value=0, max_value=100, value=10, step=10))/100


# Load TSP data and initialize the ant colony
tsp_file = 'att48-specs/att48.tsp'
environment = Environment(tsp_file, rho)
ant_colony = AntColony(ant_population, iterations, alpha, beta, rho, tsp_file)
plot_utils = PlotUtils(environment)

distances = []


st.title("🐜 Stigmergic Interaction for Smart Colony Optimization")


st.write("""
         
         Colony Optimization is a probabilistic technique used in optimization problems. Inspired by the behavior of ants searching for food. Initially proposed by Marco Dorigo in 1992, it has been applied to various problems, notably the traveling salesman problem (TSP) which aims to find the shortest possible route visiting a set of cities and returning to the origin city.
         
         """)
with st.expander("More Details on the following Colony Optimisation implementation? [click here]"): 

    st.subheader("Pheromone Trail")
    st.write("""
    Ants communicate via pheromones, leaving a trail on their path which is used by other ants to follow. 
    In ACO (Ant Colony Optimisation), this behavior is simulated to create a search algorithm where paths with stronger pheromone trails 
    are more likely to be followed by subsequent ants, influencing the collective decision-making process.
    """)
    
    st.subheader("Update Rules")
    st.write("The amount of pheromone deposited, often related to the quality of the solution (e.g., shorter paths in TSP), is defined by the pheromone update rule. Pheromone trails evaporate over time, decreasing their attractive strength unless reinforced by further successful paths.")
    
    st.write("### Pheromone Evaporation")
    st.latex(r"\tau_{xy} \leftarrow (1 - \rho) \cdot \tau_{xy}")
    st.write("where $\\tau_{xy}$ is the pheromone concentration on the path from node $x$ to $y$, and $\\rho$ (0 < $\\rho$ < 1) is the pheromone evaporation rate.")
    st.code("""def update_pheromone_map(self, ants):
            print("Updating pheromone map...")
            # Apply the evaporation to the pheromone trails
            self.pheromone_map *= (1 - self.rho)
            ....""", language='python')
    
    st.write("### Pheromone Update")
    st.latex(r"\tau_{xy} \leftarrow \tau_{xy} + \sum_{k=1}^{m} \Delta \tau_{xy}^k")
    st.latex(r"\Delta \tau_{xy}^k = \frac{Q}{L_k}")
    st.write("Here, $Q$ is a constant, and $L_k$ is the length of the tour completed by ant $k$.")
    st.code("""def update_pheromone_map(self, ants):
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
                    self.pheromone_map[tour[i+1], tour[i]] += pheromone_contribution""", language='python')
    
    st.subheader("Path Selection")
    st.write("""
    Ants probabilistically choose their paths based on the pheromone strength and the heuristic desirability of the path (e.g., inverse of the distance in TSP):
    """)
    st.latex(r"P_{xy}^k = \frac{(\tau_{xy}^\alpha) \cdot (\eta_{xy}^\beta)}{\sum (\tau_{xy}^\alpha) \cdot (\eta_{xy}^\beta)}")
    st.write("""
    where:
    - $P_{xy}^k$ is the probability that ant $k$ moves from city $x$ to $y$,
    - $\\tau_{xy}$ is the amount of pheromone on the path from $x$ to $y$,
    - $\\eta_{xy}$ is the heuristic value of the path (e.g., $\\frac{1}{\\text{distance}_{xy}}$),
    - $\\alpha$ and $\\beta$ are parameters controlling the influence of the pheromone trail and the heuristic value, respectively.
    """)
    st.code("""    def select_path(self, unvisited):
            print("Calculating path probabilities...")
            probabilities = []
            pheromones = self.environment.get_pheromone_map()
            distances = self.environment._create_distance_matrix()

            # Calculate the attractiveness of each path
            for next_city in unvisited:
                pheromone_level = pheromones[self.current_location][next_city] ** self.alpha
                distance_influence = (1 / distances[self.current_location][next_city]) ** self.beta
                probabilities.append(pheromone_level * distance_influence)

            # Normalize to create a probability distribution
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            # Choose the next city based on the defined probabilities
            return np.random.choice(list(unvisited), p=probabilities)""", language='python')
    
    st.image("aco_gif.gif")

st.write("---")

st.header("⬅️ Run Simulation with specific Parameters")
st.info("Ant Colony Optimisation - please set your parameters in the left sidebar and then start the simulation by pressing the button - Stephan Nef")


#with st.expander("See optimal Solution for this case:"):
    #st.image(r"att48-specs\att48_opt.png")
    #tourfile = ("att48-specs\att48.opt.tour")
    #with open(tourfile, 'r') as file:
    #    file_contents = file.read()
    #    st.markdown(file_contents)
        
        
start = st.button("▶️ Start Simulation:")
st.write(f"Ant Population: {ant_population} / Iterations: {iterations} / Alpha: {alpha} / Beta: {beta} / Rho: {rho}")

if start:
    with st.spinner("Simulation running..."):
        distances = []
        best_tour = None
        shortest_distance = float('inf')
    
        col1, col2  = st.columns(2)
        with col1:
            # st.subheader("Map - Best Tour:")
            tour_plot_placeholder = st.empty()
        
        with col2:
            # st.subheader("Distance Convergence:")
            distance_plot_placeholder = st.empty()
    
        for _ in range(iterations):
            
                
                for ant in ant_colony.ants:
                    ant.run()
                    if ant.travelled_distance < shortest_distance:
                        shortest_distance = ant.travelled_distance
                        best_tour = ant.tour
                        distances.append(shortest_distance)
                        
                        ant_colony.environment.update_pheromone_map(ant_colony.ants)
                        plot_utils.update_pheromone_map(opacity_scale=scale)
                        plot_utils.update_tour(best_tour, color='red')
                        plot_utils.plot_distance_convergence(distances)
                        
                        tour_plot_placeholder.plotly_chart(plot_utils.fig, use_container_width=True)
                        distance_plot_placeholder.plotly_chart(plot_utils.distance_fig, use_container_width=True)
        st.success(f"Shortest Distance: {shortest_distance}")
        st.success(f"Best Tour: {best_tour}")

st.write("---")


def run_simulation(alpha, beta, rho):
    tsp_file = 'att48-specs/att48.tsp'
    ant_colony = AntColony(ant_population=48, iterations=20, alpha=alpha, beta=beta, rho=rho, tsp_file=tsp_file)
    best_tour, shortest_distance = ant_colony.solve()
    return shortest_distance

# Define ranges for alpha, beta, and rho
alpha_range = np.arange(1, 6, 1)  
beta_range = np.arange(1, 6, 1)  
rho_range = np.arange(0.1, 1, 0.1)  


st.header("🔝 Empirical Simulation to find optimal Parameters")
st.write("Executed a simulation for  α  and  β within the range from 1 to 10 in increments of 1, and for ρ from 0.1 to 1 in increments of 0.1. The total number of simulation possibilities considering all combinations of α, β and ρ for each iteration and ant can be calculated as:")
st.latex(r"""
P = (\text{Number of values for } \alpha) \times 
    (\text{Number of values for } \beta) \times 
    (\text{Number of values for } \rho)
""")

st.info("Conducted an empirical simulation using a population of 48 ants across 20 iterations. Below, you can explore the analysis and view the simulation results across various parameters. My best solution so far alpha=4, beta=3, rho=0.3 with distance= 10'974.")
 
st.image("ACO_Parameter_Simulation.png")


file_path = 'aco_simulation_results_backup.xlsx'
data = pd.read_excel(file_path)
 
distance_min = data['distance'].min()
distance_max = data['distance'].max()


st.header("Impact of Parameters α (Alpha) and β (Beta)")

st.subheader("Alpha (α) - Pheromone Importance")
st.write("""
- **Low α**: Leads to more exploration with less reliance on pheromone trails, potentially increasing the route length due to less focused searching.
- **High α**: Results in strong adherence to pheromone trails, which can enhance the exploitation of known good paths but may result in premature convergence to suboptimal paths.
""")

plot_placeholder_alpha = st.empty()
unique_alphas = data['alpha'].unique()


def animate_alpha(plot_placeholder_alpha, unique_alphas, running_state_alpha):
    current_alpha_index = 0
    increment = 1
    zaxis_range = [distance_min, distance_max]
    
    while running_state_alpha["run_alpha"]:
        alpha = unique_alphas[current_alpha_index]
        df_filtered = data[data['alpha'] == alpha]
        pivot_df = df_filtered.pivot_table(values='distance', index='beta', columns='rho')

        fig_alpha = go.Figure(data=[go.Surface(z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index)])
        fig_alpha.update_layout(
            title=f"Optimization Results for Alpha = {alpha}",
            scene=dict(
                zaxis=dict(title='Distance', range=zaxis_range),
                xaxis=dict(title='Rho'),
                yaxis=dict(title='Beta')
            ),
            coloraxis=dict(colorscale='Viridis'),
            autosize=True
        )

        plot_placeholder_alpha.plotly_chart(fig_alpha, use_container_width=True)
        
        time.sleep(0.5)  
        
        current_alpha_index += increment
        if current_alpha_index == len(unique_alphas) or current_alpha_index < 0:
            increment *= -1
            current_alpha_index += increment

# Initialize the running state
if 'run_alpha' not in st.session_state:
    st.session_state['run_alpha'] = False

if not st.session_state['run_alpha']:
    plot_placeholder_alpha.image("alpha.png", use_column_width=True)

    

# Button to start and stop the animation
if st.button('▶️ Start/Stop Alpha Animation' if not st.session_state['run_alpha'] else '▶️ Start/Stop Alpha Animation'):
    st.session_state['run_alpha'] = not st.session_state['run_alpha']
    if st.session_state['run_alpha']:
        animate_alpha(plot_placeholder_alpha, unique_alphas, st.session_state)        
    else:
        st.session_state['run_alpha'] = False
        st.session_state['run_beta'] = False
        plot_placeholder_alpha = st.empty()
        plot_placeholder_beta = st.empty()
        plot_placeholder_alpha.image("alpha.png", use_column_width=True)



st.subheader("Beta (β) - Heuristic Information Importance")
st.write("""
- **Low β**: Places less emphasis on heuristic distance, leading to broader exploration but possibly less efficient paths.
- **High β**: Favors shorter paths strongly, which might improve the immediate solution quality but can reduce overall exploration, potentially missing longer but ultimately better routes.
""")

plot_placeholder_beta = st.empty()
unique_betas = data['beta'].unique()

def animate_beta(plot_placeholder_beta, data, unique_betas, running_state_beta):
    current_beta_index = 0
    increment = 1
    zaxis_range = [data['distance'].min(), data['distance'].max()]
    
    while running_state_beta["run_beta"]:
        beta = unique_betas[current_beta_index]
        df_filtered = data[data['beta'] == beta]
        pivot_df = df_filtered.pivot_table(values='distance', index='alpha', columns='rho')

        # Create a surface plot using graph_objects for beta
        fig_beta = go.Figure(data=[go.Surface(z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index)])
        fig_beta.update_layout(
            title=f"Optimization Results for Beta = {beta}",
            scene=dict(
                zaxis=dict(title='Distance', range=zaxis_range),
                xaxis=dict(title='Rho'),
                yaxis=dict(title='Alpha')
            ),
            coloraxis=dict(colorscale='Viridis'),
            autosize=True
        )

        plot_placeholder_beta.plotly_chart(fig_beta, use_container_width=True)
        
        time.sleep(0.7)
        
        current_beta_index += increment
        if current_beta_index == len(unique_betas) or current_beta_index < 0:
            increment *= -1
            current_beta_index += increment
            

if 'run_beta' not in st.session_state:
    st.session_state['run_beta'] = False
    
if not st.session_state['run_beta']:
    plot_placeholder_beta.image("alpha.png", use_column_width=True)

# Button to start and stop the animation
if st.button('▶️ Start/Stop Beta Animation' if not st.session_state['run_beta'] else '▶️ Start/Stop Beta Animation'):
    st.session_state['run_beta'] = not st.session_state['run_beta']
    if st.session_state['run_beta']:
        try:
            animate_beta(plot_placeholder_beta, unique_betas, st.session_state)  
        except:
            st.session_state['run_beta'] = True
    else:
        st.session_state['run_alpha'] = False
        st.session_state['run_beta'] = False
        plot_placeholder_alpha = st.empty()
        plot_placeholder_beta = st.empty()
        plot_placeholder_beta.image("alpha.png", use_column_width=True)

st.header("Impact of Evaporation Rate ρ")
st.write("""
The evaporation rate (ρ) moderates how quickly pheromones fade away, aiding in balancing exploration and exploitation by preventing the algorithm from overly fast convergence on suboptimal paths.
""")
st.write("""
- **Low ρ**: Pheromones persist longer, reinforcing existing trails and potentially leading to quicker convergence on suboptimal solutions.
- **High ρ**: Encourages more frequent exploration of new paths by reducing the influence of older pheromone trails, which may slow down convergence but fosters a more thorough search process.
""")


st.header("Adapting ACO to a Dynamic Traveling Salesman Problem (DTSP)")
st.write("""
To adapt ACO for dynamically changing scenarios such as DTSP where cities can be added or removed:
""")
st.write("""
- **Dynamic Updates**: Modify the distance and pheromone matrices in real-time as cities are added or removed. New cities start with minimal pheromones to spur exploration.
- **Reinitialization Strategies**: Consider reinitializing pheromone levels partially or entirely when significant changes occur to prevent reliance on outdated paths.
- **Real-time Adaptation**: Implement efficient mechanisms to handle updates, ensuring the algorithm can adapt without needing complete restarts unless absolutely necessary.
""")

st.write("This approach ensures that the ACO remains effective and robust even as the problem space changes dynamically, providing optimized solutions continuously.")


with st.expander("Calculate own empirical Simulation"):
    st.warning("If you start empirical simulation calculation it will calculate a simulation for α and β within the range from 1 to 10 in increments of 1, and for ρ from 0.1 to 1 in increments of 0.1. This will be heavy to compute and therefore recommended for local use on a GPU. The results will be saved in 'aco_simulation_results.xlsx'")
    
    if st.button('Calculate Empirical Simulation [< 20 hours callculation time, not recommended to run, see results above]'):
        results = []
        plot_placeholder = st.empty()  
    
        with st.status("Simulation running...", expanded=True) as status:
            for alpha in alpha_range:
                for beta in beta_range:
                    for rho in rho_range:
                        st.write(f"🏃 Calculating: alpha={alpha}, beta={beta}, rho={rho}")
                        distance = run_simulation(alpha, beta, rho)
                        results.append({'alpha': alpha, 'beta': beta, 'rho': rho, 'distance': distance})
    
                        results_df = pd.DataFrame(results)
                        file_name = 'aco_simulation_results.xlsx'
                        results_df.to_excel(file_name, index=False)
    
                        # Update the plot
                        fig = px.scatter_3d(results_df, x='alpha', y='beta', z='rho', color='distance',
                                            color_continuous_scale=px.colors.sequential.Viridis,
                                            title="Optimization Results Across Parameters")
                        plot_placeholder.plotly_chart(fig, use_container_width=True)
            status.update(label="Simulation complete!", state="complete", expanded=False)
        
        st.success("Simulation complete. Review the plot for results.")
        plot_placeholder.plotly_chart(fig, use_container_width=True)
        st.dataframe(results_df)
    


