import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# Define a random adjacency matrix with values of either 0 or 1
np.random.seed(random.randint(1, 587))  # Set seed for reproducibility
adj_matrix = np.random.randint(2, size=(4, 4))

# Remove self-connections (diagonal elements)
np.fill_diagonal(adj_matrix, 0)

# Create a directed graph from the modified adjacency matrix
G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

# Draw the directed graph with arrows
plt.figure(figsize=(8, 6))
pos = nx.circular_layout(G)  # Position nodes using a circular layout
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, font_weight='bold')
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='skyblue')

# Add edge weights as labels on the edges
edge_labels = {(i, j): adj_matrix[i, j] for i, j in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_color='red')

plt.title('Graph Visualization with Arrows (No Self-Connections)')
plt.show()

# Draw the adjacency matrix with edge weights
plt.figure(figsize=(6, 6))
plt.imshow(adj_matrix, cmap='Blues', interpolation='none', vmin=0, vmax=1)

# Add text annotations for each cell in the matrix
for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
        plt.text(j, i, adj_matrix[i, j], ha='center', va='center', color='red', fontsize=12)

plt.colorbar(label='Edge Weight')
plt.title('Adjacency Matrix with Edge Weights (No Self-Connections)')

# Adjust x and y axis to start from 0
ticks = np.arange(len(adj_matrix))
plt.xticks(ticks=ticks, labels=ticks)
plt.yticks(ticks=ticks, labels=ticks)
plt.gca().xaxis.tick_top()  # Move x-axis ticks to the top
plt.gca().invert_yaxis()  # Invert y-axis to match matrix layout
plt.show()
