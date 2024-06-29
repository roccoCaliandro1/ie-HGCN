import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# Define random adjacency matrix
np.random.seed(random.randint(1, 568))  # Set seed for reproducibility
adj_matrix = np.random.rand(4, 4)

# Ensure the matrix is symmetric to represent an undirected graph
adj_matrix = (adj_matrix + adj_matrix.T) / 2

# Remove self-connections (diagonal elements)
np.fill_diagonal(adj_matrix, 0)

# Randomly remove at least one edge to ensure the graph is not fully connected
num_edges_to_remove = np.random.randint(1, np.count_nonzero(adj_matrix) + 1)

# Flatten the output of np.where to get a 1D array of indices
nonzero_indices = np.transpose(np.where(adj_matrix > 0))
flat_indices = nonzero_indices.flatten()

# Randomly select unique indices to remove edges
indices_to_remove = np.random.choice(flat_indices, size=(num_edges_to_remove * 2), replace=False)
edges_to_remove = indices_to_remove.reshape((num_edges_to_remove, 2))

# Update adjacency matrix by setting selected edges to 0
for (i, j) in edges_to_remove:
    adj_matrix[i, j] = 0
    adj_matrix[j, i] = 0

# Create a graph from the modified adjacency matrix
G = nx.from_numpy_array(adj_matrix)

# Draw the graph with edge labels
plt.figure(figsize=(8, 6))
pos = nx.circular_layout(G)  # Position nodes using a circular layout
edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='skyblue', width=2)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', font_family='sans-serif')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red', font_size=10)

plt.title('Graph Visualization with Arrows and Edge Weights (No Self-Connections)')
plt.show()

# Draw the adjacency matrix with edge weights
plt.figure(figsize=(6, 6))
plt.imshow(adj_matrix, cmap='Blues', interpolation='none', vmin=0, vmax=1)

# Add text annotations for each cell in the matrix
for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
        plt.text(j, i, f'{adj_matrix[i, j]:.2f}', ha='center', va='center', color='red', fontsize=12)

plt.colorbar(label='Edge Weight')
plt.title('Adjacency Matrix with Edge Weights (Edges Removed)')

# Adjust x and y axis to start from 0
ticks = np.arange(len(adj_matrix))
plt.xticks(ticks=ticks, labels=ticks)
plt.yticks(ticks=ticks, labels=ticks)
plt.gca().xaxis.tick_top()  # Move x-axis ticks to the top
plt.gca().invert_yaxis()  # Invert y-axis to match matrix layout
plt.show()
