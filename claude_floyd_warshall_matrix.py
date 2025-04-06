import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Create a weighted directed graph
def create_weighted_graph():
    # Adjacency matrix representation of a weighted graph
    # We'll use infinity (np.inf) for edges that don't exist
    # For example, this represents a graph with 6 vertices
    
    n = 6  # Number of vertices
    
    # Initialize adjacency matrix with infinity values
    adj_matrix = np.full((n, n), np.inf)
    
    # Set diagonal to 0 (distance from a vertex to itself is 0)
    np.fill_diagonal(adj_matrix, 0)
    
    # Define edges and weights
    # (u, v, weight) means there's an edge from u to v with weight
    edges = [
        (0, 1, 5),
        (0, 2, 3),
        (1, 3, 6),
        (1, 2, 2),
        (2, 4, 4),
        (2, 3, 7),
        (3, 5, 1),
        (4, 3, 2),
        (4, 5, 8)
    ]
    
    # Add edges to adjacency matrix
    for u, v, w in edges:
        adj_matrix[u, v] = w
    
    return adj_matrix, edges

# Implement Floyd-Warshall algorithm
def floyd_warshall(adj_matrix):
    n = adj_matrix.shape[0]
    dist = adj_matrix.copy()
    
    # Initialize next matrix for path reconstruction
    next_v = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] != np.inf:
                next_v[i, j] = j
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    next_v[i, j] = next_v[i, k]
    
    return dist, next_v

# Reconstruct paths
def get_path(next_v, i, j):
    if next_v[i, j] == 0:
        return []
    
    path = [i]
    while i != j:
        i = next_v[i, j]
        path.append(i)
    
    return path

# Create the graph
adj_matrix, edges = create_weighted_graph()

# Print the adjacency matrix
print("Adjacency Matrix:")
print(pd.DataFrame(adj_matrix).replace(np.inf, "∞"))

# Solve for all shortest paths
dist_matrix, next_v = floyd_warshall(adj_matrix)

# Print the distance matrix
print("\nDistance Matrix (Shortest Paths):")
print(pd.DataFrame(dist_matrix).replace(np.inf, "∞"))

# Get paths for a few pairs
print("\nShortest paths between vertices:")
for i in range(6):
    for j in range(i+1, 6):
        path = get_path(next_v, i, j)
        if path:
            print(f"Path from {i} to {j}: {path} (distance: {dist_matrix[i, j]})")

# Visualize the graph
def visualize_graph(adj_matrix, edges):
    n = adj_matrix.shape[0]
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n):
        G.add_node(i)
    
    # Add edges
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    plt.figure(figsize=(12, 8))
    
    # Create a layout for our nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    
    # Draw edges
    edge_labels = {(u, v): f"{w}" for u, v, w in edges}
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=15)
    
    # Highlight shortest path from 0 to 5
    path_0_to_5 = get_path(next_v, 0, 5)
    if path_0_to_5:
        path_edges = list(zip(path_0_to_5, path_0_to_5[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=3)
    
    plt.title("Weighted Directed Graph with Shortest Path from 0 to 5 Highlighted")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize
visualize_graph(adj_matrix, edges)