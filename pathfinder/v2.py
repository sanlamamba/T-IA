import csv
import networkx as nx
import matplotlib.pyplot as plt

# Initialize the graph
G = nx.Graph()

# Dictionary to hold station coordinates
station_positions = {}

PATHFINDER_DIR = './pathfinder/'
OUTPUT_DIR = PATHFINDER_DIR + 'output/'
file_path = OUTPUT_DIR + 'prepared_station_data.csv'    


# Load data from CSV file
with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # Loop through each row in the CSV file
    for row in reader:
        station_libelle = row['LIBELLE']
        coordinates = eval(row['Geo Shape'])['coordinates']  # Extract the coordinates
        connected_to = eval(row['connected_to'])  # Convert string representation of list to actual list

        # Add the station to the graph with coordinates
        station_positions[station_libelle] = (coordinates[0], coordinates[1])

        # Add edges for each connection
        for connection in connected_to:
            connected_station = connection['libelle']
            distance = connection['distance']

            # Add the connection to the graph with distance as weight
            G.add_edge(station_libelle, connected_station, weight=distance)

# Function to display the graph with nodes, edges, and edge weights
def display_graph_with_edges(graph, positions, from_station, to_station):
    plt.figure(figsize=(12, 10))

    # Draw the nodes (stations)
    node_color_map = []
    for node in graph.nodes():
        if node == from_station:
            node_color_map.append('green')  # Color for 'from' station
        elif node == to_station:
            node_color_map.append('red')  # Color for 'to' station
        else:
            node_color_map.append('blue')  # Color for other stations

    nx.draw_networkx_nodes(graph, positions, node_size=50, node_color=node_color_map)

    # Draw the edges (connections)
    nx.draw_networkx_edges(graph, positions, width=1.5, alpha=0.7)

    # Draw the labels (station names)
    # nx.draw_networkx_labels(graph, positions, font_size=10)

    # Draw the edge labels (distances) manually
    for (u, v, d) in graph.edges(data=True):
        x = (positions[u][0] + positions[v][0]) / 2
        y = (positions[u][1] + positions[v][1]) / 2
        plt.text(x, y, f"{d['weight']:.2f}", fontsize=8, ha='center')

    # Show the graph
    plt.title("Railway Network with Distances")
    plt.show()

    # save it as image file
    plt.savefig('graph.png')

# Function to find the shortest path
def find_shortest_path(from_station, to_station):
    try:
        # Use Dijkstra's algorithm to find the shortest path
        shortest_path = nx.dijkstra_path(G, source=from_station, target=to_station, weight='weight')
        path_length = nx.dijkstra_path_length(G, source=from_station, target=to_station, weight='weight')
        return shortest_path, path_length
    except nx.NetworkXNoPath:
        return None, float('inf')

# Example usage
from_station = "La Douzillère"  # Replace with actual 'from' station
to_station = "Ste-Colombe-Septveilles"  # Replace with actual 'to' station

shortest_path, path_length = find_shortest_path(from_station, to_station)

if shortest_path:
    print(f"The shortest path from {from_station} to {to_station} is:")
    print(" -> ".join(shortest_path))
    print(f"Total distance: {path_length} km")
else:
    print(f"No path found from {from_station} to {to_station}")

# Display the graph with edges, nodes, and distances, highlighting the 'from' and 'to' stations
display_graph_with_edges(G, station_positions, from_station, to_station)
