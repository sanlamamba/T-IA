import pandas as pd
import json
import ast
import networkx as nx

# Load station data from CSV
PATHFINDER_DIR = './pathfinder/'
OUTPUT_DIR = PATHFINDER_DIR + 'output/'
df = pd.read_csv(OUTPUT_DIR + 'prepared_station_data.csv')
# Function to safely parse the 'connected_to' field
def parse_connected_to(connected_to_str):
    try:
        # Replace single quotes with double quotes for valid JSON
        connected_to_str = connected_to_str.replace("'", '"')
        
        # Try using json.loads, but if it fails, fall back to ast.literal_eval
        try:
            return json.loads(connected_to_str)
        except json.JSONDecodeError:
            # Use ast.literal_eval if the string isn't valid JSON
            return ast.literal_eval(connected_to_str)
    except Exception as e:
        print(f"Error parsing connected_to: {e}")
        return None

# Parse the 'connected_to' field for each row in the DataFrame
df['connected_to_parsed'] = df['connected_to'].apply(parse_connected_to)

# Create a NetworkX graph
G = nx.Graph()

# Function to add edges to the graph from the parsed 'connected_to' field
def add_edges_from_connected_to(row):
    current_station = row['LIBELLE']
    connected_stations = row['connected_to_parsed']
    
    if connected_stations is not None:
        for connection in connected_stations:
            connected_station = connection['libelle']
            distance = connection['distance']
            
            # Add an edge to the graph between the current station and connected station
            G.add_edge(current_station, connected_station, weight=distance)

# Add edges for each station
df.apply(add_edges_from_connected_to, axis=1)

# Function to find the shortest path between two stations
def find_shortest_path(from_station, to_station):
    try:
        # Use Dijkstra's algorithm to find the shortest path based on distance (weight)
        shortest_path = nx.dijkstra_path(G, from_station, to_station, weight='weight')
        shortest_distance = nx.dijkstra_path_length(G, from_station, to_station, weight='weight')
        
        return shortest_path, shortest_distance
    except nx.NetworkXNoPath:
        return None, float('inf')
    except nx.NodeNotFound as e:
        print(f"Error: {e}")
        return None, float('inf')

# Example: Find the shortest path between two stations
from_station = 'La Douzillère'
to_station = 'St-Césaire'
shortest_path, shortest_distance = find_shortest_path(from_station, to_station)

if shortest_path:
    print(f"Shortest path from {from_station} to {to_station}: {shortest_path}")
    print(f"Total distance: {shortest_distance:.2f} km")
else:
    print(f"No path found between {from_station} and {to_station}.")

# Save the parsed DataFrame for future reference
df.to_csv('parsed_station_data.csv', index=False)
