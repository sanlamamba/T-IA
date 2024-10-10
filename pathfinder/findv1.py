import json
import heapq

# Load the JSON file that contains the adjacency list
def load_graph(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Dijkstra's algorithm to find the shortest path between two stations
def dijkstra(graph, start, end):
    # Initialize a priority queue (min-heap)
    queue = [(0, start)]  # (distance, node)
    distances = {station['CODE_UIC']: float('infinity') for station in graph}
    distances[start] = 0
    previous_nodes = {station['CODE_UIC']: None for station in graph}
    
    # Mapping station CODE_UICs to their neighbors (adjacency list)
    station_map = {station['CODE_UIC']: station['connected_to'] for station in graph}

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        
        # If we reached the destination node, stop
        if current_node == end:
            break
        
        # Explore neighbors
        for neighbor in station_map[current_node]:
            neighbor_station = neighbor['station']
            distance = neighbor['distance']
            new_distance = current_distance + distance
            
            # Only consider this path if it's shorter than any known path
            if new_distance < distances[neighbor_station]:
                distances[neighbor_station] = new_distance
                previous_nodes[neighbor_station] = current_node
                heapq.heappush(queue, (new_distance, neighbor_station))
    
    # Reconstruct the shortest path
    path = []
    total_distance = distances[end]
    
    if total_distance == float('infinity'):
        return None, float('infinity')  # No path found
    
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    
    return path[::-1], total_distance  # Return reversed path and total distance

# Find the shortest path between two stations
def find_shortest_path(json_file, start, end):
    graph = load_graph(json_file)
    path, total_distance = dijkstra(graph, start, end)
    
    if path is None:
        print(f"No path found between {start} and {end}.")
    else:
        print(f"The shortest path from {start} to {end} is: {path}")
        print(f"Total distance: {total_distance:.2f} km")
    
    return path, total_distance

# Example usage:
json_file = './pathfinder/output/adjacency_list_stations_uic_dict.json'
start_station = 87009696  # Replace with the starting station CODE_UIC
end_station = 87491456    # Replace with the destination station CODE_UIC

# Find and print the shortest path
find_shortest_path(json_file, start_station, end_station)
