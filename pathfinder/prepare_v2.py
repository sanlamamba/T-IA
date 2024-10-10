import pandas as pd
import json
from geopy.distance import geodesic
from joblib import Parallel, delayed
from tqdm import tqdm
import folium
import colorsys  

BASE_DIR = './data/'
PATHFINDER_DIR = './pathfinder/'
gare_df = pd.read_csv(BASE_DIR + 'liste-des-gares.csv', delimiter=';')
ligne_df = pd.read_csv(BASE_DIR + 'formes-des-lignes-du-rfn.csv', delimiter=';')

include_geo_shape = False 

# Prepare the dataframe and add a 'connected_to' column
prepared_df = gare_df[['CODE_UIC', 'LIBELLE', 'CODE_LIGNE', 'Geo Shape']].copy()
prepared_df['connected_to'] = None

def calculate_connected_stations(index, row, gare_df, ligne_df, include_geo_shape):
    try:
        current_station_code = row['CODE_UIC']  # Use CODE_UIC instead of LIBELLE
        current_line_code = row['CODE_LIGNE']
    
        current_station_coords = row['Geo Shape']
        if isinstance(current_station_coords, str):
            current_station_coords = json.loads(current_station_coords)['coordinates']
        
        concerned_lines = ligne_df[ligne_df['CODE_LIGNE'] == current_line_code]
    
        connected_stations = gare_df[gare_df['CODE_LIGNE'].isin(concerned_lines['CODE_LIGNE'])]
        
        connected_info = []
        for _, connected_row in connected_stations.iterrows():
            if connected_row['CODE_UIC'] == current_station_code:
                continue  # Skip if it's the same station
            
            connected_station_coords = connected_row['Geo Shape']
            if isinstance(connected_station_coords, str):
                connected_station_coords = json.loads(connected_station_coords)['coordinates']
            
            coord_1 = (current_station_coords[1], current_station_coords[0])
            coord_2 = (connected_station_coords[1], connected_station_coords[0])
            distance = geodesic(coord_1, coord_2).kilometers
            
            # Append connected station info as tuples (CODE_UIC, distance)
            connected_info.append((connected_row['CODE_UIC'], distance))
    
        return index, connected_info
    except Exception as e:
        print(f"Error processing station {row['CODE_UIC']}: {e}")
        return index, None

# Parallelize the connected station calculation process
results = Parallel(n_jobs=-1)(
    delayed(calculate_connected_stations)(index, row, gare_df, ligne_df, include_geo_shape) 
    for index, row in tqdm(prepared_df.iterrows(), total=len(prepared_df))
)

# Store the adjacency list in the 'connected_to' column with tuple format using CODE_UIC
for index, connected_info in results:
    if connected_info is not None:
        prepared_df.at[index, 'connected_to'] = connected_info

# Convert tuples in 'connected_to' to a list of dictionaries for better readability in JSON
def convert_to_dict_format(station_data):
    return [{'station': conn[0], 'distance': conn[1]} for conn in station_data]

# Apply the conversion before saving to JSON
prepared_df['connected_to'] = prepared_df['connected_to'].apply(convert_to_dict_format)

# Save the result as a CSV where each station has a list of connected stations (adjacency list in dictionaries using CODE_UIC)
prepared_df.to_csv(PATHFINDER_DIR + "output/" 'adjacency_list_stations_uic.csv', index=False)

# Save as JSON with dictionary format
prepared_df[['CODE_UIC', 'connected_to']].to_json(PATHFINDER_DIR + "output/" 'adjacency_list_stations_uic_dict.json', orient='records')

print("Data preparation complete.")
