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

prepared_df = gare_df[['CODE_UIC', 'LIBELLE', 'CODE_LIGNE', 'Geo Shape']].copy()
prepared_df['connected_to'] = None

def generate_color(hue=0, saturation=0.5, lightness=0.5, depth=0):
    """Generates a harmonic color based on HSL model."""
    h = (hue + depth * 0.4) % 1.0 
    s = 1 * (saturation - depth * 0.5)
    l = 1 * (lightness - depth * 0.5)
    
    rgb = colorsys.hls_to_rgb(h, l, s)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

railway_colors = {}

def calculate_connected_stations(index, row, gare_df, ligne_df, include_geo_shape):
    try:
        current_station_code = row['CODE_UIC']
        current_station_libelle = row['LIBELLE']
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
            connected_station_libelle = connected_row['LIBELLE']
            if isinstance(connected_station_coords, str):
                connected_station_coords = json.loads(connected_station_coords)['coordinates']
            
            coord_1 = (current_station_coords[1], current_station_coords[0])
            coord_2 = (connected_station_coords[1], connected_station_coords[0])
            distance = geodesic(coord_1, coord_2).kilometers
            
            if include_geo_shape:
                railway_info = concerned_lines[['CODE_LIGNE', 'Geo Shape']].drop_duplicates().to_dict(orient='records')
            else:
                railway_info = None 
            
            connected_info.append({
                'libelle': connected_row['LIBELLE'],
                'distance': distance,
                'railway': railway_info if include_geo_shape else None
            })
    
        return index, connected_info
    except Exception as e:
        print(f"Error processing station {row['LIBELLE']}: {e}")
        return index, None

results = Parallel(n_jobs=-1)(
    delayed(calculate_connected_stations)(index, row, gare_df, ligne_df, include_geo_shape) 
    for index, row in tqdm(prepared_df.iterrows(), total=len(prepared_df))
)

for index, connected_info in results:
    if connected_info is not None:
        prepared_df.at[index, 'connected_to'] = connected_info

print("Fin Préparation des données.")

prepared_df.to_csv(PATHFINDER_DIR + "output/" 'prepared_station_data.csv', index=False)

print("Preparation Visualisation.")
map_france = folium.Map(location=[46.603354, 1.888334], zoom_start=6)

for _, row in prepared_df.iterrows():
    coords = json.loads(row['Geo Shape'])['coordinates']
    folium.Marker(location=[coords[1], coords[0]], popup=row['LIBELLE']).add_to(map_france)

if include_geo_shape:
    for _, row in prepared_df.iterrows():
        connected_to = row['connected_to']
        if connected_to:
            for connected in connected_to:
                if connected['railway']:
                    for railway in connected['railway']:
                        railway_coords = json.loads(railway['Geo Shape'])['coordinates']
                        railway_points = [(coord[1], coord[0]) for coord in railway_coords]  
                        
                        code_ligne = railway['CODE_LIGNE']
                        if code_ligne not in railway_colors:
                            railway_colors[code_ligne] = generate_color(hue=len(railway_colors))
                        
                        folium.PolyLine(
                            railway_points, 
                            color=railway_colors[code_ligne], 
                            weight=2.5, 
                            opacity=1
                        ).add_to(map_france)


filename = include_geo_shape and 'station_map_with_railways.html' or 'station_map.html'

map_france.save(PATHFINDER_DIR + "output/" + filename)
print("Fin")
