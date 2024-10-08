import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Fichier csv récupéré depuis le site de la SCNF
gares_df = pd.read_csv("liste-des-gares.csv", delimiter=';')


print("Aperçu du dataset des gares :")
print(gares_df.head())

# Initialiser un graphe en utilisant networkx validé par Lamamba 
graph = nx.Graph()

# Ajout les 3469 gares avec leur position
for index, row in gares_df.iterrows():
    graph.add_node(row['LIBELLE'], pos=(row['X_WGS84'], row['Y_WGS84']))

# Afficher le nombre total de nœuds dans le graphe
print(f"Nombre total de gares dans le graphe: {graph.number_of_nodes()}")

# Visualiser le graphe
pos = nx.get_node_attributes(graph, 'pos')  # Obtenir les positions des nœuds

plt.figure(figsize=(10, 8))
nx.draw(graph, pos, with_labels=True, node_size=50, node_color='blue', font_size=8)

# Afficher le graphe
plt.title("Visualisation du graphe des gares")
plt.show()