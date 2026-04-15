import osmnx as ox
import geopandas as gpd
import os
import city2graph as c2g

# This converts your Incheon GPKG into a "Pipe Graph"
pipe_graph = c2g.Graph.from_file("data/incheon_graph.gpkg")
print(f"Graph converted: {len(pipe_graph.nodes)} pipe junctions found.")

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

print("Fetching Incheon Graph (this may take 1-2 minutes)...")
# We use 'drive' as a proxy for the pipe layout
G = ox.graph_from_place('Incheon, South Korea', network_type='drive')

# Save as GeoPackage for faster loading later
ox.save_graph_geopackage(G, filepath='data/incheon_graph.gpkg')
print("✅ Incheon Graph saved to data/incheon_graph.gpkg")