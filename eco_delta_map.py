# eco_delta_map.py
# Run once: python eco_delta_map.py
# Output: data/eco_delta_graph.gpkg + data/eco_delta_nodes.csv

import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
import networkx as nx
import pickle

os.makedirs("data", exist_ok=True)

print("Downloading Eco-Delta City, Busan OSM network...")

# Eco-Delta City, Gangseo-gu, Busan — real bounding box
# Center: 35.1350° N, 128.9700° E  (Myeongji-Dong / Eco Delta area)
PLACE = "Myeongji-dong, Gangseo-gu, Busan, South Korea"

# Download walkable + drivable network (proxy for pipe network topology)
try:
    G_raw = ox.graph_from_place(PLACE, network_type="all", simplify=True)
    print(f"Downloaded: {len(G_raw.nodes)} nodes, {len(G_raw.edges)} edges")
except Exception:
    # Fallback: bounding box for Eco-Delta area
    print("Place query failed — using bounding box fallback...")
    G_raw = ox.graph_from_bbox(
        north=35.160, south=35.110,
        east=128.995, west=128.945,
        network_type="all"
    )
    print(f"Downloaded via bbox: {len(G_raw.nodes)} nodes, {len(G_raw.edges)} edges")

# Convert to undirected, project to metric CRS
G = ox.projection.project_graph(G_raw)
G_undir = ox.convert.to_undirected(G)

# Save as pickle (for fast reloading in brain)
with open("data/eco_delta_graph.pkl", "wb") as f:
    pickle.dump(G_undir, f)
print("Saved: data/eco_delta_graph.pkl")

# Save nodes as CSV with lat/lon for dashboard
nodes, edges = ox.convert.graph_to_gdfs(G_undir)
nodes_latlon = nodes.to_crs(epsg=4326)  # back to WGS84 for Folium
nodes_latlon["node_id"] = nodes_latlon.index.astype(str)
nodes_latlon["lat"] = nodes_latlon.geometry.y
nodes_latlon["lon"] = nodes_latlon.geometry.x
nodes_csv = nodes_latlon[["node_id", "lat", "lon"]].reset_index(drop=True)
nodes_csv.to_csv("data/eco_delta_nodes.csv", index=False)
print(f"Saved: data/eco_delta_nodes.csv ({len(nodes_csv)} nodes)")

# Save edges as GeoJSON for Folium overlay
edges_latlon = edges.to_crs(epsg=4326)
edges_latlon.to_file("data/eco_delta_edges.geojson", driver="GeoJSON")
print("Saved: data/eco_delta_edges.geojson")

print("\nEco-Delta map ready. Run main_pipeline.py next.")