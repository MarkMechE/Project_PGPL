import networkx as nx
import osmnx as ox
from pgpl_brain import PGPLBrain # Import your Day 1 Brain

# 1. Load the Incheon Graph
G = ox.load_graphml(filepath=None) # Or use the .gpkg from Step 10
# (Simplified for demo: using a random node from your existing brain.G)
brain = PGPLBrain()
target_node = list(brain.G.nodes())[5] # Pick a random node as the "leak" location

# 2. Process
sensor_packet = brain.sim_piezo(leak_dist=45.0)
sensor_packet.update({'flow': 15, 'pres1': 2.8}) # Trigger values

for i in range(3):
    result = brain.process(sensor_packet)
    if result['flag'] == 'DISPATCH':
        # Snap to Graph
        coords = (brain.G.nodes[target_node].get('x', 126.7), brain.G.nodes[target_node].get('y', 37.4))
        print(f"🚨 LEAK DETECTED!")
        print(f"Location: {result['loc_m']}m from sensor")
        print(f"GPS Coordinates: {coords}")