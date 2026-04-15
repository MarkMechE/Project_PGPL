import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import networkx as nx
from scipy.signal import correlate

class PGPLBrain:
    def __init__(self):
        print("Initializing Synthetic Pipe Network (Incheon Proxy)...")
        # Fix: Convert MultiGraph to Simple Graph to avoid unpack error
        temp_g = nx.scale_free_graph(n=100, seed=42).to_undirected()
        self.G = nx.Graph(temp_g) 
        
        for u, v in self.G.edges():
            self.G.edges[u,v]['length'] = np.random.uniform(40, 60)
        self.persist_count = 0
        self.c_acoustic = 1400 

    def sim_piezo(self, leak_dist):
        fs = 10000
        t = np.linspace(0, 1, fs)
        mic1 = np.sin(2*np.pi*100*t) + 0.05*np.random.randn(fs)
        delay_s = leak_dist / self.c_acoustic
        mic2 = np.sin(2*np.pi*100*(t - delay_s)) + 0.05*np.random.randn(fs)
        return {'mic1_sig': mic1, 'mic2_sig': mic2, 'fs': fs}

    def process(self, sensors):
        self.persist_count += 1
        if self.persist_count >= 3:
            corr = correlate(sensors['mic1_sig'], sensors['mic2_sig'])
            delay_idx = np.argmax(np.abs(corr)) - (len(sensors['mic1_sig']) - 1)
            delta_tau = abs(delay_idx / sensors['fs'])
            return {'flag': 'DISPATCH', 'loc_m': round(delta_tau * self.c_acoustic, 2)}
        return {'flag': 'MONITOR', 'count': self.persist_count}

if __name__ == "__main__":
    brain = PGPLBrain()
    data = brain.sim_piezo(45.0)
    print("--- Starting Pipeline ---")
    for i in range(3):
        print(f"Run {i+1}: {brain.process(data)}")
