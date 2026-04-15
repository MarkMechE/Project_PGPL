import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import networkx as nx
from scipy.signal import correlate, butter, lfilter

class PULSE_AT_Brain:
    def __init__(self):
        print("PULSE-AT: Initializing Robust Filtering Engine...")
        self.G = nx.scale_free_graph(n=100, seed=42).to_undirected()
        self.G = nx.Graph(self.G)
        self.persist_count = 0
        self.min_persist = 3
        self.c_acoustic = 1400 
        
    def _butter_bandpass_filter(self, data, lowcut=100, highcut=1000, fs=10000, order=5):
        """PULSE-AT Core: Removes urban noise frequencies."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def process(self, sensors):
        # 1. Apply Bandpass Filter to raw mic signals
        sig1 = self._butter_bandpass_filter(sensors['mic1_sig'])
        sig2 = self._butter_bandpass_filter(sensors['mic2_sig'])
        
        # 2. Gating Logic (Z-Score)
        z_score = abs(sensors.get('flow', 13) - 13) / 2.0
        
        if z_score > 2.0: # Anomaly threshold
            self.persist_count += 1
        else:
            self.persist_count = 0

        if self.persist_count >= self.min_persist:
            # 3. Time Delay Estimation (Cross-Correlation)
            corr = correlate(sig1, sig2)
            delay_idx = np.argmax(np.abs(corr)) - (len(sig1) - 1)
            delta_tau = abs(delay_idx / sensors['fs'])
            
            leak_dist = delta_tau * self.c_acoustic
            return {
                'flag': 'DISPATCH', 
                'loc_m': round(leak_dist, 2), 
                'z_score': round(z_score, 2),
                'status': 'Leak Confirmed'
            }
        
        return {'flag': 'MONITOR', 'z_score': round(z_score, 2)}

# Test the upgraded logic
if __name__ == "__main__":
    from pgpl_brain import PGPLBrain # Keep old for comparison
    brain = PULSE_AT_Brain()
    print("✅ PULSE-AT Brain Upgraded and Ready.")