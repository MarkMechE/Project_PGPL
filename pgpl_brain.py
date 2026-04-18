# pgpl_brain.py
# PULSE-AT Brain — full 4-tier pipeline with Eco-Delta graph snap

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import correlate, butter, lfilter
from collections import deque

ECO_DELTA_CENTER = (35.135, 128.970)  # Eco-Delta City, Busan


class PULSE_AT_Brain:

    def __init__(self, graph_path="data/eco_delta_graph.pkl",
                 nodes_path="data/eco_delta_nodes.csv"):

        print("PULSE-AT: Initializing...")

        # --- Load real Eco-Delta graph ---
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                self.G = pickle.load(f)
            print(f"  Graph loaded: {len(self.G.nodes)} nodes")
        else:
            print("  WARNING: Graph not found. Run eco_delta_map.py first.")
            print("  Using fallback synthetic graph.")
            self.G = nx.scale_free_graph(n=200, seed=42).to_undirected()
            self.G = nx.Graph(self.G)

        # --- Node lookup table (node_id -> lat/lon) ---
        if os.path.exists(nodes_path):
            df = pd.read_csv(nodes_path)
            self.node_coords = {
                int(row["node_id"]): (row["lat"], row["lon"])
                for _, row in df.iterrows()
            }
        else:
            self.node_coords = {}

        self.node_list = list(self.G.nodes())

        # --- PULSE-AT state ---
        self.persist_count = 0
        self.min_persist = 3
        self.psi_window = deque(maxlen=30)      # Tier 3: drift window
        self.score_history = deque(maxlen=150)  # adaptive calibration

        # --- Physical constants ---
        self.c_acoustic = 1400   # m/s in water pipe
        self.fs_default = 10000  # Hz

        # --- Adaptive threshold ---
        self.eps_base = 0.05
        self.eps_current = self.eps_base
        self.lr = 0.003

        print("  PULSE-AT Brain ready.\n")

    # ------------------------------------------------------------------ #
    #  Signal processing                                                   #
    # ------------------------------------------------------------------ #

    def _butter_bandpass(self, data, lowcut=100, highcut=1000,
                         fs=10000, order=5):
        """Tier 0 — remove tide noise and EMI below 100 Hz."""
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 1e-4)
        high = min(highcut / nyq, 1 - 1e-4)
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def _sim_piezo(self, leak_dist_m, fs=10000, noise_sigma=0.05):
        """Simulate two mic signals given a true leak distance."""
        t = np.linspace(0, 1, fs)
        mic1 = np.sin(2 * np.pi * 150 * t) + noise_sigma * np.random.randn(fs)
        delay_s = leak_dist_m / (self.c_acoustic * 2)
        delay_samples = int(delay_s * fs)
        mic2 = np.roll(mic1, delay_samples) + noise_sigma * np.random.randn(fs)
        return mic1, mic2

    # ------------------------------------------------------------------ #
    #  Tier 1 — anomaly scoring                                           #
    # ------------------------------------------------------------------ #

    def _tier1_score(self, sensors):
        """Bidirectional z-score from flow, pressure, turbulence."""
        flow = sensors.get("flow", 13.0)
        pres = sensors.get("pressure", 3.2)
        var_z = sensors.get("var_z", 0.0)

        z_flow = abs(flow - 13.0) / 2.0
        z_pres = abs(pres - 3.2) / 0.5
        anti_corr = max(0.0, (z_flow - z_pres) * 0.5)  # flow up + pres down

        score = (0.40 * z_flow +
                 0.30 * z_pres +
                 0.30 * var_z +
                 0.50 * anti_corr)
        return round(score, 4)

    # ------------------------------------------------------------------ #
    #  Tier 2 — conformal calibration                                     #
    # ------------------------------------------------------------------ #

    def _tier2_pvalue(self, score):
        """Conformal p-value against stored score history."""
        if len(self.score_history) < 10:
            return 1.0
        n = len(self.score_history)
        n_geq = sum(1 for s in self.score_history if s >= score)
        return (n_geq + 1) / (n + 1)

    # ------------------------------------------------------------------ #
    #  Tier 3 — drift gate (PSI + persistence)                           #
    # ------------------------------------------------------------------ #

    def _tier3_gate(self, score, pvalue):
        """Returns True only after persistent anomaly with drift confirmation."""
        self.psi_window.append(score)
        self.score_history.append(score)

        # PSI proxy: variance of recent window vs. baseline
        if len(self.psi_window) >= 10:
            psi = float(np.std(list(self.psi_window)) / (np.mean(list(self.psi_window)) + 1e-6))
        else:
            psi = 0.0

        # Persistence counter
        if score > 2.5 and pvalue < 0.10:
            self.persist_count += 1
        else:
            self.persist_count = max(0, self.persist_count - 1)

        gate_open = (self.persist_count >= self.min_persist and psi >= 0.20)
        return gate_open, round(psi, 4)

    # ------------------------------------------------------------------ #
    #  Tier 4 — triangulation + graph snap                               #
    # ------------------------------------------------------------------ #

    def _tier4_localize(self, sensors, sensor_node_idx=0):
        """Cross-correlation Δτ → distance → shortest path snap → GPS."""
        fs = sensors.get("fs", self.fs_default)
        mic1 = self._butter_bandpass(sensors["mic1_sig"], fs=fs)
        mic2 = self._butter_bandpass(sensors["mic2_sig"], fs=fs)

        # Cross-correlation time-delay estimation
        corr = correlate(mic1, mic2, mode="full")
        delay_idx = int(np.argmax(np.abs(corr))) - (len(mic1) - 1)
        delta_tau = abs(delay_idx / fs)
        leak_dist_m = delta_tau * self.c_acoustic

        # Graph snap: follow shortest path from sensor node
        if len(self.node_list) == 0:
            return leak_dist_m, ECO_DELTA_CENTER, None

        sensor_node = self.node_list[sensor_node_idx % len(self.node_list)]

        try:
            # Walk the graph up to leak_dist_m (use hop cutoff as proxy)
            hop_cutoff = max(1, int(leak_dist_m / 50))  # ~50 m per hop
            paths = nx.single_source_shortest_path(
                self.G, sensor_node, cutoff=hop_cutoff
            )
            # Pick the farthest reachable node
            end_node = max(paths.keys(),
                           key=lambda n: len(paths[n]))
            leak_gps = self.node_coords.get(end_node, ECO_DELTA_CENTER)
        except Exception:
            end_node = sensor_node
            leak_gps = self.node_coords.get(sensor_node, ECO_DELTA_CENTER)

        return round(leak_dist_m, 2), leak_gps, end_node

    # ------------------------------------------------------------------ #
    #  Adaptive epsilon update                                            #
    # ------------------------------------------------------------------ #

    def _update_epsilon(self, confirmed):
        if confirmed:
            self.eps_current = max(0.02, self.eps_current - self.lr)
        else:
            self.eps_current = min(0.08, self.eps_current + self.lr * 0.5)

    # ------------------------------------------------------------------ #
    #  Main process() call                                                #
    # ------------------------------------------------------------------ #

    def process(self, sensors, sensor_node_idx=0):
        """
        sensors dict keys:
          flow        (L/min)
          pressure    (bar)
          var_z       (turbulence proxy, float)
          mic1_sig    (numpy array)
          mic2_sig    (numpy array)
          fs          (int, sample rate — default 10000)
          true_dist   (optional, for error reporting in sims)
        """
        # Tier 1
        score = self._tier1_score(sensors)

        # Tier 2
        pvalue = self._tier2_pvalue(score)

        # Tier 3
        gate_open, psi = self._tier3_gate(score, pvalue)

        if gate_open:
            # Tier 4
            leak_dist, leak_gps, end_node = self._tier4_localize(
                sensors, sensor_node_idx
            )
            self._update_epsilon(confirmed=True)

            true_dist = sensors.get("true_dist", None)
            error_m = round(abs(true_dist - leak_dist), 2) if true_dist else None

            return {
                "flag":      "DISPATCH",
                "loc_m":     leak_dist,
                "gps":       leak_gps,
                "score":     score,
                "pvalue":    pvalue,
                "psi":       psi,
                "persist":   self.persist_count,
                "eps":       round(self.eps_current, 4),
                "end_node":  end_node,
                "error_m":   error_m,
                "status":    "Leak confirmed — dispatch repair crew",
            }

        self._update_epsilon(confirmed=False)
        return {
            "flag":    "MONITOR",
            "score":   score,
            "pvalue":  pvalue,
            "psi":     psi,
            "persist": self.persist_count,
            "eps":     round(self.eps_current, 4),
            "status":  "Monitoring — no confirmed leak",
        }


# ------------------------------------------------------------------ #
#  Quick self-test                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    brain = PULSE_AT_Brain()

    TRUE_LEAK = 45.0  # metres

    t = np.linspace(0, 1, 10000)
    mic1, mic2 = brain._sim_piezo(TRUE_LEAK)

    sensors = {
        "flow":     20.0,
        "pressure": 2.8,
        "var_z":    3.2,
        "mic1_sig": mic1,
        "mic2_sig": mic2,
        "fs":       10000,
        "true_dist": TRUE_LEAK,
    }

    print("Running 5 cycles to trigger persistence gate...\n")
    for i in range(5):
        result = brain.process(sensors, sensor_node_idx=0)
        print(f"  Cycle {i+1}: {result['flag']} | score={result['score']} "
              f"| persist={result['persist']} | psi={result['psi']}")
        if result["flag"] == "DISPATCH":
            print(f"\n  LEAK FOUND at {result['loc_m']} m "
                  f"| GPS: {result['gps']} "
                  f"| error: {result['error_m']} m")
            break