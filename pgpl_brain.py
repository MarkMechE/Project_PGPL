import os
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import correlate, butter, lfilter
from collections import deque

ECO_DELTA_CENTER = (35.135, 128.970)


class PULSE_AT_Brain:

    def __init__(
        self,
        graph_path="data/eco_delta_graph.pkl",
        nodes_path="data/eco_delta_nodes.csv",
        clf_path="outputs/anomaly_classifier.pkl",
    ):
        print("PULSE-AT: Initializing...")

        # ── Graph ─────────────────────────────────────────────────
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                self.G = pickle.load(f)
            print(f"  Graph loaded: {len(self.G.nodes)} nodes")
        else:
            print("  WARNING: Graph not found. Run eco_delta_map.py first.")
            print("  Using fallback synthetic graph.")
            raw = nx.scale_free_graph(n=200, seed=42)
            self.G = nx.Graph(raw.to_undirected())

        # ── Node coordinate lookup ─────────────────────────────────
        if os.path.exists(nodes_path):
            df = pd.read_csv(nodes_path)
            self.node_coords = {
                int(row["node_id"]): (float(row["lat"]), float(row["lon"]))
                for _, row in df.iterrows()
            }
        else:
            self.node_coords = {}

        self.node_list = list(self.G.nodes())

        # ── Anomaly classifier ─────────────────────────────────────
        self.clf         = None
        self.clf_encoder = None
        if os.path.exists(clf_path):
            with open(clf_path, "rb") as f:
                bundle = pickle.load(f)
            self.clf         = bundle["model"]
            self.clf_encoder = bundle["encoder"]
            print(f"  Classifier loaded: {list(self.clf_encoder.classes_)}")
        else:
            print("  WARNING: Classifier not found. Run anomaly_classifier.py first.")

        # ── PULSE-AT state ─────────────────────────────────────────
        self.persist_count = 0
        self.min_persist   = 3
        self.psi_window    = deque(maxlen=30)
        self.score_history = deque(maxlen=150)

        # ── Physical constants ─────────────────────────────────────
        self.c_acoustic  = 1400    # m/s in water pipe
        self.fs_default  = 10000   # Hz

        # ── Adaptive epsilon ───────────────────────────────────────
        self.eps_current = 0.05
        self.lr          = 0.003

        print("  PULSE-AT Brain ready.\n")

    # ---------------------------------------------------------------- #
    #  Signal utilities                                                  #
    # ---------------------------------------------------------------- #

    def _bandpass(self, data, lo=100, hi=1000, fs=10000, order=5):
        """Butterworth bandpass — removes tide noise below 100 Hz."""
        nyq  = 0.5 * fs
        low  = max(lo / nyq, 1e-4)
        high = min(hi / nyq, 1 - 1e-4)
        b, a = butter(order, [low, high], btype="band")
        return lfilter(b, a, data)

    def _sim_piezo(self, leak_dist_m, fs=10000, noise_sigma=0.05):
        """
        Simulate two mic signals for a leak at leak_dist_m.
        Used for testing and bridge data generation.
        """
        t       = np.linspace(0, 1, fs)
        mic1    = (np.sin(2 * np.pi * 150 * t)
                   + noise_sigma * np.random.randn(fs))
        delay_s = leak_dist_m / (self.c_acoustic * 2)
        shift   = int(delay_s * fs)
        mic2    = np.roll(mic1, shift) + noise_sigma * np.random.randn(fs)
        return mic1, mic2

    # ---------------------------------------------------------------- #
    #  Tier 1 — bidirectional z-score                                   #
    # ---------------------------------------------------------------- #

    def _tier1(self, sensors):
        """
        Weighted anomaly score from flow, pressure, turbulence.
        Anti-correlation bonus fires when flow rises and pressure drops.
        """
        z_flow = abs(sensors.get("flow",     13.0) - 13.0) / 2.0
        z_pres = abs(sensors.get("pressure",  3.2) -  3.2) / 0.5
        z_var  = float(sensors.get("var_z",   0.0))
        anti   = max(0.0, (z_flow - z_pres) * 0.5)
        score  = (0.40 * z_flow
                + 0.30 * z_pres
                + 0.30 * z_var
                + 0.50 * anti)
        return round(score, 4)

    # ---------------------------------------------------------------- #
    #  Tier 2 — conformal p-value                                       #
    # ---------------------------------------------------------------- #

    def _tier2(self, score):
        """
        Empirical p-value: fraction of calibration scores >= current score.
        Returns 1.0 (no evidence) until 10 scores are accumulated.
        """
        if len(self.score_history) < 10:
            return 1.0
        n     = len(self.score_history)
        n_geq = sum(1 for s in self.score_history if s >= score)
        return round((n_geq + 1) / (n + 1), 4)

    # ---------------------------------------------------------------- #
    #  Tier 3 — drift gate (PSI + persistence)                          #
    # ---------------------------------------------------------------- #

    def _tier3(self, score, pvalue):
        """
        Gate opens only after:
          - score > 2.5 AND pvalue < 0.10 for >= min_persist consecutive rows
          - PSI (coefficient of variation of recent window) >= 0.20
        """
        self.psi_window.append(score)
        self.score_history.append(score)

        if len(self.psi_window) >= 10:
            arr = list(self.psi_window)
            psi = round(
                float(np.std(arr) / (np.mean(arr) + 1e-6)),
                4,
            )
        else:
            psi = 0.0

        if score > 2.5 and pvalue < 0.10:
            self.persist_count += 1
        else:
            self.persist_count = max(0, self.persist_count - 1)

        gate = (self.persist_count >= self.min_persist and psi >= 0.20)
        return gate, psi

    # ---------------------------------------------------------------- #
    #  Tier 4 — acoustic triangulation + graph snap                     #
    # ---------------------------------------------------------------- #

    def _tier4(self, sensors, sensor_node_idx=0):
        """
        1. Bandpass filter both mic signals.
        2. Cross-correlate to find peak delay index.
        3. Convert delay to distance via speed of sound.
        4. Walk the pipe graph to snap distance to real GPS node.
        """
        fs   = int(sensors.get("fs", self.fs_default))
        sig1 = self._bandpass(sensors["mic1_sig"], fs=fs)
        sig2 = self._bandpass(sensors["mic2_sig"], fs=fs)

        corr      = correlate(sig1, sig2, mode="full")
        delay_idx = int(np.argmax(np.abs(corr))) - (len(sig1) - 1)
        delta_tau = abs(delay_idx / fs)
        dist_m    = round(delta_tau * self.c_acoustic, 2)

        if not self.node_list:
            return dist_m, ECO_DELTA_CENTER, None

        snode = self.node_list[sensor_node_idx % len(self.node_list)]

        try:
            hops  = max(1, int(dist_m / 50))
            paths = nx.single_source_shortest_path(
                self.G, snode, cutoff=hops
            )
            end_node = max(paths.keys(), key=lambda n: len(paths[n]))
            gps      = self.node_coords.get(end_node, ECO_DELTA_CENTER)
        except Exception:
            end_node = snode
            gps      = self.node_coords.get(snode, ECO_DELTA_CENTER)

        return dist_m, gps, end_node

    # ---------------------------------------------------------------- #
    #  Anomaly classification                                            #
    # ---------------------------------------------------------------- #

    def _classify(self, sensors):
        """
        Resample filtered mic1 to 512 bins and run MLP classifier.
        Returns (label, probability).
        Falls back to ('unknown', 0.0) if classifier not loaded.
        """
        if self.clf is None:
            return "unknown", 0.0

        fs  = int(sensors.get("fs", self.fs_default))
        sig = self._bandpass(sensors["mic1_sig"], fs=fs)

        feat = np.interp(
            np.linspace(0, len(sig) - 1, 512),
            np.arange(len(sig)),
            sig,
        ).reshape(1, -1)

        probs = self.clf.predict_proba(feat)[0]
        idx   = int(np.argmax(probs))
        return self.clf_encoder.classes_[idx], round(float(probs[idx]), 3)

    # ---------------------------------------------------------------- #
    #  Severity scoring + priority level                                 #
    # ---------------------------------------------------------------- #

    def _severity_score(self, result, sensors):
        """
        Severity = base_score * anomaly_confidence * type_weight * zone_weight

        type_weight:  burst=1.0, leak=0.75, drop=0.45, normal=0.10
        zone_weight:  inferred from flow rate
                      flow > 22 L/min = main trunk (1.0)
                      flow > 17       = branch     (0.6)
                      else            = lateral    (0.3)

        Priority thresholds:
          P1-CRITICAL >= 0.80
          P2-HIGH     >= 0.50
          P3-MODERATE >= 0.20
          P4-LOW       < 0.20
        """
        base       = min(1.0, result.get("score", 0) / 5.0)
        anom_prob  = float(result.get("anom_prob", 0.5))
        anomaly    = result.get("anomaly", "normal")
        flow       = float(sensors.get("flow", 13.0))

        type_weight = {
            "burst":  1.00,
            "leak":   0.75,
            "drop":   0.45,
            "normal": 0.10,
        }
        tw = type_weight.get(anomaly, 0.50)
        zw = 1.0 if flow > 22 else (0.6 if flow > 17 else 0.3)

        severity = round(base * anom_prob * tw * zw, 3)

        if severity >= 0.80:
            priority = "P1-CRITICAL"
        elif severity >= 0.50:
            priority = "P2-HIGH"
        elif severity >= 0.20:
            priority = "P3-MODERATE"
        else:
            priority = "P4-LOW"

        return severity, priority

    # ---------------------------------------------------------------- #
    #  Adaptive epsilon                                                  #
    # ---------------------------------------------------------------- #

    def _update_eps(self, confirmed):
        if confirmed:
            self.eps_current = max(0.02, self.eps_current - self.lr)
        else:
            self.eps_current = min(0.08, self.eps_current + self.lr * 0.5)

    # ---------------------------------------------------------------- #
    #  Main process() — called once per sensor reading                  #
    # ---------------------------------------------------------------- #

    def process(self, sensors, sensor_node_idx=0):
        """
        Full 4-tier pipeline.

        sensors dict expected keys:
          flow        float  L/min
          pressure    float  bar
          var_z       float  turbulence proxy
          mic1_sig    np.ndarray
          mic2_sig    np.ndarray
          fs          int    sample rate (default 10000)
          true_dist   float  optional — for error reporting in sims
          timestamp   str    optional
          regime      str    optional  'stress' | 'normal'

        Returns dict with flag='DISPATCH' or flag='MONITOR'.
        """
        score        = self._tier1(sensors)
        pvalue       = self._tier2(score)
        gate, psi    = self._tier3(score, pvalue)

        if gate:
            dist_m, gps, end_node       = self._tier4(sensors, sensor_node_idx)
            anom_label, anom_prob       = self._classify(sensors)
            self._update_eps(confirmed=True)

            true_dist = sensors.get("true_dist")
            error_m   = (round(abs(true_dist - dist_m), 2)
                         if true_dist is not None else None)

            result = {
                "flag":      "DISPATCH",
                "loc_m":     dist_m,
                "gps":       gps,
                "score":     score,
                "pvalue":    pvalue,
                "psi":       psi,
                "persist":   self.persist_count,
                "eps":       round(self.eps_current, 4),
                "end_node":  end_node,
                "anomaly":   anom_label,
                "anom_prob": anom_prob,
                "error_m":   error_m,
            }

            severity, priority  = self._severity_score(result, sensors)
            result["severity"]  = severity
            result["priority"]  = priority
            result["status"]    = (
                f"{priority} — {anom_label} "
                f"({anom_prob:.0%} confidence)"
            )
            return result

        self._update_eps(confirmed=False)
        return {
            "flag":    "MONITOR",
            "score":   score,
            "pvalue":  pvalue,
            "psi":     psi,
            "persist": self.persist_count,
            "eps":     round(self.eps_current, 4),
            "status":  "Monitoring — no confirmed leak",
        }


# -------------------------------------------------------------------- #
#  Sensor deployment calculator                                         #
# -------------------------------------------------------------------- #

def calculate_sensor_deployment(pipe_length_km, spacing_m=200):
    """
    Minimum sensor count for a pipe network.

    Rules:
      - 1 node per spacing_m on main trunk lines
      - Each node carries 2 piezo mics
      - 20% redundancy added for Eco-Delta saline / tidal conditions
    """
    base_nodes      = int((pipe_length_km * 1000) / spacing_m)
    redundancy      = int(base_nodes * 0.20)
    recommended     = base_nodes + redundancy
    total_mics      = recommended * 2

    return {
        "pipe_length_km":    pipe_length_km,
        "spacing_m":         spacing_m,
        "base_nodes":        base_nodes,
        "redundancy_nodes":  redundancy,
        "recommended_nodes": recommended,
        "total_mics":        total_mics,
        "coverage_m":        spacing_m,
    }


# -------------------------------------------------------------------- #
#  Self-test                                                            #
# -------------------------------------------------------------------- #

if __name__ == "__main__":
    brain = PULSE_AT_Brain()

    TRUE_LEAK = 45.0
    mic1, mic2 = brain._sim_piezo(TRUE_LEAK)

    sensors = {
        "flow":      20.0,
        "pressure":   2.8,
        "var_z":      3.2,
        "mic1_sig":  mic1,
        "mic2_sig":  mic2,
        "fs":        10000,
        "true_dist": TRUE_LEAK,
    }

    print("Running 5 cycles to trigger persistence gate...\n")
    for i in range(5):
        r = brain.process(sensors, sensor_node_idx=0)
        print(
            f"  Cycle {i+1}: {r['flag']} | "
            f"score={r['score']} | "
            f"persist={r['persist']} | "
            f"psi={r['psi']}"
        )
        if r["flag"] == "DISPATCH":
            print(f"\n  Status   : {r['status']}")
            print(f"  Location : {r['loc_m']} m")
            print(f"  GPS      : {r['gps']}")
            print(f"  Error    : {r['error_m']} m")
            print(f"  Severity : {r['severity']}")
            print(f"  Priority : {r['priority']}")
            break

    print("\n--- Sensor Deployment Plan ---")
    for km, label in [
        ( 8,  "Myeongji-dong core"),
        (20,  "Phase 1 mains"),
        (40,  "Full Eco-Delta"),
        (80,  "Gangseo-gu district"),
    ]:
        d = calculate_sensor_deployment(km, spacing_m=200)
        print(
            f"  {label:30s} "
            f"{d['recommended_nodes']:3d} nodes  "
            f"{d['total_mics']:3d} mics"
        )