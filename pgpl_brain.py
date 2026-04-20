import os, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import correlate, butter, lfilter
from collections import deque

ECO_DELTA_CENTER = (35.135, 128.970)


class PULSE_AT_Brain:

    def __init__(self,
                 graph_path="data/eco_delta_graph.pkl",
                 nodes_path="data/eco_delta_nodes.csv",
                 clf_path="outputs/anomaly_classifier.pkl"):

        print("PULSE-AT: Initializing...")

        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                self.G = pickle.load(f)
            print(f"  Graph: {len(self.G.nodes)} nodes")
        else:
            print("  WARNING: Run eco_delta_map.py first. Using fallback graph.")
            self.G = nx.scale_free_graph(n=200, seed=42).to_undirected()
            self.G = nx.Graph(self.G)

        if os.path.exists(nodes_path):
            df = pd.read_csv(nodes_path)
            self.node_coords = {
                int(r["node_id"]): (r["lat"], r["lon"])
                for _, r in df.iterrows()
            }
        else:
            self.node_coords = {}

        self.node_list = list(self.G.nodes())

        self.clf = None
        self.clf_encoder = None
        if os.path.exists(clf_path):
            with open(clf_path, "rb") as f:
                bundle = pickle.load(f)
            self.clf = bundle["model"]
            self.clf_encoder = bundle["encoder"]
            print(f"  Classifier: {list(self.clf_encoder.classes_)}")
        else:
            print("  WARNING: Run anomaly_classifier.py first.")

        self.persist_count = 0
        self.min_persist   = 3
        self.psi_window    = deque(maxlen=30)
        self.score_history = deque(maxlen=150)
        self.c_acoustic    = 1400
        self.fs_default    = 10000
        self.eps_current   = 0.05
        self.lr            = 0.003
        print("  Ready.\n")

    def _bandpass(self, data, lo=100, hi=1000, fs=10000, order=5):
        nyq  = 0.5 * fs
        low  = max(lo / nyq, 1e-4)
        high = min(hi / nyq, 1 - 1e-4)
        b, a = butter(order, [low, high], btype="band")
        return lfilter(b, a, data)

    def _sim_piezo(self, leak_dist_m, fs=10000, noise_sigma=0.05):
        t       = np.linspace(0, 1, fs)
        mic1    = np.sin(2 * np.pi * 150 * t) + noise_sigma * np.random.randn(fs)
        delay_s = leak_dist_m / (self.c_acoustic * 2)
        mic2    = np.roll(mic1, int(delay_s * fs)) + noise_sigma * np.random.randn(fs)
        return mic1, mic2

    def _tier1(self, sensors):
        z_flow = abs(sensors.get("flow", 13.0) - 13.0) / 2.0
        z_pres = abs(sensors.get("pressure", 3.2) - 3.2) / 0.5
        z_var  = sensors.get("var_z", 0.0)
        anti   = max(0.0, (z_flow - z_pres) * 0.5)
        return round(0.40 * z_flow + 0.30 * z_pres + 0.30 * z_var + 0.50 * anti, 4)

    def _tier2(self, score):
        if len(self.score_history) < 10:
            return 1.0
        n     = len(self.score_history)
        n_geq = sum(1 for s in self.score_history if s >= score)
        return round((n_geq + 1) / (n + 1), 4)

    def _tier3(self, score, pvalue):
        self.psi_window.append(score)
        self.score_history.append(score)
        psi = (round(float(np.std(list(self.psi_window)) /
               (np.mean(list(self.psi_window)) + 1e-6)), 4)
               if len(self.psi_window) >= 10 else 0.0)
        if score > 2.5 and pvalue < 0.10:
            self.persist_count += 1
        else:
            self.persist_count = max(0, self.persist_count - 1)
        gate = self.persist_count >= self.min_persist and psi >= 0.20
        return gate, psi

    def _tier4(self, sensors, sensor_node_idx=0):
        fs   = sensors.get("fs", self.fs_default)
        sig1 = self._bandpass(sensors["mic1_sig"], fs=fs)
        sig2 = self._bandpass(sensors["mic2_sig"], fs=fs)
        corr = correlate(sig1, sig2, mode="full")
        idx  = int(np.argmax(np.abs(corr))) - (len(sig1) - 1)
        dist = abs(idx / fs) * self.c_acoustic

        if not self.node_list:
            return round(dist, 2), ECO_DELTA_CENTER, None

        snode = self.node_list[sensor_node_idx % len(self.node_list)]
        try:
            hops  = max(1, int(dist / 50))
            paths = nx.single_source_shortest_path(self.G, snode, cutoff=hops)
            enode = max(paths.keys(), key=lambda n: len(paths[n]))
            gps   = self.node_coords.get(enode, ECO_DELTA_CENTER)
        except Exception:
            gps   = self.node_coords.get(snode, ECO_DELTA_CENTER)
            enode = snode
        return round(dist, 2), gps, enode

    def _classify(self, sensors):
        if self.clf is None:
            return "unknown", 0.0
        fs   = sensors.get("fs", self.fs_default)
        sig  = self._bandpass(sensors["mic1_sig"], fs=fs)
        feat = np.interp(
            np.linspace(0, len(sig) - 1, 512),
            np.arange(len(sig)), sig
        ).reshape(1, -1)
        probs = self.clf.predict_proba(feat)[0]
        idx   = int(np.argmax(probs))
        return self.clf_encoder.classes_[idx], round(float(probs[idx]), 3)

    def _severity_score(self, result, sensors):
        base_score  = min(1.0, result.get("score", 0) / 5.0)
        anom_prob   = result.get("anom_prob", 0.5)
        anomaly     = result.get("anomaly", "normal")

        type_weight = {"burst": 1.0, "leak": 0.75, "drop": 0.45, "normal": 0.1}
        tw          = type_weight.get(anomaly, 0.5)

        flow        = sensors.get("flow", 13.0)
        zone_weight = 1.0 if flow > 22 else (0.6 if flow > 17 else 0.3)

        severity = round(base_score * anom_prob * tw * zone_weight, 3)

        if severity >= 0.80:
            priority = "P1-CRITICAL"
        elif severity >= 0.50:
            priority = "P2-HIGH"
        elif severity >= 0.20:
            priority = "P3-MODERATE"
        else:
            priority = "P4-LOW"

        return severity, priority

    def _update_eps(self, confirmed):
        if confirmed:
            self.eps_current = max(0.02, self.eps_current - self.lr)
        else:
            self.eps_current = min(0.08, self.eps_current + self.lr * 0.5)

    def process(self, sensors, sensor_node_idx=0):
        score        = self._tier1(sensors)
        pvalue       = self._tier2(score)
        gate, psi    = self._tier3(score, pvalue)

        if gate:
            dist, gps, enode        = self._tier4(sensors, sensor_node_idx)
            anom_label, anom_prob   = self._classify(sensors)
            self._update_eps(True)

            true_dist = sensors.get("true_dist")
            result = {
                "flag":      "DISPATCH",
                "loc_m":     dist,
                "gps":       gps,
                "score":     score,
                "pvalue":    pvalue,
                "psi":       psi,
                "persist":   self.persist_count,
                "eps":       round(self.eps_current, 4),
                "end_node":  enode,
                "anomaly":   anom_label,
                "anom_prob": anom_prob,
                "error_m":   round(abs(true_dist - dist), 2) if true_dist else None,
            }
            severity, priority      = self._severity_score(result, sensors)
            result["severity"]      = severity
            result["priority"]      = priority
            result["status"]        = f"{priority} — {anom_label} ({anom_prob:.0%})"
            return result

        self._update_eps(False)
        return {
            "flag":    "MONITOR",
            "score":   score,
            "pvalue":  pvalue,
            "psi":     psi,
            "persist": self.persist_count,
            "eps":     round(self.eps_current, 4),
            "status":  "Monitoring",
        }


if __name__ == "__main__":
    brain = PULSE_AT_Brain()
    mic1, mic2 = brain._sim_piezo(45.0)
    s = {
        "flow": 20.0, "pressure": 2.8, "var_z": 3.2,
        "mic1_sig": mic1, "mic2_sig": mic2,
        "fs": 10000, "true_dist": 45.0,
    }
    print("Running 5 cycles...\n")
    for i in range(5):
        r = brain.process(s)
        print(f"  Cycle {i+1}: {r['flag']} | score={r['score']} | persist={r['persist']}")
        if r["flag"] == "DISPATCH":
            print(f"\n  {r['status']}")
            print(f"  Location: {r['loc_m']} m | GPS: {r['gps']}")
            print(f"  Severity: {r['severity']} | Priority: {r['priority']}")
            break