"""
pgpl_brain.py  —  PULSE-AT Brain  (Full Debug Release)
=======================================================
Root-cause fixes
-----------------
BUG 1 — Nodes in water
  Old code: bilinear interpolation across bounding box → random nodes in river.
  Fix: OSMnx graph_from_polygon pulls only real road nodes on land.
       Fallback uses Shapely rejection-sampling inside EDC polygon → also land-only.

BUG 2 — process() signature mismatch
  Old: process(self, sensors) — TypeError when dashboard passed sensor_node_idx.
  Fix: process(self, sensors, sensor_node_idx=0).

BUG 3 — GPS snap returned centroid fallback always
  Old: node_coords was empty. Fix: populated from OSMnx node data (y=lat, x=lon).

BUG 4 — Multi-tier result fields missing
  Old brain returned only {flag, loc_m, z_score, status}.
  Fix: full dict on every call.

BUG 5 — calculate_sensor_deployment not defined in brain
  Fix: added here (dashboard imports it).
"""

import os, math
import numpy as np
import networkx as nx
from collections import deque
from shapely.geometry import Polygon, Point
from scipy.signal import correlate, butter, lfilter

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ── Verified EDC boundary ────────────────────────────────────────────────────
EDC_CORNERS_LATLON = [
    (35.163501, 128.906182),   # NW
    (35.157991, 128.937993),   # NE
    (35.111804, 128.931679),   # SE
    (35.120786, 128.899541),   # SW
]
EDC_LAT_MIN  = 35.111804
EDC_LAT_MAX  = 35.163501
EDC_LON_MIN  = 128.899541
EDC_LON_MAX  = 128.937993
EDC_CENTROID = (35.138521, 128.918849)

# Shapely polygon (lon, lat) for containment tests
_EDC_SHAPE = Polygon([(lon, lat) for lat, lon in EDC_CORNERS_LATLON])

C_ACOUSTIC = 1400.0
_GRAPH_N   = 200


# ── Graph builders ───────────────────────────────────────────────────────────

def _build_osmnx_graph():
    import osmnx as ox
    print("  Downloading OSM network for EDC polygon ...")
    osm_poly = Polygon([(lat, lon) for lat, lon in EDC_CORNERS_LATLON])
    G_osm = ox.graph_from_polygon(osm_poly, network_type="all",
                                  retain_all=True, truncate_by_edge=True)
    G_nx = nx.Graph(G_osm)
    node_coords = {}
    for nid, data in G_osm.nodes(data=True):
        lat = data.get("y", EDC_CENTROID[0])
        lon = data.get("x", EDC_CENTROID[1])
        if _EDC_SHAPE.contains(Point(lon, lat)):
            node_coords[nid] = (round(lat, 6), round(lon, 6))
    keep = set(node_coords.keys())
    G_nx = G_nx.subgraph(keep).copy()
    node_list = list(G_nx.nodes())
    print(f"  OSMnx graph: {len(node_list)} land nodes")
    return G_nx, node_list, node_coords


def _build_synthetic_graph(n=_GRAPH_N):
    """Rejection-sample inside EDC polygon — guaranteed land-only."""
    rng = np.random.default_rng(42)
    G   = nx.Graph()
    node_coords = {}
    idx = 0
    lat_r = EDC_LAT_MAX - EDC_LAT_MIN
    lon_r = EDC_LON_MAX - EDC_LON_MIN
    attempts = 0
    while idx < n and attempts < n * 50:
        attempts += 1
        lat = EDC_LAT_MIN + rng.uniform(0, lat_r)
        lon = EDC_LON_MIN + rng.uniform(0, lon_r)
        if _EDC_SHAPE.contains(Point(lon, lat)):
            G.add_node(idx)
            node_coords[idx] = (round(lat, 6), round(lon, 6))
            idx += 1
    coords_arr = np.array([node_coords[i] for i in range(idx)])
    for i in range(len(coords_arr)):
        dists = np.linalg.norm(coords_arr - coords_arr[i], axis=1)
        dists[i] = np.inf
        for j in np.argsort(dists)[:3]:
            G.add_edge(i, int(j))
    node_list = list(G.nodes())
    print(f"  Synthetic graph: {len(node_list)} land-only nodes")
    return G, node_list, node_coords


def _load_graph():
    try:
        return _build_osmnx_graph()
    except Exception as e:
        print(f"  OSMnx unavailable ({e}), using synthetic land-only graph")
        return _build_synthetic_graph(_GRAPH_N)


# ── Deployment calculator ────────────────────────────────────────────────────

def calculate_sensor_deployment(pipe_length_km=40.0, spacing_m=200.0):
    pipe_m      = pipe_length_km * 1000.0
    base        = int(math.ceil(pipe_m / spacing_m))
    redund      = int(math.ceil(base * 0.20))
    recommended = base + redund
    return {"pipe_length_km": pipe_length_km, "spacing_m": spacing_m,
            "coverage_m": spacing_m, "base_nodes": base,
            "redundancy_nodes": redund, "recommended_nodes": recommended,
            "total_mics": recommended * 2}


# ── Anomaly helpers ──────────────────────────────────────────────────────────

def _classify_anomaly(score, z):
    if score > 0.85 and z > 3.5: return "burst",         round(min(score*1.05,1.0),3)
    if score > 0.60:              return "leak",          round(score,3)
    if z > 2.5:                   return "pressure_drop", round(score*0.9,3)
    if score > 0.35:              return "slug",          round(score*0.7,3)
    return "normal", round(score*0.3,3)

def _severity_priority(anomaly, anom_prob):
    s = round(anom_prob,3)
    if anomaly == "burst" or s >= 0.80: return s, "P1-CRITICAL"
    if s >= 0.50: return s, "P2-HIGH"
    if s >= 0.20: return s, "P3-MODERATE"
    return s, "P4-LOW"


# ── Brain ────────────────────────────────────────────────────────────────────

class PULSE_AT_Brain:
    def __init__(self):
        print("PULSE-AT: Initialising PGPL Brain ...")
        self.G, self.node_list, self.node_coords = _load_graph()
        self.persist_count  = 0
        self.min_persist    = 3
        self.psi_window     = deque(maxlen=10)
        self._eps           = 2.0
        self._score_history = deque(maxlen=50)
        self.c_acoustic     = C_ACOUSTIC
        print(f"  {len(self.node_list)} pipe-graph nodes ready | centroid {EDC_CENTROID}")

    def _butter_bandpass(self, data, lowcut=100, highcut=1000, fs=10000, order=5):
        nyq  = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
        return lfilter(b, a, np.asarray(data, dtype=float))

    def _tdoa_distance(self, sig1, sig2, fs):
        f1 = self._butter_bandpass(sig1, fs=fs)
        f2 = self._butter_bandpass(sig2, fs=fs)
        corr = correlate(f1, f2, mode="full")
        delay_idx = int(np.argmax(np.abs(corr))) - (len(f1) - 1)
        return round(abs(delay_idx / fs) * self.c_acoustic, 2)

    def _snap_to_graph(self, sensor_node_idx, leak_dist_m):
        n = len(self.node_list)
        if n == 0:
            return EDC_CENTROID
        start_key = self.node_list[sensor_node_idx % n]
        try:
            edge_len_m   = 50.0
            target_hops  = max(1, int(round(leak_dist_m / edge_len_m)))
            visited      = {start_key}
            frontier     = [start_key]
            for _ in range(target_hops):
                nxt = []
                for node in frontier:
                    for nbr in self.G.neighbors(node):
                        if nbr not in visited:
                            visited.add(nbr)
                            nxt.append(nbr)
                if not nxt:
                    break
                frontier = nxt
            end_key = frontier[0] if frontier else start_key
        except Exception:
            end_key = start_key
        return self.node_coords.get(end_key, EDC_CENTROID)

    def _update_epsilon(self, score):
        self._score_history.append(score)
        if len(self._score_history) >= 10:
            mu = float(np.mean(self._score_history))
            sg = float(np.std(self._score_history)) + 1e-6
            self._eps = round(mu + 2.0*sg, 3)
        return self._eps

    def _psi(self, p_value):
        self.psi_window.append(p_value)
        return round(float(np.mean(self.psi_window)), 4)

    def _sim_piezo(self, true_dist_m, noise_sigma=0.05, fs=10000, n_samples=2048):
        rng   = np.random.default_rng()
        delay = int(round(true_dist_m / self.c_acoustic * fs))
        t     = np.linspace(0, n_samples/fs, n_samples)
        base  = np.sin(2*np.pi*400*t) * np.exp(-5*t)
        return (base + rng.normal(0, noise_sigma, n_samples),
                np.roll(base, delay) + rng.normal(0, noise_sigma, n_samples))

    def process(self, sensors: dict, sensor_node_idx: int = 0) -> dict:
        fs       = sensors.get("fs", 10000)
        flow     = float(sensors.get("flow", 13.0))
        pressure = float(sensors.get("pressure", 3.2))
        var_z    = float(sensors.get("var_z", 0.0))

        flow_z  = abs(flow - 13.0) / 2.0
        pres_z  = abs(pressure - 3.2) / 0.3
        varz_z  = abs(var_z) / 1.5
        z_score = round(float(np.mean([flow_z, pres_z, varz_z])), 4)
        score   = round(float(np.clip(z_score / 4.0, 0.0, 1.0)), 4)

        p_value = round(float(np.exp(-max(z_score, 0.0))), 4)
        psi_val = self._psi(p_value)
        eps     = self._update_epsilon(score)

        if z_score > eps:
            self.persist_count += 1
        else:
            self.persist_count = 0

        base = {"flag": "MONITOR", "score": score, "psi": psi_val,
                "eps": eps, "persist": self.persist_count, "pvalue": p_value,
                "anomaly": "normal", "anom_prob": 0.0, "severity": 0.0,
                "priority": "P4-LOW", "loc_m": None, "error_m": None, "gps": None}

        if self.persist_count < self.min_persist or psi_val > 0.80:
            return base

        mic1 = np.asarray(sensors.get("mic1_sig", [0.0]*512), dtype=float)
        mic2 = np.asarray(sensors.get("mic2_sig", [0.0]*512), dtype=float)
        leak_dist = self._tdoa_distance(mic1, mic2, fs)
        gps       = self._snap_to_graph(sensor_node_idx, leak_dist)
        true_dist = sensors.get("true_dist")
        error_m   = round(abs(leak_dist - true_dist), 2) if true_dist is not None else None
        anomaly,  anom_prob = _classify_anomaly(score, z_score)
        severity, priority  = _severity_priority(anomaly, anom_prob)

        return {"flag": "DISPATCH", "score": score, "psi": psi_val,
                "eps": eps, "persist": self.persist_count, "pvalue": p_value,
                "anomaly": anomaly, "anom_prob": anom_prob,
                "severity": severity, "priority": priority,
                "loc_m": leak_dist, "error_m": error_m, "gps": gps}