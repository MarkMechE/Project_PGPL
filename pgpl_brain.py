"""
pgpl_brain.py  —  PGL Tier 2 Gate
Persistence-Gated Piezo Leak Localizer  |  Eco-Delta City, Busan

Replaces:
  - fixed z > 2.0 threshold  →  adaptive rolling z-score
  - no CP                    →  Mondrian conformal p-value
  - no drift detection       →  PSI population stability index
  - c_acoustic = 1400        →  Biot-Gassmann c(s) via biot_velocity.py
  - scale_free_graph         →  removed (Phase 2: OSMnx real graph)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
from collections import deque
from scipy.signal import correlate, butter, sosfilt


# ── Biot velocity (inline fallback if biot_velocity.py absent) ────────────
try:
    from biot_velocity import get_biot_velocity
except ImportError:
    def get_biot_velocity(salinity_psu=7.0, **_):
        return float(np.clip(1450.0 + salinity_psu * 1.8, 1450, 1520))


# ══════════════════════════════════════════════════════════════════════════
# Sub-components
# ══════════════════════════════════════════════════════════════════════════

class AdaptiveZScore:
    """Rolling mean/std normaliser — no fixed threshold."""
    def __init__(self, window=100):
        self._buf = deque(maxlen=window)

    def score(self, value: float) -> float:
        if len(self._buf) < 5:
            self._buf.append(value)
            return 0.0
        mu = float(np.mean(self._buf))
        sigma = float(np.std(self._buf)) + 1e-9
        self._buf.append(value)
        return abs(value - mu) / sigma


class MondirianCP:
    """
    Mondrian conformal predictor.
    p-value = fraction of calibration scores >= current score.
    Low p-value (<=alpha) = anomalous.
    """
    def __init__(self):
        self._cal: list = []

    def calibrate(self, score: float) -> None:
        self._cal.append(score)

    def p_value(self, score: float) -> float:
        if len(self._cal) < 10:
            return 1.0
        return float(np.mean(np.array(self._cal) >= score))


class PSIDrift:
    """Population Stability Index — detects tidal-phase distribution shifts."""
    _BINS = np.linspace(0.0, 10.0, 11)

    def __init__(self, ref_window=50):
        self._ref = deque(maxlen=ref_window)
        self._cur = deque(maxlen=ref_window)
        self._ref_ready = False

    def update(self, z: float) -> float:
        if not self._ref_ready:
            self._ref.append(z)
            if len(self._ref) == self._ref.maxlen:
                self._ref_ready = True
            return 0.0
        self._cur.append(z)
        if len(self._cur) < 10:
            return 0.0
        r, _ = np.histogram(np.array(self._ref), bins=self._BINS, density=True)
        c, _ = np.histogram(np.array(self._cur), bins=self._BINS, density=True)
        r = np.clip(r, 1e-6, None)
        c = np.clip(c, 1e-6, None)
        return float(np.sum((c - r) * np.log(c / r)))


# ══════════════════════════════════════════════════════════════════════════
# Main brain class
# ══════════════════════════════════════════════════════════════════════════

class PULSE_AT_Brain:
    """
    4-tier PGL pipeline brain.

    Tier 1  —  bandpass filter (200-800 Hz, 2 kHz Fs)
    Tier 2  —  adaptive z-score + Mondrian CP + PSI drift + N-window persistence
    Tier 3  —  6-class heuristic classifier + severity scoring (P1-P4)
    Tier 4  —  TDOA distance estimate using Biot velocity c(s)
    """

    # 6 event classes + type weights
    TYPE_WEIGHTS = {
        "Burst":        1.0,
        "Crack":        0.8,
        "Micro":        0.6,
        "PressureDrop": 0.4,
        "Pump":         0.1,
        "Tidal":        0.0,
    }

    def __init__(self,
                 alpha: float = 0.10,
                 persistence_n: int = 3,
                 psi_threshold: float = 0.20,
                 zone_weight: float = 0.5):

        print("PGL Brain: initialising Persistence-Gated pipeline...")

        self.alpha          = alpha
        self.min_persist    = persistence_n
        self.psi_threshold  = psi_threshold
        self.zone_weight    = zone_weight

        # Tier 2 components
        self._z    = AdaptiveZScore(window=100)
        self._cp   = MondirianCP()
        self._psi  = PSIDrift(ref_window=50)
        self._flag_buf = deque(maxlen=persistence_n)
        self._cal_n    = 0

        # expose for dashboard compatibility
        self.persist_count = 0
        self.psi_window    = deque(maxlen=50)

    # ── Tier 1: bandpass filter ───────────────────────────────────────────
    @staticmethod
    def _bandpass(signal: np.ndarray, fs: int = 2000,
                  low: float = 200.0, high: float = 800.0) -> np.ndarray:
        sos = butter(4, [low / (0.5 * fs), high / (0.5 * fs)],
                     btype='band', output='sos')
        return sosfilt(sos, signal)

    # ── Tier 3: 6-class heuristic classifier ─────────────────────────────
    @staticmethod
    def _classify(signal: np.ndarray, salinity: float,
                  pressure_z: float = 0.0) -> tuple:
        """Returns (event_label, confidence)."""
        rms    = float(np.sqrt(np.mean(signal ** 2)))
        peak   = float(np.max(np.abs(signal)))
        crest  = peak / (rms + 1e-9)
        n      = len(signal)
        # crude narrowband proxy: ratio of even-indexed samples
        mid_r  = float(np.sum(np.abs(signal[::2])) /
                       (np.sum(np.abs(signal)) + 1e-9))
        rmsn   = rms / (np.std(signal) * 3 + 1e-9)   # normalised energy proxy
        pp     = (peak - float(np.min(np.abs(signal)))) / (rms + 1e-9)

        if   rmsn > 0.65 and pp > 1.2:
            return "Burst",        min(0.95, 0.75 + 0.04 * pp)
        elif rmsn > 0.35 and mid_r > 0.55 and crest < 1.8:
            return "Crack",        min(0.90, 0.60 + 0.25 * mid_r)
        elif rmsn > 0.20 and salinity < 9.0:
            return "Micro",        min(0.88, 0.55 + 0.25 * rmsn)
        elif rmsn > 0.15 and salinity >= 9.0:
            return "Micro",        min(0.72, 0.45 + 0.20 * rmsn)
        elif salinity > 7.0 and rmsn < 0.10 and pressure_z > 1.5:
            return "PressureDrop", 0.70
        elif pressure_z > 0.6:
            return "Pump",         min(0.80, 0.55 + 0.15 * pressure_z)
        else:
            return "Tidal",        max(0.50, 1.0 - rmsn * 5)

    # ── Severity + priority tier ──────────────────────────────────────────
    def _severity(self, conf: float, event: str) -> tuple:
        tw  = self.TYPE_WEIGHTS.get(event, 0.0)
        sev = round(0.4 * conf + 0.3 * tw + 0.3 * self.zone_weight, 3)
        if   sev >= 0.80: tier = "P1-CRITICAL"
        elif sev >= 0.60: tier = "P2-HIGH"
        elif sev >= 0.40: tier = "P3-MODERATE"
        else:             tier = "P4-LOW"
        return sev, tier

    # ── Tier 4: TDOA distance (single-sensor proxy) ───────────────────────
    @staticmethod
    def _tdoa_distance(signal: np.ndarray,
                       velocity: float, fs: int = 2000) -> float:
        env      = np.abs(signal)
        onset    = int(np.argmax(env > 0.5 * np.max(env)))
        prop_s   = onset / fs
        return float(np.clip(velocity * prop_s, 0.5, 200.0))

    # ── Two-sensor cross-correlation distance (when mic2 available) ───────
    @staticmethod
    def _xcorr_distance(sig1: np.ndarray, sig2: np.ndarray,
                        velocity: float, fs: int = 2000) -> float:
        corr      = correlate(sig1, sig2, mode='full')
        delay_idx = int(np.argmax(np.abs(corr))) - (len(sig1) - 1)
        delta_tau = abs(delay_idx / fs)
        return float(np.clip(delta_tau * velocity, 0.0, 200.0))

    # ── Main process entry point ──────────────────────────────────────────
    def process(self, sensors: dict, sensor_node_idx: int = 0) -> dict:
        """
        sensors keys (all optional except mic1_sig):
            mic1_sig    : np.ndarray  (required)
            mic2_sig    : np.ndarray  (optional — enables 2-sensor xcorr)
            fs          : int         (default 2000)
            salinity    : float       (psu, default 7.0)
            flow        : float       (L/s, used as pressure_z proxy)
            pressure    : float
            true_dist   : float       (for MAE when ground truth known)
        """
        fs         = int(sensors.get('fs', 2000))
        salinity   = float(sensors.get('salinity', 7.0))
        flow       = float(sensors.get('flow', 13.0))
        pressure_z = abs(flow - 13.0) / 2.0     # simple z proxy

        # Tier 1 — filter
        sig1 = self._bandpass(np.array(sensors['mic1_sig']), fs)
        sig2 = (self._bandpass(np.array(sensors['mic2_sig']), fs)
                if 'mic2_sig' in sensors else None)

        # Tier 1 — Biot velocity
        velocity = get_biot_velocity(salinity)

        # Tier 2 — adaptive gate
        energy = float(np.sqrt(np.mean(sig1 ** 2)))
        z      = self._z.score(energy)

        if self._cal_n < 100:
            self._cp.calibrate(z)
            self._cal_n += 1

        p_val = self._cp.p_value(z)
        psi   = self._psi.update(z)

        is_anomalous = (p_val <= self.alpha)
        self._flag_buf.append(is_anomalous)
        gated = (len(self._flag_buf) == self.min_persist
                 and all(self._flag_buf))

        # expose for dashboard
        self.persist_count = int(is_anomalous)
        self.psi_window.append(psi)

        score = round(float(p_val <= self.alpha) * z / max(z, 1), 3)
        eps   = round(self.alpha, 3)

        base = {
            'flag':    'MONITOR',
            'score':   round(z, 3),
            'psi':     round(psi, 4),
            'persist': sum(self._flag_buf),
            'pvalue':  round(p_val, 4),
            'eps':     eps,
            'velocity_ms': round(velocity, 1),
        }

        if not gated:
            return base

        # Tier 3 — classify
        event, conf = self._classify(sig1, salinity, pressure_z)
        sev, tier   = self._severity(conf, event)

        # Tier 4 — localise
        if sig2 is not None:
            dist = self._xcorr_distance(sig1, sig2, velocity, fs)
        else:
            dist = self._tdoa_distance(sig1, velocity, fs)

        true_dist = sensors.get('true_dist')
        error     = round(abs(dist - true_dist), 2) if true_dist else None

        # Eco-Delta City approx GPS centre
        lat0, lon0 = 35.135, 128.970
        gps = (round(lat0 + dist * 0.000009, 6),
               round(lon0 + dist * 0.000009, 6))

        return {
            **base,
            'flag':      'DISPATCH',
            'anomaly':   event,
            'anom_prob': round(conf, 3),
            'severity':  sev,
            'priority':  tier,
            'loc_m':     round(dist, 2),
            'error_m':   error,
            'gps':       gps,
        }


# ── Sensor deployment helper (used by dashboard) ──────────────────────────
def calculate_sensor_deployment(pipe_length_km: float,
                                spacing_m: int = 200) -> dict:
    base  = int(np.ceil(pipe_length_km * 1000 / spacing_m))
    redun = int(np.ceil(base * 0.20))
    rec   = base + redun
    return {
        'pipe_length_km':   pipe_length_km,
        'spacing_m':        spacing_m,
        'base_nodes':       base,
        'redundancy_nodes': redun,
        'recommended_nodes': rec,
        'total_mics':       rec * 2,
        'coverage_m':       spacing_m,
    }


# ── Standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    brain = PULSE_AT_Brain()

    # warm-up: 120 normal windows
    rng = np.random.default_rng(0)
    for _ in range(120):
        brain.process({
            'mic1_sig': rng.normal(0, 0.03, 2000),
            'salinity': float(rng.uniform(3, 10)),
        })

    # test: 5 leak windows
    print("\n--- Leak detection test ---")
    for i in range(5):
        leak_sig = rng.normal(0, 0.03, 2000)
        leak_sig[800:850] += rng.uniform(0.3, 0.8, 50)   # inject burst
        result = brain.process({
            'mic1_sig':  leak_sig,
            'mic2_sig':  leak_sig * 0.9 + rng.normal(0, 0.005, 2000),
            'salinity':  float(rng.uniform(3, 10)),
            'flow':      float(rng.uniform(16, 22)),
            'true_dist': float(rng.uniform(5, 40)),
        })
        print(f"  [{i+1}] flag={result['flag']:8s}  "
              f"priority={result.get('priority','-'):12s}  "
              f"event={result.get('anomaly','-'):12s}  "
              f"dist={result.get('loc_m','-')} m  "
              f"v={result['velocity_ms']} m/s")

    print("\n✅ PGL Brain ready.")