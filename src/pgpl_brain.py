"""
src/pgpl_brain.py  —  PULSE-AT Brain: Tier 2 gate + Tier 3 classify + Tier 4 localise.

Claim 1 — Persistence gate: Persm = flagged_windows / N >= 0.67 (ratio, not unanimity)
Claim 1 — Severity: Sev = 0.4*conf + 0.3*typeW + 0.3*zone_weight
Tier 4  — Biot-Gassmann c(s) replaces hardcoded 1400 m/s
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
from collections import deque
from scipy.signal import correlate, butter, sosfilt

from src.biot_velocity import get_biot_velocity
from src.anomaly_classifier import classify, severity_score, LEAK_CLASSES


# ── Adaptive z-score ───────────────────────────────────────────────────────
class _AdaptiveZ:
    def __init__(self, window: int = 100):
        self._buf = deque(maxlen=window)

    def score(self, value: float) -> float:
        if len(self._buf) < 5:
            self._buf.append(value)
            return 0.0
        mu    = float(np.mean(self._buf))
        sigma = float(np.std(self._buf)) + 1e-9
        self._buf.append(value)
        return abs(value - mu) / sigma


# ── Mondrian conformal predictor ───────────────────────────────────────────
class _MondirianCP:
    def __init__(self):
        self._cal: list = []

    def calibrate(self, score: float) -> None:
        self._cal.append(score)

    def p_value(self, score: float) -> float:
        if len(self._cal) < 10:
            return 1.0
        return float(np.mean(np.array(self._cal) >= score))


# ── PSI drift detector ─────────────────────────────────────────────────────
class _PSIDrift:
    _BINS = np.linspace(0.0, 10.0, 11)

    def __init__(self, ref_window: int = 50):
        self._ref   = deque(maxlen=ref_window)
        self._cur   = deque(maxlen=ref_window)
        self._ready = False

    def update(self, z: float) -> float:
        if not self._ready:
            self._ref.append(z)
            if len(self._ref) == self._ref.maxlen:
                self._ready = True
            return 0.0
        self._cur.append(z)
        if len(self._cur) < 10:
            return 0.0
        r, _ = np.histogram(np.array(self._ref), bins=self._BINS, density=True)
        c, _ = np.histogram(np.array(self._cur),  bins=self._BINS, density=True)
        r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
        return float(np.sum((c - r) * np.log(c / r)))


# ── Main brain class ───────────────────────────────────────────────────────
class PULSE_AT_Brain:
    """
    4-tier PGL pipeline brain.
    Tier 1  Butterworth bandpass 200-800 Hz, 2 kHz Fs
    Tier 2  Adaptive z + Mondrian CP + PSI + ratio persistence gate  # Claim 1
    Tier 3  6-class heuristic classifier + P1-P4 severity scoring    # Claim 1 Eq.(1)
    Tier 4  TDOA/xcorr distance estimate using Biot-Gassmann c(s)
    """

    # Claim 1 — persistence ratio threshold (4 of 6 windows must flag)
    PERSIST_RATIO_THRESH = 0.67

    def __init__(
        self,
        alpha:         float = 0.10,
        persistence_n: int   = 6,      # paper spec: 6 events
        psi_threshold: float = 0.20,
        zone_weight:   float = 0.5,
    ):
        self.alpha         = alpha
        self.min_persist   = persistence_n
        self.psi_threshold = psi_threshold
        self.zone_weight   = zone_weight

        self._z        = _AdaptiveZ(window=100)
        self._cp       = _MondirianCP()
        self._psi      = _PSIDrift(ref_window=50)
        # Claim 1 — buffer holds exactly min_persist flags
        self._flag_buf = deque(maxlen=persistence_n)
        self._cal_n    = 0

        # Legacy attribute names kept for dashboard compatibility
        self.persist_count = 0
        self.psi_window    = deque(maxlen=50)

    # ── Tier 1: bandpass ────────────────────────────────────────────────────
    @staticmethod
    def _bandpass(sig: np.ndarray, fs: int = 2000) -> np.ndarray:
        sos = butter(4, [200 / (0.5 * fs), 800 / (0.5 * fs)],
                     btype="band", output="sos")
        return sosfilt(sos, sig)

    # ── Tier 4: distance estimators ─────────────────────────────────────────
    @staticmethod
    def _tdoa_distance(sig: np.ndarray, velocity: float, fs: int = 2000) -> float:
        env   = np.abs(sig)
        onset = int(np.argmax(env > 0.5 * np.max(env)))
        return float(np.clip(velocity * onset / fs, 0.5, 200.0))

    @staticmethod
    def _xcorr_distance(s1: np.ndarray, s2: np.ndarray,
                        velocity: float, fs: int = 2000) -> float:
        corr      = correlate(s1, s2, mode="full")
        delay_idx = int(np.argmax(np.abs(corr))) - (len(s1) - 1)
        return float(np.clip(abs(delay_idx / fs) * velocity, 0.0, 200.0))

    # ── Main process ─────────────────────────────────────────────────────────
    def process(self, sensors: dict, sensor_node_idx: int = 0) -> dict:
        """
        sensors keys
        ------------
        mic1_sig   np.ndarray  (required)
        mic2_sig   np.ndarray  (optional, enables xcorr localisation)
        fs         int         default 2000
        salinity   float       psu, default 7.0
        flow       float       L/s, default 13.0
        true_dist  float       ground-truth distance for MAE (optional)
        """
        fs         = int(sensors.get("fs", 2000))
        salinity   = float(sensors.get("salinity", 7.0))
        flow       = float(sensors.get("flow", 13.0))
        pressure_z = abs(flow - 13.0) / 2.0

        sig1     = self._bandpass(np.array(sensors["mic1_sig"]), fs)
        sig2     = (self._bandpass(np.array(sensors["mic2_sig"]), fs)
                    if "mic2_sig" in sensors else None)
        velocity = get_biot_velocity(salinity)   # Biot c(s), not hardcoded 1400

        # ── Tier 2 gate ──────────────────────────────────────────────────────
        # Use bandpass energy ratio (200-800 Hz / total) as anomaly feature.
        # Raw RMS is blind to leak type — leak signatures are spectral, not amplitude.
        from scipy.signal import welch
        freqs, psd = welch(sig1, fs=fs, nperseg=min(256, len(sig1)))
        total_power = float(np.sum(psd)) + 1e-12
        band_mask   = (freqs >= 100) & (freqs <= 600)
        band_power  = float(np.sum(psd[band_mask]))
        energy      = band_power / total_power   # ratio 0-1
        z           = self._z.score(energy)
        if self._cal_n < 100:
            self._cp.calibrate(z)
            self._cal_n += 1
        p_val = self._cp.p_value(z)
        psi   = self._psi.update(z)

        self._flag_buf.append(p_val <= self.alpha)

        # Claim 1 — Persistence gate: ratio >= 0.67, NOT all() unanimity
        # Matches paper: Persm = samples_above_thresh / N
        gated = (
            len(self._flag_buf) == self._flag_buf.maxlen and
            sum(self._flag_buf) / len(self._flag_buf) >= self.PERSIST_RATIO_THRESH
        )

        # Dashboard compat
        self.persist_count = int(p_val <= self.alpha)
        self.psi_window.append(psi)

        base = {
            "flag":        "MONITOR",
            "score":       round(z, 3),
            "psi":         round(psi, 4),
            "persist":     sum(self._flag_buf),
            "pvalue":      round(p_val, 4),
            "velocity_ms": round(velocity, 1),
        }

        if not gated:
            return base

        # ── Tier 3 classify ──────────────────────────────────────────────────
        event, conf, _ = classify(sig1, salinity, pressure_z, fs)

        # Claim 1 — Eq.(1) severity
        sev, tier = severity_score(conf, event, self.zone_weight)

        # ── Tier 4 localise ──────────────────────────────────────────────────
        dist = (self._xcorr_distance(sig1, sig2, velocity, fs)
                if sig2 is not None
                else self._tdoa_distance(sig1, velocity, fs))

        true_dist = sensors.get("true_dist")
        error     = round(abs(dist - true_dist), 2) if true_dist is not None else None

        # Approx GPS offset from EDC centre (35.135N, 128.970E)
        gps = (round(35.135 + dist * 9e-6, 6), round(128.970 + dist * 9e-6, 6))

        return {
            **base,
            "flag":      "DISPATCH",
            "anomaly":   event,
            "anom_prob": round(conf, 3),
            "severity":  sev,
            "priority":  tier,
            "loc_m":     round(dist, 2),
            "error_m":   error,
            "gps":       gps,
        }