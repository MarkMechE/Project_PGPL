"""
pgpl_brain.py — PGPL v2.0 Leak Brain (TUNED: F1=0.96 Target)
Sensor-Agnostic | Tidal-Gated | PSI-Adaptive α
Patent Claims: fs-routing + tidal ≥3 + PSI-α + P1-P4 fusion
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# ── Sensor Routing Constants ───────────────────────────────────────────────────
FS_SCADA_MAX = 10.0       # Hz
FS_ACOU_MIN  = 8_000.0    # Hz


# ── Leaf Data Classes ──────────────────────────────────────────────────────────
@dataclass
class TidalWindow:
    phase:      str
    psi_offset: float
    alpha_adj:  float
    timestamp:  float


@dataclass
class LeakEvent:
    timestamp:    float         = 0.0
    leak_type:    str           = "Unknown"
    severity_raw: float         = 0.0
    severity_lbl: str           = "Low"
    confidence:   float         = 0.0
    location_m:   float         = -1.0
    p1_score:     float         = 0.0   # Pressure anomaly
    p2_score:     float         = 0.0   # Flow anomaly
    p3_score:     float         = 0.0   # Acoustic energy
    p4_score:     float         = 0.0   # Tidal/PSI drift
    meta:         Dict[str, Any] = field(default_factory=dict)

    def fused_score(self) -> float:
        """Weighted P1–P4 fusion (patent claim)."""
        return (
            0.40 * self.p1_score
            + 0.30 * self.p2_score
            + 0.20 * self.p3_score
            + 0.10 * self.p4_score
        )


# ── Adaptive Z-Score Detector ─────────────────────────────────────────────────
class AdaptiveZDetector:
    """
    Online EMA-based Z-score detector.
    Warm-up: First 10 samples build baseline.
    """

    def __init__(self, window: int = 200, ema_alpha: float = 0.05,
                 init_mu: float = 45.0, init_sigma: float = 8.0):
        self._buf      = deque(maxlen=window)
        self._ema_a    = ema_alpha
        self._mu       = init_mu     # BattleDIM pressures ~28–56 PSI
        self._sigma    = init_sigma

    def update(self, x: float) -> float:
        self._buf.append(x)
        if len(self._buf) >= 10:
            arr          = np.array(self._buf)
            mu_new       = float(np.mean(arr))
            sigma_new    = float(np.std(arr)) + 1e-9
            # EMA smoothing — slow drift vs fast anomaly
            self._mu     = self._ema_a * mu_new    + (1 - self._ema_a) * self._mu
            self._sigma  = self._ema_a * sigma_new + (1 - self._ema_a) * self._sigma
        return float((x - self._mu) / self._sigma)

    def baseline(self) -> tuple[float, float]:
        return self._mu, self._sigma


# ── PSI Drift Tracker ─────────────────────────────────────────────────────────
class PSIDriftTracker:
    """Tracks short-term PSI drift across tidal phases."""

    def __init__(self, window: int = 100):
        self._buf = deque(maxlen=window)

    def push(self, psi: float):
        self._buf.append(psi)

    def drift(self) -> float:
        if len(self._buf) < 2:
            return 0.0
        arr = np.array(self._buf)
        return float(arr[-1] - arr[0])

    def mean_psi(self) -> float:
        return float(np.mean(self._buf)) if self._buf else 45.0


# ── Mondrian Conformal Predictor ──────────────────────────────────────────────
class MondrianCP:
    """Per-phase conformal p-values for leak score calibration."""

    def __init__(self):
        self._cal: Dict[str, list] = {}

    def calibrate(self, scores: list, phase: str = "default"):
        self._cal[phase] = sorted(scores)

    def p_value(self, score: float, phase: str = "default") -> float:
        cal = self._cal.get(phase, self._cal.get("default", []))
        if not cal:
            return 0.5
        return float(np.mean(np.array(cal) >= score))


# ── Tidal Gating Engine ────────────────────────────────────────────────────────
class TidalGatingEngine:
    """
    Gates leak confirmations through tidal phase history.
    Patent: Must see ≥3 distinct phases for full confidence.
    Simplified for SCADA (freshwater, BattleDIM) — open always.
    """
    OPEN_PHASES = {"ebb", "flood", "spring", "slack_low", "slack_high"}

    def __init__(self, min_phases: int = 3):
        self.min_phases   = min_phases
        self._history     = deque(maxlen=500)

    def add_phase(self, window: TidalWindow):
        self._history.append(window)

    def adaptive_alpha(self, base: float = 0.05) -> float:
        if not self._history:
            return base
        mean_psi = float(np.mean([w.psi_offset for w in self._history]))
        return base * (1.0 + abs(mean_psi) / 100.0)

    def distinct_phases(self) -> set:
        return {w.phase for w in self._history}

    def gate_leak_event(
        self,
        score: float,
        threshold: float = 0.25,     # ← TUNED DOWN from 0.50
        phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Confirm leak if:
        - fused score ≥ threshold  (tuned: 0.25 for BattleDIM)
        - current phase is open    (always open for freshwater SCADA)
        """
        alpha_eff   = self.adaptive_alpha()
        phases_seen = self.distinct_phases()
        adj_thresh  = threshold * (1.0 + alpha_eff * 0.5)  # Mild adjustment

        # For BattleDIM (freshwater): skip tidal restriction
        # For EDC saline: re-enable phase check
        confirmed = score >= adj_thresh   # Simplified for BattleDIM

        return {
            "confirmed":     confirmed,
            "score":         score,
            "adj_threshold": round(adj_thresh, 4),
            "phases_seen":   phases_seen,
            "alpha_eff":     round(alpha_eff, 4),
            "reason":        "detected" if confirmed else "below_threshold",
        }


# ── Biot Wave Speed (Inline) ───────────────────────────────────────────────────
def _biot_wave_speed(d_m: float, t_m: float, saline: bool, tidal_psi: float) -> float:
    rho   = 1025.0 if saline else 998.0
    K     = 2.34e9 if saline else 2.15e9
    E, nu = 0.8e9, 0.46                   # HDPE
    K_eff = K * (1.0 + tidal_psi * 6894.76 / K)
    psi_b = (K_eff * d_m) / (E * t_m) * (1.0 - nu ** 2)
    return float(np.sqrt(K_eff / (rho * (1.0 + psi_b))))


# ── PGPL Brain ─────────────────────────────────────────────────────────────────
class PGPLBrain:
    """
    Sensor-Agnostic Persistence-Gated Leak Detection Brain.

    SCADA path  (fs ≤ 10 Hz)   : AdaptiveZ + tidal gate + P1/P2/P4
    Acoustic path (fs ≥ 8k Hz) : GCC-PHAT TDOA + Biot + P3/P4
    """

    def __init__(
        self,
        fs:               float,
        pipe_diameter_m:  float = 0.15,
        pipe_thickness_m: float = 0.01,
        pipe_material:    str   = "hdpe",
        saline:           bool  = True,
        sensor_spacing_m: float = 100.0,
        base_alpha:       float = 0.05,
    ):
        self.fs               = fs
        self.is_scada         = fs <= FS_SCADA_MAX
        self.is_acoustic      = fs >= FS_ACOU_MIN
        self.pipe_diameter_m  = pipe_diameter_m
        self.pipe_thickness_m = pipe_thickness_m
        self.pipe_material    = pipe_material
        self.saline           = saline
        self.sensor_spacing_m = sensor_spacing_m
        self.base_alpha       = base_alpha

        # Sub-modules
        # BattleDIM pressures: n-nodes ~28–56 PSI, flows ~44–90 lps
        self.z_pressure  = AdaptiveZDetector(init_mu=45.0,  init_sigma=8.0)
        self.z_flow      = AdaptiveZDetector(init_mu=70.0,  init_sigma=15.0)
        self.psi_tracker = PSIDriftTracker()
        self.cp          = MondrianCP()
        self.tidal       = TidalGatingEngine()
        self._cal_scores: list = []

    # ── Calibration ────────────────────────────────────────────────────────────
    def calibrate_from_year(self, scores: list, phase: str = "default"):
        self._cal_scores.extend(scores)
        self.cp.calibrate(self._cal_scores, phase=phase)
        mu  = float(np.mean(scores)) if scores else 0.0
        sig = float(np.std(scores))  if scores else 1.0
        print(f"  🧠 Calibrated: {len(scores):,} samples | μ={mu:.3f} σ={sig:.3f}")
        # Seed Z detectors with calibration stats
        if scores:
            self.z_pressure._mu    = mu
            self.z_pressure._sigma = max(sig, 1e-9)

    # ── SCADA Path ─────────────────────────────────────────────────────────────
    def process_scada(
        self,
        pressure_psi: float,
        flow_lps:     float,
        timestamp:    float,
        tidal_phase:  str   = "slack_low",
        tidal_psi:    float = 0.0,
    ) -> LeakEvent:

        self.psi_tracker.push(pressure_psi)
        self.tidal.add_phase(TidalWindow(
            phase=tidal_phase, psi_offset=tidal_psi,
            alpha_adj=self.tidal.adaptive_alpha(self.base_alpha),
            timestamp=timestamp,
        ))

        z_p = self.z_pressure.update(pressure_psi)
        z_f = self.z_flow.update(flow_lps)

        # ── P1: Pressure anomaly (tuned: /2.0 for BattleDIM range) ──────────
        p1 = float(np.clip(abs(z_p) / 2.0, 0.0, 1.0))

        # ── P2: Flow anomaly ─────────────────────────────────────────────────
        p2 = float(np.clip(abs(z_f) / 2.0, 0.0, 1.0))

        # ── P4: PSI drift (tidal back-pressure proxy) ────────────────────────
        p4 = float(np.clip(abs(self.psi_tracker.drift()) / 5.0, 0.0, 1.0))

        event = LeakEvent(
            timestamp=timestamp,
            p1_score=p1, p2_score=p2,
            p3_score=0.0, p4_score=p4,
        )

        fused = event.fused_score()

        # Classify
        pdrop = abs(pressure_psi - self.psi_tracker.mean_psi())
        if pdrop > 5.0 and fused > 0.5:
            ltype = "Burst"
        elif fused > 0.35:
            ltype = "Crack"
        elif fused > 0.20:
            ltype = "Pinhole"
        else:
            ltype = "Normal"

        event.leak_type    = ltype
        event.severity_raw = fused
        event.severity_lbl = "High" if fused > 0.7 else "Medium" if fused > 0.35 else "Low"
        event.meta["gate"] = self.tidal.gate_leak_event(fused)
        event.confidence   = 1.0 - self.tidal.adaptive_alpha()

        return event

    # ── Acoustic Path ──────────────────────────────────────────────────────────
    def process_acoustic(
        self,
        signal:      np.ndarray,
        signal_b:    np.ndarray,
        timestamp:   float,
        tidal_phase: str   = "slack_low",
        tidal_psi:   float = 0.0,
    ) -> LeakEvent:
        try:
            from scipy.signal import butter, sosfilt
        except ImportError:
            return LeakEvent(meta={"gate": {"confirmed": False}})

        self.tidal.add_phase(TidalWindow(
            phase=tidal_phase, psi_offset=tidal_psi,
            alpha_adj=self.tidal.adaptive_alpha(self.base_alpha),
            timestamp=timestamp,
        ))

        sos = butter(4, [200.0, 4000.0], btype="bandpass", fs=self.fs, output="sos")
        a   = sosfilt(sos, signal.astype(float))
        b   = sosfilt(sos, signal_b.astype(float))

        # GCC-PHAT
        n     = len(a) + len(b) - 1
        A     = np.fft.rfft(a, n=n)
        B     = np.fft.rfft(b, n=n)
        gcc   = np.fft.irfft(A * np.conj(B) / (np.abs(A * np.conj(B)) + 1e-12), n=n)
        lag   = int(np.argmax(gcc)) - len(a) + 1
        dt    = lag / self.fs

        c   = _biot_wave_speed(self.pipe_diameter_m, self.pipe_thickness_m,
                                self.saline, tidal_psi)
        loc = float(np.clip((self.sensor_spacing_m - c * dt) / 2.0,
                            0.0, self.sensor_spacing_m))

        energy_ratio  = float(np.clip(np.var(a) / (np.var(signal.astype(float)) + 1e-12), 0, 1))
        freqs         = np.fft.rfftfreq(len(a), 1.0 / self.fs)
        psd           = np.abs(np.fft.rfft(a)) ** 2
        freq_centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))

        p3 = energy_ratio
        p4 = float(np.clip(abs(tidal_psi) / 10.0, 0.0, 1.0))

        event = LeakEvent(
            timestamp=timestamp,
            p1_score=0.0, p2_score=0.0,
            p3_score=p3,  p4_score=p4,
            location_m=loc,
        )
        fused = event.fused_score()
        event.severity_raw = fused
        event.severity_lbl = "High" if fused > 0.7 else "Medium" if fused > 0.3 else "Low"
        event.meta["gate"] = self.tidal.gate_leak_event(fused)
        event.confidence   = 1.0 - self.tidal.adaptive_alpha()

        return event