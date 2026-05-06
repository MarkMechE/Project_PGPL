"""
pgpl_brain.py — Sensor-Agnostic Persistence-Gated Leak Brain
PGPL v2.0 | fs-routing: SCADA <10 Hz | Acoustic >8 kHz
Patent Claim: Unified gating + tidal fusion + PSI-adaptive α
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
from collections import deque
from scipy.signal import butter, sosfilt, correlate

from .biot_velocity      import biot_wave_speed, tdoa_distance
from .anomaly_classifier import classify_leak, score_severity, LeakEvent
from .tidal_gating       import TidalGatingEngine, TidalWindow

# ── Sensor Routing Constants ───────────────────────────────────────────────────
FS_SCADA_MAX  = 10.0     # Hz  — SCADA upper bound
FS_ACOU_MIN   = 8_000.0  # Hz  — acoustic lower bound

# ── Adaptive Z-Score Detector ─────────────────────────────────────────────────
class AdaptiveZDetector:
    """
    Online Z-score anomaly detector with exponential moving stats.
    Used for SCADA pressure/flow signals (fs < 10 Hz).
    """

    def __init__(self, window: int = 200, alpha: float = 0.05):
        self.window   = window
        self.alpha    = alpha           # EMA smoothing
        self._buf     = deque(maxlen=window)
        self._mu      = 0.0
        self._sigma   = 1.0

    def update(self, x: float) -> float:
        """Return z-score for new sample x."""
        self._buf.append(x)
        if len(self._buf) >= 10:
            arr         = np.array(self._buf)
            self._mu    = float(np.mean(arr))
            self._sigma = float(np.std(arr))  + 1e-9
        z = (x - self._mu) / self._sigma
        return float(z)


# ── Mondrian Conformal Predictor ──────────────────────────────────────────────
class MondrianCP:
    """
    Mondrian conformal prediction for leak score calibration.
    Produces valid p-values per tidal phase (stratum).
    """

    def __init__(self):
        self._cal: dict[str, list[float]] = {}

    def calibrate(self, scores: list[float], phase: str = "default"):
        self._cal[phase] = sorted(scores)

    def p_value(self, score: float, phase: str = "default") -> float:
        cal = self._cal.get(phase, self._cal.get("default", []))
        if not cal:
            return 0.5
        return float(np.mean(np.array(cal) >= score))


# ── PSI Drift Tracker ─────────────────────────────────────────────────────────
class PSIDriftTracker:
    """
    Tracks PSI baseline drift across tidal phases.
    Feeds adaptive α into TidalGatingEngine.
    """

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
        return float(np.mean(self._buf)) if self._buf else 0.0


# ── PGPL Brain ─────────────────────────────────────────────────────────────────
class PGPLBrain:
    """
    Main sensor-agnostic leak detection brain.

    Routing logic:
    ─ fs < 10 Hz     → SCADA path (AdaptiveZ on pressure/flow)
    ─ fs > 8000 Hz   → Acoustic path (bandpass + TDOA)
    ─ Both present   → Fused P1–P4 scoring

    Patent claims:
    1. fs-routing gate
    2. Tidal phase ≥3 confirmation
    3. PSI-adaptive α via MondrianCP
    4. Fused P1–P4 severity
    """

    def __init__(
        self,
        fs: float,
        pipe_diameter_m: float   = 0.15,
        pipe_thickness_m: float  = 0.01,
        pipe_material: str       = "hdpe",
        saline: bool             = True,
        sensor_spacing_m: float  = 100.0,
        base_alpha: float        = 0.05,
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
        self.z_pressure  = AdaptiveZDetector(window=200, alpha=base_alpha)
        self.z_flow      = AdaptiveZDetector(window=200, alpha=base_alpha)
        self.cp          = MondrianCP()
        self.psi_tracker = PSIDriftTracker()
        self.tidal       = TidalGatingEngine()

        # Cross-year calibration buffer (BattLeDIM 2018 → 2019)
        self._cal_scores: list[float] = []

    # ── SCADA Path ─────────────────────────────────────────────────────────────
    def process_scada(
        self,
        pressure_psi: float,
        flow_lps: float,
        timestamp: float,
        tidal_phase: str     = "slack_low",
        tidal_psi: float     = 0.0,
    ) -> LeakEvent:
        """Process one SCADA timestep. Returns LeakEvent."""
        if not self.is_scada:
            raise ValueError(f"fs={self.fs} Hz is not SCADA (need ≤ {FS_SCADA_MAX} Hz)")

        self.psi_tracker.push(pressure_psi)
        self.tidal.add_phase(TidalWindow(
            phase=tidal_phase,
            psi_offset=tidal_psi,
            alpha_adj=self.tidal.adaptive_alpha(self.base_alpha),
            timestamp=timestamp,
        ))

        z_p = self.z_pressure.update(pressure_psi)
        z_f = self.z_flow.update(flow_lps)

        p1 = float(np.clip(abs(z_p) / 5.0, 0, 1))  # pressure anomaly score
        p2 = float(np.clip(abs(z_f) / 5.0, 0, 1))  # flow anomaly score
        p4 = float(np.clip(abs(self.psi_tracker.drift()) / 10.0, 0, 1))  # tidal correlation

        event = LeakEvent(
            timestamp=timestamp,
            p1_score=p1,
            p2_score=p2,
            p3_score=0.0,   # no acoustic in SCADA path
            p4_score=p4,
        )

        fused = event.fused_score()
        leak_type, _ = classify_leak(
            energy_ratio=fused,
            freq_centroid_hz=0.0,
            pressure_drop_psi=abs(pressure_psi - self.psi_tracker.mean_psi()),
        )
        event.leak_type    = leak_type
        event.severity_raw = fused
        event.severity_lbl = score_severity(fused)

        # Gate through tidal filter
        gate = self.tidal.gate_leak_event(fused)
        event.meta["gate"]    = gate
        event.confidence      = 1.0 - gate["alpha_eff"]

        return event

    # ── Acoustic Path ──────────────────────────────────────────────────────────
    def process_acoustic(
        self,
        signal: np.ndarray,
        signal_b: np.ndarray,
        timestamp: float,
        tidal_phase: str  = "slack_low",
        tidal_psi: float  = 0.0,
    ) -> LeakEvent:
        """
        Process one acoustic window (two sensors A+B).
        Applies bandpass 200–4000 Hz → GCC-PHAT TDOA → Biot location.
        """
        if not self.is_acoustic:
            raise ValueError(f"fs={self.fs} Hz is not acoustic (need ≥ {FS_ACOU_MIN} Hz)")

        self.tidal.add_phase(TidalWindow(
            phase=tidal_phase,
            psi_offset=tidal_psi,
            alpha_adj=self.tidal.adaptive_alpha(self.base_alpha),
            timestamp=timestamp,
        ))

        # Bandpass 200–4000 Hz
        sos = butter(4, [200, 4000], btype="bandpass", fs=self.fs, output="sos")
        a   = sosfilt(sos, signal.astype(float))
        b   = sosfilt(sos, signal_b.astype(float))

        # GCC-PHAT TDOA
        n      = len(a) + len(b) - 1
        A      = np.fft.rfft(a, n=n)
        B      = np.fft.rfft(b, n=n)
        denom  = np.abs(A * np.conj(B)) + 1e-12
        gcc    = np.fft.irfft(A * np.conj(B) / denom, n=n)
        lag    = int(np.argmax(gcc)) - len(a) + 1
        dt_sec = lag / self.fs

        # Biot wave speed + TDOA location
        c   = biot_wave_speed(
            self.pipe_diameter_m,
            self.pipe_thickness_m,
            self.pipe_material,
            self.saline,
            tidal_psi,
        )
        loc = tdoa_distance(c, dt_sec, self.sensor_spacing_m)

        # Energy + frequency features
        energy_ratio   = float(np.var(a) / (np.var(signal.astype(float)) + 1e-12))
        energy_ratio   = np.clip(energy_ratio, 0, 1)
        freqs          = np.fft.rfftfreq(len(a), 1 / self.fs)
        psd            = np.abs(np.fft.rfft(a)) ** 2
        freq_centroid  = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))

        p3 = float(np.clip(energy_ratio, 0, 1))
        p4 = float(np.clip(abs(tidal_psi) / 10.0, 0, 1))

        event = LeakEvent(
            timestamp=timestamp,
            p1_score=0.0,   # no SCADA in acoustic path
            p2_score=0.0,
            p3_score=p3,
            p4_score=p4,
            location_m=loc,
        )

        fused = event.fused_score()
        leak_type, _ = classify_leak(
            energy_ratio=energy_ratio,
            freq_centroid_hz=freq_centroid,
            pressure_drop_psi=0.0,
        )
        event.leak_type    = leak_type
        event.severity_raw = fused
        event.severity_lbl = score_severity(fused)

        gate             = self.tidal.gate_leak_event(fused)
        event.meta["gate"]   = gate
        event.confidence     = 1.0 - gate["alpha_eff"]

        return event

    # ── Cross-Year Calibration ─────────────────────────────────────────────────
    def calibrate_from_year(self, scores: list[float], phase: str = "default"):
        """Feed 2018 calibration scores → MondrianCP."""
        self._cal_scores.extend(scores)
        self.cp.calibrate(self._cal_scores, phase=phase)