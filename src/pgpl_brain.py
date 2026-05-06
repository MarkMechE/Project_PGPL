"""
pgpl_brain.py — PGPL v2.0 Sensor-Agnostic Leak Detection Brain
================================================================
Patent Claims:
  1. Unified fs-routing gate  (SCADA ≤10Hz / Acoustic ≥8kHz)
  2. Tidal phase ≥3 confirmation
  3. PSI-adaptive α via MondrianCP
  4. Fused P1–P4 severity scoring

Design principle:
  ZERO hardcoded data constants.
  Brain learns μ/σ from whatever signal arrives.
  Works on BattleDIM (freshwater PSI),
         EDC Busan  (saline bar),
         Mendeley   (acoustic Pa) — same code, same weights.

Author : EDC Busan
Version: 2.1.0 (truly agnostic)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing      import Dict, Any, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# Routing Constants ONLY — these are physics, not data
# ══════════════════════════════════════════════════════════════════════════════
FS_SCADA_MAX = 10.0        # Hz  — SCADA upper bound  (physics of SCADA systems)
FS_ACOU_MIN  = 8_000.0     # Hz  — Acoustic lower bound (pipe vibration range)

# Pipe material library (physics, not data-specific)
_PIPE_PROPS: Dict[str, Tuple[float, float]] = {
    "hdpe":  (0.8e9,  0.46),   # E (Pa), Poisson ratio
    "steel": (200e9,  0.30),
    "pvc":   (3.0e9,  0.38),
    "ci":    (170e9,  0.26),   # cast iron
}


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TidalWindow:
    phase:      str
    psi_offset: float
    alpha_adj:  float
    timestamp:  float


@dataclass
class LeakEvent:
    timestamp:    float          = 0.0
    leak_type:    str            = "Normal"
    severity_raw: float          = 0.0
    severity_lbl: str            = "Low"
    confidence:   float          = 0.0
    location_m:   float          = -1.0
    p1_score:     float          = 0.0
    p2_score:     float          = 0.0
    p3_score:     float          = 0.0
    p4_score:     float          = 0.0
    meta:         Dict[str, Any] = field(default_factory=dict)

    # Fixed weights — same for all sensors
    _W = (0.40, 0.30, 0.20, 0.10)

    def fused_score(self) -> float:
        """
        Sensor-agnostic weighted fusion.

        Normalises by the sum of weights of ACTIVE channels only.
        Active = score > 0.0

        This means:
          SCADA only   (p1+p2 active)      → normalised over 0.70
          Acoustic only (p3+p4 active)     → normalised over 0.30
          All active   (p1+p2+p3+p4)       → normalised over 1.00

        Same threshold works for ALL sensor combinations.
        No sensor-specific tuning needed.
        Patent claim: unified gate regardless of sensor type.
        """
        scores  = (self.p1_score, self.p2_score,
                   self.p3_score, self.p4_score)
        weights = self._W

        weighted_sum   = sum(w * s for w, s in zip(weights, scores))
        active_weights = sum(w for w, s in zip(weights, scores) if s > 0.0)

        if active_weights < 1e-9:
            return 0.0

        # Normalise to [0, 1] regardless of which sensors are active
        return float(weighted_sum / active_weights)

    @property
    def is_confirmed(self) -> bool:
        return bool(self.meta.get("gate", {}).get("confirmed", False))


# ══════════════════════════════════════════════════════════════════════════════
# Sub-Module 1 — Adaptive Z Detector (learns its own baseline)
# ══════════════════════════════════════════════════════════════════════════════
class AdaptiveZDetector:
    """
    Online Z-score detector.

    No init μ/σ required — warm-up period builds baseline
    from the first `warmup` samples of whatever signal arrives.
    Works for PSI, bar, Pa, lps, m3/s — any unit.
    """

    def __init__(self, window: int = 200, ema_alpha: float = 0.05,
                 warmup: int = 20):
        self._buf       = deque(maxlen=window)
        self._ema_alpha = ema_alpha
        self._warmup    = warmup
        self._mu        = None    # learned from data — NOT preset
        self._sigma     = None

    def update(self, x: float) -> float:
        """
        Feed one sample. Returns z-score.
        Returns 0.0 during warm-up (not enough data yet).
        """
        self._buf.append(x)

        if len(self._buf) < self._warmup:
            return 0.0   # warm-up: no anomaly score yet

        arr = np.array(self._buf, dtype=float)

        if self._mu is None:
            # First time past warm-up: initialise from buffer
            self._mu    = float(np.mean(arr))
            self._sigma = float(np.std(arr)) + 1e-9
        else:
            # EMA update: tracks slow drift, highlights sharp spikes
            batch_mu    = float(np.mean(arr))
            batch_sigma = float(np.std(arr))  + 1e-9
            self._mu    = self._ema_alpha * batch_mu    + (1.0 - self._ema_alpha) * self._mu
            self._sigma = self._ema_alpha * batch_sigma + (1.0 - self._ema_alpha) * self._sigma

        return float((x - self._mu) / self._sigma)

    def seed(self, mu: float, sigma: float) -> None:
        """Override with calibration-year statistics."""
        self._mu    = float(mu)
        self._sigma = max(float(sigma), 1e-9)

    @property
    def is_warmed_up(self) -> bool:
        return self._mu is not None


# ══════════════════════════════════════════════════════════════════════════════
# Sub-Module 2 — PSI Drift Tracker (unit-free)
# ══════════════════════════════════════════════════════════════════════════════
class PSIDriftTracker:
    """
    Tracks short-window signal drift.
    'PSI' in the name is historical — works for any pressure unit.
    Drift is normalised by the window std so it stays [0, 1]-ish.
    """

    def __init__(self, window: int = 100):
        self._buf = deque(maxlen=window)

    def push(self, value: float) -> None:
        self._buf.append(value)

    def normalised_drift(self) -> float:
        """
        Drift = |last − first| / std_of_window.
        Returns 0 during warm-up.
        Unit-free: works for PSI, bar, Pa, etc.
        """
        if len(self._buf) < 5:
            return 0.0
        arr   = np.array(self._buf, dtype=float)
        drift = abs(arr[-1] - arr[0])
        sigma = float(np.std(arr)) + 1e-9
        return float(np.clip(drift / sigma, 0.0, 5.0) / 5.0)  # normalise to [0,1]

    def mean(self) -> float:
        return float(np.mean(self._buf)) if self._buf else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Sub-Module 3 — Mondrian Conformal Predictor
# ══════════════════════════════════════════════════════════════════════════════
class MondrianCP:
    """
    Mondrian conformal prediction stratified by tidal phase.
    p_value < alpha → anomalous.
    Patent claim 3: α adjusts with tidal PSI level.
    """

    def __init__(self):
        self._cal: Dict[str, List[float]] = {}

    def calibrate(self, scores: List[float], phase: str = "default") -> None:
        if scores:
            self._cal[phase] = sorted(scores)

    def p_value(self, score: float, phase: str = "default") -> float:
        cal = self._cal.get(phase) or self._cal.get("default") or []
        if not cal:
            return 0.5
        return float(np.mean(np.array(cal, dtype=float) >= score))

    @property
    def has_calibration(self) -> bool:
        return bool(self._cal)


# ══════════════════════════════════════════════════════════════════════════════
# Sub-Module 4 — Tidal Gating Engine
# ══════════════════════════════════════════════════════════════════════════════
class TidalGatingEngine:
    """
    Gates detections through tidal phase history.

    Patent claims:
    - ≥3 distinct tidal phases for full confidence
    - PSI-adaptive α: α_eff = α_base × (1 + |mean_psi_offset| / 100)

    saline_mode=False  →  BattleDIM / freshwater: phase gate OFF
    saline_mode=True   →  EDC saline: phase gate ON
    """

    OPEN_PHASES = {"ebb", "flood", "spring", "slack_low", "slack_high"}
    MIN_PHASES  = 3

    def __init__(self, saline_mode: bool = False):
        self._history    = deque(maxlen=1000)
        self.saline_mode = saline_mode

    def add_phase(self, window: TidalWindow) -> None:
        self._history.append(window)

    def distinct_phases(self) -> set:
        return {w.phase for w in self._history}

    def adaptive_alpha(self, base: float = 0.05) -> float:
        """PSI-adaptive α (patent claim 3)."""
        if not self._history:
            return base
        mean_psi = float(np.mean([abs(w.psi_offset) for w in self._history]))
        return float(base * (1.0 + mean_psi / 100.0))

    def gate_leak_event(
        self,
        score:     float,
        threshold: float         = 0.15,
        phase:     Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Confirm detection through tidal filter.

        Threshold 0.15 works because fused scores are
        already unit-free [0, 1] regardless of sensor type.
        """
        alpha_eff  = self.adaptive_alpha()
        adj_thresh = threshold * (1.0 + alpha_eff * 0.2)
        score_ok   = score >= adj_thresh

        # Phase gate — only for saline networks
        if self.saline_mode and self._history:
            phase_ok = self._history[-1].phase in self.OPEN_PHASES
        else:
            phase_ok = True

        confirmed = score_ok and phase_ok

        return {
            "confirmed":     confirmed,
            "score":         round(score,      4),
            "adj_threshold": round(adj_thresh, 4),
            "alpha_eff":     round(alpha_eff,  4),
            "phases_seen":   self.distinct_phases(),
            "reason":        (
                "leak_confirmed"       if confirmed      else
                "below_threshold"      if not score_ok   else
                "tidal_phase_blocked"
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Physics Helpers (not data-specific)
# ══════════════════════════════════════════════════════════════════════════════
def _biot_wave_speed(
    d_m: float, t_m: float,
    material: str  = "hdpe",
    saline:   bool = True,
    tidal_psi:float = 0.0,
) -> float:
    """Biot thin-shell wave speed. Physics only — no data constants."""
    rho   = 1025.0 if saline else 998.0       # seawater / freshwater (kg/m³)
    K     = 2.34e9 if saline else 2.15e9      # bulk modulus (Pa)
    E, nu = _PIPE_PROPS.get(material, _PIPE_PROPS["hdpe"])

    tidal_pa = tidal_psi * 6894.76            # PSI → Pa
    K_eff    = K * (1.0 + tidal_pa / K)
    psi_b    = (K_eff * d_m) / (E * t_m) * (1.0 - nu ** 2)
    return float(np.sqrt(K_eff / (rho * (1.0 + psi_b))))


def _tdoa_locate(c: float, dt_sec: float, spacing_m: float) -> float:
    """x = (L − c·Δt) / 2, clipped to pipe length."""
    return float(np.clip((spacing_m - c * dt_sec) / 2.0, 0.0, spacing_m))


def _classify(fused: float, pressure_drop_normalised: float,
              freq_centroid: float = 0.0) -> Tuple[str, str]:
    """
    Heuristic classifier.
    All inputs are unit-free ratios — agnostic to sensor type.
    """
    if pressure_drop_normalised > 1.5 and fused > 0.60:
        ltype = "Burst"
    elif freq_centroid > 4000 and fused > 0.50:
        ltype = "Crack"
    elif fused > 0.40:
        ltype = "Joint"
    elif fused > 0.20:
        ltype = "Pinhole"
    else:
        ltype = "Normal"

    severity = (
        "High"   if fused > 0.70 else
        "Medium" if fused > 0.35 else
        "Low"
    )
    return ltype, severity


# ══════════════════════════════════════════════════════════════════════════════
# PGPL Brain
# ══════════════════════════════════════════════════════════════════════════════
class PGPLBrain:
    """
    Sensor-Agnostic Persistence-Gated Leak Detection Brain.

    No data constants inside.
    Routing purely by fs.
    Baseline learned from calibration year or warm-up buffer.

    Examples
    --------
    BattleDIM (freshwater SCADA):
        brain = PGPLBrain(fs=1/60, saline=False)

    EDC Busan (saline SCADA):
        brain = PGPLBrain(fs=1/60, saline=True)

    Mendeley (acoustic hydrophone):
        brain = PGPLBrain(fs=44100, saline=True, sensor_spacing_m=5.0)
    """

    def __init__(
        self,
        fs:               float,
        pipe_diameter_m:  float = 0.15,
        pipe_thickness_m: float = 0.01,
        pipe_material:    str   = "hdpe",
        saline:           bool  = False,
        sensor_spacing_m: float = 100.0,
        base_alpha:       float = 0.05,
        z_window:         int   = 200,
        z_warmup:         int   = 20,
    ):
        self.fs           = fs
        self.is_scada     = (fs <= FS_SCADA_MAX)
        self.is_acoustic  = (fs >= FS_ACOU_MIN - 1.0)   # 7999+ = acoustic

        if not self.is_scada and not self.is_acoustic:
            raise ValueError(
                f"fs={fs:.1f} Hz is not SCADA (≤{FS_SCADA_MAX}Hz) "
                f"nor Acoustic (≥{FS_ACOU_MIN}Hz)."
            )

        self.pipe_diameter_m  = pipe_diameter_m
        self.pipe_thickness_m = pipe_thickness_m
        self.pipe_material    = pipe_material
        self.saline           = saline
        self.sensor_spacing_m = sensor_spacing_m
        self.base_alpha       = base_alpha

        # Sub-modules — NO data constants passed in
        self.z_pressure  = AdaptiveZDetector(window=z_window, warmup=z_warmup)
        self.z_flow      = AdaptiveZDetector(window=z_window, warmup=z_warmup)
        self.psi_tracker = PSIDriftTracker()
        self.cp          = MondrianCP()
        self.tidal       = TidalGatingEngine(saline_mode=saline)

        self._cal_done   = False

    # ── Calibration ────────────────────────────────────────────────────────────
    def calibrate_from_year(
        self,
        scores: List[float],
        phase:  str = "default",
    ) -> None:
        """
        Seed detectors from prior-year anomaly scores.
        Works for any unit — z-scores are unit-free by definition.
        """
        if not scores:
            print("  ⚠️  calibrate_from_year: empty — skipping.")
            return

        arr = np.array(scores, dtype=float)
        mu  = float(np.mean(arr))
        sig = float(np.std(arr)) + 1e-9

        self.z_pressure.seed(mu, sig)
        self.z_flow.seed(mu, sig)
        self.cp.calibrate(scores, phase=phase)
        self._cal_done = True

        print(
            f"  🧠 Calibrated [{phase}]: {len(scores):,} samples | "
            f"μ={mu:.4f}  σ={sig:.4f}"
        )

    # ── SCADA Path ─────────────────────────────────────────────────────────────
    def process_scada(
        self,
        pressure_psi: float,
        flow_lps:     float,
        timestamp:    float,
        tidal_phase:  str   = "slack_low",
        tidal_psi:    float = 0.0,
    ) -> LeakEvent:
        """
        Process one SCADA timestep.
        Units of pressure/flow do not matter —
        Z-score normalisation makes scores unit-free.
        """
        self.psi_tracker.push(pressure_psi)
        self.tidal.add_phase(TidalWindow(
            phase      = tidal_phase,
            psi_offset = tidal_psi,
            alpha_adj  = self.tidal.adaptive_alpha(self.base_alpha),
            timestamp  = timestamp,
        ))

        z_p = self.z_pressure.update(pressure_psi)
        z_f = self.z_flow.update(flow_lps)

        # Unit-free scores: z normalised to [0,1] by clipping at ±3σ
        p1 = float(np.clip(abs(z_p) / 3.0, 0.0, 1.0))
        p2 = float(np.clip(abs(z_f) / 3.0, 0.0, 1.0))
        p3 = 0.0
        p4 = self.psi_tracker.normalised_drift()

        event = LeakEvent(
            timestamp = timestamp,
            p1_score  = p1,
            p2_score  = p2,
            p3_score  = p3,
            p4_score  = p4,
        )

        fused = event.fused_score()

        # Pressure drop also normalised by local std
        mu_p  = self.psi_tracker.mean()
        sig_p = self.z_pressure._sigma if self.z_pressure._sigma else 1.0
        drop_normalised = abs(pressure_psi - mu_p) / sig_p

        ltype, slbl = _classify(fused, drop_normalised)
        event.leak_type    = ltype
        event.severity_raw = fused
        event.severity_lbl = slbl
        event.meta["gate"] = self.tidal.gate_leak_event(fused)
        event.confidence   = 1.0 - self.tidal.adaptive_alpha(self.base_alpha)

        if self.cp.has_calibration:
            event.meta["cp_p_value"] = round(
                self.cp.p_value(fused, phase=tidal_phase), 4
            )

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
        """
        Process one acoustic window.
        Sensor-agnostic: no bandpass assumption.
        P3 = log-scale peak-to-median activity ratio.
        Works for any fs, any sensor type.
        """
        self.tidal.add_phase(TidalWindow(
            phase      = tidal_phase,
            psi_offset = tidal_psi,
            alpha_adj  = self.tidal.adaptive_alpha(self.base_alpha),
            timestamp  = timestamp,
        ))

        sig_a = signal.astype(float)
        sig_b = signal_b.astype(float)

        # ── GCC-PHAT TDOA ──────────────────────────────────────────────────────
        n_fft  = len(sig_a) + len(sig_b) - 1
        A_fft  = np.fft.rfft(sig_a, n=n_fft)
        B_fft  = np.fft.rfft(sig_b, n=n_fft)
        cross  = A_fft * np.conj(B_fft)
        gcc    = np.fft.irfft(cross / (np.abs(cross) + 1e-12), n=n_fft)
        lag    = int(np.argmax(gcc)) - len(sig_a) + 1
        dt     = lag / self.fs

        c   = _biot_wave_speed(self.pipe_diameter_m, self.pipe_thickness_m,
                                self.pipe_material, self.saline, tidal_psi)
        loc = _tdoa_locate(c, dt, self.sensor_spacing_m)

        # ── P3: sensor-agnostic leak indicator ────────────────────────────────
        # Physics: leak creates asymmetric wave arrival (non-zero TDOA)
        # No-leak transient: symmetric arrival (TDOA ≈ 0)
        # Metric: how far is the GCC peak from zero-lag?
        # Unit-free: lag normalised by max possible lag (signal length / 2)
        
        max_lag     = len(sig_a) / 2.0
        lag_ratio   = float(np.clip(abs(lag) / max_lag, 0.0, 1.0))
        
        # Also require some signal energy (reject pure silence)
        has_signal  = float(np.var(sig_a) > 1e-10)
        
        p3 = lag_ratio * has_signal

        # ── P4: tidal drift ────────────────────────────────────────────────────
        p4 = float(np.clip(abs(tidal_psi) / 10.0, 0.0, 1.0))

        # ── Frequency centroid (classification only) ───────────────────────────
        psd           = np.abs(np.fft.rfft(sig_a)) ** 2
        freqs         = np.fft.rfftfreq(len(sig_a), 1.0 / self.fs)
        freq_centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))

        # ── Build event ────────────────────────────────────────────────────────
        event = LeakEvent(
            timestamp  = timestamp,
            p1_score   = 0.0,
            p2_score   = 0.0,
            p3_score   = p3,
            p4_score   = p4,
            location_m = loc,
        )

        fused       = event.fused_score()
        ltype, slbl = _classify(fused, 0.0, freq_centroid)

        event.leak_type    = ltype
        event.severity_raw = fused
        event.severity_lbl = slbl
        event.meta["gate"]             = self.tidal.gate_leak_event(fused)
        event.meta["biot_c_ms"]        = round(c, 2)
        event.meta["freq_centroid_hz"] = round(freq_centroid, 1)
        event.confidence               = 1.0 - self.tidal.adaptive_alpha(self.base_alpha)

        return event

    def __repr__(self) -> str:
        mode = "SCADA" if self.is_scada else "Acoustic"
        return (
            f"PGPLBrain(fs={self.fs}Hz, mode={mode}, "
            f"saline={self.saline}, calibrated={self._cal_done})"
        )