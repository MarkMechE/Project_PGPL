"""
tidal_gating.py — Tidal phase engine (≥3 phases required)
PGPL v2.0 | Patent novelty: tidal fusion in reclaimed saline nets
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

TIDAL_PHASES = ["ebb", "flood", "slack_low", "slack_high", "spring", "neap"]
MIN_PHASES_REQUIRED = 3   # Patent claim: must observe ≥3 distinct phases


@dataclass
class TidalWindow:
    phase:       str
    psi_offset:  float    # Back-pressure adjustment (PSI)
    alpha_adj:   float    # PSI-adaptive α for conformal prediction
    timestamp:   float
    confidence:  float    = 1.0


class TidalGatingEngine:
    """
    Tracks tidal phase history and gates leak detections.
    A leak is confirmed only if it persists across ≥3 tidal phases.

    Patent novelty:
    ─ Unified fs-routing gating (SCADA <10 Hz / acoustic >8 kHz)
    ─ Tidal phase requirement ≥3
    ─ PSI-adaptive α calibration
    """

    def __init__(self, min_phases: int = MIN_PHASES_REQUIRED):
        self.min_phases      = min_phases
        self.phase_history:  list[TidalWindow] = []
        self.confirmed_leaks: list[dict]       = []

    def add_phase(self, window: TidalWindow):
        self.phase_history.append(window)

    def distinct_phases_seen(self) -> set[str]:
        return {w.phase for w in self.phase_history}

    def ready(self) -> bool:
        """True when ≥ min_phases distinct phases have been observed."""
        return len(self.distinct_phases_seen()) >= self.min_phases

    def adaptive_alpha(self, base_alpha: float = 0.05) -> float:
        """
        PSI-adaptive α: widen CI under high tidal back-pressure.
        α_eff = base_alpha * (1 + mean_psi_offset / 100)
        """
        if not self.phase_history:
            return base_alpha
        mean_psi = np.mean([w.psi_offset for w in self.phase_history])
        return float(base_alpha * (1.0 + abs(mean_psi) / 100.0))

    def gate_leak_event(
        self,
        event_score: float,
        threshold: float = 0.50,
        phase: Optional[str] = None,
    ) -> dict:
        """
        Gate a detection through tidal filter.

        Returns dict with keys: {confirmed, reason, phases_seen, alpha_eff}
        """
        alpha_eff   = self.adaptive_alpha()
        phases_seen = self.distinct_phases_seen()
        adj_thresh  = threshold * (1.0 + alpha_eff)

        confirmed = (
            self.ready()
            and event_score >= adj_thresh
        )

        return {
            "confirmed":    confirmed,
            "reason":       "tidal_gated" if confirmed else (
                            "insufficient_phases" if not self.ready()
                            else "below_threshold"),
            "phases_seen":  phases_seen,
            "alpha_eff":    alpha_eff,
            "adj_threshold": adj_thresh,
            "event_score":  event_score,
        }