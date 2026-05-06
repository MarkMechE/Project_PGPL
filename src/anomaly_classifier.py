"""
anomaly_classifier.py — 6-class leak classifier + severity scorer
PGPL v2.0 | Reused + cleaned from PROJECT_PGPL
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Literal

# ── Leak Types & Weights ───────────────────────────────────────────────────────
LeakType = Literal["Burst", "Crack", "Joint", "Corrosion", "Pinhole", "Unknown"]

TYPE_WEIGHTS: dict[str, float] = {
    "Burst":     1.00,
    "Crack":     0.85,
    "Joint":     0.70,
    "Corrosion": 0.60,
    "Pinhole":   0.45,
    "Unknown":   0.30,
}

SEVERITY_THRESHOLDS = {
    "Critical": 0.80,
    "High":     0.60,
    "Medium":   0.40,
    "Low":      0.00,
}


@dataclass
class LeakEvent:
    timestamp:    float
    leak_type:    LeakType      = "Unknown"
    severity_raw: float         = 0.0       # 0–1
    severity_lbl: str           = "Low"
    confidence:   float         = 0.0       # conformal prediction interval width
    location_m:   float         = -1.0      # TDOA result; -1 = unknown
    p1_score:     float         = 0.0       # Pressure anomaly
    p2_score:     float         = 0.0       # Flow anomaly
    p3_score:     float         = 0.0       # Acoustic energy
    p4_score:     float         = 0.0       # Tidal correlation
    meta:         dict          = field(default_factory=dict)

    def fused_score(self) -> float:
        """Weighted P1–P4 fusion (EDC patent claim)."""
        return float(
            0.30 * self.p1_score
            + 0.30 * self.p2_score
            + 0.25 * self.p3_score
            + 0.15 * self.p4_score
        )


def classify_leak(
    energy_ratio: float,
    freq_centroid_hz: float,
    pressure_drop_psi: float,
) -> tuple[LeakType, float]:
    """
    Heuristic 6-class classifier based on signal features.

    Returns (leak_type, type_weight)
    """
    if pressure_drop_psi > 15.0 and energy_ratio > 0.85:
        return "Burst",     TYPE_WEIGHTS["Burst"]
    elif freq_centroid_hz > 4000 and energy_ratio > 0.70:
        return "Crack",     TYPE_WEIGHTS["Crack"]
    elif 1000 < freq_centroid_hz <= 4000 and energy_ratio > 0.55:
        return "Joint",     TYPE_WEIGHTS["Joint"]
    elif energy_ratio > 0.40 and pressure_drop_psi < 5.0:
        return "Corrosion", TYPE_WEIGHTS["Corrosion"]
    elif freq_centroid_hz < 500 and energy_ratio > 0.25:
        return "Pinhole",   TYPE_WEIGHTS["Pinhole"]
    else:
        return "Unknown",   TYPE_WEIGHTS["Unknown"]


def score_severity(fused: float) -> str:
    """Map fused 0–1 score → severity label."""
    for label, threshold in SEVERITY_THRESHOLDS.items():
        if fused >= threshold:
            return label
    return "Low"