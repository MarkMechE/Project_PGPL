"""
PGL Tier 3 — 6-class heuristic classifier + severity scorer.
Phase 1 baseline. Labeled for replacement by CNN-Transformer in Phase 2.

Classes
-------
Tidal        type_weight = 0.0   (environmental noise, no action)
Pump         type_weight = 0.1   (pump harmonic artefact)
PressureDrop type_weight = 0.4   (hydraulic event, no acoustic leak)
Micro        type_weight = 0.6   (micro-leak, sustained low-amplitude hiss)
Crack        type_weight = 0.8   (narrowband crack signal)
Burst        type_weight = 1.0   (high-energy broadband rupture)
"""
import numpy as np
from scipy.signal import welch

TYPE_WEIGHTS = {
    "Burst":        1.0,
    "Crack":        0.8,
    "Micro":        0.6,
    "PressureDrop": 0.4,
    "Pump":         0.1,
    "Tidal":        0.0,
}
LEAK_CLASSES = {"Burst", "Crack", "Micro"}


def extract_features(signal: np.ndarray, fs: int = 2000) -> dict:
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    total = float(np.sum(psd)) + 1e-12

    def band(f_lo, f_hi):
        return float(np.sum(psd[(freqs >= f_lo) & (freqs < f_hi)])) / total

    rms   = float(np.sqrt(np.mean(signal ** 2)))
    peak  = float(np.max(np.abs(signal)))
    crest = peak / (rms + 1e-9)

    return {
        "rms":       rms,
        "crest":     crest,
        "b_0_50":    band(0,   50),
        "b_50_200":  band(50,  200),
        "b_200_800": band(200, 800),
        "centroid":  float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12)),
    }


def classify(
    signal: np.ndarray,
    salinity_psu: float = 7.0,
    pressure_z:   float = 0.0,
    fs:           int   = 2000,
) -> tuple:
    """Returns (event_label: str, confidence: float, features: dict)."""
    f = extract_features(signal, fs)
    rms, crest, b_mid, b_pmp = f["rms"], f["crest"], f["b_200_800"], f["b_50_200"]

    if rms > 0.08 and crest > 4.5:
        cls, conf = "Burst", min(0.95, 0.72 + 0.04 * (crest - 4.5))
    elif b_mid > 0.45 and crest < 2.5 and rms > 0.015:
        cls, conf = "Crack",  min(0.90, 0.60 + 0.55 * b_mid)
    elif b_mid > 0.38 and rms > 0.008 and salinity_psu < 9.0:
        cls, conf = "Micro",  min(0.88, 0.52 + 0.28 * b_mid + 0.08 * max(pressure_z, 0))
    elif b_mid > 0.30 and rms > 0.005:
        cls, conf = "Micro",  min(0.72, 0.42 + 0.22 * b_mid)
    elif salinity_psu > 7.0 and rms < 0.012 and pressure_z > 1.2:
        cls, conf = "PressureDrop", min(0.80, 0.55 + 0.08 * pressure_z)
    elif b_pmp > 0.50:
        cls, conf = "Pump",  min(0.85, 0.55 + 0.55 * b_pmp)
    else:
        cls, conf = "Tidal", max(0.50, 1.0 - rms * 8.0)

    return cls, float(np.clip(conf, 0.0, 1.0)), f


def severity_score(
    confidence: float,
    class_name: str,
    zone_weight: float = 0.5,
) -> tuple:
    """
    Severity = 0.4 × confidence + 0.3 × type_weight + 0.3 × zone_weight
    Returns (score: float, priority_tier: str)
    """
    tw  = TYPE_WEIGHTS.get(class_name, 0.0)
    sev = float(np.clip(0.4 * confidence + 0.3 * tw + 0.3 * zone_weight, 0.0, 1.0))

    if   sev >= 0.80: tier = "P1-CRITICAL"
    elif sev >= 0.60: tier = "P2-HIGH"
    elif sev >= 0.40: tier = "P3-MODERATE"
    else:             tier = "P4-LOW"

    return round(sev, 3), tier