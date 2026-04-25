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
        "b_0_200":   band(0,   200),
        "centroid":  float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12)),
    }


def classify(
    signal: np.ndarray,
    salinity_psu: float = 7.0,
    pressure_z:   float = 0.0,
    fs:           int   = 2000,
) -> tuple:
    """
    Called ONLY after the persistence gate fires — signal is confirmed anomalous.
    Role: determine leak TYPE, not re-gate.

    Measured Mendeley feature ranges (steady-state windows):
      NL : b_200_800=0.004-0.037, b_0_200=0.957-0.995, rms=0.002-0.015
      CC : b_200_800=0.003-0.094, b_0_200=0.901-0.995, rms=0.001-0.013
      OL : b_200_800 highest, b_0_200 lowest among leak types
      LC : b_200_800=0.046, moderate
      GL : b_200_800=0.021, subtle (gasket damps)

    Strategy: gate already confirmed anomaly. Classifier ranks by spectral shape.
    Primary discriminant: b_200_800 ratio (higher = more turbulent leak).
    Secondary: b_0_200 (lower = more energy shifted to mid/high freq).
    """
    f = extract_features(signal, fs)
    rms      = f["rms"]
    b_mid    = f["b_200_800"]
    b_low    = f["b_0_200"]
    b_pmp    = f["b_50_200"]
    centroid = f["centroid"]

    # ── Orifice Leak (OL): strongest turbulence ───────────────────────────────
    # Highest b_mid, lowest b_low among all types
    if b_mid > 0.070:
        cls, conf = "Burst", min(0.92, 0.65 + 0.30 * b_mid)

    # ── Circumferential / Longitudinal Crack ──────────────────────────────────
    # b_mid clearly elevated vs NL baseline (~0.037)
    elif b_mid > 0.048:
        cls, conf = "Crack", min(0.88, 0.52 + 0.42 * b_mid)

    # ── Weaker crack signal ───────────────────────────────────────────────────
    # b_mid > NL mean but below strong crack
    elif b_mid > 0.030 and b_low < 0.960:
        cls, conf = "Crack", min(0.74, 0.44 + 0.38 * b_mid)

    # ── Gasket Leak: low b_mid but gate confirmed anomaly ────────────────────
    # GL has b_mid ~0.021 (actually LOWER than NL mean 0.037)
    # Gate fired on energy shift — classify as Micro (subtle sustained leak)
    elif b_mid < 0.030 and centroid > 200:
        cls, conf = "Micro", min(0.68, 0.48 + rms * 12.0)

    # ── Pressure Drop ─────────────────────────────────────────────────────────
    elif salinity_psu > 7.0 and pressure_z > 1.2:
        cls, conf = "PressureDrop", min(0.80, 0.55 + 0.08 * pressure_z)

    # ── Pump harmonic ─────────────────────────────────────────────────────────
    elif b_pmp > 0.72:
        cls, conf = "Pump", min(0.85, 0.55 + 0.40 * b_pmp)

    # ── Anomaly confirmed by gate but type unclear → default to Micro ─────────
    # Gate fired = something is wrong. Safer to flag than miss.
    else:
        cls, conf = "Micro", min(0.62, 0.45 + b_mid * 5.0)

    return cls, float(np.clip(conf, 0.0, 1.0)), f


def severity_score(
    confidence: float,
    class_name: str,
    zone_weight: float = 0.5,
) -> tuple:
    tw  = TYPE_WEIGHTS.get(class_name, 0.0)
    sev = float(np.clip(0.4 * confidence + 0.3 * tw + 0.3 * zone_weight, 0.0, 1.0))
    if   sev >= 0.80: tier = "P1-CRITICAL"
    elif sev >= 0.60: tier = "P2-HIGH"
    elif sev >= 0.40: tier = "P3-MODERATE"
    else:             tier = "P4-LOW"
    return round(sev, 3), tier
