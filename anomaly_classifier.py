"""
anomaly_classifier.py  —  PGL Tier 3 heuristic classifier
6 event classes  |  Phase 1 baseline (CNN-Transformer in Phase 2)

Classes
-------
0  Tidal        — tidal/environmental noise, no action
1  Pump         — pump harmonic artefact
2  PressureDrop — hydraulic event, no acoustic leak signature
3  Micro        — micro-leak, sustained low-amplitude hiss
4  Crack        — narrowband crack propagation signal
5  Burst        — high-energy broadband rupture

Priority dispatch: only classes 3-5 trigger DISPATCH (leak events).
"""

import numpy as np
from scipy.signal import welch

# Class metadata
CLASS_NAMES = {
    0: "Tidal",
    1: "Pump",
    2: "PressureDrop",
    3: "Micro",
    4: "Crack",
    5: "Burst",
}

# Type weights for severity formula (0.4*conf + 0.3*typeW + 0.3*zoneW)
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
    """Spectral + temporal features from a 1-D acoustic window."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    total = float(np.sum(psd)) + 1e-12

    def band(f_lo, f_hi):
        return float(np.sum(psd[(freqs >= f_lo) & (freqs < f_hi)])) / total

    rms    = float(np.sqrt(np.mean(signal ** 2)))
    peak   = float(np.max(np.abs(signal)))
    crest  = peak / (rms + 1e-9)
    sc     = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))

    return {
        'rms':        rms,
        'peak':       peak,
        'crest':      crest,
        'b_0_50':     band(0,   50),
        'b_50_200':   band(50,  200),
        'b_200_800':  band(200, 800),
        'b_800_1000': band(800, 1000),
        'centroid':   sc,
    }


def classify(signal: np.ndarray,
             salinity_psu: float = 7.0,
             pressure_z:   float = 0.0,
             fs:           int   = 2000) -> tuple:
    """
    Returns (class_name: str, confidence: float, features: dict)

    Decision rules (heuristic — Phase 1 baseline):
      Burst        : high RMS + high peak-to-peak
      Crack        : mid-band energy dominant + low crest
      Micro        : 200-800 Hz band dominant + moderate RMS
      PressureDrop : high salinity + low acoustic energy + pressure anomaly
      Pump         : 50-200 Hz band dominant
      Tidal        : default
    """
    f = extract_features(signal, fs)

    rms   = f['rms']
    crest = f['crest']
    b_mid = f['b_200_800']
    b_pmp = f['b_50_200']

    # Burst: broad energy + spiky
    if rms > 0.08 and crest > 4.5:
        cls  = "Burst"
        conf = min(0.95, 0.72 + 0.04 * (crest - 4.5))

    # Crack: narrowband mid, smooth waveform
    elif b_mid > 0.45 and crest < 2.5 and rms > 0.015:
        cls  = "Crack"
        conf = min(0.90, 0.60 + 0.55 * b_mid)

    # Micro-leak: mid-band, lower salinity
    elif b_mid > 0.38 and rms > 0.008 and salinity_psu < 9.0:
        cls  = "Micro"
        conf = min(0.88, 0.52 + 0.28 * b_mid + 0.08 * max(pressure_z, 0))

    # Micro-leak: attenuated at higher salinity
    elif b_mid > 0.30 and rms > 0.005:
        cls  = "Micro"
        conf = min(0.72, 0.42 + 0.22 * b_mid)

    # Pressure drop: hydraulic signature, low acoustic
    elif salinity_psu > 7.0 and rms < 0.012 and pressure_z > 1.2:
        cls  = "PressureDrop"
        conf = min(0.80, 0.55 + 0.08 * pressure_z)

    # Pump harmonic
    elif b_pmp > 0.50:
        cls  = "Pump"
        conf = min(0.85, 0.55 + 0.55 * b_pmp)

    # Default: tidal noise
    else:
        cls  = "Tidal"
        conf = max(0.50, 1.0 - rms * 8.0)

    return cls, float(np.clip(conf, 0.0, 1.0)), f


def severity_score(confidence: float,
                   class_name: str,
                   zone_weight: float = 0.5) -> tuple:
    """
    Severity = 0.4 * confidence + 0.3 * type_weight + 0.3 * zone_weight

    Returns (score: float, priority_tier: str)
    """
    tw  = TYPE_WEIGHTS.get(class_name, 0.0)
    sev = 0.4 * confidence + 0.3 * tw + 0.3 * zone_weight
    sev = float(np.clip(sev, 0.0, 1.0))

    if   sev >= 0.80: tier = "P1-CRITICAL"
    elif sev >= 0.60: tier = "P2-HIGH"
    elif sev >= 0.40: tier = "P3-MODERATE"
    else:             tier = "P4-LOW"

    return round(sev, 3), tier


def is_leak_event(class_name: str) -> bool:
    return class_name in LEAK_CLASSES


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(1)

    print("Classifier self-test:")
    for label, sig in [
        ("burst",  np.concatenate([rng.normal(0,.02,900), rng.uniform(.4,.8,100), rng.normal(0,.02,1000)])),
        ("noise",  rng.normal(0, 0.02, 2000)),
    ]:
        cls, conf, _ = classify(sig, salinity_psu=7.0, fs=2000)
        sev, tier    = severity_score(conf, cls)
        print(f"  input={label:<6}  class={cls:<12}  conf={conf:.2f}  "
              f"sev={sev:.2f}  {tier}")