"""
Physics-informed synthetic signal generator — PGL Phase 1.

All signals are SYNTHETIC. No real sensor recordings are used.
Tidal modulation + Biot attenuation + Butterworth bandpass.
"""
import numpy as np
from scipy.signal import butter, sosfilt
from src.biot_velocity import get_biot_velocity

_RNG = np.random.default_rng(42)

LEAK_CLASSES = {"Burst", "Crack", "Micro"}


def _bandpass(signal: np.ndarray, fs: int,
              low: float = 200.0, high: float = 800.0) -> np.ndarray:
    sos = butter(4, [low / (0.5 * fs), high / (0.5 * fs)],
                 btype="band", output="sos")
    return sosfilt(sos, signal)


def generate_scenario(
    fs: int = 2000,
    duration_s: float = 10.0,
    is_leak: bool = False,
    distance_m: float = 10.0,
    salinity_psu: float = 7.0,
    tidal_phase_rad: float = 0.0,
) -> dict:
    """
    Returns dict with keys:
        signal        np.ndarray  bandpass-filtered acoustic window
        label         int         1=leak, 0=no leak
        distance_m    float       ground-truth distance (m)
        salinity_psu  float
        tidal_phase   float
        velocity_ms   float       Biot c(s)
    """
    n = int(fs * duration_s)
    t = np.linspace(0.0, duration_s, n)

    # Tidal cyclostationary noise (semidiurnal period ~44640 s)
    tidal_mod = 1.0 + 0.45 * np.sin(2.0 * np.pi * t / 44640.0 + tidal_phase_rad)
    signal = _RNG.normal(0.0, 0.04, n) * tidal_mod

    # Pump harmonic (50 Hz + 3rd overtone)
    signal += 0.015 * np.sin(2.0 * np.pi * 50.0 * t)
    signal += 0.008 * np.sin(2.0 * np.pi * 150.0 * t)

    if is_leak:
        alpha       = 0.018 * (salinity_psu / 10.0) + 0.005
        attenuation = np.exp(-alpha * distance_m)
        leak_src    = _RNG.uniform(-1.0, 1.0, n) * 0.6
        if distance_m < 8.0:
            idx = n // 2
            leak_src[idx: idx + int(0.01 * fs)] += 4.0  # burst spike
        signal += leak_src * attenuation

    return {
        "signal":       _bandpass(signal, fs),
        "label":        int(is_leak),
        "distance_m":   distance_m,
        "salinity_psu": salinity_psu,
        "tidal_phase":  tidal_phase_rad,
        "velocity_ms":  get_biot_velocity(salinity_psu),
    }


def make_dataset(n_samples: int = 500, fs: int = 2000) -> list:
    """Random-assignment balanced dataset (allows consecutive runs for gate)."""
    rng    = np.random.default_rng(42)
    labels = rng.choice([True, False], size=n_samples, replace=True, p=[0.5, 0.5])
    return [
        generate_scenario(
            fs=fs,
            is_leak=bool(lbl),
            distance_m=float(rng.uniform(2.0, 40.0)),
            salinity_psu=float(rng.uniform(3.0, 10.0)),
            tidal_phase_rad=float(rng.uniform(0.0, 2 * np.pi)),
        )
        for lbl in labels
    ]