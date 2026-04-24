"""
data_generator.py  —  Physics-informed synthetic signal generator
PGL Phase 1  |  Eco-Delta City, Busan

Generates tidal-modulated, Biot-attenuated, bandpass-filtered acoustic
scenarios for pipeline evaluation.  No real sensor data required.
"""

import numpy as np
from scipy.signal import butter, sosfilt

try:
    from biot_velocity import get_biot_velocity
except ImportError:
    def get_biot_velocity(s=7.0, **_):
        return float(np.clip(1450.0 + s * 1.8, 1450, 1520))

_RNG = np.random.default_rng(42)


def _bandpass(signal: np.ndarray, fs: int,
              low: float = 200.0, high: float = 800.0) -> np.ndarray:
    sos = butter(4, [low / (0.5 * fs), high / (0.5 * fs)],
                 btype='band', output='sos')
    return sosfilt(sos, signal)


def generate_scenario(
    fs: int           = 2000,
    duration_s: float = 10.0,
    is_leak: bool     = False,
    distance_m: float = 10.0,
    salinity_psu: float = 7.0,
    tidal_phase_rad: float = 0.0,
) -> dict:
    """
    Returns
    -------
    dict with keys:
        signal       : np.ndarray  — bandpass-filtered acoustic window
        label        : int         — 1=leak, 0=no leak
        distance_m   : float       — ground-truth distance
        salinity_psu : float
        tidal_phase  : float
        velocity_ms  : float       — Biot c(s)
    """
    n = int(fs * duration_s)
    t = np.linspace(0.0, duration_s, n)

    # Tidal cyclostationary noise (semidiurnal period ~44640 s)
    tidal_mod = 1.0 + 0.45 * np.sin(
        2.0 * np.pi * t / 44640.0 + tidal_phase_rad)
    base_noise = _RNG.normal(0.0, 0.04, n) * tidal_mod

    # Pump harmonic interference (50 Hz + 150 Hz overtone)
    pump = (0.015 * np.sin(2.0 * np.pi * 50.0 * t) +
            0.008 * np.sin(2.0 * np.pi * 150.0 * t))
    signal = base_noise + pump

    if is_leak:
        # Biot-Stoll attenuation: alpha increases with salinity
        alpha       = 0.018 * (salinity_psu / 10.0) + 0.005
        attenuation = np.exp(-alpha * distance_m)

        leak_src = _RNG.uniform(-1.0, 1.0, n) * 0.6
        # Burst spike for close-range events
        if distance_m < 8.0:
            idx = n // 2
            leak_src[idx: idx + int(0.01 * fs)] += 4.0

        signal = signal + leak_src * attenuation

    signal = _bandpass(signal, fs)
    return {
        'signal':      signal,
        'label':       int(is_leak),
        'distance_m':  distance_m,
        'salinity_psu': salinity_psu,
        'tidal_phase': tidal_phase_rad,
        'velocity_ms': get_biot_velocity(salinity_psu),
    }


def make_dataset(n_samples: int = 500, fs: int = 2000) -> list:
    """
    Generate balanced random-assignment dataset.
    Random assignment (not alternating) allows consecutive leak windows
    so the persistence gate can fire naturally.
    """
    rng    = np.random.default_rng(42)
    labels = rng.choice([True, False], size=n_samples,
                        replace=True, p=[0.50, 0.50])
    out = []
    for is_leak in labels:
        out.append(generate_scenario(
            fs           = fs,
            is_leak      = bool(is_leak),
            distance_m   = float(rng.uniform(2.0, 40.0)),
            salinity_psu = float(rng.uniform(3.0, 10.0)),
            tidal_phase_rad = float(rng.uniform(0.0, 2.0 * np.pi)),
        ))
    return out


if __name__ == "__main__":
    ds = make_dataset(10)
    for i, s in enumerate(ds):
        print(f"  [{i}] label={s['label']}  "
              f"dist={s['distance_m']:.1f} m  "
              f"sal={s['salinity_psu']:.1f} psu  "
              f"v={s['velocity_ms']:.1f} m/s  "
              f"rms={np.sqrt(np.mean(s['signal']**2)):.4f}")