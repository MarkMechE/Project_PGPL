"""
pulse_at_bridge.py  —  Data bridge for dashboard.py
Phase 1 stub: generates synthetic PULSE-AT rows in the format
the dashboard expects.  Replace with real KOHO/K-water data in Phase 2.
"""

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(99)


def load_or_generate_pulse_at_data(n: int = 300) -> pd.DataFrame:
    """
    Returns a DataFrame with columns matching what the dashboard expects.
    50% stress (leak) / 50% normal rows.
    """
    regimes = _RNG.choice(['stress', 'normal'], size=n, p=[0.5, 0.5])
    rows = []
    for regime in regimes:
        is_leak = regime == 'stress'
        flow     = float(_RNG.uniform(15, 24) if is_leak else _RNG.uniform(10, 14))
        pressure = float(_RNG.uniform(2.5, 3.0) if is_leak else _RNG.uniform(2.9, 3.2))
        var_z    = float(_RNG.uniform(2.5, 4.5) if is_leak else _RNG.uniform(0.5, 1.8))
        salinity = float(_RNG.uniform(3.0, 10.0))
        rows.append({
            'regime':    regime,
            'flow':      round(flow, 2),
            'pressure':  round(pressure, 3),
            'var_z':     round(var_z, 3),
            'salinity':  round(salinity, 2),
            'timestamp': f"2026-04-{(_RNG.integers(1,30)):02d} "
                         f"{(_RNG.integers(0,24)):02d}:00",
        })
    return pd.DataFrame(rows)


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Passthrough — column names already match dashboard expectations."""
    return df.reset_index(drop=True)


def row_to_sensors(row, brain, fs: int = 2000, n: int = 20000) -> dict:
    """
    Convert one DataFrame row into the sensors dict that brain.process() needs.
    Injects a synthetic acoustic signal matching the regime.
    """
    is_leak  = row.get('regime', 'normal') == 'stress'
    salinity = float(row.get('salinity', 7.0))

    # Base tidal noise
    sig = _RNG.normal(0, 0.03, n)
    sig += 0.015 * np.sin(2 * np.pi * 50 * np.arange(n) / fs)

    if is_leak:
        # Inject attenuated leak burst
        alpha = 0.018 * (salinity / 10.0) + 0.005
        dist  = float(_RNG.uniform(5, 35))
        atten = np.exp(-alpha * dist)
        leak  = _RNG.uniform(-1, 1, n) * 0.6 * atten
        idx   = n // 2
        leak[idx: idx + int(0.01 * fs)] += 2.5
        sig   = sig + leak

    return {
        'mic1_sig':  sig,
        'mic2_sig':  sig * 0.9 + _RNG.normal(0, 0.005, n),
        'fs':        fs,
        'flow':      float(row.get('flow', 13.0)),
        'pressure':  float(row.get('pressure', 3.0)),
        'var_z':     float(row.get('var_z', 1.0)),
        'salinity':  salinity,
        'regime':    row.get('regime', 'normal'),
        'timestamp': row.get('timestamp', ''),
    }
    