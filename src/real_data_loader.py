"""
src/real_data_loader.py  —  Mendeley testbed dataset loader + salinity assignment.

Dataset: "Acoustic leak detection in water pipe" (Mendeley Data, DOI: 10.17632/tbrnp6vrnj/1)
Sensors: Accelerometer (CSV), Hydrophone (.raw), Dynamic Pressure Sensor (CSV)

Salinity source:
  Lee et al. (2023), Marine Geology, DOI: 10.1016/j.margeo.2023.107089
  Nakdong Estuary reclaimed aquifer (EDC pipe zone): brackish 3-10 psu.
  Mean 7.0 psu used as default; KHOA open-seawater values (25-35 psu) excluded.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from math import gcd

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False
    print("[real_data_loader] soundfile not installed — hydrophone .raw loader disabled.")
    print("  pip install soundfile")

from config import (
    ACCEL_DIR, PRESSURE_DIR, HYDRO_DIR,
    SALINITY_MEAN_PSU, SALINITY_RANGE_PSU, FS,
)

# ── Sensor sample rates (Mendeley dataset spec) ────────────────────────────
ACCEL_FS    = 3000   # Accelerometer: 0.5-3000 Hz range
PRESSURE_FS = 100    # Dynamic Pressure Sensor: 100 Hz
HYDRO_FS    = 8000   # Hydrophone: 8000 Hz

# ── Label mapping ──────────────────────────────────────────────────────────
LABEL_MAP = {
    "OL": "Burst",
    "LC": "Crack",
    "CC": "Crack",
    "GL": "Micro",
    "NL": "Tidal",
}
LEAK_CLASSES = {"Burst", "Crack", "Micro"}


def assign_salinity(rng: np.random.Generator = None) -> float:
    if rng is None:
        return SALINITY_MEAN_PSU
    lo, hi = SALINITY_RANGE_PSU
    return float(rng.uniform(lo, hi))


def _parse_label(filename: str) -> str:
    fname = os.path.basename(filename)
    for ext in (".csv", ".raw"):
        fname = fname.replace(ext, "")
    parts = fname.split("_")
    if len(parts) < 2:
        return "Tidal"
    return LABEL_MAP.get(parts[1].upper(), "Tidal")


def _resample(signal: np.ndarray, src_fs: int, tgt_fs: int = FS) -> np.ndarray:
    if src_fs == tgt_fs:
        return signal.astype(np.float32)
    g = gcd(int(src_fs), int(tgt_fs))
    return resample_poly(signal, tgt_fs // g, src_fs // g).astype(np.float32)


def _normalize(signal: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(signal))
    if peak < 1e-12:
        return signal.astype(np.float32)
    return (signal / peak).astype(np.float32)


def _read_mendeley_csv(filepath: str) -> np.ndarray:
    """
    Robustly read a Mendeley CSV with or without a 'Value' header row.
    Tries header=0 first (skips text header); extracts first fully numeric column.
    """
    df = pd.read_csv(filepath, header=0, low_memory=False)

    signal = None
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        valid = converted.dropna()
        if len(valid) > 10:
            signal = valid.values.astype(float)
            break

    if signal is None or len(signal) == 0:
        raise ValueError(f"No numeric data found in {os.path.basename(filepath)}")

    return signal


def load_mendeley_accelerometer(filepath: str,
                                salinity_psu: float = SALINITY_MEAN_PSU) -> dict:
    label  = _parse_label(filepath)
    signal = _read_mendeley_csv(filepath)
    signal = _normalize(_resample(signal, ACCEL_FS, FS))
    return {
        "mic1_sig":    signal,
        "fs":          FS,
        "salinity":    salinity_psu,
        "flow":        13.0,
        "label":       label,
        "is_leak":     label in LEAK_CLASSES,
        "source_file": os.path.basename(filepath),
        "sensor_type": "accelerometer",
    }


def load_mendeley_pressure(filepath: str,
                           salinity_psu: float = SALINITY_MEAN_PSU) -> dict:
    label    = _parse_label(filepath)
    pressure = _read_mendeley_csv(filepath)

    p_mean     = float(np.mean(pressure))
    p_std      = float(np.std(pressure)) + 1e-9
    pressure_z = float(np.mean(np.abs(pressure - p_mean) / p_std))
    flow_proxy = float(np.clip(13.0 - pressure_z * 2.0, 0.0, 30.0))

    signal = _normalize(_resample(pressure, PRESSURE_FS, FS))
    return {
        "mic1_sig":    signal,
        "fs":          FS,
        "salinity":    salinity_psu,
        "flow":        flow_proxy,
        "label":       label,
        "is_leak":     label in LEAK_CLASSES,
        "source_file": os.path.basename(filepath),
        "sensor_type": "pressure",
        "pressure_z":  round(pressure_z, 4),
    }


def load_mendeley_hydrophone(filepath: str,
                             salinity_psu: float = SALINITY_MEAN_PSU,
                             channels: int = 1,
                             subtype: str = "PCM_32",
                             endian: str = "LITTLE") -> dict:
    if not _SF_AVAILABLE:
        raise ImportError("soundfile required: pip install soundfile")

    label = _parse_label(filepath)
    signal_data, _ = sf.read(
        filepath, channels=channels, samplerate=HYDRO_FS,
        subtype=subtype, endian=endian, format="RAW",
    )
    signal = np.array(signal_data, dtype=float)
    if signal.ndim > 1:
        signal = signal[:, 0]
    signal = _normalize(_resample(signal, HYDRO_FS, FS))
    return {
        "mic1_sig":    signal,
        "fs":          FS,
        "salinity":    salinity_psu,
        "flow":        13.0,
        "label":       label,
        "is_leak":     label in LEAK_CLASSES,
        "source_file": os.path.basename(filepath),
        "sensor_type": "hydrophone",
    }


def load_real_dataset(
    use_accelerometer: bool = True,
    use_pressure:      bool = True,
    use_hydrophone:    bool = False,
    rng: np.random.Generator = None,
    verbose: bool = True,
) -> list:
    samples = []

    loaders = []
    if use_accelerometer:
        loaders.append((ACCEL_DIR,    "*.csv", load_mendeley_accelerometer, "Accelerometer"))
    if use_pressure:
        loaders.append((PRESSURE_DIR, "*.csv", load_mendeley_pressure,      "Dynamic Pressure Sensor"))
    if use_hydrophone and _SF_AVAILABLE:
        loaders.append((HYDRO_DIR,    "*.raw", load_mendeley_hydrophone,    "Hydrophone"))

    for base_dir, pattern, loader_fn, sensor_name in loaders:
        if not os.path.isdir(base_dir):
            print(f"  [WARN] Not found: {base_dir}")
            print(f"         Edit DATASET_ROOT in config.py")
            continue

        files  = sorted(glob.glob(os.path.join(base_dir, "**", pattern), recursive=True))
        n_ok   = 0
        n_skip = 0

        for f in files:
            sal = assign_salinity(rng)
            try:
                sample = loader_fn(f, salinity_psu=sal)
                if len(sample["mic1_sig"]) < FS:
                    raise ValueError("Signal too short (<1 s)")
                samples.append(sample)
                n_ok += 1
            except Exception as e:
                if verbose:
                    print(f"  [skip] {os.path.basename(f)}: {e}")
                n_skip += 1

        if verbose:
            print(f"  {sensor_name}: {n_ok} loaded, {n_skip} skipped")

    if verbose:
        n_leak   = sum(1 for s in samples if s["is_leak"])
        n_normal = len(samples) - n_leak
        sal_desc = (f"fixed {SALINITY_MEAN_PSU} psu"
                    if rng is None else f"random {SALINITY_RANGE_PSU} psu")
        print(f"\n  Total   : {len(samples)} samples")
        print(f"  Leak    : {n_leak}   Normal: {n_normal}")
        print(f"  Salinity: {sal_desc}")

    return samples