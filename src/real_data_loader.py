"""
src/real_data_loader.py  —  Mendeley testbed dataset loader + salinity assignment.
Salinity: Lee et al. (2023), Marine Geology, DOI: 10.1016/j.margeo.2023.107089
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

from config import (
    ACCEL_DIR, PRESSURE_DIR, HYDRO_DIR,
    SALINITY_MEAN_PSU, SALINITY_RANGE_PSU, FS,
)

ACCEL_FS    = 3000
PRESSURE_FS = 100
HYDRO_FS    = 8000

LABEL_MAP = {
    "OL": "Burst",
    "LC": "Crack",
    "CC": "Crack",
    "GL": "Micro",
    "NL": "Tidal",
}
LEAK_CLASSES = {"Burst", "Crack", "Micro"}


def assign_salinity(rng=None):
    if rng is None:
        return SALINITY_MEAN_PSU
    lo, hi = SALINITY_RANGE_PSU
    return float(rng.uniform(lo, hi))


def _parse_label(filename):
    fname = os.path.basename(filename)
    for ext in (".csv", ".raw"):
        fname = fname.replace(ext, "")
    parts = fname.split("_")
    if len(parts) < 2:
        return "Tidal"
    return LABEL_MAP.get(parts[1].upper(), "Tidal")


def _resample(signal, src_fs, tgt_fs=FS):
    if src_fs == tgt_fs:
        return signal.astype(np.float32)
    g = gcd(int(src_fs), int(tgt_fs))
    return resample_poly(signal, tgt_fs // g, src_fs // g).astype(np.float32)


def _read_mendeley_csv(filepath):
    """Read Mendeley CSV — picks the Value column, skips Sample/timestamp."""
    df = pd.read_csv(filepath, header=0, low_memory=False)
    # Prefer explicit "Value" column if present
    if "Value" in df.columns:
        col = pd.to_numeric(df["Value"], errors="coerce").dropna().values.astype(float)
        if len(col) > 10:
            return col
    # Fallback: find first numeric column with real variance
    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce").dropna().values.astype(float)
        if len(converted) > 10 and converted.std() > 1e-6:
            return converted
    raise ValueError(f"No usable numeric column in {os.path.basename(filepath)}")


def load_mendeley_accelerometer(filepath, salinity_psu=SALINITY_MEAN_PSU):
    label  = _parse_label(filepath)
    raw    = _read_mendeley_csv(filepath)
    signal = _resample(raw, ACCEL_FS, FS)
    # Skip first 30s transient (pump startup)
    trim = int(30 * FS)
    if len(signal) > trim + FS:
        signal = signal[trim:]
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


def load_mendeley_pressure(filepath, salinity_psu=SALINITY_MEAN_PSU):
    label    = _parse_label(filepath)
    raw      = _read_mendeley_csv(filepath)
    p_mean   = float(np.mean(raw))
    p_std    = float(np.std(raw)) + 1e-9
    pressure_z = float(np.mean(np.abs(raw - p_mean) / p_std))
    flow_proxy = float(np.clip(13.0 - pressure_z * 2.0, 0.0, 30.0))
    signal   = _resample(raw, PRESSURE_FS, FS)
    trim = int(30 * FS)
    if len(signal) > trim + FS:
        signal = signal[trim:]
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


def load_mendeley_hydrophone(filepath, salinity_psu=SALINITY_MEAN_PSU,
                              channels=1, subtype="PCM_32", endian="LITTLE"):
    if not _SF_AVAILABLE:
        raise ImportError("soundfile required: pip install soundfile")
    label = _parse_label(filepath)
    signal_data, _ = sf.read(filepath, channels=channels, samplerate=HYDRO_FS,
                              subtype=subtype, endian=endian, format="RAW")
    signal = np.array(signal_data, dtype=float)
    if signal.ndim > 1:
        signal = signal[:, 0]
    signal = _resample(signal, HYDRO_FS, FS)
    trim = int(30 * FS)
    if len(signal) > trim + FS:
        signal = signal[trim:]
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


def load_real_dataset(use_accelerometer=True, use_pressure=True,
                       use_hydrophone=False, rng=None, verbose=True):
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
            continue
        files  = sorted(glob.glob(os.path.join(base_dir, "**", pattern), recursive=True))
        n_ok = n_skip = 0
        for f in files:
            sal = assign_salinity(rng)
            try:
                sample = loader_fn(f, salinity_psu=sal)
                if len(sample["mic1_sig"]) < FS:
                    raise ValueError("Signal too short")
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
        sal_desc = f"fixed {SALINITY_MEAN_PSU} psu" if rng is None else f"random {SALINITY_RANGE_PSU} psu"
        print(f"\n  Total   : {len(samples)} samples")
        print(f"  Leak    : {n_leak}   Normal: {n_normal}")
        print(f"  Salinity: {sal_desc}")

    return samples
