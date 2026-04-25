"""
src/real_data_loader.py  —  Mendeley testbed dataset loader + salinity assignment.

Dataset: "Acoustic leak detection in water pipe" (Mendeley Data, DOI: 10.17632/tbrnp6vrnj/1)
Sensors: Accelerometer (CSV), Hydrophone (.raw), Dynamic Pressure Sensor (CSV)

Salinity source:
  Lee et al. (2023), Marine Geology, DOI: 10.1016/j.margeo.2023.107089
  Nakdong Estuary reclaimed aquifer (EDC pipe zone): brackish 3–10 psu.
  Mean 7.0 psu used as default; KHOA open-seawater values (25–35 psu) excluded.
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

# ── Label mapping ──────────────────────────────────────────────────────────
# Maps Mendeley filename codes → PGL classifier class names
# Mendeley naming: <pipe_type>_<leak_code>_<flow>LPS_<sensor_id>.csv
#   Pipe types: BR = Branched, LP = Looped
#   Leak codes: OL = Orifice Leak, LC = Longitudinal Crack,
#               CC = Circumferential Crack, GL = Gasket Leak, NL = No Leak
LABEL_MAP = {
    "OL": "Burst",   # Orifice leak  → sudden high-energy broadband
    "LC": "Crack",   # Longitudinal crack → sustained narrowband
    "CC": "Crack",   # Circumferential crack → sustained narrowband
    "GL": "Micro",   # Gasket leak   → low-amplitude, sustained
    "NL": "Tidal",   # No leak       → treated as ambient/tidal baseline
}
LEAK_CLASSES = {"Burst", "Crack", "Micro"}


# ── Salinity helpers ───────────────────────────────────────────────────────
def assign_salinity(rng: np.random.Generator = None) -> float:
    """
    Return salinity in psu for Nakdong EDC aquifer zone.
    Fixed mean (7.0) for reproducibility; pass rng for robustness sweep.

    Source: Lee et al. (2023), Marine Geology, DOI: 10.1016/j.margeo.2023.107089
    """
    if rng is None:
        return SALINITY_MEAN_PSU
    lo, hi = SALINITY_RANGE_PSU
    return float(rng.uniform(lo, hi))


def _parse_label(filename: str) -> str:
    """Extract leak-type label from Mendeley filename."""
    fname = os.path.basename(filename).replace(".csv", "").replace(".raw", "")
    parts = fname.split("_")
    if len(parts) < 2:
        return "Tidal"
    code = parts[1].upper()
    return LABEL_MAP.get(code, "Tidal")


def _resample(signal: np.ndarray, src_fs: int, tgt_fs: int = FS) -> np.ndarray:
    """Resample signal from src_fs → tgt_fs using polyphase filter."""
    if src_fs == tgt_fs:
        return signal
    g = gcd(src_fs, tgt_fs)
    return resample_poly(signal, tgt_fs // g, src_fs // g).astype(np.float32)


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] peak-amplitude."""
    peak = np.max(np.abs(signal))
    return signal / (peak + 1e-9)


# ── Accelerometer loader ───────────────────────────────────────────────────
# Mendeley accelerometer fs: confirmed 3000 Hz (sensor range 0.5–3000 Hz)
ACCEL_FS = 3000

def load_mendeley_accelerometer(filepath: str,
                                salinity_psu: float = SALINITY_MEAN_PSU) -> dict:
    label = _parse_label(filepath)

    df = pd.read_csv(filepath, header=0, low_memory=False)   # skip "Value" header
    # After reading with header, the signal is in the first (and only) data column
    signal = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values.astype(float)

    signal = _normalize(_resample(signal, ACCEL_FS, FS))

    return {
        "mic1_sig":     signal,
        "fs":           FS,
        "salinity":     salinity_psu,
        "label":        label,
        "is_leak":      label in LEAK_CLASSES,
        "source_file":  os.path.basename(filepath),
        "sensor_type":  "accelerometer",
    }


def load_mendeley_pressure(filepath: str,
                           salinity_psu: float = SALINITY_MEAN_PSU) -> dict:
    label = _parse_label(filepath)

    df = pd.read_csv(filepath, header=0, low_memory=False)
    pressure = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values.astype(float)

    signal = _normalize(_resample(pressure, PRESSURE_FS, FS))

    pressure_mean = float(np.mean(pressure))
    pressure_std  = float(np.std(pressure)) + 1e-9
    pressure_z    = float(np.mean(np.abs(pressure - pressure_mean) / pressure_std))

    return {
        "mic1_sig":     signal,
        "fs":           FS,
        "salinity":     salinity_psu,
        "flow":         13.0 - pressure_z * 2.0,
        "label":        label,
        "is_leak":      label in LEAK_CLASSES,
        "source_file":  os.path.basename(filepath),
        "sensor_type":  "pressure",
        "pressure_z":   round(pressure_z, 4),
    }
# ── Hydrophone loader ──────────────────────────────────────────────────────
# Mendeley hydrophone fs: 8000 Hz (per dataset spec)
HYDRO_FS = 8000

def load_mendeley_hydrophone(filepath: str,
                              salinity_psu: float = SALINITY_MEAN_PSU,
                              channels: int = 1,
                              subtype: str = "PCM_32",
                              endian: str = "LITTLE") -> dict:
    """
    Load one Mendeley hydrophone .raw file.
    Requires soundfile: pip install soundfile

    Uses the official converter from the Mendeley dataset authors.
    """
    if not _SF_AVAILABLE:
        raise ImportError("soundfile required: pip install soundfile")

    label = _parse_label(filepath)

    signal_data, _ = sf.read(
        filepath,
        channels=channels,
        samplerate=HYDRO_FS,
        subtype=subtype,
        endian=endian,
        format="RAW",
    )
    signal = signal_data.astype(float)
    if signal.ndim > 1:
        signal = signal[:, 0]   # mono

    signal = _normalize(_resample(signal, HYDRO_FS, FS))

    return {
        "mic1_sig":     signal,
        "fs":           FS,
        "salinity":     salinity_psu,
        "label":        label,
        "is_leak":      label in LEAK_CLASSES,
        "source_file":  os.path.basename(filepath),
        "sensor_type":  "hydrophone",
    }


# ── Dataset loader (all sensors) ───────────────────────────────────────────
def load_real_dataset(
    use_accelerometer: bool = True,
    use_pressure:      bool = True,
    use_hydrophone:    bool = False,   # needs soundfile; set True when installed
    rng: np.random.Generator = None,
    verbose: bool = True,
) -> list:
    """
    Load all available Mendeley files into a list of sample dicts.

    Each sample is ready for run_pipeline.py's sequential evaluation loop.
    Salinity is drawn from the Nakdong EDC aquifer range [3–10 psu] per sample.

    Args:
        use_accelerometer:  load Accelerometer/ CSVs
        use_pressure:       load Dynamic Pressure Sensor/ CSVs
        use_hydrophone:     load Hydrophone/ .raw files (requires soundfile)
        rng:                numpy Generator for salinity variation; None = fixed mean
        verbose:            print loading summary

    Returns:
        list of dicts with keys: mic1_sig, fs, salinity, label, is_leak, source_file
    """
    samples = []

    loaders = []
    if use_accelerometer:
        loaders.append((ACCEL_DIR,    "*.csv", load_mendeley_accelerometer))
    if use_pressure:
        loaders.append((PRESSURE_DIR, "*.csv", load_mendeley_pressure))
    if use_hydrophone and _SF_AVAILABLE:
        loaders.append((HYDRO_DIR,    "*.raw", load_mendeley_hydrophone))

    for base_dir, pattern, loader_fn in loaders:
        if not os.path.isdir(base_dir):
            print(f"  [WARN] Directory not found: {base_dir}")
            print(f"         Check DATASET_ROOT in config.py")
            continue

        files = glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)
        n_ok = n_skip = 0
        for f in sorted(files):
            sal = assign_salinity(rng)
            try:
                samples.append(loader_fn(f, salinity_psu=sal))
                n_ok += 1
            except Exception as e:
                if verbose:
                    print(f"  [skip] {os.path.basename(f)}: {e}")
                n_skip += 1

        if verbose:
            print(f"  {os.path.basename(base_dir)}: {n_ok} loaded, {n_skip} skipped")

    if verbose:
        n_leak   = sum(1 for s in samples if s["is_leak"])
        n_normal = len(samples) - n_leak
        print(f"\n  Total: {len(samples)} samples  |  Leak: {n_leak}  |  Normal: {n_normal}")
        print(f"  Salinity: {'fixed ' + str(SALINITY_MEAN_PSU) + ' psu' if rng is None else str(SALINITY_RANGE_PSU) + ' psu (random)'}")

    return samples