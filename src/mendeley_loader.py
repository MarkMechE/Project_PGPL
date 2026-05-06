"""
mendeley_loader.py — Real Mendeley acoustic loader
PGPL v2.0 | NO synthetic fallback — errors if missing
DOI: 10.17632/tbrnp6vrnj.1
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import HYDRO_DIR, ACCEL_DIR, PRESSURE_DIR

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


def _scan_wav(directory: str, label: str) -> list[str]:
    """Return sorted list of .wav files. Exit if directory missing."""
    if not os.path.isdir(directory):
        print(f"[FATAL] Missing {label} directory: {directory}")
        print("        Sync iCloud → Project_PGPL_Dataset/Mendeley/")
        sys.exit(1)
    wavs = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".wav")
    ])
    if not wavs:
        print(f"[WARN]  No .wav files found in {label}: {directory}")
    else:
        print(f"  ✅  Found {len(wavs)} .wav in {label}")
    return wavs


def load_hydrophone_files() -> list[str]:
    """Return list of hydrophone .wav paths (no loading yet — lazy)."""
    return _scan_wav(HYDRO_DIR, "Hydrophone")


def load_accelerometer_files() -> list[str]:
    """Return list of accelerometer .wav paths."""
    return _scan_wav(ACCEL_DIR, "Accelerometer")


def load_pressure_files() -> list[str]:
    """Return list of dynamic pressure sensor .wav paths."""
    return _scan_wav(PRESSURE_DIR, "Dynamic Pressure Sensor")


def read_wav_pair(
    path_a: str,
    path_b: str,
    max_samples: int = 80_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load two .wav files (sensor A and B) for TDOA.

    Returns (signal_a, signal_b, sample_rate)
    """
    if not HAS_SF:
        raise ImportError("soundfile not installed. Run: pip install soundfile")
    for p in (path_a, path_b):
        if not os.path.exists(p):
            print(f"[FATAL] Missing WAV: {p}")
            sys.exit(1)

    a, fs_a = sf.read(path_a, always_2d=False)
    b, fs_b = sf.read(path_b, always_2d=False)

    if fs_a != fs_b:
        raise ValueError(f"Sample rate mismatch: {fs_a} vs {fs_b}")

    # Trim to mono + max_samples
    if a.ndim > 1: a = a[:, 0]
    if b.ndim > 1: b = b[:, 0]
    a = a[:max_samples]
    b = b[:max_samples]

    return a, b, float(fs_a)