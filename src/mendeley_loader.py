"""
mendeley_loader.py — Mendeley Acoustic Dataset Loader
======================================================
RAW format (confirmed from official dataset script):
  - PCM_32 (int32), 1 channel, 8000 Hz, little-endian
  - Reader: soundfile sf.read() with raw params

Dataset structure:
  Hydrophone/  {Branched,Looped} / {leak_type} / *_H1.raw + *_H2.raw
  Accelerometer/ ...             / {leak_type} / *_A1.csv + *_A2.csv
  Dynamic Pressure Sensor/ ...   / {leak_type} / *_P1.csv + *_P2.csv

GT embedded in folder name:
  "No-leak" folder → label 0
  Any leak type    → label 1

DOI: 10.17632/tbrnp6vrnj.1
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing  import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import HYDRO_DIR, ACCEL_DIR, PRESSURE_DIR

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    print("[WARN] soundfile not installed. Run: pip install soundfile")

# ── Official RAW parameters (from Mendeley dataset script) ────────────────────
RAW_FS         = 8_000.0    # Hz  — confirmed from official script
RAW_CHANNELS   = 1          # mono
RAW_SUBTYPE    = "PCM_32"   # 32-bit signed int
RAW_ENDIAN     = "LITTLE"   # little-endian
RAW_MAX_SAMPLES = 240_000   # confirmed from official script (df.head(240000))

# ── fs check: 8000 Hz < FS_ACOU_MIN(8000) edge case ─────────────────────────
# Brain acoustic path requires fs >= 8000. RAW is exactly 8000.
# Set brain fs=8000.0 → is_acoustic = (8000 >= 8000) = True ✓


# ══════════════════════════════════════════════════════════════════════════════
# GT + Metadata from path
# ══════════════════════════════════════════════════════════════════════════════
def _label_from_path(path: str) -> int:
    """
    GT label from folder name.
    'No-leak' → 0  |  anything else → 1
    Case-insensitive.
    """
    for part in Path(path).parts:
        if part.lower() == "no-leak":
            return 0
    return 1


def _leak_type_from_path(path: str) -> str:
    """Extract leak type string from folder name."""
    known = {
        "circumferential crack": "CircumferentialCrack",
        "gasket leak":           "GasketLeak",
        "longitudinal crack":    "LongitudinalCrack",
        "orifice leak":          "OrificeLeak",
        "no-leak":               "NoLeak",
        "background noise":      "BackgroundNoise",
    }
    for part in Path(path).parts:
        key = part.lower()
        if key in known:
            return known[key]
    return "Unknown"


def _topology_from_path(path: str) -> str:
    """Branched or Looped network topology."""
    for part in Path(path).parts:
        if part.lower() == "branched":
            return "Branched"
        if part.lower() == "looped":
            return "Looped"
    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Pair Scanner — builds list with GT labels from folder names
# ══════════════════════════════════════════════════════════════════════════════
def _scan_pairs(
    root_dir:   str,
    suffix_a:   str,
    suffix_b:   str,
    extensions: Tuple[str, ...],
    label:      str = "sensor",
) -> List[Dict]:
    """
    Recursively find sensor A+B pairs under root_dir.

    Returns list of dicts:
      path_a, path_b, label (0/1), leak_type, topology, stem
    """
    if not os.path.isdir(root_dir):
        print(f"  [WARN] Directory not found: {root_dir}")
        return []

    # Collect all matching files
    all_files: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in extensions:
                all_files.append(os.path.join(dirpath, fname))

    # Match A files → find B partner
    a_files = sorted([f for f in all_files if suffix_a in Path(f).stem])

    pairs = []
    missing = 0
    for path_a in a_files:
        path_b = path_a.replace(suffix_a, suffix_b)
        if not os.path.exists(path_b):
            missing += 1
            continue
        pairs.append({
            "path_a":    path_a,
            "path_b":    path_b,
            "label":     _label_from_path(path_a),
            "leak_type": _leak_type_from_path(path_a),
            "topology":  _topology_from_path(path_a),
            "stem":      Path(path_a).stem.replace(suffix_a, ""),
        })

    leak_count    = sum(p["label"]     for p in pairs)
    no_leak_count = sum(1 - p["label"] for p in pairs)

    print(f"  ✅  {label}: {len(pairs)} pairs "
          f"({leak_count} leak / {no_leak_count} no-leak)"
          + (f"  ⚠️  {missing} missing partners" if missing else ""))

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════════════════
def load_hydrophone_pairs() -> List[Dict]:
    """All H1+H2 .raw pairs with embedded GT labels."""
    return _scan_pairs(HYDRO_DIR, "_H1", "_H2", (".raw",), "Hydrophone")


def load_accelerometer_pairs() -> List[Dict]:
    """All A1+A2 .csv pairs with embedded GT labels."""
    return _scan_pairs(ACCEL_DIR, "_A1", "_A2", (".csv",), "Accelerometer")


def load_pressure_pairs() -> List[Dict]:
    """All P1+P2 .csv pairs with embedded GT labels."""
    return _scan_pairs(PRESSURE_DIR, "_P1", "_P2", (".csv",), "DynamicPressure")


# Legacy shim — keeps old run_pipeline.py working
def load_hydrophone_files() -> List[str]:
    return [p["path_a"] for p in load_hydrophone_pairs()]


# ══════════════════════════════════════════════════════════════════════════════
# RAW Reader — uses official soundfile approach
# ══════════════════════════════════════════════════════════════════════════════
def read_raw(
    path:        str,
    max_samples: int   = RAW_MAX_SAMPLES,
    fs:          float = RAW_FS,
) -> Tuple[np.ndarray, float]:
    """
    Read one Mendeley .raw file using official soundfile approach.
    Parameters match the official dataset conversion script exactly.

    Returns (signal_float32_normalised, sample_rate).
    """
    if not HAS_SF:
        raise ImportError("soundfile required. Run: pip install soundfile")
    if not os.path.exists(path):
        raise FileNotFoundError(f"RAW not found: {path}")

    # Exact params from official Mendeley script
    sig, _ = sf.read(
        path,
        channels   = RAW_CHANNELS,
        samplerate = int(fs),
        subtype    = RAW_SUBTYPE,
        endian     = RAW_ENDIAN,
    )

    sig = np.array(sig, dtype=np.float32).flatten()
    sig = sig[:max_samples]

    # Normalise to [-1, 1]
    peak = np.max(np.abs(sig)) + 1e-9
    sig  = sig / peak

    return sig, fs


def read_raw_pair(
    path_a:      str,
    path_b:      str,
    max_samples: int = RAW_MAX_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load H1 + H2 pair for TDOA.
    Returns (sig_a, sig_b, sample_rate).
    """
    sig_a, fs = read_raw(path_a, max_samples)
    sig_b, _  = read_raw(path_b, max_samples)

    # Equal length padding
    n     = max(len(sig_a), len(sig_b))
    sig_a = np.pad(sig_a, (0, n - len(sig_a)))
    sig_b = np.pad(sig_b, (0, n - len(sig_b)))

    return sig_a, sig_b, fs


# ══════════════════════════════════════════════════════════════════════════════
# CSV Reader (Accelerometer + Dynamic Pressure)
# ══════════════════════════════════════════════════════════════════════════════
def read_csv_signal(
    path:        str,
    max_samples: int = RAW_MAX_SAMPLES,
) -> Tuple[np.ndarray, float]:
    """
    Read Mendeley CSV sensor file.
    Auto-detects first numeric column.
    Normalises to [-1, 1].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, sep=None, engine="python")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError(f"No numeric columns in {path}")

    sig  = df[num_cols[0]].dropna().values[:max_samples].astype(np.float32)
    peak = np.max(np.abs(sig)) + 1e-9
    sig  = sig / peak

    return sig, RAW_FS


def read_csv_pair(
    path_a:      str,
    path_b:      str,
    max_samples: int = RAW_MAX_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load CSV sensor pair for TDOA."""
    sig_a, fs = read_csv_signal(path_a, max_samples)
    sig_b, _  = read_csv_signal(path_b, max_samples)

    n     = max(len(sig_a), len(sig_b))
    sig_a = np.pad(sig_a, (0, n - len(sig_a)))
    sig_b = np.pad(sig_b, (0, n - len(sig_b)))

    return sig_a, sig_b, fs


# ══════════════════════════════════════════════════════════════════════════════
# Legacy shim
# ══════════════════════════════════════════════════════════════════════════════
def read_wav_pair(
    path_a:      str,
    path_b:      str,
    max_samples: int = RAW_MAX_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Routes to read_raw_pair or read_csv_pair by extension."""
    ext = Path(path_a).suffix.lower()
    if ext == ".raw":
        return read_raw_pair(path_a, path_b, max_samples)
    elif ext == ".csv":
        return read_csv_pair(path_a, path_b, max_samples)
    else:
        raise ValueError(f"Unsupported extension: {ext}")