"""
config.py — Project PGPL path configuration.
Auto-detects Windows (iCloudDrive) vs Mac (Library/Mobile Documents).
"""
import os
import sys
from typing import Optional

# ── Auto-detect iCloud path (fallback to env var) ───────────────────────────
DATASET_ROOT = os.environ.get("PGPL_DATASET_ROOT")  # Override via env
if not DATASET_ROOT:
    if sys.platform == "win32":
        DATASET_ROOT = os.path.join(os.environ["USERPROFILE"], "iCloudDrive", "Project_PGPL_Dataset")
    else:
        DATASET_ROOT = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Project_PGPL_Dataset")

# ── Mendeley dirs ───────────────────────────────────────────────────────────
MENDELEY_DIR = os.path.join(DATASET_ROOT, "Mendeley")
ACCEL_DIR    = os.path.join(MENDELEY_DIR, "Accelerometer")
PRESSURE_DIR = os.path.join(MENDELEY_DIR, "Dynamic Pressure Sensor")
HYDRO_DIR    = os.path.join(MENDELEY_DIR, "Hydrophone")

# ── BattleDIM ───────────────────────────────────────────────────────────────
BATTLEDIM_DIR = os.path.join(DATASET_ROOT, "BattleDIM")

# ── BattleDIM Files (Optional: Set to None if not needed) ───────────────────
FLOWS_2019    = os.path.join(BATTLEDIM_DIR, "2019_SCADA_Flows.csv")
PRESSURES_2019 = os.path.join(BATTLEDIM_DIR, "2019_SCADA_Pressures.csv")
LEAKAGES_2019 = os.path.join(BATTLEDIM_DIR, "2019_Leakages.csv")
FLOWS_2018    = os.path.join(BATTLEDIM_DIR, "2018_SCADA_Flows.csv")
PRESSURES_2018 = os.path.join(BATTLEDIM_DIR, "2018_SCADA_Pressures.csv")
LTOWN_INP     = os.path.join(BATTLEDIM_DIR, "L-TOWN.inp")

# ── Salinity (Lee et al. 2023, Marine Geology) ──────────────────────────────
SALINITY_MEAN_PSU  = 7.0
SALINITY_RANGE_PSU = (3.0, 10.0)

# ── Pipeline constants ──────────────────────────────────────────────────────
FS            = 2000
ALPHA         = 0.05
PERSISTENCE_N = 6
PERSIST_RATIO = 0.67
PSI_THRESHOLD = 0.20
ZONE_WEIGHT   = 0.5

# ── Output paths ────────────────────────────────────────────────────────────
OUTPUTS_DIR = "outputs"
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "real_data_results.csv")


def validate_paths(verbose: bool = False) -> bool:
    """Check key paths exist. Returns True if OK."""
    paths = [DATASET_ROOT, MENDELEY_DIR, BATTLEDIM_DIR]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        if verbose:
            print(f"❌ Missing: {missing}")
        return False
    if verbose:
        print("✅ All key paths exist.")
    return True


# Auto-validate on import (quiet by default)
validate_paths(verbose=False)