"""
config.py  —  Project PULSE-AT path configuration.

Edit DATASET_ROOT to point to your local copy of the Mendeley dataset.
Default assumes iCloud sync on macOS.
"""
import os

# ── Dataset root ───────────────────────────────────────────────────────────
# Change this line if your dataset is somewhere else.
DATASET_ROOT = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/Project_PGPL_Dataset"
)

ACCEL_DIR    = os.path.join(DATASET_ROOT, "Accelerometer")
PRESSURE_DIR = os.path.join(DATASET_ROOT, "Dynamic Pressure Sensor")
HYDRO_DIR    = os.path.join(DATASET_ROOT, "Hydrophone")

# ── Salinity ───────────────────────────────────────────────────────────────
# Source: Lee et al. (2023), Marine Geology, DOI: 10.1016/j.margeo.2023.107089
# Nakdong Estuary reclaimed aquifer (EDC pipe zone): brackish 3–10 psu
# (KHOA 2020 tidal data cited therein; open seawater 25–35 psu is NOT used)
SALINITY_MEAN_PSU  = 7.0
SALINITY_RANGE_PSU = (3.0, 10.0)

# ── Pipeline constants ─────────────────────────────────────────────────────
FS               = 2000    # Hz — resample target for classifier
ALPHA            = 0.10    # Mondrian CP significance level
PERSISTENCE_N    = 6       # windows per evaluation block
PERSIST_RATIO    = 0.67    # Claim 1: ≥ 4/6 windows must flag
PSI_THRESHOLD    = 0.20
ZONE_WEIGHT      = 0.5     # default; override per pipe segment if known

# ── Output paths ───────────────────────────────────────────────────────────
OUTPUTS_DIR      = "outputs"
RESULTS_CSV      = os.path.join(OUTPUTS_DIR, "real_data_results.csv")