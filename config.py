"""
config.py — iCloud path config + validation
PGPL v2.0 | EDC Busan
"""
import os
import sys

# ── iCloud Root ────────────────────────────────────────────────────────────────
DATASET_ROOT   = r"C:\Users\macad\iCloudDrive\Project_PGPL_Dataset"
BATTLEDIM_DIR  = os.path.join(DATASET_ROOT, "BattleDIM")
MENDELEY_DIR   = os.path.join(DATASET_ROOT, "Mendeley")
HYDRO_DIR      = os.path.join(MENDELEY_DIR, "Hydrophone")
ACCEL_DIR      = os.path.join(MENDELEY_DIR, "Accelerometer")
PRESSURE_DIR   = os.path.join(MENDELEY_DIR, "Dynamic Pressure Sensor")

# ── BattLeDIM Files ────────────────────────────────────────────────────────────
FLOWS_2019     = os.path.join(BATTLEDIM_DIR, "2019_SCADA_Flows.csv")
PRESSURES_2019 = os.path.join(BATTLEDIM_DIR, "2019_SCADA_Pressures.csv")
LEAKAGES_2019  = os.path.join(BATTLEDIM_DIR, "2019_Leakages.csv")
FLOWS_2018     = os.path.join(BATTLEDIM_DIR, "2018_SCADA_Flows.csv")
PRESSURES_2018 = os.path.join(BATTLEDIM_DIR, "2018_SCADA_Pressures.csv")
LTOWN_INP      = os.path.join(BATTLEDIM_DIR, "L-TOWN.inp")

# ── Output Dir ─────────────────────────────────────────────────────────────────
OUTPUTS_DIR    = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Validation ─────────────────────────────────────────────────────────────────
REQUIRED_PATHS = {
    "BattleDIM dir":    BATTLEDIM_DIR,
    "Mendeley dir":     MENDELEY_DIR,
    "Flows 2019":       FLOWS_2019,
    "Pressures 2019":   PRESSURES_2019,
    "Leakages 2019":    LEAKAGES_2019,
    "Flows 2018":       FLOWS_2018,
    "Pressures 2018":   PRESSURES_2018,
}

def validate_paths(strict: bool = True) -> bool:
    """Check all required data paths exist. Errors if strict=True."""
    all_ok = True
    for name, path in REQUIRED_PATHS.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status}  {name}: {path}")
        if not exists:
            all_ok = False
    if not all_ok and strict:
        print("\n[FATAL] Missing data files. Sync iCloud before running.")
        sys.exit(1)
    return all_ok

if __name__ == "__main__":
    print("── PGPL v2.0 Path Validation ──")
    ok = validate_paths(strict=False)
    print(f"\nPaths OK: {ok}")