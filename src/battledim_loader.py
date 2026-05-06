"""
battledim_loader.py — Real BattLeDIM SCADA loader
PGPL v2.0 | NO synthetic fallback — errors if missing
DOI: 10.5281/zenodo.4017659
"""
import pandas as pd
import numpy as np
import sys
import os

# ── Append project root to sys.path if needed ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import FLOWS_2019, PRESSURES_2019, LEAKAGES_2019, FLOWS_2018, PRESSURES_2018


def _load_or_die(path: str, label: str) -> pd.DataFrame:
    """Load CSV or exit with clear error — no synthetic fallback."""
    if not os.path.exists(path):
        print(f"[FATAL] Missing {label}: {path}")
        print("        Sync iCloud → Project_PGPL_Dataset/BattleDIM/")
        sys.exit(1)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"  ✅  Loaded {label}: {df.shape}")
    return df


def load_battledim_2019() -> dict[str, pd.DataFrame]:
    """
    Load 2019 BattLeDIM SCADA data (test year).

    Returns dict: {flows, pressures, leakages}
    """
    print("── Loading BattLeDIM 2019 ──")
    flows     = _load_or_die(FLOWS_2019,     "Flows 2019")
    pressures = _load_or_die(PRESSURES_2019, "Pressures 2019")
    leakages  = _load_or_die(LEAKAGES_2019,  "Leakages 2019")
    return {"flows": flows, "pressures": pressures, "leakages": leakages}


def load_battledim_2018() -> dict[str, pd.DataFrame]:
    """
    Load 2018 BattLeDIM SCADA data (calibration year).

    Returns dict: {flows, pressures}
    """
    print("── Loading BattLeDIM 2018 (calibration) ──")
    flows     = _load_or_die(FLOWS_2018,     "Flows 2018")
    pressures = _load_or_die(PRESSURES_2018, "Pressures 2018")
    return {"flows": flows, "pressures": pressures}


def build_ground_truth(leakages: pd.DataFrame) -> pd.Series:
    """
    Build binary ground truth series from leakage events table.
    Aligns to minutely timestamps.

    Returns pd.Series (1 = leak active, 0 = normal)
    """
    if leakages.empty:
        raise ValueError("Leakages dataframe is empty.")

    # Expand each row's [Start, End] range into per-minute flags
    idx = pd.date_range(
        start=leakages.index.min(),
        end=leakages.index.max(),
        freq="1min",
    )
    gt = pd.Series(0, index=idx, name="leak_active")

    for _, row in leakages.iterrows():
        try:
            start = pd.Timestamp(row.get("Start Time", row.name))
            end   = pd.Timestamp(row.get("End Time",   row.name))
            gt.loc[start:end] = 1
        except Exception:
            continue

    return gt