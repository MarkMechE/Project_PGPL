"""
battledim_loader.py — BattLeDIM SCADA loader (EU FORMAT FIXED v2.1)
PGPL v2.0 | sep=';' + decimal=',' | Robust parsing
DOI: 10.5281/zenodo.4017659
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from pandas.errors import ParserError

# ── Append project root ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import FLOWS_2019, PRESSURES_2019, LEAKAGES_2019, FLOWS_2018, PRESSURES_2018


def _load_or_die(path: Path, name: str) -> pd.DataFrame:
    """Robust loader: Auto-detects sep/decimal + 6 fallbacks."""
    print(f"  🔍 Loading {name}: {path}")
    
    # Preview for sep detection
    with open(path, 'r', encoding='utf-8') as f:
        preview = [next(f).strip() for _ in range(3)]
    print(f"  📄 Preview: {preview}")
    
    # Auto-detect separator
    sep = ';' if ';' in preview[0] else ','
    decimal = ',' if ',' in preview[1] and not preview[1].count(',') == preview[1].count(';') + 1 else '.'
    print(f"  🎯 Detected: sep='{sep}', decimal='{decimal}'")
    
    strategies = [
        # 1: Standard comma
        lambda: pd.read_csv(path, sep=',', decimal='.', index_col=0, parse_dates=True, low_memory=False),
        # 2: Semicolon + comma decimal (EU!)
        lambda: pd.read_csv(path, sep=sep, decimal=decimal, index_col=0, parse_dates=True, low_memory=False),
        # 3: Skip spaces + bad lines
        lambda: pd.read_csv(path, sep=sep, decimal=decimal, index_col=0, parse_dates=True, 
                           skipinitialspace=True, on_bad_lines='skip', low_memory=False),
        # 4: Skip junk header
        lambda: pd.read_csv(path, sep=sep, decimal=decimal, index_col=0, parse_dates=True, 
                           skiprows=1, skipinitialspace=True, on_bad_lines='skip', low_memory=False),
        # 5: No index_col → Infer ts col
        lambda: _infer_timestamp_col(path, sep, decimal),
        # 6: Raw + manual split (multi-value cells)
        lambda: _split_multivalue_cells(path, sep, decimal)
    ]
    
    for i, strat in enumerate(strategies, 1):
        try:
            df = strat()
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0 and df.shape[1] > 0:
                df = df.dropna(how='all').dropna(axis=1, how='all')
                print(f"  ✅ SUCCESS (Strat {i}): {df.shape[0]:,} rows × {df.shape[1]} cols")
                print(f"  📊 Cols: {list(df.columns[:6])}...")
                print(f"  ⏰ Range: {df.index.min()} → {df.index.max()}")
                return df
        except Exception as e:
            print(f"  ❌ Strat {i}: {str(e)[:80]}...")
    
    raise ValueError(f"❌ All 6 strats failed for {name}. Manual CSV fix needed.")


def _infer_timestamp_col(path: Path, sep: str, decimal: str) -> pd.DataFrame:
    """Infer ts col name."""
    df = pd.read_csv(path, sep=sep, decimal=decimal, skipinitialspace=True, low_memory=False, on_bad_lines='skip')
    ts_cols = [c for c in df.columns if any(kw in str(c).lower() for kw in ['time', 'date', 'timestamp'])]
    if not ts_cols:
        ts_cols = [df.columns[0]]
    df[ts_cols[0]] = pd.to_datetime(df[ts_cols[0]], errors='coerce')
    df = df.dropna(subset=[ts_cols[0]])
    df.set_index(ts_cols[0], inplace=True)
    return df


def _split_multivalue_cells(path: Path, sep: str, decimal: str) -> pd.DataFrame:
    """Split '77,77' cells into sub-cols (e.g., p227_a, p227_b)."""
    df = pd.read_csv(path, sep=sep, decimal=decimal, skipinitialspace=True, low_memory=False)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')  # First col ts
    df.set_index(df.columns[0], inplace=True)
    
    # Split multi-value cols (e.g., "77,77" → 77.77, 77)
    new_cols = {}
    for col in df.columns:
        split = df[col].astype(str).str.split(',', expand=True).apply(pd.to_numeric, errors='coerce')
        if split.shape[1] > 1:
            for i in range(split.shape[1]):
                new_cols[f"{col}_{chr(97+i)}"] = split[i]  # a,b,c...
        else:
            new_cols[col] = split[0]
    df = pd.DataFrame(new_cols, index=df.index)
    return df


# ── Loaders & GT (unchanged) ─────────────────────────────────────────────────
def load_battledim_2019() -> dict[str, pd.DataFrame]:
    print("── Loading BattLeDIM 2019 (test) ──")
    flows = _load_or_die(FLOWS_2019, "Flows 2019")
    pressures = _load_or_die(PRESSURES_2019, "Pressures 2019")
    leakages = _load_or_die(LEAKAGES_2019, "Leakages 2019")
    return {"flows": flows, "pressures": pressures, "leakages": leakages}

def load_battledim_2018() -> dict[str, pd.DataFrame]:
    print("── Loading BattLeDIM 2018 (calibration) ──")
    flows = _load_or_die(FLOWS_2018, "Flows 2018")
    pressures = _load_or_die(PRESSURES_2018, "Pressures 2018")
    return {"flows": flows, "pressures": pressures}

def build_ground_truth(leakages: pd.DataFrame) -> pd.Series:
    """
    BattleDIM GT: time-series of per-pipe leak magnitudes.
    Label = 1 if ANY pipe node actively leaking > 0.5 lps.

    Why threshold = 0.5:
      - Zero-leak rows in BattleDIM = exactly 0.0
      - Smallest real leak in dataset ≈ 1.5 lps
      - 0.5 gives clean binary separation → ~10-15% positive rate
    """
    print(f"  🏗️  Raw leaks shape: {leakages.shape}")

    leakages.index = pd.to_datetime(leakages.index, errors="coerce")
    leakages       = leakages.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    LEAK_THRESHOLD = 0.5   # lps

    gt            = (leakages > LEAK_THRESHOLD).any(axis=1).astype(int)
    gt.index.name = "Timestamp"
    gt.name       = "leak_active"

    print(
        f"  ✅  GT: {gt.sum():,} leak steps / "
        f"{len(gt):,} total  ({gt.mean():.1%} positive rate)"
    )
    return gt