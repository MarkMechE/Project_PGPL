"""
tests/test_real_data_loader.py  —  Loader + salinity unit tests.

Run: python -m pytest tests/test_real_data_loader.py -v
  or: python tests/test_real_data_loader.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import tempfile
import csv

from src.real_data_loader import (
    _parse_label, assign_salinity, _resample, _normalize,
    LABEL_MAP, LEAK_CLASSES, load_mendeley_accelerometer,
    ACCEL_FS,
)
from config import SALINITY_MEAN_PSU, SALINITY_RANGE_PSU, FS


# ── Label parsing ──────────────────────────────────────────────────────────
def test_parse_label_orifice():
    assert _parse_label("BR_OL_0.18LPS_A1.csv") == "Burst"

def test_parse_label_longitudinal_crack():
    assert _parse_label("LP_LC_0.12LPS_A2.csv") == "Crack"

def test_parse_label_circumferential_crack():
    assert _parse_label("BR_CC_0.10LPS_A3.csv") == "Crack"

def test_parse_label_gasket():
    assert _parse_label("LP_GL_0.05LPS_A1.csv") == "Micro"

def test_parse_label_no_leak():
    assert _parse_label("BR_NL_0.18LPS_A1.csv") == "Tidal"

def test_parse_label_unknown_defaults_tidal():
    assert _parse_label("UNKNOWN.csv") == "Tidal"


# ── Salinity ───────────────────────────────────────────────────────────────
def test_salinity_fixed_default():
    sal = assign_salinity(rng=None)
    assert sal == SALINITY_MEAN_PSU, f"Expected {SALINITY_MEAN_PSU}, got {sal}"

def test_salinity_random_in_range():
    rng = np.random.default_rng(0)
    lo, hi = SALINITY_RANGE_PSU
    for _ in range(50):
        sal = assign_salinity(rng=rng)
        assert lo <= sal <= hi, f"Salinity {sal} out of range [{lo}, {hi}]"

def test_salinity_not_seawater():
    """Confirm we are NOT using KHOA open-seawater values (25–35 psu)."""
    lo, hi = SALINITY_RANGE_PSU
    assert hi <= 15.0, (
        f"Upper salinity bound {hi} psu looks like open seawater — "
        "should be ≤ 15 psu for EDC brackish aquifer"
    )


# ── Signal helpers ─────────────────────────────────────────────────────────
def test_resample_length():
    sig = np.random.randn(ACCEL_FS * 2)   # 2 seconds at 3000 Hz
    out = _resample(sig, ACCEL_FS, FS)    # → 2000 Hz
    expected_len = int(len(sig) * FS / ACCEL_FS)
    # Allow ±1 sample tolerance from polyphase filter
    assert abs(len(out) - expected_len) <= 1, (
        f"Resampled length {len(out)} != expected {expected_len}"
    )

def test_normalize_peak_one():
    sig = np.array([0.0, 3.5, -7.0, 2.0])
    n   = _normalize(sig)
    assert abs(np.max(np.abs(n)) - 1.0) < 1e-6


# ── Accelerometer loader ───────────────────────────────────────────────────
def test_load_accel_csv_two_column():
    """Create a synthetic Mendeley-format CSV and check loader output."""
    n = ACCEL_FS * 3     # 3 seconds
    t = np.linspace(0, 3, n)
    sig = np.sin(2 * np.pi * 500 * t) * 0.5

    # Write to temp file with Mendeley naming: BR_LC_0.18LPS_A1.csv
    with tempfile.NamedTemporaryFile(
        suffix=".csv", prefix="BR_LC_0.18LPS_A1", delete=False, mode="w"
    ) as f:
        w = csv.writer(f)
        for ti, si in zip(t, sig):
            w.writerow([round(ti, 6), round(si, 6)])
        tmp_path = f.name

    try:
        result = load_mendeley_accelerometer(tmp_path, salinity_psu=7.0)
        assert result["label"]    == "Crack"
        assert result["is_leak"]  == True
        assert result["fs"]       == FS
        assert result["salinity"] == 7.0
        assert len(result["mic1_sig"]) > 0
        assert np.max(np.abs(result["mic1_sig"])) <= 1.0 + 1e-6
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_parse_label_orifice()
    test_parse_label_longitudinal_crack()
    test_parse_label_circumferential_crack()
    test_parse_label_gasket()
    test_parse_label_no_leak()
    test_parse_label_unknown_defaults_tidal()
    test_salinity_fixed_default()
    test_salinity_random_in_range()
    test_salinity_not_seawater()
    test_resample_length()
    test_normalize_peak_one()
    test_load_accel_csv_two_column()
    print("All loader + salinity tests passed.")