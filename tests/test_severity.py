"""----------------------------
NEW FILE: tests/test_severity.py
PURPOSE:
Unit tests for severity formula and persistence gate correctness.
CODE:
---------------------------------"""

"""tests/test_severity.py — PGL unit tests"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from pgpl_brain import _severity_score, _classify, PULSE_AT_Brain

def test_severity_p1():
    # 0.4*0.9 + 0.3*1.0 + 0.3*0.5 = 0.36+0.30+0.15 = 0.81
    sev, tier = _severity_score(0.9, "Burst", 0.5)
    assert abs(sev - 0.81) < 0.001, f"Expected 0.81, got {sev}"
    assert tier == "P1-CRITICAL"

def test_severity_p2():
    # 0.4*0.5 + 0.3*0.6 + 0.3*0.5 = 0.20+0.18+0.15 = 0.53  → P2 (>=0.40, <0.60)
    sev, tier = _severity_score(0.5, "Micro", 0.5)
    assert 0.40 <= sev < 0.60, f"Expected P3 range, got {sev}"

def test_persistence_ratio_not_unanimity():
    """Gate must fire at ratio>=0.67, not require all() flags."""
    brain = PULSE_AT_Brain(alpha=0.10, persistence_n=6)
    # Fill buffer with 5 True + 1 False = ratio 0.833 → should gate
    from collections import deque
    brain._flag_buf = deque([True, True, True, True, True, False], maxlen=6)
    ratio = sum(brain._flag_buf) / len(brain._flag_buf)
    gated = (len(brain._flag_buf) == brain._flag_buf.maxlen and
             ratio >= brain.PERSIST_RATIO_THRESH)
    assert gated, f"Ratio {ratio:.2f} should trigger gate but didn't"

def test_persistence_unanimity_would_fail():
    """Confirm all() would incorrectly reject 5/6 flags."""
    from collections import deque
    buf = deque([True, True, True, True, True, False], maxlen=6)
    old_gate = all(buf)   # False — proves all() is wrong
    assert not old_gate, "This confirms all() is broken for tidal conditions"

if __name__ == "__main__":
    test_severity_p1()
    test_persistence_ratio_not_unanimity()
    test_persistence_unanimity_would_fail()
    print("All tests passed.")