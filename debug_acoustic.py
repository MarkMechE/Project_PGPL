import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pgpl_brain import PGPLBrain, TidalWindow
from src.mendeley_loader import load_hydrophone_pairs, read_raw_pair
from scipy.signal import butter, sosfilt

pairs = load_hydrophone_pairs()
print("Total pairs:", len(pairs))
print()

# Separate into leak and no-leak
leaks    = [p for p in pairs if p["label"] == 1]
no_leaks = [p for p in pairs if p["label"] == 0]

print("=== NO-LEAK PAIRS (label=0) — should NOT be detected ===")
print("stem | p3 | fused | thresh | conf")
print("-" * 75)

for j, info in enumerate(no_leaks):
    try:
        sig_a, sig_b, fs = read_raw_pair(info["path_a"], info["path_b"])
        brain = PGPLBrain(fs=fs, pipe_diameter_m=0.05,
                          pipe_thickness_m=0.005, saline=False,
                          sensor_spacing_m=1.5)
        for ph in ["ebb","flood","slack_low","slack_high","spring"]:
            brain.tidal.add_phase(TidalWindow(
                phase=ph, psi_offset=0.0,
                alpha_adj=0.05, timestamp=float(j)))
        event = brain.process_acoustic(sig_a, sig_b, float(j), "ebb", 0.0)
        gate  = event.meta.get("gate", {})
        conf  = "WRONG-YES" if gate.get("confirmed") else "correct-NO"
        print(info["stem"], "|",
              round(event.p3_score, 4), "|",
              round(event.fused_score(), 4), "|",
              round(gate.get("adj_threshold", 0), 4), "|",
              conf)
    except Exception as e:
        print("ERR", info["stem"], str(e))

print()
print("=== LEAK PAIRS (label=1) — should BE detected ===")
print("stem | p3 | fused | thresh | conf")
print("-" * 75)

for j, info in enumerate(leaks):
    try:
        sig_a, sig_b, fs = read_raw_pair(info["path_a"], info["path_b"])
        brain = PGPLBrain(fs=fs, pipe_diameter_m=0.05,
                          pipe_thickness_m=0.005, saline=False,
                          sensor_spacing_m=1.5)
        for ph in ["ebb","flood","slack_low","slack_high","spring"]:
            brain.tidal.add_phase(TidalWindow(
                phase=ph, psi_offset=0.0,
                alpha_adj=0.05, timestamp=float(j)))
        event = brain.process_acoustic(sig_a, sig_b, float(j), "ebb", 0.0)
        gate  = event.meta.get("gate", {})
        conf  = "correct-YES" if gate.get("confirmed") else "MISSED-NO"
        print(info["stem"], "|",
              round(event.p3_score, 4), "|",
              round(event.fused_score(), 4), "|",
              round(gate.get("adj_threshold", 0), 4), "|",
              conf)
    except Exception as e:
        print("ERR", info["stem"], str(e))

print()
print("=== SUMMARY ===")