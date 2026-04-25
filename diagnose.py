"""
diagnose.py — inspect what the gate sees on real Mendeley data.
Run: python diagnose.py
"""
import os, sys, numpy as np
from config import FS, ALPHA, PERSISTENCE_N, PSI_THRESHOLD, ZONE_WEIGHT
from src.pgpl_brain import PULSE_AT_Brain, LEAK_CLASSES
from src.real_data_loader import load_real_dataset, _chunk_signal

# Load a handful of samples
samples = load_real_dataset(use_accelerometer=True, use_pressure=False,
                            use_hydrophone=False, verbose=False)

# Separate normal and leak
normal = [s for s in samples if not s["is_leak"]][:4]
leaks  = [s for s in samples if s["is_leak"]][:4]

def inspect(sample, label):
    brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                           psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)
    chunks = _chunk_signal(sample["mic1_sig"], fs=sample["fs"])
    print(f"\n  [{label}] {sample['source_file']}  ({len(chunks)} windows)")
    print(f"  {'win':>3}  {'energy':>10}  {'z':>8}  {'pval':>8}  {'flag_buf':>12}  {'gated'}")
    for i, chunk in enumerate(chunks[:20]):
        r = brain.process({"mic1_sig": chunk, "fs": FS,
                           "salinity": 7.0, "flow": 13.0})
        buf = list(brain._flag_buf)
        print(f"  {i:>3}  {r['score']:>10.4f}  "
              f"{'n/a':>8}  {r['pvalue']:>8.4f}  "
              f"{str(buf):>12}  {r['flag']}")

print("\n=== NORMAL SAMPLES ===")
for s in normal:
    inspect(s, "NORMAL")

print("\n=== LEAK SAMPLES ===")
for s in leaks:
    inspect(s, "LEAK")