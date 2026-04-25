
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")
from config import FS, ALPHA, PERSISTENCE_N, PSI_THRESHOLD, ZONE_WEIGHT
from src.pgpl_brain import PULSE_AT_Brain, LEAK_CLASSES
from src.real_data_loader import load_real_dataset

def chunk(sig, fs=FS):
    w = int(fs)
    return [sig[i:i+w] for i in range(0, len(sig)-w, w)]

samples = load_real_dataset(use_accelerometer=True, use_pressure=False,
                            use_hydrophone=False, verbose=False)
normal = [s for s in samples if not s["is_leak"]][:2]
leaks  = [s for s in samples if s["is_leak"]][:2]

def inspect(s, label):
    brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                           psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)
    chunks = chunk(s["mic1_sig"])
    print(f"\n[{label}] {s['source_file']}  ({len(chunks)} windows)")
    print(f"  win   energy      z       pval    flags          flag")
    for i, c in enumerate(chunks[:20]):
        r = brain.process({"mic1_sig": c, "fs": FS, "salinity": 7.0})
        print(f"  {i:>3}   {r['score']:>8.5f}  {'?':>6}  {r['pvalue']:>8.4f}  "
              f"{str(list(brain._flag_buf)):>14}  {r['flag']}")

print("\n=== NORMAL ===")
for s in normal: inspect(s, "NL")
print("\n=== LEAK ===")
for s in leaks: inspect(s, "LEAK")
