"""
run_pipeline.py  —  PULSE-AT Phase 2: Real data evaluation (Mendeley Testbed)

Dataset : "Acoustic leak detection in water pipe" (Mendeley Data)
          DOI: 10.17632/tbrnp6vrnj/1
Salinity: Nakdong EDC aquifer brackish zone 3-10 psu
          (Lee et al. 2023, Marine Geology, DOI: 10.1016/j.margeo.2023.107089)

NOTE: Results are from real acoustic sensor recordings on a 47 m PVC testbed.
      Ground-truth leak labels provided by dataset authors.
      Localization MAE not reported (no true_dist in Mendeley data).

Usage:
    python run_pipeline.py                  # accel + pressure
    python run_pipeline.py --accel-only     # accelerometer only
    python run_pipeline.py --pressure-only  # pressure only
    python run_pipeline.py --random-salinity
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

from config import (
    FS, ALPHA, PERSISTENCE_N, PSI_THRESHOLD, ZONE_WEIGHT,
    RESULTS_CSV,
)
from src.pgpl_brain import PULSE_AT_Brain, LEAK_CLASSES
from src.real_data_loader import load_real_dataset

# ── CLI args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="PULSE-AT real data evaluation")
parser.add_argument("--accel-only",      action="store_true")
parser.add_argument("--pressure-only",   action="store_true")
parser.add_argument("--hydro",           action="store_true")
parser.add_argument("--random-salinity", action="store_true")
parser.add_argument("--seq-len", type=int, default=6,
                    help="Windows per sequence (default 6 = persistence_n)")
parser.add_argument("--warmup",  type=int, default=120)
args = parser.parse_args()

SEQ_LEN  = args.seq_len
N_WARMUP = args.warmup
RNG      = np.random.default_rng(42) if args.random_salinity else None

use_accel    = not args.pressure_only
use_pressure = not args.accel_only
use_hydro    = args.hydro


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(tp, fp, fn, tn) -> dict:
    precision = tp / (tp + fp)  if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)  if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    far       = fp / (fp + tn)  if (fp + tn) > 0 else 0.0
    return dict(
        Precision=round(precision, 3),
        Recall=round(recall, 3),
        F1=round(f1, 3),
        FAR=round(far, 3),
        TP=tp, FP=fp, FN=fn, TN=tn,
    )


# ── Warm-up ────────────────────────────────────────────────────────────────
def warmup_brain(n_requested: int = N_WARMUP) -> PULSE_AT_Brain:
    """
    Always loads ALL sensor types for warm-up regardless of eval mode.
    This maximises normal-condition calibration samples.
    Warm-up brain's CP cal is transferred to every eval brain.
    """
    # Load all sensors for warm-up (accel + pressure + hydro if available)
    all_samples = load_real_dataset(
        use_accelerometer=True,
        use_pressure=True,
        use_hydrophone=use_hydro,
        rng=None,           # fixed salinity for reproducible calibration
        verbose=False,
    )

    normal_sigs = [s["mic1_sig"] for s in all_samples if not s["is_leak"]]

    brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                           psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)

    if len(normal_sigs) == 0:
        print("  [WARN] No normal samples found for warm-up — CP uncalibrated.")
        return brain

    # Cycle through normal recordings, feeding 1-second windows
    count = 0
    window_size = FS
    idx = 0
    while count < n_requested:
        sig = normal_sigs[idx % len(normal_sigs)]
        for start in range(0, len(sig) - window_size, window_size):
            chunk = sig[start: start + window_size]
            brain.process({"mic1_sig": chunk, "fs": FS, "salinity": 7.0})
            count += 1
            if count >= n_requested:
                break
        idx += 1

    print(f"  CP warm-up: {brain._cal_n} calibration points from "
          f"{len(normal_sigs)} normal recordings.")
    return brain


# ── Sequence builder ───────────────────────────────────────────────────────
def _chunk_signal(signal: np.ndarray, fs: int = FS) -> list:
    """Split full recording into non-overlapping 1-second windows."""
    win = int(fs)
    return [signal[i: i + win] for i in range(0, len(signal) - win, win)]


def build_sequences(samples: list) -> list:
    """
    Each Mendeley recording → sliding windows fed sequentially to one brain.
    The brain accumulates flags across ALL windows (not just last seq_len).
    A DISPATCH at any window = positive prediction for that recording.
    seq_len is used as the minimum windows before gate can fire.
    """
    sequences = []
    for s in samples:
        chunks = _chunk_signal(s["mic1_sig"], fs=s["fs"])
        if len(chunks) < 2:
            continue
        sequences.append({
            "windows":      chunks,          # ALL windows, not truncated
            "label":        int(s["is_leak"]),
            "salinity_psu": s["salinity"],
            "sensor_type":  s.get("sensor_type", "unknown"),
            "source_file":  s.get("source_file", "?"),
            "flow":         s.get("flow", 13.0),
        })
    return sequences


# ── Evaluation loop ────────────────────────────────────────────────────────
def run():
    print("\n[PULSE-AT] Phase 2 — Real Data Evaluation (Mendeley Testbed)")
    print(f"  Sensors : accel={use_accel}, pressure={use_pressure}, hydro={use_hydro}")
    print(f"  Salinity: {'random [3-10] psu' if args.random_salinity else 'fixed 7.0 psu'}")
    print(f"  Gate    : persistence_n={PERSISTENCE_N}, ratio>={0.67}, alpha={ALPHA}\n")

    # 1. Load ALL sensor types for warm-up calibration
    print(f"  Loading all sensors for CP warm-up...")
    all_samples = load_real_dataset(
        use_accelerometer=True,
        use_pressure=True,
        use_hydrophone=use_hydro,
        rng=None,
        verbose=False,
    )
    normal_sigs = [s["mic1_sig"] for s in all_samples if not s["is_leak"]]
    print(f"  Warming up CP on {len(normal_sigs)} normal recordings ({N_WARMUP} windows)...")
    ref_brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                               psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)
    count = 0
    idx = 0
    while count < N_WARMUP and len(normal_sigs) > 0:
        sig = normal_sigs[idx % len(normal_sigs)]
        for start in range(0, len(sig) - FS, FS):
            chunk = sig[start: start + FS]
            ref_brain.process({"mic1_sig": chunk, "fs": FS, "salinity": 7.0})
            count += 1
            if count >= N_WARMUP:
                break
        idx += 1
    print(f"  CP warm-up: {ref_brain._cal_n} calibration points.")

    # 2. Load eval samples (only selected sensor types)
    print(f"\n  Loading eval samples...")
    samples = load_real_dataset(
        use_accelerometer=use_accel,
        use_pressure=use_pressure,
        use_hydrophone=use_hydro,
        rng=RNG,
        verbose=True,
    )
    if len(samples) == 0:
        print("\n  [ERROR] No samples loaded. Check DATASET_ROOT in config.py.")
        sys.exit(1)

    # 3. Build sequences
    sequences = build_sequences(samples)
    print(f"\n  Sequences: {len(sequences)} recordings → all windows fed sequentially")

    # 4. Evaluate — one fresh brain per recording
    tp = fp = fn = tn = 0
    records = []

    for seq_i, seq in enumerate(sequences):
        true_label = seq["label"]
        salinity   = seq["salinity_psu"]
        flow       = seq["flow"]

        # Fresh brain per recording; transfer CP calibration from warmup
        brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                               psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)
        brain._cp._cal = ref_brain._cp._cal.copy()
        brain._cal_n   = ref_brain._cal_n
        brain._z._buf  = type(ref_brain._z._buf)(
                             ref_brain._z._buf,
                             maxlen=ref_brain._z._buf.maxlen)

        dispatched   = False
        final_result = None

        # Feed ALL windows — gate fires when ratio >= 0.67 across buffer
        for win in seq["windows"]:
            result = brain.process({
                "mic1_sig": win,
                "fs":       FS,
                "salinity": salinity,
                "flow":     flow,
            })
            if result["flag"] == "DISPATCH" and not dispatched:
                dispatched   = True
                final_result = result
                # Don't break — let remaining windows run for PSI tracking

        is_leak_pred = (dispatched and
                        final_result is not None and
                        final_result.get("anomaly", "") in LEAK_CLASSES)

        if   is_leak_pred and true_label == 1: tp += 1
        elif is_leak_pred and true_label == 0: fp += 1
        elif not is_leak_pred and true_label == 1: fn += 1
        else:                                       tn += 1

        records.append({
            "seq":          seq_i + 1,
            "source_file":  seq["source_file"],
            "sensor_type":  seq["sensor_type"],
            "true_label":   true_label,
            "salinity_psu": round(salinity, 2),
            "n_windows":    len(seq["windows"]),
            "dispatched":   dispatched,
            "pred_leak":    is_leak_pred,
            "anomaly":      final_result.get("anomaly",  "-") if final_result else "-",
            "priority":     final_result.get("priority", "-") if final_result else "-",
            "severity":     final_result.get("severity", "-") if final_result else "-",
            "pvalue":       final_result.get("pvalue",   "-") if final_result else "-",
            "velocity_ms":  final_result.get("velocity_ms", "-") if final_result else "-",
        })

    # 5. Report
    m = compute_metrics(tp, fp, fn, tn)
    print("\n" + "-" * 50)
    print("  RESULTS  [Real Data - Mendeley Testbed, 47 m PVC]")
    print("-" * 50)
    for k, v in m.items():
        print(f"  {k:<14}: {v}")
    print("-" * 50)
    print("  Localisation MAE : n/a (no ground-truth distance in Mendeley data)")
    print("  Salinity source  : Lee et al. (2023), DOI: 10.1016/j.margeo.2023.107089")
    print("-" * 50)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n  Saved: {RESULTS_CSV}")

    # Per-sensor breakdown (only shown when multiple sensor types evaluated)
    if df["sensor_type"].nunique() > 1:
        print("\n  Per-sensor breakdown:")
        for stype, grp in df.groupby("sensor_type"):
            t = int((grp["pred_leak"] & (grp["true_label"] == 1)).sum())
            f = int((grp["pred_leak"] & (grp["true_label"] == 0)).sum())
            fn_ = int((~grp["pred_leak"] & (grp["true_label"] == 1)).sum())
            g = int((~grp["pred_leak"] & (grp["true_label"] == 0)).sum())
            sm = compute_metrics(t, f, fn_, g)
            print(f"    {stype:<22}  F1={sm['F1']}  "
                  f"Recall={sm['Recall']}  Precision={sm['Precision']}  "
                  f"FAR={sm['FAR']}  n={len(grp)}")

    # Anomaly class distribution
    if "anomaly" in df.columns:
        dispatched_df = df[df["dispatched"] == True]
        if len(dispatched_df) > 0:
            print("\n  Anomaly class distribution (dispatched only):")
            for cls, cnt in dispatched_df["anomaly"].value_counts().items():
                print(f"    {cls:<14}: {cnt}")

    print()


if __name__ == "__main__":
    run()