"""
run_pipeline.py  —  PULSE-AT Phase 2: Real data evaluation (Mendeley Testbed)

Dataset : "Acoustic leak detection in water pipe" (Mendeley Data)
          DOI: 10.17632/tbrnp6vrnj/1
Salinity: Nakdong EDC aquifer brackish zone 3–10 psu
          (Lee et al. 2023, Marine Geology, DOI: 10.1016/j.margeo.2023.107089)

NOTE: Results are from real acoustic sensor recordings on a 47 m PVC testbed.
      Ground-truth leak labels are provided by the dataset authors.
      Localization MAE is not reported (no true_dist in Mendeley data).

Usage:
    python run_pipeline.py
    python run_pipeline.py --pressure-only
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
parser.add_argument("--accel-only",     action="store_true")
parser.add_argument("--pressure-only",  action="store_true")
parser.add_argument("--hydro",          action="store_true",
                    help="Include hydrophone (requires soundfile)")
parser.add_argument("--random-salinity", action="store_true",
                    help="Draw salinity from [3–10 psu] range per sample")
parser.add_argument("--seq-len", type=int, default=12,
                    help="Windows per sequence for persistence gate (default 12)")
parser.add_argument("--warmup",  type=int, default=120,
                    help="CP warm-up windows (default 120)")
args = parser.parse_args()

SEQ_LEN  = args.seq_len
N_WARMUP = args.warmup
RNG      = np.random.default_rng(42) if args.random_salinity else None

use_accel    = not args.pressure_only
use_pressure = not args.accel_only
use_hydro    = args.hydro


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(tp, fp, fn, tn) -> dict:
    precision = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    far       = fp / (fp + tn)   if (fp + tn) > 0 else 0.0
    return dict(
        Precision=round(precision, 3),
        Recall=round(recall, 3),
        F1=round(f1, 3),
        FAR=round(far, 3),
        TP=tp, FP=fp, FN=fn, TN=tn,
    )


# ── Warm-up: calibrate Mondrian CP on normal (NL) samples ─────────────────
def warmup_brain(samples: list, n: int = N_WARMUP) -> PULSE_AT_Brain:
    """
    Run n normal-condition windows through a single brain to warm up the
    Mondrian conformal predictor calibration set.
    Returns the calibrated brain (its _cp._cal is copied to eval brains).
    """
    normal_sigs = [
        s["mic1_sig"] for s in samples
        if not s["is_leak"] and len(s["mic1_sig"]) > 0
    ]
    if len(normal_sigs) < n:
        print(f"  [WARN] Only {len(normal_sigs)} normal samples for warm-up "
              f"(requested {n}). Using all.")
        n = len(normal_sigs)

    brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                           psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)
    _rng = np.random.default_rng(0)
    for i in range(n):
        sig = normal_sigs[i % len(normal_sigs)]
        # Chunk into windows of 1 s for warm-up
        window_size = FS
        for start in range(0, len(sig) - window_size, window_size):
            chunk = sig[start: start + window_size]
            brain.process({"mic1_sig": chunk, "fs": FS, "salinity": 7.0})

    print(f"  CP warm-up: {brain._cal_n} calibration points accumulated.")
    return brain


# ── Sequence builder ───────────────────────────────────────────────────────
def _chunk_signal(signal: np.ndarray, window_s: float = 1.0,
                  fs: int = FS) -> list:
    """Split a full recording into non-overlapping 1-second windows."""
    win = int(window_s * fs)
    return [signal[i: i + win]
            for i in range(0, len(signal) - win, win)]


def build_sequences(samples: list, seq_len: int = SEQ_LEN) -> list:
    """
    Each sample (full recording) → one sequence of seq_len consecutive windows.
    Sequences shorter than seq_len are zero-padded at start.

    Returns list of dicts:
        windows, label (0/1), salinity_psu, sensor_type, source_file
    """
    sequences = []
    for s in samples:
        chunks = _chunk_signal(s["mic1_sig"], fs=s["fs"])
        if len(chunks) == 0:
            continue
        # Take the last seq_len chunks (most likely to contain leak transient)
        if len(chunks) >= seq_len:
            windows = chunks[-seq_len:]
        else:
            # Pad with first chunk repeated
            pad = [chunks[0]] * (seq_len - len(chunks))
            windows = pad + chunks

        sequences.append({
            "windows":      windows,
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
    print(f"  Salinity: {'random [3–10] psu' if args.random_salinity else 'fixed 7.0 psu'}")
    print(f"  Seq len : {SEQ_LEN} windows × 1 s | FS: {FS} Hz\n")

    # 1. Load real data
    samples = load_real_dataset(
        use_accelerometer=use_accel,
        use_pressure=use_pressure,
        use_hydrophone=use_hydro,
        rng=RNG,
        verbose=True,
    )
    if len(samples) == 0:
        print("\n  [ERROR] No samples loaded. Check DATASET_ROOT in config.py.")
        print("  Make sure iCloud files are fully downloaded (no .icloud placeholders).")
        sys.exit(1)

    # 2. Warm-up
    print(f"\n  Warming up CP ({N_WARMUP} normal windows)...")
    ref_brain = warmup_brain(samples, N_WARMUP)

    # 3. Build sequences
    sequences = build_sequences(samples, SEQ_LEN)
    print(f"\n  Sequences built: {len(sequences)} total")

    # 4. Evaluate
    tp = fp = fn = tn = 0
    records = []

    for seq_i, seq in enumerate(sequences):
        true_label = seq["label"]
        salinity   = seq["salinity_psu"]
        flow       = seq["flow"]

        # Fresh brain per sequence; transfer CP calibration from warmup
        brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                               psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)
        brain._cp._cal = ref_brain._cp._cal.copy()
        brain._cal_n   = ref_brain._cal_n
        brain._z._buf  = type(ref_brain._z._buf)(
                             ref_brain._z._buf,
                             maxlen=ref_brain._z._buf.maxlen)

        dispatched   = False
        final_result = None

        for win in seq["windows"]:
            result = brain.process({
                "mic1_sig": win,
                "fs":       FS,
                "salinity": salinity,
                "flow":     flow,
                # no true_dist → error_m = None (honest; Mendeley has no GT distance)
            })
            if result["flag"] == "DISPATCH" and not dispatched:
                dispatched   = True
                final_result = result

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
            "dispatched":   dispatched,
            "pred_leak":    is_leak_pred,
            "anomaly":      final_result.get("anomaly",  "—") if final_result else "—",
            "priority":     final_result.get("priority", "—") if final_result else "—",
            "severity":     final_result.get("severity", "—") if final_result else "—",
            "pvalue":       final_result.get("pvalue",   "—") if final_result else "—",
            "velocity_ms":  final_result.get("velocity_ms", "—") if final_result else "—",
        })

    # 5. Report
    m = compute_metrics(tp, fp, fn, tn)
    print("\n" + "─" * 50)
    print("  RESULTS  [Real Data — Mendeley Testbed, 47 m PVC]")
    print("─" * 50)
    for k, v in m.items():
        print(f"  {k:<14}: {v}")
    print("─" * 50)
    print("  Localisation MAE: n/a (no ground-truth distance in Mendeley data)")
    print("  Salinity source : Lee et al. (2023), DOI: 10.1016/j.margeo.2023.107089")
    print("─" * 50)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n  Saved: {RESULTS_CSV}")

    # Per-sensor-type breakdown
    if df["sensor_type"].nunique() > 1:
        print("\n  Per-sensor breakdown:")
        for stype, grp in df.groupby("sensor_type"):
            t = int((grp["pred_leak"] & (grp["true_label"]==1)).sum())
            f = int((grp["pred_leak"] & (grp["true_label"]==0)).sum())
            n = int((~grp["pred_leak"] & (grp["true_label"]==1)).sum())
            g = int((~grp["pred_leak"] & (grp["true_label"]==0)).sum())
            sm = compute_metrics(t, f, n, g)
            print(f"    {stype:<14} F1={sm['F1']}  Recall={sm['Recall']}  "
                  f"FAR={sm['FAR']}  n={len(grp)}")

    print("\n  Next: BattLeDIM pressure/flow validation → Phase 3.\n")


if __name__ == "__main__":
    run()