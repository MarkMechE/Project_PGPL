"""
main_pipeline.py  —  PGL Phase 1 evaluation (time-series sequential)
python main_pipeline.py

NOTE: All results are SYNTHETIC physics-informed simulation.
      Design-phase indicators only — not validated field performance.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

os.makedirs("outputs", exist_ok=True)

from pgpl_brain import PULSE_AT_Brain, LEAK_CLASSES

FS       = 2000
N_WARMUP = 120
N_SEQ    = 50    # sequences
SEQ_LEN  = 12    # windows per sequence (> persistence_n=6)

_RNG = np.random.default_rng(42)


# ── Physics-informed signal generator (inline, no extra file needed) ───────
def _bandpass(sig, fs, low=200.0, high=800.0):
    sos = butter(4, [low / (0.5 * fs), high / (0.5 * fs)],
                 btype="band", output="sos")
    return sosfilt(sos, sig)

def _generate_window(fs, is_leak, distance_m, salinity_psu, tidal_phase):
    n   = int(fs * 10.0)
    t   = np.linspace(0.0, 10.0, n)
    sig = _RNG.normal(0.0, 0.04, n)
    sig *= 1.0 + 0.45 * np.sin(2.0 * np.pi * t / 44640.0 + tidal_phase)
    sig += 0.015 * np.sin(2.0 * np.pi * 50.0 * t)
    sig += 0.008 * np.sin(2.0 * np.pi * 150.0 * t)
    if is_leak:
        alpha = 0.018 * (salinity_psu / 10.0) + 0.005
        att   = np.exp(-alpha * distance_m)
        leak  = _RNG.uniform(-1.0, 1.0, n) * 0.6
        if distance_m < 8.0:
            idx = n // 2
            leak[idx: idx + int(0.01 * fs)] += 4.0
        sig += leak * att
    return _bandpass(sig, fs)

def make_timeseries_dataset(n_sequences=N_SEQ, seq_length=SEQ_LEN, fs=FS):
    """
    Each sequence = seq_length consecutive windows from the SAME pipe state.
    This matches how the persistence gate operates in deployment.
    """
    sequences = []
    for _ in range(n_sequences):
        is_leak  = bool(_RNG.choice([True, False]))
        salinity = float(_RNG.uniform(3.0, 10.0))
        distance = float(_RNG.uniform(2.0, 40.0))
        windows  = [
            _generate_window(fs, is_leak, distance, salinity,
                             float(_RNG.uniform(0.0, 6.283)))
            for _ in range(seq_length)
        ]
        sequences.append({
            "windows":      windows,
            "label":        int(is_leak),
            "distance_m":   distance,
            "salinity_psu": salinity,
        })
    return sequences


def compute_metrics(tp, fp, fn, tn, errors):
    precision = tp / (tp + fp)       if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)       if (tp + fn) > 0 else 0.0
    # Standard F1
    f1  = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    far = fp / (fp + tn)             if (fp + tn) > 0 else 0.0
    mae = float(np.mean(errors))     if errors else float("nan")
    return dict(Precision=round(precision, 3), Recall=round(recall, 3),
                F1=round(f1, 3), FAR=round(far, 3),
                MAE_m=round(mae, 2) if not np.isnan(mae) else "n/a",
                TP=tp, FP=fp, FN=fn, TN=tn)


def run():
    print("\n[SYNTHETIC SIMULATION — Phase 1 prototype]")
    print(f"  Sequences: {N_SEQ} × {SEQ_LEN} windows | FS: {FS} Hz | "
          f"CP warm-up: {N_WARMUP}\n")

    # CP warm-up on a single brain instance with clean noise
    warmup_brain = PULSE_AT_Brain(alpha=0.10, persistence_n=6,
                                  psi_threshold=0.20, zone_weight=0.5)
    for _ in range(N_WARMUP):
        sig = _generate_window(FS, False,
                               float(_RNG.uniform(2, 40)),
                               float(_RNG.uniform(3, 10)),
                               float(_RNG.uniform(0, 6.283)))
        warmup_brain.process({"mic1_sig": sig, "fs": FS, "salinity": 7.0})

    dataset = make_timeseries_dataset()
    tp = fp = fn = tn = 0
    errors, records = [], []

    for seq_i, seq in enumerate(dataset):
        true_label = seq["label"]
        salinity   = seq["salinity_psu"]
        distance   = seq["distance_m"]

        # Fresh brain per sequence (simulates independent pipe segment)
        brain = PULSE_AT_Brain(alpha=0.10, persistence_n=6,
                               psi_threshold=0.20, zone_weight=0.5)
        # Transfer calibration from warmup
        brain._cp._cal  = warmup_brain._cp._cal.copy()
        brain._cal_n    = warmup_brain._cal_n
        brain._z._buf   = type(warmup_brain._z._buf)(
                              warmup_brain._z._buf,
                              maxlen=warmup_brain._z._buf.maxlen)

        dispatched    = False
        final_result  = None

        for win in seq["windows"]:
            result = brain.process({
                "mic1_sig":  win,
                "fs":        FS,
                "salinity":  salinity,
                "true_dist": distance,
            })
            if result["flag"] == "DISPATCH" and not dispatched:
                dispatched   = True
                final_result = result

        is_leak_pred = dispatched and final_result.get("anomaly", "") in LEAK_CLASSES

        if   is_leak_pred and true_label == 1:
            tp += 1
            if final_result.get("error_m") is not None:
                errors.append(final_result["error_m"])
        elif is_leak_pred and true_label == 0: fp += 1
        elif not is_leak_pred and true_label == 1: fn += 1
        else: tn += 1

        records.append({
            "sequence":      seq_i + 1,
            "true_label":    true_label,
            "true_dist_m":   round(distance, 2),
            "salinity_psu":  round(salinity, 2),
            "dispatched":    dispatched,
            "predicted_leak": is_leak_pred,
            "anomaly":       final_result.get("anomaly", "—") if final_result else "—",
            "priority":      final_result.get("priority", "—") if final_result else "—",
            "severity":      final_result.get("severity", "—") if final_result else "—",
            "loc_m":         final_result.get("loc_m", "—")    if final_result else "—",
            "error_m":       final_result.get("error_m", "—")  if final_result else "—",
            "velocity_ms":   final_result.get("velocity_ms", "—") if final_result else "—",
        })

    m = compute_metrics(tp, fp, fn, tn, errors)

    print("─" * 45)
    print("  RESULTS  [SYNTHETIC — design-phase targets]")
    print("─" * 45)
    for k, v in m.items():
        print(f"  {k:<14}: {v}")
    print("─" * 45)

    path = "outputs/simulation_results.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"\n  Saved: {path}")
    print("\n  Next: Phase 2 — BattLeDIM data + lab bench validation.\n")


if __name__ == "__main__":
    run()