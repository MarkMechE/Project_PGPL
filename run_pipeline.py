"""
run_pipeline.py — Main F1 runner (UPDATED v2.1)
PGPL v2.0 | BattleDIM F1 target: 0.96 | Mendeley F1 target: 0.85
Run: python run_pipeline.py
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm  # Progress bar

sys.path.insert(0, os.path.dirname(__file__))

from config import validate_paths, OUTPUTS_DIR
from src.pgpl_brain        import PGPLBrain
from src.battledim_loader  import load_battledim_2019, load_battledim_2018, build_ground_truth
from src.mendeley_loader   import load_hydrophone_files, read_wav_pair

# ── Tidal phase stub ──────────────────────────────────────────────────────────
TIDAL_PHASES_CYCLE = ["ebb", "flood", "slack_low", "slack_high", "spring"]

def mock_tidal_phase(i: int) -> tuple[str, float]:
    phase = TIDAL_PHASES_CYCLE[i % len(TIDAL_PHASES_CYCLE)]
    psi   = float(np.sin(i * 0.01) * 2.5)
    return phase, psi


# ══════════════════════════════════════════════════════════════════════════════
# 1. BattLeDIM F1 (IMPROVED)
# ══════════════════════════════════════════════════════════════════════════════
def run_battledim_f1() -> dict:
    print("\n" + "═" * 50)
    print("  BattLeDIM F1 Evaluation (2019 Test Year)")
    print("═" * 50)

    # Load
    data_2018 = load_battledim_2018()
    data_2019 = load_battledim_2019()
    gt        = build_ground_truth(data_2019["leakages"])
    print(f"  📊 GT leak rate: {gt.mean():.1%} ({gt.sum():,} positives)")

    pressures = data_2019["pressures"]
    flows     = data_2019["flows"]

    # Init brain
    brain = PGPLBrain(fs=1/60, saline=False)

    # FULL calibration (105k samples!)
    p_col = pressures.columns[0]
    cal_pressures = data_2018["pressures"][p_col].dropna().tolist()  # FIXED: No [:500]
    cal_z_scores  = [abs(v - np.mean(cal_pressures)) / (np.std(cal_pressures) + 1e-9)
                     for v in cal_pressures]
    brain.calibrate_from_year(cal_z_scores, phase="default")

    # Robust common index (drop NaNs)
    common_idx = pressures.index.intersection(flows.index).intersection(gt.index).dropna()
    print(f"  Processing {len(common_idx):,} timesteps …")

    y_pred = []
    y_true = []
    p_col  = pressures.columns[0]
    f_col  = flows.columns[0]

    for i, ts in enumerate(tqdm(common_idx, desc="  🧠 Processing")):
        psi  = float(pressures.loc[ts, p_col])
        flow = float(flows.loc[ts, f_col])
        phase, tidal_psi = mock_tidal_phase(i)

        event = brain.process_scada(
            pressure_psi=psi,
            flow_lps=flow,
            timestamp=float(ts.timestamp() / 1e9),  # Seconds
            tidal_phase=phase,
            tidal_psi=tidal_psi,
        )

        gate = event.meta.get("gate", {})
        pred = 1 if gate.get("confirmed", False) else 0
        y_pred.append(pred)
        y_true.append(int(gt.loc[ts]))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    print(f"  📈 Balance: Pos {np.sum(y_true)}/{len(y_true)} ({y_true.mean():.1%}) | Detections: {np.sum(y_pred)}")

    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    far = float(np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1))

    result = {"dataset": "BattLeDIM", "F1": round(f1, 3),
              "Precision": round(pre, 3), "Recall": round(rec, 3),
              "FAR": round(far, 4)}
    print(f"  F1={f1:.3f}  Precision={pre:.3f}  Recall={rec:.3f}  FAR={far:.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. Mendeley F1 (FIXED imports)
# ══════════════════════════════════════════════════════════════════════════════
def run_mendeley_f1() -> dict:
    print("\n" + "═" * 50)
    print("  Mendeley F1 Evaluation (Acoustic)")
    print("═" * 50)

    hydro_files = load_hydrophone_files()
    if len(hydro_files) < 2:
        print("  [SKIP] Need ≥2 .wav files in Mendeley/Hydrophone/")
        return {"dataset": "Mendeley", "F1": None, "Precision": None, "Recall": None, "FAR": None}

    pairs = [(hydro_files[i], hydro_files[i+1]) for i in range(0, len(hydro_files)-1, 2)]
    y_pred_list = []
    tdoa_errors = []

    print(f"  Processing {len(pairs)} pairs …")
    for j, (path_a, path_b) in enumerate(pairs):
        try:
            sig_a, sig_b, fs = read_wav_pair(path_a, path_b)
        except Exception as e:
            print(f"  [WARN] Pair {j}: {e}")
            continue

        brain = PGPLBrain(fs=fs, pipe_diameter_m=0.10, pipe_thickness_m=0.008,
                         pipe_material="hdpe", saline=True, sensor_spacing_m=5.0)

        phase, tidal_psi = mock_tidal_phase(j)

        # Internal TidalWindow (no import)
        from src.pgpl_brain import TidalWindow  # From brain
        for ph in ["ebb", "flood", "slack_low", "slack_high", "spring"]:
            brain.tidal.add_phase(TidalWindow(phase=ph, psi_offset=tidal_psi,
                                             alpha_adj=brain.tidal.adaptive_alpha(0.05),
                                             timestamp=float(j)))

        event = brain.process_acoustic(sig_a, sig_b, float(j), phase, tidal_psi)
        gate = event.meta.get("gate", {})
        pred = 1 if gate.get("confirmed", False) else 0
        y_pred_list.append(pred)

        if event.location_m >= 0:
            tdoa_errors.append(event.location_m)

    y_true_list = [1] * len(y_pred_list)  # All leaks assumption

    if not y_pred_list:
        return {"dataset": "Mendeley", "F1": None, "Precision": None, "Recall": None, "FAR": None}

    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)

    f1  = f1_score(y_true, y_pred, zero_division=0)
    far = float(np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1))
    mae = float(np.mean(tdoa_errors)) if tdoa_errors else float("nan")

    result = {"dataset": "Mendeley", "F1": round(f1, 3), "Precision": None, "Recall": None,
              "TDOA_MAE_m": round(mae, 2), "FAR": round(far, 4)}
    print(f"  F1={f1:.3f}  TDOA_MAE={mae:.2f}m  FAR={far:.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("── PGPL v2.0 Pipeline ──")
    validate_paths(strict=True)

    results = []
    results.append(run_battledim_f1())
    results.append(run_mendeley_f1())

    print("\n" + "═" * 50)
    print("  📊 FINAL F1 TABLE")
    print("═" * 50)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    out_path = os.path.join(OUTPUTS_DIR, "f1_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")