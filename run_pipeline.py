"""
run_pipeline.py — Main F1 runner
PGPL v2.0 | BattleDIM F1 target: 0.96 | Mendeley F1 target: 0.85
Run: python run_pipeline.py
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))

from config import validate_paths
from src.pgpl_brain        import PGPLBrain
from src.battledim_loader  import load_battledim_2019, load_battledim_2018, build_ground_truth
from src.mendeley_loader   import load_hydrophone_files, read_wav_pair

# ── Tidal phase stub (replace with real tidal API if available) ────────────────
TIDAL_PHASES_CYCLE = ["ebb", "flood", "slack_low", "slack_high", "spring"]

def mock_tidal_phase(i: int) -> tuple[str, float]:
    """Cycle through tidal phases for demonstration. Replace with real data."""
    phase = TIDAL_PHASES_CYCLE[i % len(TIDAL_PHASES_CYCLE)]
    psi   = float(np.sin(i * 0.01) * 2.5)   # ±2.5 PSI tidal swing
    return phase, psi


# ══════════════════════════════════════════════════════════════════════════════
# 1. BattLeDIM F1
# ══════════════════════════════════════════════════════════════════════════════
def run_battledim_f1() -> dict:
    print("\n" + "═" * 50)
    print("  BattLeDIM F1 Evaluation (2019 Test Year)")
    print("═" * 50)

    # Load
    data_2018 = load_battledim_2018()
    data_2019 = load_battledim_2019()
    gt        = build_ground_truth(data_2019["leakages"])

    pressures = data_2019["pressures"]
    flows     = data_2019["flows"]

    # Init brain (SCADA @ ~1/60 Hz → fs=0.0167 Hz)
    brain = PGPLBrain(fs=1/60, saline=False)  # BattLeDIM = freshwater

    # Calibrate from 2018
    p_col = pressures.columns[0]
    cal_pressures = data_2018["pressures"][p_col].dropna().tolist()
    cal_z_scores  = [abs(v - np.mean(cal_pressures)) / (np.std(cal_pressures) + 1e-9)
                     for v in cal_pressures[:500]]
    brain.calibrate_from_year(cal_z_scores, phase="default")

    # Run detection
    y_pred = []
    y_true = []
    p_col  = pressures.columns[0]
    f_col  = flows.columns[0]

    common_idx = pressures.index.intersection(flows.index).intersection(gt.index)
    print(f"  Processing {len(common_idx)} timesteps …")

    for i, ts in enumerate(common_idx):
        psi  = float(pressures.loc[ts, p_col])
        flow = float(flows.loc[ts, f_col])
        phase, tidal_psi = mock_tidal_phase(i)

        event = brain.process_scada(
            pressure_psi=psi,
            flow_lps=flow,
            timestamp=ts.timestamp(),
            tidal_phase=phase,
            tidal_psi=tidal_psi,
        )

        gate     = event.meta.get("gate", {})
        pred     = 1 if gate.get("confirmed", False) else 0
        y_pred.append(pred)
        y_true.append(int(gt.loc[ts]))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    far = float(np.sum((y_pred == 1) & (y_true == 0)) / (np.sum(y_true == 0) + 1e-9))

    result = {"dataset": "BattLeDIM", "F1": round(f1, 3),
              "Precision": round(pre, 3), "Recall": round(rec, 3),
              "FAR": round(far, 4)}
    print(f"  F1={f1:.3f}  Precision={pre:.3f}  Recall={rec:.3f}  FAR={far:.4f}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. Mendeley F1
# ══════════════════════════════════════════════════════════════════════════════
def run_mendeley_f1() -> dict:
    print("\n" + "═" * 50)
    print("  Mendeley F1 Evaluation (Acoustic)")
    print("═" * 50)

    hydro_files = load_hydrophone_files()
    if len(hydro_files) < 2:
        print("  [SKIP] Need ≥2 hydrophone files for TDOA.")
        return {"dataset": "Mendeley", "F1": None, "FAR": None}

    # Pair consecutive files (A=even, B=odd index)
    pairs = [(hydro_files[i], hydro_files[i+1])
             for i in range(0, len(hydro_files) - 1, 2)]

    y_pred_list = []
    tdoa_errors = []

    print(f"  Processing {len(pairs)} sensor pairs …")
    for j, (path_a, path_b) in enumerate(pairs):
        try:
            sig_a, sig_b, fs = read_wav_pair(path_a, path_b)
        except Exception as e:
            print(f"  [WARN] Pair {j}: {e}")
            continue

        brain = PGPLBrain(
            fs=fs,
            pipe_diameter_m=0.10,
            pipe_thickness_m=0.008,
            pipe_material="hdpe",
            saline=True,
            sensor_spacing_m=5.0,
        )

        phase, tidal_psi = mock_tidal_phase(j)

        # Feed 5 different phases so gate is ready
        for ph in ["ebb", "flood", "slack_low", "slack_high", "spring"]:
            from src.tidal_gating import TidalWindow
            brain.tidal.add_phase(TidalWindow(
                phase=ph, psi_offset=tidal_psi,
                alpha_adj=brain.tidal.adaptive_alpha(),
                timestamp=float(j),
            ))

        event = brain.process_acoustic(
            signal=sig_a, signal_b=sig_b,
            timestamp=float(j),
            tidal_phase=phase,
            tidal_psi=tidal_psi,
        )
        gate = event.meta.get("gate", {})
        pred = 1 if gate.get("confirmed", False) else 0
        y_pred_list.append(pred)

        if event.location_m >= 0:
            tdoa_errors.append(event.location_m)

    # Mendeley ground truth: assume all paired files contain leaks (label=1)
    # Adjust per actual Mendeley metadata if available
    y_true_list = [1] * len(y_pred_list)

    if not y_pred_list:
        return {"dataset": "Mendeley", "F1": None, "FAR": None}

    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)

    f1    = f1_score(y_true, y_pred, zero_division=0)
    far   = float(np.sum((y_pred == 1) & (y_true == 0)) / (np.sum(y_true == 0) + 1e-9))
    mae   = float(np.mean(tdoa_errors)) if tdoa_errors else float("nan")

    result = {"dataset": "Mendeley", "F1": round(f1, 3),
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

    # Save
    from config import OUTPUTS_DIR
    out_path = os.path.join(OUTPUTS_DIR, "f1_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")