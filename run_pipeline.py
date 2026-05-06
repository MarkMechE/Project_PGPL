"""
run_pipeline.py — Main F1 Runner (v2.1 — Tuned)
PGPL v2.0 | BattleDIM F1 target: 0.96 | Mendeley F1 target: 0.85
Run: python run_pipeline.py
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw):
        return x

sys.path.insert(0, os.path.dirname(__file__))

from config                 import validate_paths, OUTPUTS_DIR
from src.pgpl_brain         import PGPLBrain
from src.battledim_loader   import load_battledim_2019, load_battledim_2018, build_ground_truth
from src.mendeley_loader    import load_hydrophone_files, read_wav_pair

# ── Tidal Phase Stub ──────────────────────────────────────────────────────────
TIDAL_PHASES_CYCLE = ["ebb", "flood", "slack_low", "slack_high", "spring"]

def mock_tidal_phase(i: int) -> tuple[str, float]:
    phase = TIDAL_PHASES_CYCLE[i % len(TIDAL_PHASES_CYCLE)]
    psi   = float(np.sin(i * 0.01) * 2.5)
    return phase, psi


# ══════════════════════════════════════════════════════════════════════════════
# 1. BattLeDIM F1
# ══════════════════════════════════════════════════════════════════════════════
def run_battledim_f1() -> dict:
    print("\n" + "═" * 52)
    print("  BattLeDIM F1 Evaluation  (2019 Test Year)")
    print("═" * 52)

    # ── Load data ─────────────────────────────────────────────────────────────
    data_2018 = load_battledim_2018()
    data_2019 = load_battledim_2019()
    gt        = build_ground_truth(data_2019["leakages"])

    pressures = data_2019["pressures"]
    flows     = data_2019["flows"]

    # ── Init Brain ────────────────────────────────────────────────────────────
    brain = PGPLBrain(fs=1/60, saline=False)   # BattleDIM = freshwater

    # ── Full 2018 calibration (all 105k samples → robust baseline) ───────────
    p_col         = data_2018["pressures"].columns[0]
    cal_vals      = data_2018["pressures"][p_col].dropna().values
    cal_mean      = float(np.mean(cal_vals))
    cal_std       = float(np.std(cal_vals)) + 1e-9
    cal_z_scores  = [abs(v - cal_mean) / cal_std for v in cal_vals]
    brain.calibrate_from_year(cal_z_scores, phase="default")

    # ── Common index (drop NaT) ───────────────────────────────────────────────
    p_col = pressures.columns[0]
    f_col = flows.columns[0]

    common_idx = (
        pressures.index
        .intersection(flows.index)
        .intersection(gt.index)
    )
    common_idx = common_idx[~common_idx.isna()]
    print(f"  Processing {len(common_idx):,} timesteps …")

    y_pred, y_true = [], []

    for i, ts in enumerate(tqdm(common_idx, desc="  🧠 BattleDIM")):
        try:
            psi  = float(pressures.at[ts, p_col])
            flow = float(flows.at[ts, f_col])
        except Exception:
            continue

        if np.isnan(psi) or np.isnan(flow):
            continue

        phase, tidal_psi = mock_tidal_phase(i)

        event = brain.process_scada(
            pressure_psi = psi,
            flow_lps     = flow,
            timestamp    = float(i),
            tidal_phase  = phase,
            tidal_psi    = tidal_psi,
        )

        gate = event.meta.get("gate", {})
        y_pred.append(1 if gate.get("confirmed", False) else 0)
        y_true.append(int(gt.at[ts]))

    y_pred = np.array(y_pred, dtype=int)
    y_true = np.array(y_true, dtype=int)

    # ── Debug balance ─────────────────────────────────────────────────────────
    print(f"  📊 y_true pos: {y_true.sum():,}/{len(y_true):,}  ({y_true.mean():.1%})")
    print(f"  📊 y_pred pos: {y_pred.sum():,}/{len(y_pred):,}  ({y_pred.mean():.1%})")

    if y_pred.sum() == 0:
        print("  ⚠️  WARNING: Brain detected 0 leaks. "
              "Lower gate threshold in pgpl_brain.py TidalGatingEngine.")

    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    far = float(np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1))

    print(f"  F1={f1:.3f}  Precision={pre:.3f}  Recall={rec:.3f}  FAR={far:.4f}")
    return {
        "dataset":   "BattleDIM",
        "F1":        round(f1,  3),
        "Precision": round(pre, 3),
        "Recall":    round(rec, 3),
        "FAR":       round(far, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Mendeley F1 (Acoustic)
# ══════════════════════════════════════════════════════════════════════════════
def run_mendeley_f1() -> dict:
    print("\n" + "═" * 52)
    print("  Mendeley F1 Evaluation  (Acoustic — RAW PCM_32)")
    print("═" * 52)

    from src.mendeley_loader import load_hydrophone_pairs, read_raw_pair
    from src.pgpl_brain     import TidalWindow

    pairs = load_hydrophone_pairs()

    if len(pairs) < 2:
        print("  [SKIP] No .raw pairs found.")
        return {"dataset": "Mendeley", "F1": None,
                "Precision": None, "Recall": None, "FAR": None}

    y_pred_list = []
    y_true_list = []
    tdoa_errors = []

    for j, pair_info in enumerate(tqdm(pairs, desc="  🎙  Mendeley")):
        try:
            sig_a, sig_b, fs = read_raw_pair(
                pair_info["path_a"],
                pair_info["path_b"],
            )
        except Exception as e:
            print(f"  [WARN] {pair_info['stem']}: {e}")
            continue

        # fs=8000 → exactly at FS_ACOU_MIN boundary → acoustic path ✓
        brain = PGPLBrain(
            fs               = fs,          # 8000 Hz
            pipe_diameter_m  = 0.05,        # lab pipe ~50mm
            pipe_thickness_m = 0.005,
            pipe_material    = "hdpe",
            saline           = False,       # lab = freshwater
            sensor_spacing_m = 1.5,         # lab sensor spacing
        )

        # Lab setting: no real tidal — use neutral phases
        for ph in ["ebb", "flood", "slack_low", "slack_high", "spring"]:
            brain.tidal.add_phase(TidalWindow(
                phase=ph, psi_offset=0.0,
                alpha_adj=brain.tidal.adaptive_alpha(),
                timestamp=float(j),
            ))

        phase, tidal_psi = mock_tidal_phase(j)
        event = brain.process_acoustic(
            sig_a, sig_b, float(j), phase, tidal_psi,
        )

        gate = event.meta.get("gate", {})
        y_pred_list.append(1 if gate.get("confirmed", False) else 0)
        y_true_list.append(pair_info["label"])   # ← from folder name

        if event.location_m >= 0:
            tdoa_errors.append(event.location_m)

    if not y_pred_list:
        return {"dataset": "Mendeley", "F1": None,
                "Precision": None, "Recall": None, "FAR": None}

    y_pred = np.array(y_pred_list, dtype=int)
    y_true = np.array(y_true_list, dtype=int)

    print(f"  📊 y_true: {y_true.sum()} leak / {(1-y_true).sum()} no-leak "
          f"({y_true.mean():.1%})")
    print(f"  📊 y_pred: {y_pred.sum()} detected / {len(y_pred)} total")

    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    far = float(
        np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1)
    )
    mae = float(np.mean(tdoa_errors)) if tdoa_errors else float("nan")

    print(f"  F1={f1:.3f}  Precision={pre:.3f}  "
          f"Recall={rec:.3f}  FAR={far:.4f}  TDOA_MAE={mae:.2f}m")

    return {
        "dataset":    "Mendeley",
        "F1":         round(f1,  3),
        "Precision":  round(pre, 3),
        "Recall":     round(rec, 3),
        "FAR":        round(far, 4),
        "TDOA_MAE_m": round(mae, 2) if not np.isnan(mae) else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("── PGPL v2.0 Pipeline ──")
    validate_paths(strict=True)

    results = [run_battledim_f1(), run_mendeley_f1()]

    print("\n" + "═" * 52)
    print("  📊  FINAL F1 TABLE")
    print("═" * 52)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out = os.path.join(OUTPUTS_DIR, "f1_results.csv")
    df.to_csv(out, index=False)
    print(f"\n  ✅  Saved → {out}")