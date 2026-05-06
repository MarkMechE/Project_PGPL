"""
run_battledim.py  —  PGL Phase 3: BattLeDIM Real Municipal Network Validation
Dataset : L-TOWN (DOI: 10.5281/zenodo.4017659)
No synthetic data. Real SCADA pressure/flow only.
"""
import sys
import os
import argparse
from collections import deque

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config (auto-detects Windows vs Mac iCloud path) ───────────────────────
from config import (
    BATTLEDIM_DIR,
    PERSISTENCE_N, PSI_THRESHOLD, ZONE_WEIGHT, SALINITY_MEAN_PSU,
)

from src.pgpl_brain import PULSE_AT_Brain
from src.battledim_loader import (
    load_battledim_pressures,
    load_battledim_flows,
    load_battledim_leakages,
    build_sequences,
    get_pipe_distances,
    _compute_deviation,
    BATTLEDIM_FS,
    BACKGROUND_PIPES,
)

# ── Run constants ───────────────────────────────────────────────────────────
ALPHA             = 0.15
CP_CALIB_END      = "2019-01-10 23:55:00"
NORMAL_EVAL_START = "2019-01-11 00:00:00"
NORMAL_EVAL_END   = "2019-01-15 23:55:00"
LEAK_EVAL_START   = "2019-01-16 00:00:00"
BEST_NODES        = ["n288", "n105", "n114"]
SEQ_LENGTH        = 6
WINDOW_SIZE       = 72
STEP              = 36

# ── File paths (from config, works on Windows + Mac) ────────────────────────
BATTLEDIM_DIR_2018 = BATTLEDIM_DIR   # same folder has both 2018 and 2019

FILES = {
    "pressure_2019": os.path.join(BATTLEDIM_DIR, "2019_SCADA_Pressures.csv"),
    "flow_2019":     os.path.join(BATTLEDIM_DIR, "2019_SCADA_Flows.csv"),
    "leakages_2019": os.path.join(BATTLEDIM_DIR, "2019_Leakages.csv"),
    "pressure_2018": os.path.join(BATTLEDIM_DIR, "2018_SCADA_Pressures.csv"),
    "flow_2018":     os.path.join(BATTLEDIM_DIR, "2018_SCADA_Flows.csv"),
    "ltown_inp":     os.path.join(BATTLEDIM_DIR, "L-TOWN.inp"),
}


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(tp, fp, fn, tn, errors):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    far       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    mae       = float(np.mean(errors)) if errors else float("nan")
    return dict(
        Precision = round(precision, 3),
        Recall    = round(recall, 3),
        F1        = round(f1, 3),
        FAR       = round(far, 3),
        MAE_m     = round(mae, 2) if not np.isnan(mae) else "n/a",
        TP=tp, FP=fp, FN=fn, TN=tn,
    )


# ── Window builder ───────────────────────────────────────────────────────────
def _make_windows(dev_series, flow_series, label,
                  t_start, t_end, window_size, step, node_name):
    block_dev  = dev_series[t_start:t_end]
    block_flow = flow_series.reindex(block_dev.index, method="nearest")
    windows = []
    for i in range(0, len(block_dev) - window_size, step):
        sig = block_dev.iloc[i:i + window_size].values.astype("float32")
        if np.any(np.isnan(sig)):
            continue
        fv = float(np.nanmean(block_flow.iloc[i:i + window_size].values))
        windows.append({
            "mic1_sig":  sig,
            "fs":        BATTLEDIM_FS,
            "salinity":  SALINITY_MEAN_PSU,
            "flow":      fv,
            "label":     label,
            "is_leak":   label == 1,
            "timestamp": str(block_dev.index[i]),
            "node":      node_name,
        })
    return windows


# ── Transfer CP calibration between brain instances ──────────────────────────
def _transfer_calibration(ref_brain, new_brain):
    new_brain._cp._cal = list(ref_brain._cp._cal)
    new_brain._cal_n   = ref_brain._cal_n
    new_brain._z._buf  = deque(ref_brain._z._buf,
                                maxlen=ref_brain._z._buf.maxlen)


# ── Main run ─────────────────────────────────────────────────────────────────
def run(node=None, all_nodes=False):
    print("\n[PGL Phase 3] BattLeDIM Municipal Network Validation")
    print("  Dataset  : L-TOWN (DOI: 10.5281/zenodo.4017659)")
    print("  Data dir :", BATTLEDIM_DIR)
    print("  Calib    : 2018 Jan-Mar  (clean baseline)")
    print("  Normal   : 2019 Jan 11-15")
    print("  Leak eval: 2019 Jan 16 onward")
    print(f"  Salinity : {SALINITY_MEAN_PSU} psu (Lee et al. 2023)")
    print(f"  Window   : {WINDOW_SIZE} x 5 min = {WINDOW_SIZE * 5} min")
    print(f"  Seq      : {SEQ_LENGTH} windows = {SEQ_LENGTH * WINDOW_SIZE * 5 // 60} hrs")
    print(f"  Gate     : persistence_n={PERSISTENCE_N}, alpha={ALPHA}")

    # ── Check files exist ────────────────────────────────────────────────────
    missing = [k for k, v in FILES.items()
               if not os.path.isfile(v) and k != "ltown_inp"]
    if missing:
        print("\n  [ERROR] Missing files:")
        for m in missing:
            print(f"    {FILES[m]}")
        sys.exit(1)

    # ── Load 2019 data ───────────────────────────────────────────────────────
    print("\n  Loading 2019 SCADA data...")
    pres_2019  = load_battledim_pressures(FILES["pressure_2019"])
    flow_2019  = load_battledim_flows(FILES["flow_2019"])
    leaks_2019 = load_battledim_leakages(FILES["leakages_2019"])

    event_pipes = [c for c in leaks_2019.columns if c not in BACKGROUND_PIPES]
    active_ts   = int((leaks_2019[event_pipes] > 0).any(axis=1).sum())
    print(f"  2019 pressure shape : {pres_2019.shape}")
    print(f"  Event pipes         : {len(event_pipes)} | "
          f"Active timesteps: {active_ts}/{len(leaks_2019)}")

    # ── Select nodes ─────────────────────────────────────────────────────────
    available_nodes = list(pres_2019.columns)
    if all_nodes:
        nodes_to_run = available_nodes
    elif node and node in available_nodes:
        nodes_to_run = [node]
    else:
        nodes_to_run = [n for n in BEST_NODES if n in available_nodes]
        if not nodes_to_run:
            nodes_to_run = list(
                pres_2019.std().sort_values(ascending=False).index[:3]
            )
    print(f"  Nodes to evaluate   : {nodes_to_run}")

    # ── Optional pipe distances ──────────────────────────────────────────────
    pipe_distances = {}
    if os.path.isfile(FILES["ltown_inp"]):
        pipe_distances = get_pipe_distances(FILES["ltown_inp"], nodes_to_run)

    # ── Load 2018 calibration data once ──────────────────────────────────────
    print("\n  Loading 2018 calibration data...")
    pres_2018 = load_battledim_pressures(FILES["pressure_2018"])
    flow_2018 = load_battledim_flows(FILES["flow_2018"])
    print(f"  2018 pressure shape : {pres_2018.shape}")

    all_records  = []
    node_results = {}

    # ── Per-node evaluation ──────────────────────────────────────────────────
    for eval_node in nodes_to_run:
        print(f"\n  === Node: {eval_node} ===")

        # 2019 series
        pressure_series = pres_2019[eval_node].dropna()
        flow_col        = flow_2019.columns[0]
        flow_series     = flow_2019[flow_col].reindex(
            pressure_series.index, method="nearest"
        )
        deviation  = _compute_deviation(pressure_series)
        leak_active = (leaks_2019[event_pipes] > 0).any(axis=1)
        leak_mask   = (leak_active
                       .reindex(pressure_series.index, method="nearest")
                       .fillna(False)
                       .astype(bool))

        # 2018 calibration windows
        if eval_node not in pres_2018.columns:
            print(f"  [WARN] {eval_node} not in 2018 data — skipping")
            continue

        dev_2018   = _compute_deviation(pres_2018[eval_node].dropna())
        flow_2018s = flow_2018[flow_2018.columns[0]].reindex(
            dev_2018.index, method="nearest"
        )
        calib_dev  = dev_2018["2018-01-01":"2018-03-31"]
        calib_flow = flow_2018s["2018-01-01":"2018-03-31"]

        calib_wins = []
        for i in range(144, len(calib_dev) - WINDOW_SIZE, WINDOW_SIZE):
            sig = calib_dev.iloc[i:i + WINDOW_SIZE].values.astype("float32")
            if not np.any(np.isnan(sig)):
                fv = float(np.nanmean(calib_flow.iloc[i:i + WINDOW_SIZE].values))
                calib_wins.append({
                    "mic1_sig": sig,
                    "fs":       BATTLEDIM_FS,
                    "salinity": SALINITY_MEAN_PSU,
                    "flow":     fv,
                })

        # Build reference brain with calibration
        ref_brain = PULSE_AT_Brain(
            alpha=ALPHA,
            persistence_n=PERSISTENCE_N,
            psi_threshold=PSI_THRESHOLD,
            zone_weight=ZONE_WEIGHT,
        )
        for w in calib_wins:
            e = float(np.sqrt(np.mean(np.array(w["mic1_sig"]) ** 2)))
            z = ref_brain._z.score(e)
            ref_brain._cp.calibrate(z)
        ref_brain._cal_n = len(calib_wins)

        print(f"  CP calib : {len(calib_wins)} windows (2018 Jan-Mar clean)")
        if ref_brain._cp._cal:
            cal = ref_brain._cp._cal
            print(f"  cal range: [{min(cal):.3f}, {max(cal):.3f}]")

        # Normal windows (2019 Jan 11-15)
        normal_wins = _make_windows(
            deviation, flow_series, 0,
            NORMAL_EVAL_START, NORMAL_EVAL_END,
            WINDOW_SIZE, STEP, eval_node,
        )
        normal_seqs = build_sequences(normal_wins, SEQ_LENGTH)

        # Leak windows (2019 Jan 16 onward)
        leak_dev_s  = deviation[LEAK_EVAL_START:]
        leak_flow_s = flow_series[LEAK_EVAL_START:]
        leak_mask_s = leak_mask[LEAK_EVAL_START:]
        leak_wins   = []
        for i in range(0, len(leak_dev_s) - WINDOW_SIZE, STEP):
            sig = leak_dev_s.iloc[i:i + WINDOW_SIZE].values.astype("float32")
            if np.any(np.isnan(sig)):
                continue
            fv  = float(np.nanmean(leak_flow_s.iloc[i:i + WINDOW_SIZE].values))
            lbl = 1 if leak_mask_s.iloc[i:i + WINDOW_SIZE].sum() > WINDOW_SIZE * 0.5 else 0
            leak_wins.append({
                "mic1_sig":  sig,
                "fs":        BATTLEDIM_FS,
                "salinity":  SALINITY_MEAN_PSU,
                "flow":      fv,
                "label":     lbl,
                "is_leak":   lbl == 1,
                "timestamp": str(leak_dev_s.index[i]),
                "node":      eval_node,
            })
        leak_seqs = build_sequences(leak_wins, SEQ_LENGTH)

        all_seqs = normal_seqs + leak_seqs
        n_lk = sum(1 for s in all_seqs if s["is_leak"])
        print(f"  Sequences: {len(all_seqs)} | "
              f"normal={len(all_seqs) - n_lk} | leak={n_lk}")

        # ── Evaluate sequences ────────────────────────────────────────────────
        tp = fp = fn = tn = 0
        errors = []

        for seq in all_seqs:
            brain = PULSE_AT_Brain(
                alpha=ALPHA,
                persistence_n=PERSISTENCE_N,
                psi_threshold=PSI_THRESHOLD,
                zone_weight=ZONE_WEIGHT,
            )
            _transfer_calibration(ref_brain, brain)

            dispatched   = False
            final_result = None

            for win in seq["windows"]:
                r = brain.process({
                    "mic1_sig": win["mic1_sig"],
                    "fs":       BATTLEDIM_FS,
                    "salinity": win["salinity"],
                    "flow":     win["flow"],
                })
                if r["flag"] == "DISPATCH" and not dispatched:
                    dispatched   = True
                    final_result = r

            is_leak_pred = dispatched
            true_label   = seq["label"]

            if   is_leak_pred and true_label == 1: tp += 1
            elif is_leak_pred and true_label == 0: fp += 1
            elif not is_leak_pred and true_label == 1: fn += 1
            else: tn += 1

            all_records.append({
                "node":       eval_node,
                "timestamp":  seq.get("timestamp", ""),
                "true_label": true_label,
                "pred_leak":  is_leak_pred,
                "anomaly":    final_result.get("anomaly", "") if final_result else "",
                "priority":   final_result.get("priority", "") if final_result else "",
            })

        m = compute_metrics(tp, fp, fn, tn, errors)
        node_results[eval_node] = m
        print(f"  P={m['Precision']}  R={m['Recall']}  F1={m['F1']}  "
              f"TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")

    # ── Final results ────────────────────────────────────────────────────────
    sep = "=" * 57
    print(f"\n{sep}")
    print("  RESULTS  [BattLeDIM L-TOWN — Real Municipal Network]")
    print(sep)

    if not node_results:
        print("  No results — check node names and data files.")
        return

    best = max(node_results, key=lambda nd: node_results[nd]["F1"])
    print(f"  Best node : {best}")
    for k in ["Precision", "Recall", "F1", "FAR", "MAE_m", "TP", "FP", "FN", "TN"]:
        print(f"  {k:<14}: {node_results[best][k]}")

    if len(nodes_to_run) > 1:
        print("\n  Per-node summary:")
        for nd, m in node_results.items():
            print(f"  {nd:<8}  F1={m['F1']}  "
                  f"P={m['Precision']}  R={m['Recall']}  FAR={m['FAR']}")

    print(sep)

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    out = "outputs/battledim_results.csv"
    pd.DataFrame(all_records).to_csv(out, index=False)
    print(f"\n  Saved: {out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PGPL pipeline on BattleDIM real SCADA data."
    )
    parser.add_argument("--node",      type=str,  default=None,
                        help="Single pressure node to evaluate (e.g. n288)")
    parser.add_argument("--all-nodes", action="store_true",
                        help="Evaluate all 33 pressure nodes")
    args = parser.parse_args()
    run(node=args.node, all_nodes=args.all_nodes)