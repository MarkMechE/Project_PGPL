"""
run_battledim.py  —  PGL Phase 3: BattLeDIM Real Municipal Network Validation
Dataset : L-TOWN (DOI: 10.5281/zenodo.4017659)
Usage   : python run_battledim.py
          python run_battledim.py --node n1
          python run_battledim.py --all-nodes
"""

import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ALPHA, PERSISTENCE_N, PSI_THRESHOLD, ZONE_WEIGHT, SALINITY_MEAN_PSU
from src.pgpl_brain import PULSE_AT_Brain, LEAK_CLASSES = {"Burst", "Crack", "Micro", "PressureDrop"}  # SCADA mode includes PressureDrop
from src.battledim_loader import (
    load_battledim_pressures, load_battledim_flows,
    load_battledim_leakages, build_pressure_windows,
    build_sequences, build_warmup_windows, get_pipe_distances,
    BATTLEDIM_FS,
)

def _get_battledim_dir():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", os.path.join(os.path.dirname(__file__), "config.py"))
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        base = getattr(cfg, "DATASET_ROOT", None)
        if base:
            return os.path.join(base, "BattLeDIM")
    except Exception:
        pass
    return os.path.join(os.path.dirname(__file__), "data", "BattLeDIM")

BATTLEDIM_DIR = _get_battledim_dir()

FILES = {
    "pressure_2018": os.path.join(BATTLEDIM_DIR, "2018_SCADA_Pressures.csv"),
    "flow_2018":     os.path.join(BATTLEDIM_DIR, "2018_SCADA_Flows.csv"),
    "leakages_2018": os.path.join(BATTLEDIM_DIR, "2018_Fixed_Leakages_Report.txt"),
    "pressure_2019": os.path.join(BATTLEDIM_DIR, "2019_SCADA_Pressures.csv"),
    "flow_2019":     os.path.join(BATTLEDIM_DIR, "2019_SCADA_Flows.csv"),
    "leakages_2019": os.path.join(BATTLEDIM_DIR, "2019_Leakages.csv"),
    "ltown_inp":     os.path.join(BATTLEDIM_DIR, "L-TOWN.inp"),
}

N_WARMUP    = 200
SEQ_LENGTH  = 12
WINDOW_SIZE = 72   # 72 x 5min = 6 hours per window
STEP        = 36  # 50% overlap


def compute_metrics(tp, fp, fn, tn, errors):
    precision = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    far = fp / (fp + tn)         if (fp + tn) > 0 else 0.0
    mae = float(np.mean(errors)) if errors else float("nan")
    return dict(
        Precision=round(precision, 3), Recall=round(recall, 3),
        F1=round(f1, 3), FAR=round(far, 3),
        MAE_m=round(mae, 2) if not np.isnan(mae) else "n/a",
        TP=tp, FP=fp, FN=fn, TN=tn,
    )


def run(node=None, all_nodes=False):
    print("\n[PGL Phase 3] BattLeDIM Municipal Network Validation")
    print(f"  Dataset : L-TOWN (DOI: 10.5281/zenodo.4017659)")
    print(f"  Salinity: {SALINITY_MEAN_PSU} psu (Lee et al. 2023)")
    print(f"  Window  : {WINDOW_SIZE} x 5 min = {WINDOW_SIZE*5} min per window")
    print(f"  Gate    : persistence_n={PERSISTENCE_N}, "
          f"ratio>={PULSE_AT_Brain.PERSIST_RATIO_THRESH}, alpha={ALPHA}")

    missing = [k for k, v in FILES.items()
               if not os.path.isfile(v) and k != "ltown_inp"]
    if missing:
        print(f"\n  [ERROR] Missing files in {BATTLEDIM_DIR}:")
        for m in missing:
            print(f"    {FILES[m]}")
        print("  Download from: https://zenodo.org/records/4017659")
        sys.exit(1)

    print("\n  Loading 2018 baseline...")
    pres_2018  = load_battledim_pressures(FILES["pressure_2018"])
    flow_2018  = load_battledim_flows(FILES["flow_2018"])
    print("  Loading 2019 evaluation data...")
    pres_2019  = load_battledim_pressures(FILES["pressure_2019"])
    flow_2019  = load_battledim_flows(FILES["flow_2019"])
    leaks_2019 = load_battledim_leakages(FILES["leakages_2019"])
    print(f"  2018: {pres_2018.shape} | 2019: {pres_2019.shape}")
    BACKGROUND_PIPES = {"p257", "p427", "p654", "p810"}
event_pipes = [c for c in leaks_2019.columns if c not in BACKGROUND_PIPES]
active_times = (leaks_2019[event_pipes] > 0).any(axis=1).sum()
print(f"  Leak event timesteps: {active_times} / {len(leaks_2019)} "
      f"({100*active_times/len(leaks_2019):.1f}%)")
    available_nodes = list(pres_2019.columns)
    if all_nodes:
        nodes_to_run = available_nodes
    elif node and node in available_nodes:
        nodes_to_run = [node]
    else:
        stds = pres_2019.std().sort_values(ascending=False)
        nodes_to_run = list(stds.index[:3])
    print(f"  Nodes: {nodes_to_run}")

    pipe_distances = {}
    if os.path.isfile(FILES["ltown_inp"]):
        pipe_distances = get_pipe_distances(FILES["ltown_inp"], nodes_to_run)

    print(f"\n  Building CP warm-up ({N_WARMUP} windows from Jan-Mar 2018)...")
    warmup_windows = build_warmup_windows(
        pres_2018, flow_2018, FILES["leakages_2018"],
        n_warmup=N_WARMUP, node=nodes_to_run[0])
    print(f"  Warm-up windows: {len(warmup_windows)}")

    ref_brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                               psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)
    for w in warmup_windows:
        ref_brain.process({"mic1_sig": w["mic1_sig"], "fs": BATTLEDIM_FS,
                           "salinity": SALINITY_MEAN_PSU, "flow": w["flow"]})
    print(f"  CP calibrated: {ref_brain._cal_n} points")

    all_records  = []
    node_results = {}

    for eval_node in nodes_to_run:
        print(f"\n  Evaluating node: {eval_node}")
        windows   = build_pressure_windows(pres_2019, flow_2019, leaks_2019,
                                           window_size=WINDOW_SIZE,
                                           step=STEP, node=eval_node)
        sequences = build_sequences(windows, seq_length=SEQ_LENGTH)
        n_leak    = sum(1 for s in sequences if s["is_leak"])
        print(f"    Sequences: {len(sequences)} ({n_leak} leak, {len(sequences)-n_leak} normal)")

        tp = fp = fn = tn = 0
        errors = []

        for seq in sequences:
            brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                                   psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)
            brain._cp._cal = ref_brain._cp._cal.copy()
            brain._cal_n   = ref_brain._cal_n
            brain._z._buf  = type(ref_brain._z._buf)(
                                 ref_brain._z._buf, maxlen=ref_brain._z._buf.maxlen)

            dispatched   = False
            final_result = None
            for win in seq["windows"]:
                r = brain.process({"mic1_sig": win["mic1_sig"], "fs": BATTLEDIM_FS,
                                   "salinity": win["salinity"], "flow": win["flow"]})
                if r["flag"] == "DISPATCH" and not dispatched:
                    dispatched   = True
                    final_result = r

            is_leak_pred = (dispatched and final_result is not None and
                            final_result.get("anomaly", "") in LEAK_CLASSES)
            true_label   = seq["label"]

            if   is_leak_pred and true_label == 1:
                tp += 1
                if pipe_distances and final_result.get("loc_m"):
                    gt = float(np.mean(list(pipe_distances.values())))
                    errors.append(abs(final_result["loc_m"] - gt))
            elif is_leak_pred and true_label == 0: fp += 1
            elif not is_leak_pred and true_label == 1: fn += 1
            else: tn += 1

            all_records.append({
                "node": eval_node, "timestamp": seq["timestamp"],
                "true_label": true_label, "dispatched": dispatched,
                "pred_leak": is_leak_pred,
                "anomaly":   final_result.get("anomaly","—") if final_result else "—",
                "severity":  final_result.get("severity","—") if final_result else "—",
                "priority":  final_result.get("priority","—") if final_result else "—",
                "loc_m":     final_result.get("loc_m","—")    if final_result else "—",
                "pvalue":    final_result.get("pvalue","—")   if final_result else "—",
            })

        m = compute_metrics(tp, fp, fn, tn, errors)
        node_results[eval_node] = m
        print(f"    P={m['Precision']}  R={m['Recall']}  F1={m['F1']}  "
              f"FAR={m['FAR']}  MAE={m['MAE_m']}m  "
              f"TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")

    print("\n" + "="*55)
    print("  RESULTS  [BattLeDIM — L-TOWN Real Municipal Network]")
    print("="*55)
    if len(nodes_to_run) == 1:
        for k, v in node_results[nodes_to_run[0]].items():
            print(f"  {k:<14}: {v}")
    else:
        best = max(node_results, key=lambda n: node_results[n]["F1"])
        print(f"  Best node: {best}")
        for k, v in node_results[best].items():
            print(f"  {k:<14}: {v}")
        print("\n  Per-node:")
        for n, m in node_results.items():
            print(f"    {n:<6}  F1={m['F1']}  R={m['Recall']}  FAR={m['FAR']}")
    print("="*55)
    print(f"  Salinity: Lee et al. (2023), DOI: 10.1016/j.margeo.2023.107089")
    print(f"  Dataset : Vrachimis et al. (2022), DOI: 10.5281/zenodo.4017659")

    os.makedirs("outputs", exist_ok=True)
    out = "outputs/battledim_results.csv"
    pd.DataFrame(all_records).to_csv(out, index=False)
    print(f"\n  Saved: {out}")

    print("\n  Leak events (2019 ground truth, first 5):")
    time_cols = [c for c in leaks_2019.columns if "time" in c.lower()]
    id_cols   = [c for c in leaks_2019.columns
                 if "pipe" in c.lower() or "id" in c.lower()]
    for _, row in leaks_2019.head(5).iterrows():
        print(f"    Pipe {row[id_cols[0]] if id_cols else "?"}: "
              f"{row[time_cols[0]] if time_cols else "?"} -> "
              f"{row[time_cols[1]] if len(time_cols)>1 else "?"}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node",      type=str,  default=None)
    parser.add_argument("--all-nodes", action="store_true")
    args = parser.parse_args()
    run(node=args.node, all_nodes=args.all_nodes)
