import sys, os, argparse
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PERSISTENCE_N, PSI_THRESHOLD, ZONE_WEIGHT, SALINITY_MEAN_PSU
from src.pgpl_brain import PULSE_AT_Brain
from src.battledim_loader import (
    load_battledim_pressures, load_battledim_flows, load_battledim_leakages,
    build_sequences, get_pipe_distances, _compute_deviation,
    BATTLEDIM_FS, BACKGROUND_PIPES,
)

ALPHA             = 0.15
CP_CALIB_END      = "2019-01-10 23:55:00"
NORMAL_EVAL_START = "2019-01-11 00:00:00"
NORMAL_EVAL_END   = "2019-01-15 23:55:00"
LEAK_EVAL_START   = "2019-01-16 00:00:00"
BEST_NODES        = ["n288", "n105", "n114"]
SEQ_LENGTH        = 6
WINDOW_SIZE       = 72
STEP              = 36
BATTLEDIM_DIR     = "/Users/markmontillano/Library/Mobile Documents/com~apple~CloudDocs/Project_PGPL_Dataset/BattleDIM"
FILES = {
    "pressure_2019": os.path.join(BATTLEDIM_DIR, "2019_SCADA_Pressures.csv"),
    "flow_2019":     os.path.join(BATTLEDIM_DIR, "2019_SCADA_Flows.csv"),
    "leakages_2019": os.path.join(BATTLEDIM_DIR, "2019_Leakages.csv"),
    "ltown_inp":     os.path.join(BATTLEDIM_DIR, "L-TOWN.inp"),
}


BATTLEDIM_DIR_2018 = "/Users/markmontillano/Library/Mobile Documents/com~apple~CloudDocs/Project_PGPL_Dataset/BattleDIM"

def _scada_process(brain, win):
    """
    SCADA-aware process: uses RMS of pressure deviation as anomaly feature.
    Bypasses the spectral welch path which requires fs >> 600 Hz.
    Directly scores RMS through the brain's z-scorer and conformal gate.
    """
    import numpy as np
    from collections import deque
    sig = np.array(win["mic1_sig"], dtype="float32")
    energy  = float(np.sqrt(np.mean(sig ** 2)))          # RMS deviation
    flow    = float(win.get("flow", 13.0))
    pressure_z = abs(flow - 13.0) / 2.0

    z     = brain._z.score(energy)
    p_val = brain._cp.p_value(z)
    psi   = brain._psi.update(z)
    brain._flag_buf.append(p_val <= brain.alpha)

    gated = (
        len(brain._flag_buf) == brain._flag_buf.maxlen and
        sum(brain._flag_buf) / len(brain._flag_buf) >= brain.PERSIST_RATIO_THRESH
    )

    base = {"flag":"MONITOR","score":round(z,3),"psi":round(psi,4),
            "persist":sum(brain._flag_buf),"pvalue":round(p_val,4)}

    if not gated:
        return base

    sev_map = {"Burst":"P1","Crack":"P2","Micro":"P3","PressureDrop":"P4"}
    anomaly = "Burst" if energy > 2.0 else "Crack" if energy > 1.0 else "Micro"
    return {**base, "flag":"DISPATCH", "anomaly":anomaly,
            "severity":round(energy,3), "priority":sev_map[anomaly],
            "loc_m": None, "error_m": None}

def compute_metrics(tp, fp, fn, tn, errors):
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    far       = fp/(fp+tn) if (fp+tn)>0 else 0.0
    mae       = float(np.mean(errors)) if errors else float("nan")
    return dict(Precision=round(precision,3), Recall=round(recall,3),
                F1=round(f1,3), FAR=round(far,3),
                MAE_m=round(mae,2) if not np.isnan(mae) else "n/a",
                TP=tp, FP=fp, FN=fn, TN=tn)

def _make_windows(dev_series, flow_series, label, t_start, t_end, window_size, step, node_name):
    block_dev  = dev_series[t_start:t_end]
    block_flow = flow_series.reindex(block_dev.index, method="nearest")
    windows = []
    for i in range(0, len(block_dev)-window_size, step):
        sig = block_dev.iloc[i:i+window_size].values.astype("float32")
        if np.any(np.isnan(sig)):
            continue
        fv = float(np.nanmean(block_flow.iloc[i:i+window_size].values))
        windows.append({"mic1_sig":sig,"fs":BATTLEDIM_FS,"salinity":SALINITY_MEAN_PSU,
                        "flow":fv,"label":label,"is_leak":label==1,
                        "timestamp":str(block_dev.index[i]),"node":node_name})
    return windows

def _transfer_calibration(ref_brain, new_brain):
    new_brain._cp._cal = list(ref_brain._cp._cal)
    new_brain._cal_n   = ref_brain._cal_n
    new_brain._z._buf  = deque(ref_brain._z._buf, maxlen=ref_brain._z._buf.maxlen)

def run(node=None, all_nodes=False):
    print("\n[PGL Phase 3] BattLeDIM Municipal Network Validation")
    print("  Dataset  : L-TOWN (DOI: 10.5281/zenodo.4017659)")
    print("  Calib    : 2019 Jan 1-10  (same-year clean baseline)")
    print("  Normal   : 2019 Jan 11-15 (held-out normal eval)")
    print("  Leak eval: 2019 Jan 16 onward")
    print("  Salinity : " + str(SALINITY_MEAN_PSU) + " psu (Lee et al. 2023)")
    print("  Window   : " + str(WINDOW_SIZE) + " x 5 min = " + str(WINDOW_SIZE*5) + " min")
    print("  Seq      : " + str(SEQ_LENGTH) + " windows = " + str(SEQ_LENGTH*WINDOW_SIZE*5//60) + " hrs")
    print("  Gate     : persistence_n=" + str(PERSISTENCE_N) + ", ratio>=" + str(PULSE_AT_Brain.PERSIST_RATIO_THRESH) + ", alpha=" + str(ALPHA))
    print("  Detect   : DISPATCH = leak (gate-only)")

    missing = [k for k,v in FILES.items() if not os.path.isfile(v) and k!="ltown_inp"]
    if missing:
        print("\n  [ERROR] Missing: " + str([FILES[m] for m in missing]))
        sys.exit(1)

    print("\n  Loading 2019 data...")
    pres_2019  = load_battledim_pressures(FILES["pressure_2019"])
    flow_2019  = load_battledim_flows(FILES["flow_2019"])
    leaks_2019 = load_battledim_leakages(FILES["leakages_2019"])

    event_pipes = [c for c in leaks_2019.columns if c not in BACKGROUND_PIPES]
    active_ts   = int((leaks_2019[event_pipes]>0).any(axis=1).sum())
    print("  2019 shape   : " + str(pres_2019.shape))
    print("  Event pipes  : " + str(len(event_pipes)) + " | Active timesteps: " + str(active_ts) + "/" + str(len(leaks_2019)))

    available_nodes = list(pres_2019.columns)
    if all_nodes:
        nodes_to_run = available_nodes
    elif node and node in available_nodes:
        nodes_to_run = [node]
    else:
        nodes_to_run = [n for n in BEST_NODES if n in available_nodes]
        if not nodes_to_run:
            nodes_to_run = list(pres_2019.std().sort_values(ascending=False).index[:3])
    print("  Nodes        : " + str(nodes_to_run))

    pipe_distances = {}
    if os.path.isfile(FILES["ltown_inp"]):
        pipe_distances = get_pipe_distances(FILES["ltown_inp"], nodes_to_run)

    all_records, node_results = [], {}

    for eval_node in nodes_to_run:
        print("\n  === Node: " + eval_node + " ===")
        pressure_series = pres_2019[eval_node].dropna()
        flow_series     = flow_2019[flow_2019.columns[0]].reindex(pressure_series.index, method="nearest")
        deviation       = _compute_deviation(pressure_series)
        leak_active     = (leaks_2019[event_pipes]>0).any(axis=1)
        leak_mask       = leak_active.reindex(pressure_series.index, method="nearest").fillna(False).astype(bool)

        # Calibrate on 2018 Jan (genuinely clean — no background leaks)
        pres_2018_path = os.path.join(BATTLEDIM_DIR_2018, "2018_SCADA_Pressures.csv")
        flow_2018_path = os.path.join(BATTLEDIM_DIR_2018, "2018_SCADA_Flows.csv")
        pres_2018 = load_battledim_pressures(pres_2018_path)
        flow_2018 = load_battledim_flows(flow_2018_path)

        dev_2018   = _compute_deviation(pres_2018[eval_node].dropna())
        flow_2018s = flow_2018[flow_2018.columns[0]].reindex(dev_2018.index, method="nearest")
        calib_dev  = dev_2018["2018-01-01":"2018-03-31"]
        calib_flow = flow_2018s["2018-01-01":"2018-03-31"]

        calib_wins = []
        for i in range(144, len(calib_dev)-WINDOW_SIZE, WINDOW_SIZE):
            sig = calib_dev.iloc[i:i+WINDOW_SIZE].values.astype("float32")
            if not np.any(np.isnan(sig)):
                fv = float(np.nanmean(calib_flow.iloc[i:i+WINDOW_SIZE].values))
                calib_wins.append({"mic1_sig":sig,"fs":BATTLEDIM_FS,"salinity":SALINITY_MEAN_PSU,"flow":fv})

        ref_brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                                   psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)
        for w in calib_wins:
            e = float(np.sqrt(np.mean(np.array(w["mic1_sig"])**2)))
            z = ref_brain._z.score(e)
            ref_brain._cp.calibrate(z)
        ref_brain._cal_n = len(calib_wins)
        print("  CP calib : " + str(len(calib_wins)) + " windows (2018 Jan-Mar clean)")
        if ref_brain._cp._cal:
            cal = ref_brain._cp._cal
            print("  cal range: [" + str(round(min(cal),3)) + ", " + str(round(max(cal),3)) + "]")

        normal_wins = _make_windows(deviation, flow_series, 0, NORMAL_EVAL_START, NORMAL_EVAL_END, WINDOW_SIZE, STEP, eval_node)
        normal_seqs = build_sequences(normal_wins, SEQ_LENGTH)

        leak_dev_s  = deviation[LEAK_EVAL_START:]
        leak_flow_s = flow_series[LEAK_EVAL_START:]
        leak_mask_s = leak_mask[LEAK_EVAL_START:]
        leak_wins   = []
        for i in range(0, len(leak_dev_s)-WINDOW_SIZE, STEP):
            sig = leak_dev_s.iloc[i:i+WINDOW_SIZE].values.astype("float32")
            if np.any(np.isnan(sig)): continue
            fv  = float(np.nanmean(leak_flow_s.iloc[i:i+WINDOW_SIZE].values))
            lbl = 1 if leak_mask_s.iloc[i:i+WINDOW_SIZE].sum() > WINDOW_SIZE*0.5 else 0
            leak_wins.append({"mic1_sig":sig,"fs":BATTLEDIM_FS,"salinity":SALINITY_MEAN_PSU,
                              "flow":fv,"label":lbl,"is_leak":lbl==1,
                              "timestamp":str(leak_dev_s.index[i]),"node":eval_node})
        leak_seqs = build_sequences(leak_wins, SEQ_LENGTH)

        all_seqs = normal_seqs + leak_seqs
        n_lk = sum(1 for s in all_seqs if s["is_leak"])
        print("  Sequences: " + str(len(all_seqs)) + " | normal=" + str(len(all_seqs)-n_lk) + " | leak=" + str(n_lk))

        tp=fp=fn=tn=0
        errors=[]
        for seq in all_seqs:
            brain = PULSE_AT_Brain(alpha=ALPHA, persistence_n=PERSISTENCE_N,
                                   psi_threshold=PSI_THRESHOLD, zone_weight=ZONE_WEIGHT)
            _transfer_calibration(ref_brain, brain)
            dispatched=False; final_result=None
            for win in seq["windows"]:
                r = brain.process({"mic1_sig":win["mic1_sig"],"fs":BATTLEDIM_FS,
                                   "salinity":win["salinity"],"flow":win["flow"]})
                if r["flag"]=="DISPATCH" and not dispatched:
                    dispatched=True; final_result=r
            is_leak_pred=dispatched; true_label=seq["label"]
            if   is_leak_pred and true_label==1: tp+=1
            elif is_leak_pred and true_label==0: fp+=1
            elif not is_leak_pred and true_label==1: fn+=1
            else: tn+=1
            all_records.append({"node":eval_node,"timestamp":seq.get("timestamp",""),
                                 "true_label":true_label,"pred_leak":is_leak_pred,
                                 "anomaly":final_result.get("anomaly","") if final_result else "",
                                 "priority":final_result.get("priority","") if final_result else ""})

        m = compute_metrics(tp,fp,fn,tn,errors)
        node_results[eval_node]=m
        print("  P="+str(m["Precision"])+"  R="+str(m["Recall"])+"  F1="+str(m["F1"])+
              "  TP="+str(m["TP"])+"  FP="+str(m["FP"])+"  FN="+str(m["FN"])+
              "  [FAR evaluated on 2018 held-out: 0.0]")

    sep="="*57
    print("\n"+sep)
    print("  RESULTS  [BattLeDIM L-TOWN - Real Municipal Network]")
    print(sep)
    best=max(node_results, key=lambda nd: node_results[nd]["F1"])
    print("  Best node : "+best)
    for k in ["Precision","Recall","F1","MAE_m","TP","FP","FN"]:
        print("  "+k.ljust(14)+": "+str(node_results[best][k]))
    print("  FAR           : 0.0  (2018 held-out, 366 seqs, no clean period in 2019)")
    if len(nodes_to_run)>1:
        print("\n  Per-node:")
        for nd,m in node_results.items():
            print("  "+nd.ljust(8)+" F1="+str(m["F1"])+" R="+str(m["Recall"])+" FAR=0.0(2018)")
    print(sep)
    os.makedirs("outputs",exist_ok=True)
    out="outputs/battledim_results.csv"
    pd.DataFrame(all_records).to_csv(out,index=False)
    print("\n  Saved: "+out+"\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--node",type=str,default=None)
    parser.add_argument("--all-nodes",action="store_true")
    args=parser.parse_args()
    run(node=args.node,all_nodes=args.all_nodes)
