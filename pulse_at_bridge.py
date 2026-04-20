# pulse_at_bridge.py
# Loads PULSE-AT CSV and feeds each row through PGPL brain

import os
import numpy as np
import pandas as pd
from pgpl_brain import PULSE_AT_Brain

EXPORT_PATH = "pulse_at_exports/pulse_at_results.csv"

def load_or_generate_pulse_at_data(n=200):
    if os.path.exists(EXPORT_PATH):
        print(f"Loading: {EXPORT_PATH}")
        return pd.read_csv(EXPORT_PATH)
    print("Generating synthetic PULSE-AT data...")
    np.random.seed(42)
    regime = (["normal"]*80 + ["stress"]*60 + ["normal"]*40 + ["stress"]*20)[:n]
    rows = []
    for i, reg in enumerate(regime):
        if reg == "stress":
            flow, pres, var_z = (np.random.normal(21,2.5),
                                 np.random.normal(2.75,0.15),
                                 np.random.normal(3.4,0.6))
        else:
            flow, pres, var_z = (np.random.normal(13,0.8),
                                 np.random.normal(3.20,0.08),
                                 np.random.normal(0.8,0.3))
        rows.append({"timestamp": str(pd.Timestamp("2026-01-01") +
                                    pd.Timedelta(minutes=i*5)),
                     "flow": round(max(0,flow),3),
                     "pressure": round(max(0,pres),3),
                     "variance_z": round(max(0,var_z),3),
                     "regime": reg})
    df = pd.DataFrame(rows)
    os.makedirs("pulse_at_exports", exist_ok=True)
    df.to_csv(EXPORT_PATH, index=False)
    print(f"  Saved {len(df)} rows to {EXPORT_PATH}")
    return df

def map_columns(df):
    rename = {}
    for src, tgt in [
        (["flow","flow_rate","flow_lpm","q_flow"], "flow"),
        (["pressure","pres","pressure_bar","p_bar"], "pressure"),
        (["variance_z","var_z","turbulence","z_var"], "var_z"),
    ]:
        for col in src:
            if col in df.columns:
                rename[col] = tgt
                break
    return df.rename(columns=rename)

def row_to_sensors(row, brain):
    flow     = float(row.get("flow", 13.0))
    pressure = float(row.get("pressure", 3.2))
    var_z    = float(row.get("var_z", 0.5))
    dp       = max(0.0, 3.2 - pressure)
    dist     = max(5.0, 120.0 - dp * 150.0)
    amp      = 1.0 + max(0.0, (flow - 13.0) / 10.0)
    mic1, mic2 = brain._sim_piezo(dist, noise_sigma=0.04)
    return {"flow": flow, "pressure": pressure, "var_z": var_z,
            "mic1_sig": mic1 * amp, "mic2_sig": mic2 * amp,
            "fs": 10000, "true_dist": dist,
            "timestamp": str(row.get("timestamp","")),
            "regime": str(row.get("regime","unknown"))}

def run_bridge(n_sensors=4):
    brain = PULSE_AT_Brain()
    df    = map_columns(load_or_generate_pulse_at_data())
    max_n = max(1, len(brain.node_list))
    snodes = [int(i * max_n / n_sensors) for i in range(n_sensors)]
    print(f"Sensor nodes: {n_sensors} — indices {snodes}")
    results = []
    for idx, row in df.iterrows():
        sensors = row_to_sensors(row, brain)
        result  = brain.process(sensors, sensor_node_idx=idx % n_sensors)
        results.append({
            "row": idx, "timestamp": sensors["timestamp"],
            "regime": sensors["regime"],
            "flow": sensors["flow"], "pressure": sensors["pressure"],
            "var_z": sensors["var_z"],
            "pgpl_score": result["score"], "pgpl_flag": result["flag"],
            "pgpl_loc_m": result.get("loc_m"),
            "pgpl_error_m": result.get("error_m"),
            "pgpl_psi": result["psi"], "pgpl_persist": result["persist"],
            "pgpl_anomaly": result.get("anomaly","—"),
            "gps_lat": result.get("gps",(None,None))[0],
            "gps_lon": result.get("gps",(None,None))[1],
        })
    os.makedirs("outputs", exist_ok=True)
    out = pd.DataFrame(results)
    out.to_csv("outputs/bridge_results.csv", index=False)
    print(f"Saved: outputs/bridge_results.csv")
    _summary(out)
    return out

def _summary(df):
    tp = len(df[(df["pgpl_flag"]=="DISPATCH") & (df["regime"]=="stress")])
    fp = len(df[(df["pgpl_flag"]=="DISPATCH") & (df["regime"]!="stress")])
    fn = len(df[df["regime"]=="stress"]) - tp
    p  = tp/(tp+fp) if (tp+fp)>0 else 0
    r  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
    me = pd.to_numeric(df["pgpl_error_m"],errors="coerce").dropna().mean()
    print(f"\n=== Bridge Summary ===")
    print(f"  F1={f1:.3f}  Precision={p:.3f}  Recall={r:.3f}  MeanErr={me:.1f}m")

if __name__ == "__main__":
    run_bridge(n_sensors=4)