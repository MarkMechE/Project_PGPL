# main_pipeline.py
# 100-scenario batch simulation — produces thesis Exhibit A

import os
import numpy as np
import pandas as pd
from pgpl_brain import PULSE_AT_Brain

os.makedirs("outputs", exist_ok=True)
brain = PULSE_AT_Brain()
results = []

print("Running 100 simulation scenarios...\n")
for i in range(100):
    true_dist    = np.random.uniform(5, 120)
    noise_sigma  = np.random.uniform(0.02, 0.12)
    flow_anomaly = np.random.uniform(14, 25)
    pres_drop    = 3.2 - np.random.uniform(0.1, 0.5)
    var_z_val    = np.random.uniform(1.5, 4.5)
    sidx         = np.random.randint(0, max(1, len(brain.node_list)))

    mic1, mic2 = brain._sim_piezo(true_dist, noise_sigma=noise_sigma)
    sensors = {"flow": flow_anomaly, "pressure": pres_drop,
               "var_z": var_z_val, "mic1_sig": mic1, "mic2_sig": mic2,
               "fs": 10000, "true_dist": true_dist}

    brain.persist_count = 0
    brain.psi_window.clear()
    dispatched, final = False, None

    for _ in range(6):
        r = brain.process(sensors, sensor_node_idx=sidx)
        if r["flag"] == "DISPATCH":
            dispatched, final = True, r
            break

    results.append({
        "scenario": i+1, "true_dist_m": round(true_dist,2),
        "dispatched": dispatched,
        "loc_m":    final["loc_m"]    if dispatched else None,
        "error_m":  final["error_m"]  if dispatched else None,
        "anomaly":  final.get("anomaly","—") if dispatched else None,
        "score": r["score"], "psi": r["psi"],
        "gps_lat": final["gps"][0] if dispatched else None,
        "gps_lon": final["gps"][1] if dispatched else None,
    })

df = pd.DataFrame(results)
df.to_csv("outputs/simulation_results.csv", index=False)
tp = df["dispatched"].sum()
me = df["error_m"].dropna().mean()
f1 = 2*tp / (2*tp + (100-tp))
print(f"F1={f1:.3f}  Dispatched={tp}/100  MeanError={me:.1f}m")
print("Saved: outputs/simulation_results.csv")