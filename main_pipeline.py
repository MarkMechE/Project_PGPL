# main_pipeline.py
# Runs 100-scenario simulation, saves CSV metrics for thesis exhibit

import os
import numpy as np
import pandas as pd
from pgpl_brain import PULSE_AT_Brain

os.makedirs("outputs", exist_ok=True)

brain = PULSE_AT_Brain()

SCENARIOS = 100
results = []

print(f"Running {SCENARIOS} simulation scenarios...\n")

for i in range(SCENARIOS):
    # Vary leak distance, noise, flow anomaly
    true_dist = np.random.uniform(5, 120)
    noise_sigma = np.random.uniform(0.02, 0.12)
    flow_anomaly = np.random.uniform(14, 25)
    pres_drop = 3.2 - np.random.uniform(0.1, 0.5)
    var_z_val = np.random.uniform(1.5, 4.5)
    sensor_idx = np.random.randint(0, max(1, len(brain.node_list)))

    mic1, mic2 = brain._sim_piezo(true_dist, noise_sigma=noise_sigma)

    sensors = {
        "flow":      flow_anomaly,
        "pressure":  pres_drop,
        "var_z":     var_z_val,
        "mic1_sig":  mic1,
        "mic2_sig":  mic2,
        "fs":        10000,
        "true_dist": true_dist,
    }

    # Reset persist for each fresh scenario
    brain.persist_count = 0
    brain.psi_window.clear()

    dispatched = False
    final_result = None

    # Up to 6 cycles per scenario (simulate time windows)
    for cycle in range(6):
        result = brain.process(sensors, sensor_node_idx=sensor_idx)
        if result["flag"] == "DISPATCH":
            dispatched = True
            final_result = result
            break

    row = {
        "scenario":     i + 1,
        "true_dist_m":  round(true_dist, 2),
        "noise_sigma":  round(noise_sigma, 3),
        "flow_anomaly": round(flow_anomaly, 2),
        "dispatched":   dispatched,
        "loc_m":        final_result["loc_m"] if dispatched else None,
        "error_m":      final_result["error_m"] if dispatched else None,
        "score":        result["score"],
        "psi":          result["psi"],
        "persist":      result["persist"],
        "gps_lat":      final_result["gps"][0] if dispatched else None,
        "gps_lon":      final_result["gps"][1] if dispatched else None,
    }
    results.append(row)

df = pd.DataFrame(results)
df.to_csv("outputs/simulation_results.csv", index=False)

# --- Summary metrics ---
detected = df[df["dispatched"] == True]
total = len(df)
tp = len(detected)
fp = len(df[(df["dispatched"] == True) & (df["flow_anomaly"] < 15)])
fn = total - tp

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall / (precision + recall)
      if (precision + recall) > 0 else 0)

mean_err = detected["error_m"].mean() if len(detected) > 0 else None
far = fp / total

print("\n=== PGPL Simulation Results ===")
print(f"  Scenarios run   : {total}")
print(f"  Dispatched (TP) : {tp}")
print(f"  False alarms    : {fp}")
print(f"  F1 score        : {f1:.3f}")
print(f"  Mean loc error  : {mean_err:.2f} m" if mean_err else "  Mean loc error  : N/A")
print(f"  FAR             : {far:.3f}")
print(f"\n  Saved: outputs/simulation_results.csv")