"""
main_pipeline.py  —  PGL Phase 1 evaluation pipeline
Run: python main_pipeline.py

Flow
----
1. Generate 500 physics-informed synthetic scenarios
2. CP warm-up (120 normal windows)
3. For each scenario: Biot velocity → gate → classify → TDOA distance
4. Report correct F1, FAR, MAE
5. Save outputs/simulation_results.csv
"""

import os
import numpy as np
import pandas as pd

from biot_velocity    import get_biot_velocity
from data_generator   import make_dataset, generate_scenario
from pgpl_brain       import PULSE_AT_Brain
from anomaly_classifier import classify, severity_score, is_leak_event

os.makedirs("outputs", exist_ok=True)

FS       = 2000
N_WARMUP = 120
N_EVAL   = 500


# ── Correct metric formulas ───────────────────────────────────────────────
def compute_metrics(tp, fp, fn, tn, errors):
    precision = tp / (tp + fp)       if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)       if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    far       = fp / (fp + tn)       if (fp + tn) > 0 else 0.0
    mae       = float(np.mean(errors)) if errors else float('nan')
    return {
        'TP':        tp,
        'FP':        fp,
        'FN':        fn,
        'TN':        tn,
        'Precision': round(precision, 3),
        'Recall':    round(recall,    3),
        'F1':        round(f1,        3),
        'FAR':       round(far,       3),
        'MAE_m':     round(mae, 2) if not np.isnan(mae) else 'n/a',
    }


def run(n_eval=N_EVAL):
    print(f"\n{'='*55}")
    print(f"  PGL Phase 1  —  Physics-Informed Simulation")
    print(f"  Scenarios: {n_eval}  |  FS: {FS} Hz  |  Warm-up: {N_WARMUP}")
    print(f"{'='*55}\n")

    brain  = PULSE_AT_Brain(alpha=0.10, persistence_n=3,
                            psi_threshold=0.20, zone_weight=0.5)

    # ── CP warm-up on clean noise ─────────────────────────────────────────
    rng_wu = np.random.default_rng(0)
    for _ in range(N_WARMUP):
        wu = generate_scenario(
            fs=FS, is_leak=False,
            salinity_psu    = float(rng_wu.uniform(3.0, 10.0)),
            tidal_phase_rad = float(rng_wu.uniform(0.0, 6.283)),
        )
        brain.process({'mic1_sig': wu['signal'], 'fs': FS,
                       'salinity': wu['salinity_psu']})

    # ── Evaluation ────────────────────────────────────────────────────────
    dataset = make_dataset(n_samples=n_eval, fs=FS)
    tp = fp = fn = tn = 0
    errors  = []
    records = []

    for i, sample in enumerate(dataset):
        signal     = sample['signal']
        true_label = sample['label']        # 1=leak
        true_dist  = sample['distance_m']
        salinity   = sample['salinity_psu']
        velocity   = get_biot_velocity(salinity)

        result = brain.process({
            'mic1_sig':  signal,
            'fs':        FS,
            'salinity':  salinity,
            'true_dist': true_dist,
        })

        gated      = result['flag'] == 'DISPATCH'
        pred_leak  = gated and is_leak_event(result.get('anomaly', ''))

        if pred_leak and true_label == 1:
            tp += 1
            if result.get('error_m') is not None:
                errors.append(result['error_m'])
        elif pred_leak and true_label == 0:
            fp += 1
        elif not pred_leak and true_label == 1:
            fn += 1
        else:
            tn += 1

        records.append({
            'scenario':    i + 1,
            'true_label':  true_label,
            'true_dist_m': round(true_dist, 2),
            'salinity_psu': round(salinity, 2),
            'velocity_ms': round(velocity, 1),
            'gated':       gated,
            'pred_leak':   pred_leak,
            'anomaly':     result.get('anomaly', '-'),
            'priority':    result.get('priority', '-'),
            'severity':    result.get('severity', '-'),
            'loc_m':       result.get('loc_m', '-'),
            'error_m':     result.get('error_m', '-'),
            'p_value':     result.get('pvalue', '-'),
            'psi':         result.get('psi', '-'),
            'score':       result.get('score', '-'),
        })

    # ── Metrics ───────────────────────────────────────────────────────────
    m = compute_metrics(tp, fp, fn, tn, errors)

    print(f"{'─'*45}")
    print(f"  RESULTS  (n={n_eval})")
    print(f"{'─'*45}")
    for k, v in m.items():
        print(f"  {k:<14}: {v}")
    print(f"{'─'*45}")

    # ── Sample output ─────────────────────────────────────────────────────
    dispatched = [r for r in records if r['gated']]
    if dispatched:
        print(f"\n  Dispatched events ({len(dispatched)} total, first 8):\n")
        hdr = (f"  {'#':>3}  {'True':>4}  {'Event':<14}  "
               f"{'Tier':>12}  {'Sev':>5}  {'Dist':>6}  "
               f"{'Err':>6}  {'v m/s':>7}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for r in dispatched[:8]:
            print(
                f"  {r['scenario']:>3}  {r['true_label']:>4}  "
                f"{str(r['anomaly']):<14}  "
                f"{str(r['priority']):>12}  "
                f"{str(r['severity']):>5}  "
                f"{str(r['loc_m']):>6}  "
                f"{str(r['error_m']):>6}  "
                f"{r['velocity_ms']:>7}"
            )

    # ── Save CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    path = "outputs/simulation_results.csv"
    df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")
    print(f"\n  Note: Phase 1 heuristic baseline on synthetic data.")
    print(f"  CNN-Transformer + real sensor validation → Phase 2.\n")

    return m


if __name__ == "__main__":
    run()