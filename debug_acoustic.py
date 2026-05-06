import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pgpl_brain      import PGPLBrain, TidalWindow
from src.mendeley_loader import load_hydrophone_pairs, read_raw_pair

pairs = load_hydrophone_pairs()
print(f"\nTotal pairs: {len(pairs)}\n")

print(f"{'stem':<40} {'GT'} {'p3':>7} {'p4':>7} {'fused':>7} {'thresh':>7} {'conf'}")
print('-' * 90)

for j, pair_info in enumerate(pairs[:20]):
    try:
        sig_a, sig_b, fs = read_raw_pair(
            pair_info['path_a'],
            pair_info['path_b'],
        )
    except Exception as e:
        print(f'  [ERR] {pair_info[''stem'']}: {e}')
        continue

    brain = PGPLBrain(
        fs               = fs,
        pipe_diameter_m  = 0.05,
        pipe_thickness_m = 0.005,
        pipe_material    = 'hdpe',
        saline           = False,
        sensor_spacing_m = 1.5,
    )

    for ph in ['ebb', 'flood', 'slack_low', 'slack_high', 'spring']:
        brain.tidal.add_phase(TidalWindow(
            phase      = ph,
            psi_offset = 0.0,
            alpha_adj  = brain.tidal.adaptive_alpha(),
            timestamp  = float(j),
        ))

    event = brain.process_acoustic(sig_a, sig_b, float(j), 'ebb', 0.0)
    gate  = event.meta.get('gate', {})

    print(
        f"  {pair_info['stem']:<38} "
        f"  {pair_info['label']}  "
        f"{event.p3_score:7.4f}  "
        f"{event.p4_score:7.4f}  "
        f"{event.fused_score():7.4f}  "
        f"{gate.get('adj_threshold', 0):7.4f}  "
        f"{'YES' if gate.get('confirmed') else 'NO'}"
    )

print('\n-- Signal Stats (first 3 pairs) --')
for j, pair_info in enumerate(pairs[:3]):
    try:
        sig_a, sig_b, fs = read_raw_pair(
            pair_info['path_a'],
            pair_info['path_b'],
        )
        from scipy.signal import butter, sosfilt
        sos    = butter(4, [200.0, 4000.0], btype='bandpass', fs=fs, output='sos')
        a_filt = sosfilt(sos, sig_a.astype(float))

        print(f"\n  {pair_info['stem']} (label={pair_info['label']})")
        print(f"    fs             = {fs} Hz")
        print(f"    sig_a length   = {len(sig_a)}")
        print(f"    sig_a min/max  = [{sig_a.min():.6f}, {sig_a.max():.6f}]")
        print(f"    raw variance   = {np.var(sig_a):.8f}")
        print(f"    filt variance  = {np.var(a_filt):.8f}")
        print(f"    ratio filt/raw = {np.var(a_filt)/(np.var(sig_a)+1e-12):.6f}")
    except Exception as e:
        print(f'  [ERR] {e}')
