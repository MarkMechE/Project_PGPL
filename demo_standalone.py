"""
pgpl_demo.py — PGPL v2.0 Live Demo Dashboard
Mirrors run_pipeline.py exactly:
  - 2018 Pressures → calibrate_from_year()
  - 2019 Pressures + Flows + Leakages → detect
Run: streamlit run pgpl_demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import time
from scipy.io import wavfile
from sklearn.metrics import f1_score, precision_score, recall_score
from src.pgpl_brain import PGPLBrain, TidalWindow

st.set_page_config(page_title="PGPL v2.0 Demo", layout="wide", page_icon="🧠")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;600&display=swap');
html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }
h1, h2, h3 { font-family: 'Share Tech Mono', monospace !important; }
.dispatch-box {
    background: #0f2a1a; border: 2px solid #00ff88;
    padding: 1rem 1.5rem; border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    color: #00ff88; font-size: 1.4rem; text-align: center;
    animation: blink 1s infinite;
}
.monitor-box {
    background: #1a1a2e; border: 2px solid #4a4a6a;
    padding: 1rem 1.5rem; border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    color: #aaaacc; font-size: 1.4rem; text-align: center;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.5} }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PHASES = ["ebb", "flood", "slack_low", "slack_high", "spring"]

def mock_tidal(i):
    return PHASES[i % len(PHASES)], float(np.sin(i * 0.01) * 2.5)

def compute_far(y_pred, y_true):
    denom = int(np.sum(y_true == 0))
    return float(np.sum((y_pred == 1) & (y_true == 0)) / max(denom, 1))

def _parse_eu_csv(file) -> pd.DataFrame:
    """
    Robust parser for BattleDIM EU-format CSVs (sep=';', decimal=',').
    Falls back to standard comma CSV.
    Returns DataFrame with DatetimeIndex, all columns numeric.
    """
    file.seek(0)
    first = file.read(512).decode('utf-8', errors='replace')
    file.seek(0)
    sep     = ';' if first.count(';') > first.count(',') else ','
    decimal = ',' if sep == ';' else '.'

    for kwargs in [
        dict(sep=sep, decimal=decimal, index_col=0, parse_dates=True, low_memory=False),
        dict(sep=',', decimal='.',     index_col=0, parse_dates=True, low_memory=False),
        dict(sep=sep, decimal=decimal, index_col=0, parse_dates=True,
             skipinitialspace=True, on_bad_lines='skip', low_memory=False),
    ]:
        try:
            file.seek(0)
            df = pd.read_csv(file, **kwargs)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(how='all').dropna(axis=1, how='all')
            if len(df) > 0 and df.shape[1] > 0:
                return df
        except Exception:
            continue
    raise ValueError("Could not parse CSV. Check sep/decimal format.")

def load_wav(file):
    fs_data, sig = wavfile.read(file)
    if sig.ndim > 1: sig = sig[:, 0]
    sig = sig.astype(np.float64) / (np.max(np.abs(sig)) + 1e-9)
    return sig, fs_data

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    mode         = st.radio("Sensor Mode", ["SCADA (CSV)", "Acoustic (WAV pair)"])
    base_alpha   = st.slider("Base α", 0.01, 0.20, 0.05, 0.01)
    window_size  = st.slider("Window (rows/samples)", 100, 2000, 500, 100)
    stream_speed = st.slider("Stream delay (s/window)", 0.0, 1.0, 0.2, 0.05)
    saline_mode  = st.checkbox("Saline / Tidal gating (EDC Busan)", value=False)
    st.markdown("---")
    st.caption("BattleDIM: saline=OFF | EDC Busan: saline=ON")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🧠 PGPL v2.0 — Leak Detection Demo")
st.caption("Real PGPLBrain · Adaptive-Z · MondrianCP · Tidal Gating · P1–P4 Fusion")

# ── Upload UI ─────────────────────────────────────────────────────────────────
f_press18 = f_press19 = f_flows19 = f_leaks19 = None
wav_a = wav_b = None
gt_label = 0

if mode == "SCADA (CSV)":
    st.markdown("### 📂 Upload BattleDIM files")
    st.caption("Mirrors `run_pipeline.py` exactly. All 4 files needed for accurate F1/FAR.")

    row1a, row1b = st.columns(2)
    with row1a:
        st.markdown("**Calibration year**")
        f_press18 = st.file_uploader("Pressures 2018 (calibration)", type=["csv"], key="p18")
    with row1b:
        st.markdown("**Detection year**")
        f_press19 = st.file_uploader("Pressures 2019", type=["csv"], key="p19")

    row2a, row2b = st.columns(2)
    with row2a:
        f_flows19 = st.file_uploader("Flows 2019", type=["csv"], key="f19")
    with row2b:
        f_leaks19 = st.file_uploader("Leakages 2019 (GT — required for F1)", type=["csv"], key="l19")

    ready = (f_press19 is not None and f_flows19 is not None and f_leaks19 is not None)
    if not ready:
        missing = []
        if f_press19 is None: missing.append("Pressures 2019")
        if f_flows19 is None: missing.append("Flows 2019")
        if f_leaks19 is None: missing.append("Leakages 2019")
        st.info(f"👆 Still needed: {', '.join(missing)}")

else:
    st.markdown("### 📂 Upload WAV pair")
    col_a, col_b = st.columns(2)
    with col_a: wav_a = st.file_uploader("Sensor A (.wav)", type=["wav"])
    with col_b: wav_b = st.file_uploader("Sensor B (.wav)", type=["wav"])
    gt_label = st.selectbox("Ground truth for this pair", [0, 1],
                             format_func=lambda x: "1 = Leak" if x else "0 = No Leak")
    ready = wav_a is not None and wav_b is not None

if not ready:
    st.stop()

if not st.button("🚀 Run Detection", type="primary"):
    st.stop()

status_ph  = st.empty()
flag_ph    = st.empty()
chart_ph   = st.empty()
metrics_ph = st.empty()
table_ph   = st.empty()
results    = []

# ══════════════════════════════════════════════════════════════════════════════
# SCADA PATH — mirrors run_pipeline.py run_battledim_f1()
# ══════════════════════════════════════════════════════════════════════════════
if mode == "SCADA (CSV)":

    # 1. Parse 2019 files
    with st.spinner("Parsing 2019 CSV files …"):
        pressures19 = _parse_eu_csv(f_press19)
        flows19     = _parse_eu_csv(f_flows19)
        leaks19     = _parse_eu_csv(f_leaks19)

        # Align on common timestamps (same as run_pipeline.py common_idx)
        common = pressures19.index.intersection(flows19.index)
        common = common.dropna()

        p_col = pressures19.columns[0]
        f_col = flows19.columns[0]

        # Build GT — same as build_ground_truth()
        leaks19 = leaks19.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        gt_series = (leaks19 > 0.5).any(axis=1).astype(int)
        gt_series = gt_series.reindex(common, fill_value=0)

        df19 = pd.DataFrame({
            'pressure': pressures19.loc[common, p_col].values,
            'flow':     flows19.loc[common, f_col].values,
            'anomaly':  gt_series.values,
        }).dropna().reset_index(drop=True)

    leak_pct = df19['anomaly'].mean()
    st.success(
        f"✅ 2019: {len(df19):,} timesteps | "
        f"leak rate: {leak_pct:.1%} | "
        f"pressure {df19['pressure'].min():.1f}–{df19['pressure'].max():.1f}"
    )
    if leak_pct == 0:
        st.warning("⚠️ GT leak rate = 0%. Leakages CSV may not have aligned timestamps with Pressures/Flows.")

    # 2. Init brain — fs=1/60 Hz (BattleDIM 1 sample/min)
    brain = PGPLBrain(fs=1/60, saline=saline_mode, base_alpha=base_alpha)

    # 3. Calibrate — use 2018 if uploaded, else fallback to first 20% of 2019
    if f_press18 is not None:
        with st.spinner("Calibrating from 2018 pressures …"):
            press18  = _parse_eu_csv(f_press18)
            p18_col  = press18.columns[0]
            cal_vals = press18[p18_col].dropna().values
            cal_mean = float(np.mean(cal_vals))
            cal_std  = float(np.std(cal_vals)) + 1e-9
            cal_z    = [abs(v - cal_mean) / cal_std for v in cal_vals]
            brain.calibrate_from_year(cal_z, phase="default")
        st.success(f"✅ Calibrated on {len(cal_vals):,} 2018 samples")
    else:
        cal_end  = max(len(df19) // 5, 1)
        cal_vals = df19['pressure'].iloc[:cal_end].values
        cal_mean = float(np.mean(cal_vals))
        cal_std  = float(np.std(cal_vals)) + 1e-9
        cal_z    = [abs(v - cal_mean) / cal_std for v in cal_vals]
        brain.calibrate_from_year(cal_z, phase="default")
        df19     = df19.iloc[cal_end:].reset_index(drop=True)
        st.info("ℹ️ No 2018 file — calibrated on first 20% of 2019 data.")

    # 4. Stream detection
    n_windows = len(df19) // window_size
    status_ph.info(f"📡 Streaming {n_windows} windows …")

    for i, start in enumerate(range(0, len(df19) - window_size + 1, window_size)):
        end   = start + window_size
        chunk = df19.iloc[start:end]
        phase, tidal_psi = mock_tidal(i)

        event = None
        for row_i, row in chunk.iterrows():
            event = brain.process_scada(
                pressure_psi = float(row['pressure']),
                flow_lps     = float(row['flow']),
                timestamp    = float(start + row_i),
                tidal_phase  = phase,
                tidal_psi    = tidal_psi,
            )

        gate      = event.meta.get("gate", {})
        confirmed = gate.get("confirmed", False)
        gt_win    = int(chunk['anomaly'].mean() > 0.5)

        results.append({
            "window":     f"W{i+1}",
            "flag":       "DISPATCH" if confirmed else "MONITOR",
            "score":      round(event.severity_raw, 4),
            "p1":         round(event.p1_score, 4),
            "p2":         round(event.p2_score, 4),
            "p4_drift":   round(event.p4_score, 4),
            "confidence": round(event.confidence, 3),
            "phase":      phase,
            "gt_anomaly": gt_win,
        })

        res_df = pd.DataFrame(results)
        y_p = (res_df['flag'] == 'DISPATCH').astype(int).values
        y_t = res_df['gt_anomaly'].values

        if confirmed:
            flag_ph.markdown('<div class="dispatch-box">🚨 DISPATCH — LEAK DETECTED</div>',
                             unsafe_allow_html=True)
        else:
            flag_ph.markdown('<div class="monitor-box">🟢 MONITOR</div>',
                             unsafe_allow_html=True)

        with chart_ph.container():
            st.subheader("📈 Live Signal")
            st.line_chart(res_df[['score','p1','p2','p4_drift','gt_anomaly']],
                          use_container_width=True)

        if y_t.sum() > 0 or y_p.sum() > 0:
            with metrics_ph.container():
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("F1",        f"{f1_score(y_t,y_p,zero_division=0):.3f}")
                mc2.metric("Precision", f"{precision_score(y_t,y_p,zero_division=0):.3f}")
                mc3.metric("Recall",    f"{recall_score(y_t,y_p,zero_division=0):.3f}")
                mc4.metric("FAR",       f"{compute_far(y_p,y_t):.3f}")

        with table_ph.container():
            st.dataframe(
                res_df[['window','flag','score','p1','p2','p4_drift','gt_anomaly']].tail(8),
                use_container_width=True)

        time.sleep(stream_speed)

# ══════════════════════════════════════════════════════════════════════════════
# ACOUSTIC PATH
# ══════════════════════════════════════════════════════════════════════════════
else:
    sig_a, fs_a = load_wav(wav_a)
    sig_b, _    = load_wav(wav_b)

    if fs_a < 8000:
        st.error(f"WAV fs={fs_a}Hz — acoustic path requires ≥8000Hz.")
        st.stop()

    brain = PGPLBrain(
        fs=float(fs_a), pipe_diameter_m=0.05, pipe_thickness_m=0.005,
        pipe_material="hdpe", saline=saline_mode,
        sensor_spacing_m=1.5, base_alpha=base_alpha,
    )
    for ph in PHASES:
        brain.tidal.add_phase(TidalWindow(
            phase=ph, psi_offset=0.0,
            alpha_adj=brain.tidal.adaptive_alpha(), timestamp=0.0))

    min_len   = min(len(sig_a), len(sig_b))
    n_windows = min_len // window_size
    status_ph.info(f"🎙 Streaming {n_windows} acoustic windows (fs={fs_a}Hz) …")

    for i in range(n_windows):
        s = i * window_size; e = s + window_size
        phase, tidal_psi = mock_tidal(i)
        event = brain.process_acoustic(sig_a[s:e], sig_b[s:e], float(i), phase, tidal_psi)

        gate      = event.meta.get("gate", {})
        confirmed = gate.get("confirmed", False)

        results.append({
            "window":     f"W{i+1}",
            "flag":       "DISPATCH" if confirmed else "MONITOR",
            "score":      round(event.severity_raw, 4),
            "p3_tdoa":    round(event.p3_score, 4),
            "p4_tidal":   round(event.p4_score, 4),
            "location_m": round(event.location_m, 3),
            "freq_hz":    event.meta.get("freq_centroid_hz", 0),
            "confidence": round(event.confidence, 3),
            "gt_anomaly": gt_label,
        })

        res_df = pd.DataFrame(results)
        y_p = (res_df['flag'] == 'DISPATCH').astype(int).values
        y_t = res_df['gt_anomaly'].values

        if confirmed:
            flag_ph.markdown('<div class="dispatch-box">🚨 DISPATCH — LEAK DETECTED</div>',
                             unsafe_allow_html=True)
        else:
            flag_ph.markdown('<div class="monitor-box">🟢 MONITOR</div>',
                             unsafe_allow_html=True)

        with chart_ph.container():
            st.subheader("📈 Live Signal")
            st.line_chart(res_df[['score','p3_tdoa','p4_tidal']], use_container_width=True)

        if len(res_df) > 2:
            with metrics_ph.container():
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("F1",        f"{f1_score(y_t,y_p,zero_division=0):.3f}")
                mc2.metric("Precision", f"{precision_score(y_t,y_p,zero_division=0):.3f}")
                mc3.metric("Recall",    f"{recall_score(y_t,y_p,zero_division=0):.3f}")
                mc4.metric("FAR",       f"{compute_far(y_p,y_t):.3f}")

        with table_ph.container():
            st.dataframe(
                res_df[['window','flag','score','p3_tdoa','location_m','freq_hz','gt_anomaly']].tail(8),
                use_container_width=True)

        time.sleep(stream_speed)

# ── Final summary ─────────────────────────────────────────────────────────────
status_ph.success("✅ Detection complete.")
res_df = pd.DataFrame(results)
y_p = (res_df['flag'] == 'DISPATCH').astype(int).values
y_t = res_df['gt_anomaly'].values

st.subheader("📊 Final Metrics")
fc1, fc2, fc3, fc4 = st.columns(4)
fc1.metric("F1",        f"{f1_score(y_t,y_p,zero_division=0):.3f}")
fc2.metric("Precision", f"{precision_score(y_t,y_p,zero_division=0):.3f}")
fc3.metric("Recall",    f"{recall_score(y_t,y_p,zero_division=0):.3f}")
fc4.metric("FAR",       f"{compute_far(y_p,y_t):.3f}")

st.subheader("📋 Full Results")
st.dataframe(res_df, use_container_width=True)
st.download_button("⬇️ Download results CSV",
                   res_df.to_csv(index=False).encode(),
                   "pgpl_results.csv", "text/csv")
