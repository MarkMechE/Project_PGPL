"""
pgpl_demo.py — PGPL v2.0 Live Demo Dashboard
Uses real PGPLBrain from src/pgpl_brain.py
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

# ── Real Brain ────────────────────────────────────────────────────────────────
from src.pgpl_brain import PGPLBrain, TidalWindow

st.set_page_config(page_title="PGPL v2.0 Demo", layout="wide", page_icon="🧠")

# ── Styling ───────────────────────────────────────────────────────────────────
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
    animation: pulse 1s infinite;
}
.monitor-box {
    background: #1a1a2e; border: 2px solid #4a4a6a;
    padding: 1rem 1.5rem; border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    color: #aaaacc; font-size: 1.4rem; text-align: center;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
.stMetric { background: #0d1117; border-radius: 8px; padding: 0.5rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Tidal phase helper ────────────────────────────────────────────────────────
PHASES = ["ebb", "flood", "slack_low", "slack_high", "spring"]
def mock_tidal(i): return PHASES[i % len(PHASES)], float(np.sin(i * 0.01) * 2.5)

# ── FAR helper ────────────────────────────────────────────────────────────────
def compute_far(y_pred, y_true):
    denom = int(np.sum(y_true == 0))
    return float(np.sum((y_pred == 1) & (y_true == 0)) / max(denom, 1))

# ── File loaders ──────────────────────────────────────────────────────────────
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Normalise common column name variants
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ('pressure','psi','p','press'): rename[c] = 'pressure'
        elif cl in ('flow','lps','q','flowrate'): rename[c] = 'flow'
        elif cl in ('anomaly','label','leak','gt'): rename[c] = 'anomaly'
    df.rename(columns=rename, inplace=True)
    if 'pressure' not in df.columns: df['pressure'] = 0.0
    if 'flow' not in df.columns:     df['flow'] = 0.0
    if 'anomaly' not in df.columns:  df['anomaly'] = 0
    return df

def load_wav(file):
    fs_data, sig = wavfile.read(file)
    if sig.ndim > 1: sig = sig[:, 0]
    sig = sig.astype(np.float64) / (np.max(np.abs(sig)) + 1e-9)
    return sig, fs_data

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 PGPL v2.0 — Leak Detection Demo")
st.caption("Real PGPLBrain · Adaptive-Z · MondrianCP · Tidal Gating · P1–P4 Fusion")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Sensor Mode", ["SCADA (CSV)", "Acoustic (WAV pair)"])
    base_alpha  = st.slider("Base α", 0.01, 0.20, 0.05, 0.01)
    window_size = st.slider("Window size (rows/samples)", 100, 2000, 500, 100)
    stream_speed = st.slider("Stream delay (s/window)", 0.0, 1.0, 0.2, 0.05)
    saline_mode = st.checkbox("Saline / Tidal gating (EDC Busan)", value=False)
    st.markdown("---")
    st.caption("BattleDIM: saline=OFF  |  EDC Busan: saline=ON")

# ── File upload ───────────────────────────────────────────────────────────────
if mode == "SCADA (CSV)":
    uploaded = st.file_uploader("Upload SCADA CSV (pressure, flow, anomaly cols)", type=["csv"])
else:
    col_a, col_b = st.columns(2)
    with col_a: wav_a = st.file_uploader("Sensor A (.wav)", type=["wav"])
    with col_b: wav_b = st.file_uploader("Sensor B (.wav)", type=["wav"])
    gt_label = st.selectbox("Ground truth label for this file pair", [0, 1],
                             format_func=lambda x: "1 = Leak" if x else "0 = No Leak")

ready = (mode == "SCADA (CSV)" and uploaded is not None) or \
        (mode == "Acoustic (WAV pair)" and wav_a and wav_b)

if not ready:
    st.info("👆 Upload file(s) above to begin.")
    st.stop()

# ── Run ───────────────────────────────────────────────────────────────────────
if st.button("🚀 Run Detection", type="primary"):

    # ── Placeholders ──────────────────────────────────────────────────────────
    status_box   = st.empty()
    flag_box     = st.empty()
    chart_ph     = st.empty()
    metrics_ph   = st.empty()
    table_ph     = st.empty()

    results = []

    # ══════════════════════════════════════════════════════════════════════════
    # SCADA PATH
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "SCADA (CSV)":
        df = load_csv(uploaded)
        brain = PGPLBrain(fs=1/60, saline=saline_mode, base_alpha=base_alpha)

        # Warm up brain on first 20% as calibration
        cal_end = max(len(df) // 5, 1)
        cal_pressures = df['pressure'].iloc[:cal_end].dropna().tolist()
        if cal_pressures:
            mu  = float(np.mean(cal_pressures))
            sig = float(np.std(cal_pressures)) + 1e-9
            cal_z = [abs(v - mu) / sig for v in cal_pressures]
            brain.calibrate_from_year(cal_z, phase="default")

        n_windows = (len(df) - cal_end) // window_size
        detect_df = df.iloc[cal_end:].reset_index(drop=True)

        status_box.info(f"📡 Streaming {n_windows} windows (SCADA) …")

        for i, start in enumerate(range(0, len(detect_df) - window_size + 1, window_size)):
            end   = start + window_size
            chunk = detect_df.iloc[start:end]

            phase, tidal_psi = mock_tidal(i)
            # Feed each row through brain, keep last event per window
            event = None
            for row_i, row in chunk.iterrows():
                event = brain.process_scada(
                    pressure_psi = float(row['pressure']),
                    flow_lps     = float(row['flow']),
                    timestamp    = float(i * window_size + row_i),
                    tidal_phase  = phase,
                    tidal_psi    = tidal_psi,
                )

            gate       = event.meta.get("gate", {})
            confirmed  = gate.get("confirmed", False)
            gt_win     = int(chunk['anomaly'].mean() > 0.5)

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

            # Live flag
            if confirmed:
                flag_box.markdown('<div class="dispatch-box">🚨 DISPATCH — LEAK DETECTED</div>',
                                  unsafe_allow_html=True)
            else:
                flag_box.markdown('<div class="monitor-box">🟢 MONITOR</div>',
                                  unsafe_allow_html=True)

            # Live chart
            with chart_ph.container():
                st.subheader("📈 Live Signal")
                chart_cols = [c for c in ['score','p1','p2','p4_drift','gt_anomaly']
                              if c in res_df.columns]
                st.line_chart(res_df[chart_cols], use_container_width=True)

            # Running metrics
            if len(res_df) > 2:
                y_p = (res_df['flag'] == 'DISPATCH').astype(int).values
                y_t = res_df['gt_anomaly'].values
                with metrics_ph.container():
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("F1",        f"{f1_score(y_t,y_p,zero_division=0):.3f}")
                    mc2.metric("Precision", f"{precision_score(y_t,y_p,zero_division=0):.3f}")
                    mc3.metric("Recall",    f"{recall_score(y_t,y_p,zero_division=0):.3f}")
                    mc4.metric("FAR",       f"{compute_far(y_p,y_t):.3f}")

            with table_ph.container():
                st.dataframe(res_df[['window','flag','score','p1','p2','confidence','gt_anomaly']].tail(8),
                             use_container_width=True)

            time.sleep(stream_speed)

    # ══════════════════════════════════════════════════════════════════════════
    # ACOUSTIC PATH
    # ══════════════════════════════════════════════════════════════════════════
    else:
        sig_a, fs_a = load_wav(wav_a)
        sig_b, fs_b = load_wav(wav_b)
        fs = fs_a  # use sensor A fs

        if fs < 8000:
            st.error(f"WAV fs={fs}Hz is below 8000Hz minimum for acoustic path.")
            st.stop()

        brain = PGPLBrain(
            fs               = float(fs),
            pipe_diameter_m  = 0.05,
            pipe_thickness_m = 0.005,
            pipe_material    = "hdpe",
            saline           = saline_mode,
            sensor_spacing_m = 1.5,
            base_alpha       = base_alpha,
        )

        # Pre-load 5 tidal phases so gate isn't blocked from window 1
        for ph in PHASES:
            brain.tidal.add_phase(TidalWindow(
                phase=ph, psi_offset=0.0,
                alpha_adj=brain.tidal.adaptive_alpha(),
                timestamp=0.0,
            ))

        min_len    = min(len(sig_a), len(sig_b))
        n_windows  = min_len // window_size
        status_box.info(f"🎙 Streaming {n_windows} acoustic windows (fs={fs}Hz) …")

        for i in range(n_windows):
            s = i * window_size
            e = s + window_size
            chunk_a = sig_a[s:e]
            chunk_b = sig_b[s:e]

            phase, tidal_psi = mock_tidal(i)
            event = brain.process_acoustic(chunk_a, chunk_b, float(i), phase, tidal_psi)

            gate      = event.meta.get("gate", {})
            confirmed = gate.get("confirmed", False)

            results.append({
                "window":      f"W{i+1}",
                "flag":        "DISPATCH" if confirmed else "MONITOR",
                "score":       round(event.severity_raw, 4),
                "p3_tdoa":     round(event.p3_score, 4),
                "p4_tidal":    round(event.p4_score, 4),
                "location_m":  round(event.location_m, 3),
                "freq_hz":     event.meta.get("freq_centroid_hz", 0),
                "confidence":  round(event.confidence, 3),
                "gt_anomaly":  gt_label,
            })

            res_df = pd.DataFrame(results)

            if confirmed:
                flag_box.markdown('<div class="dispatch-box">🚨 DISPATCH — LEAK DETECTED</div>',
                                  unsafe_allow_html=True)
            else:
                flag_box.markdown('<div class="monitor-box">🟢 MONITOR</div>',
                                  unsafe_allow_html=True)

            with chart_ph.container():
                st.subheader("📈 Live Signal")
                st.line_chart(res_df[['score','p3_tdoa','p4_tidal']], use_container_width=True)

            # Aggregate metrics (all windows so far, same GT label)
            if len(res_df) > 2:
                y_p = (res_df['flag'] == 'DISPATCH').astype(int).values
                y_t = res_df['gt_anomaly'].values
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

    # ── Final summary ─────────────────────────────────────────────────────────
    status_box.success("✅ Detection complete.")
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

    csv = res_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download results CSV", csv, "pgpl_results.csv", "text/csv")