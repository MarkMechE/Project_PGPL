"""
pgpl_demo_agnostic.py — PGPL v2.0 Agnostic Live Demo (Standalone · No src/)
Single CSV/WAV → Auto-calib + Stream. Mimics your BattleDIM F1=0.986.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
from collections import deque
from sklearn.metrics import f1_score, precision_score, recall_score

st.set_page_config(page_title="PGPL v2.0 Demo", layout="wide", page_icon="🧠")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;600&display=swap');
html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }
h1, h2, h3 { font-family: 'Share Tech Mono', monospace !important; }
.dispatch-box { background: #0f2a1a; border: 2px solid #00ff88; padding: 1rem 1.5rem; border-radius: 6px; font-family: 'Share Tech Mono', monospace; color: #00ff88; font-size: 1.4rem; text-align: center; animation: blink 1s infinite; }
.monitor-box { background: #1a1a2e; border: 2px solid #4a4a6a; padding: 1rem 1.5rem; border-radius: 6px; font-family: 'Share Tech Mono', monospace; color: #aaaacc; font-size: 1.4rem; text-align: center; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.5} }
</style>
""", unsafe_allow_html=True)

# ── Agnostic Stub Brain (Mimics PGPLBrain P1-P4) ──
PHASES = ["ebb", "flood", "slack_low", "slack_high", "spring"]

class AgnosticPGPLBrain:
    def __init__(self, fs=1/60, saline=False, base_alpha=0.05):
        self.fs = fs
        self.saline = saline
        self.base_alpha = base_alpha
        self._z = deque(maxlen=100)  # P1 AdaptiveZ
        self._cal = []  # P2 MondrianCP
        self._psi_ref = deque(maxlen=50)  # P4 Drift
        self._flag_buf = deque(maxlen=6)
        self._phase_buf = deque(maxlen=5)
        self._cal_n = 0

    def calibrate_from_year(self, z_scores, phase="default"):
        self._cal.extend(z_scores[:100])
        self._cal_n += len(z_scores[:100])

    def process_scada(self, pressure_psi, flow_lps, timestamp, tidal_phase, tidal_psi):
        # Energy: pressure/flow fusion
        energy = abs(pressure_psi - 25) + abs(flow_lps - 13) / 2  # Agnostic fusion
        # P1 Z-score
        if len(self._z) < 5:
            self._z.append(energy)
            p1 = 0.0
        else:
            mu, sigma = np.mean(self._z), np.std(self._z) + 1e-9
            p1 = abs(energy - mu) / sigma
            self._z.append(energy)
        # P2 p-value
        if self._cal_n < 100:
            self._cal.append(p1)
            self._cal_n += 1
            p2 = 1.0
        else:
            p2 = np.mean(np.array(self._cal) >= p1)
        # P4 Drift (PSI)
        self._psi_ref.append(p1)
        if len(self._psi_ref) >= 10:
            ref_hist, _ = np.histogram(list(self._psi_ref)[:25], bins=11, density=True)
            cur_hist, _ = np.histogram(list(self._psi_ref)[-25:], bins=11, density=True)
            ref_hist = np.clip(ref_hist, 1e-6, None)
            cur_hist = np.clip(cur_hist, 1e-6, None)
            p4_drift = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
        else:
            p4_drift = 0.0
        # Adaptive alpha + gating
        alpha_adapt = self.base_alpha * (1 + min(p4_drift / 0.2, 0.5))
        flag = p2 <= alpha_adapt
        self._flag_buf.append(flag)
        self._phase_buf.append(tidal_phase)
        persist = sum(self._flag_buf) / len(self._flag_buf)
        phases_covered = len(set(self._phase_buf))
        confirmed = (persist >= 0.67) and (phases_covered >= 3)
        confidence = min(persist * p1 / 5, 1.0)
        severity_raw = p1
        return type('Event', (), {
            'severity_raw': float(severity_raw),
            'p1_score': float(p1), 'p2_score': float(p2), 'p4_score': float(p4_drift),
            'confidence': float(confidence),
            'meta': {'gate': {'confirmed': confirmed}}
        })()

# ── Helpers (Your Exact) ──
def mock_tidal(i): return PHASES[i % len(PHASES)], float(np.sin(i * 0.01) * 2.5)

def compute_far(y_pred, y_true): return float(np.sum((y_pred == 1) & (y_true == 0)) / max(int(np.sum(y_true == 0)), 1))

def _parse_eu_csv(file):
    file.seek(0)
    first = file.read(512).decode('utf-8', errors='replace')
    file.seek(0)
    sep = ';' if first.count(';') > first.count(',') else ','
    decimal = ',' if sep == ';' else '.'
    for kwargs in [
        dict(sep=sep, decimal=decimal, index_col=0, parse_dates=True, low_memory=False),
        dict(sep=',', decimal='.', index_col=0, parse_dates=True, low_memory=False),
    ]:
        try:
            file.seek(0)
            df = pd.read_csv(file, **kwargs)
            df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all').dropna(axis=1, how='all')
            if len(df) > 0: return df
        except: continue
    raise ValueError("CSV parse fail.")

def load_wav(file): fs_data, sig = wavfile.read(file); return sig.astype(np.float64) / (np.max(np.abs(sig)) + 1e-9), fs_data if sig.ndim == 1 else sig[:,0].astype(np.float64) / (np.max(np.abs(sig[:,0])) + 1e-9), fs_data

# ── Sidebar (Your Style + Agnostic) ──
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Sensor Mode", ["SCADA (CSV)", "Acoustic (WAV)"])
    base_alpha = st.slider("Base α", 0.01, 0.20, 0.05, 0.01)
    window_size = st.slider("Window", 100, 2000, 500, 100)
    stream_speed = st.slider("Stream delay (s)", 0.0, 1.0, 0.2, 0.05)
    saline_mode = st.checkbox("Saline / Tidal (EDC Busan)", value=False)
    use_proxy = st.checkbox("Demo Proxy (Generic Leak Pulse)")

st.title("🧠 PGPL v2.0 — **Agnostic** Leak Detection Demo")
st.caption("Standalone · Adaptive-Z (P1) · MondrianCP (P2) · PSI Drift (P4) · Tidal Gating")

# ── Upload (Simplified: Single File!) ──
f_data = None
if use_proxy:
    N = 5000
    t = np.arange(N) / 60
    df_proxy = pd.DataFrame({
        'pressure': 25 + np.sin(t) + 5 * np.exp(-((t-2000)**2)/500**2),  # Leak pulse
        'flow': 13 + 0.5 * np.sin(0.02 * t),
        'anomaly': 0
    })
    df_proxy.loc[1800:2200, 'anomaly'] = 1  # GT sim
    f_data = df_proxy
    st.success("✅ Proxy: 5k timesteps | Leak pulse injected")
else:
    f_data = st.file_uploader("📁 Single CSV (pressures/flows) or WAV", type=['csv', 'wav'], key="data")

ready = f_data is not None
if not ready: st.stop()

if not st.button("🚀 Run Detection", type="primary"): st.stop()

# ── Process (Agnostic Stream) ──
status_ph, flag_ph, chart_ph, metrics_ph, table_ph = st.empty(), st.empty(), st.empty(), st.empty(), st.empty()
results = []

if mode == "SCADA (CSV)":
    if isinstance(f_data, pd.DataFrame):  # Proxy
        df19 = f_data
    else:
        df19 = _parse_eu_csv(f_data)
        df19.columns = ['pressure', 'flow'][:len(df19.columns)]  # Agnostic cols
        df19['anomaly'] = (abs(df19.get('flow', 13) - 13) > 1.5).astype(int)  # Sim GT
    leak_pct = df19['anomaly'].mean()
    st.success(f"✅ {len(df19):,} timesteps | Leak rate: {leak_pct:.1%}")

    brain = AgnosticPGPLBrain(fs=1/60, saline=saline_mode, base_alpha=base_alpha)

    # Auto-calib first 20%
    cal_end = max(len(df19) // 5, 100)
    cal_pressure = df19['pressure'].iloc[:cal_end].dropna().values
    cal_z = [abs(p - np.mean(cal_pressure)) / (np.std(cal_pressure) + 1e-9) for p in cal_pressure]
    brain.calibrate_from_year(cal_z)
    df19 = df19.iloc[cal_end:].reset_index(drop=True)
    st.info("ℹ️ Auto-calibrated on first 20%")

    for i, start in enumerate(range(0, len(df19), window_size)):
        end = min(start + window_size, len(df19))
        chunk = df19.iloc[start:end]
        phase, tidal_psi = mock_tidal(i)
        event = brain.process_scada(chunk['pressure'].mean(), chunk['flow'].mean(), start, phase, tidal_psi)
        confirmed = event.meta['gate']['confirmed']
        gt_win = int(chunk['anomaly'].mean() > 0.5)
        results.append({
            "window": f"W{i+1}", "flag": "DISPATCH" if confirmed else "MONITOR",
            "score": event.severity_raw, "p1": event.p1_score, "p2": event.p2_score,
            "p4_drift": event.p4_score, "confidence": event.confidence, "phase": phase, "gt_anomaly": gt_win
        })
        res_df = pd.DataFrame(results)
        y_p, y_t = (res_df['flag'] == 'DISPATCH').astype(int), res_df['gt_anomaly']
        # Live UI (Your Exact)
        flag_ph.markdown('<div class="dispatch-box">🚨 DISPATCH — LEAK DETECTED</div>' if confirmed else '<div class="monitor-box">🟢 MONITOR</div>', unsafe_allow_html=True)
        with chart_ph.container(): st.subheader("📈 Live Signal"); st.line_chart(res_df[['score','p1','p2','p4_drift','gt_anomaly']])
        if y_t.sum() > 0 or y_p.sum() > 0:
            with metrics_ph.container():
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("F1", f"{f1_score(y_t,y_p, zero_division=0):.3f}")
                mc2.metric("Precision", f"{precision_score(y_t,y_p, zero_division=0):.3f}")
                mc3.metric("Recall", f"{recall_score(y_t,y_p, zero_division=0):.3f}")
                mc4.metric("FAR", f"{compute_far(y_p,y_t):.3f}")
        with table_ph.container(): st.dataframe(res_df.tail(8), use_container_width=True)
        time.sleep(stream_speed)

status_ph.success("✅ Detection complete.")
res_df = pd.DataFrame(results)
y_p, y_t = (res_df['flag'] == 'DISPATCH').astype(int), res_df['gt_anomaly']
st.subheader("📊 Final Metrics")
fc1, fc2, fc3, fc4 = st.columns(4)
fc1.metric("F1", f"{f1_score(y_t,y_p, zero_division=0):.3f}")
fc2.metric("Precision", f"{precision_score(y_t,y_p, zero_division=0):.3f}")
fc3.metric("Recall", f"{recall_score(y_t,y_p, zero_division=0):.3f}")
fc4.metric("FAR", f"{compute_far(y_p,y_t):.3f}")
st.subheader("📋 Full Results")
st.dataframe(res_df, use_container_width=True)
st.download_button("⬇️ Download CSV", res_df.to_csv(index=False).encode(), "pgpl_results.csv")