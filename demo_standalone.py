import streamlit as st
import pandas as pd
import numpy as np
from collections import deque
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

st.set_page_config(page_title="PGPL v2.0 Demo", layout="wide")

# ── [Brain Classes Unchanged – Same as Before] ──
class _AdaptiveZ:
    def __init__(self, window=100):
        self._buf = deque(maxlen=window)

    def score(self, value):
        if len(self._buf) < 5:
            self._buf.append(value)
            return 0.0
        mu = np.mean(self._buf)
        sigma = np.std(self._buf) + 1e-9
        self._buf.append(value)
        return abs(value - mu) / sigma

class _MondirianCP:
    def __init__(self):
        self._cal = []

    def calibrate(self, score):
        self._cal.append(score)

    def p_value(self, score):
        if len(self._cal) < 10:
            return 1.0
        return np.mean(np.array(self._cal) >= score)

class _PSIDrift:
    _BINS = np.linspace(0.0, 10.0, 11)

    def __init__(self, ref_window=50):
        self._ref = deque(maxlen=ref_window)
        self._cur = deque(maxlen=ref_window)
        self._ready = False

    def update(self, z):
        if not self._ready:
            self._ref.append(z)
            if len(self._ref) == self._ref.maxlen:
                self._ready = True
            return 0.0
        self._cur.append(z)
        if len(self._cur) < 10:
            return 0.0
        r, _ = np.histogram(np.array(self._ref), bins=self._BINS, density=True)
        c, _ = np.histogram(np.array(self._cur), bins=self._BINS, density=True)
        r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
        return np.sum((c - r) * np.log(c / r))

class _TidalPhaseTracker:
    def __init__(self, buf_n=6):
        self.phase_buf = deque(maxlen=buf_n)

    def update(self, phase):
        self.phase_buf.append(phase)

    def covered(self):
        return len(set(self.phase_buf))

class PULSE_AT_Brain:
    def __init__(self, alpha=0.10, persistence_n=6, psi_threshold=0.20, zone_weight=0.5, min_phases=3):
        self.alpha = alpha
        self.min_persist = persistence_n
        self.psi_threshold = psi_threshold
        self.zone_weight = zone_weight
        self.min_phases = min_phases
        self._z = _AdaptiveZ()
        self._cp = _MondirianCP()
        self._psi = _PSIDrift()
        self._flag_buf = deque(maxlen=persistence_n)
        self._phase_tracker = _TidalPhaseTracker(persistence_n)
        self._cal_n = 0
        self.persist_count = 0
        self.psi_window = deque(maxlen=50)

    @staticmethod
    def _bandpass(sig, fs=2000):
        sos = butter(4, [200 / (0.5 * fs), 800 / (0.5 * fs)], btype="band", output="sos")
        return sosfilt(sos, sig)

    def process(self, sensors, tide_phase='unknown'):
        fs = int(sensors.get("fs", 2000))
        salinity = float(sensors.get("salinity", 7.0))
        flow = float(sensors.get("flow", 13.0))
        pressure_z = abs(flow - 13.0) / 2.0

        if fs < 10:  # SCADA
            energy = abs(pressure_z)
        else:  # Acoustic
            sig1 = self._bandpass(np.array(sensors["mic1_sig"]), fs)
            energy = np.sqrt(np.mean(sig1 ** 2))
            velocity = 1480.0  # Biot stub

        z = self._z.score(energy)
        if self._cal_n < 100:
            self._cp.calibrate(z)
            self._cal_n += 1
        p_val = self._cp.p_value(z)
        psi = self._psi.update(z)
        alpha_adapt = self.alpha * (1 + min(psi / self.psi_threshold, 0.5))
        self._phase_tracker.update(tide_phase)
        flag = p_val <= alpha_adapt
        self._flag_buf.append(flag)
        persist_ratio = sum(self._flag_buf) / len(self._flag_buf)
        phases_ok = self._phase_tracker.covered() >= self.min_phases
        gated = (persist_ratio >= 0.67) and phases_ok

        self.persist_count = int(flag)
        self.psi_window.append(psi)

        return {
            "flag": "DISPATCH" if gated else "MONITOR",
            "score": round(z, 3),
            "p_val": round(p_val, 3),
            "psi": round(psi, 3),
            "energy": round(energy, 3),
            "alpha_adapt": round(alpha_adapt, 3),
            "persist_ratio": round(persist_ratio, 2),
            "gated": gated,
            "phases_covered": self._phase_tracker.covered(),
            "salinity": round(salinity, 2),
            "flow": round(flow, 2),
            "tide_phase": tide_phase
        }

# ── Proxy Data Loader (Unchanged) ──
@st.cache_data
def load_proxy_data():
    fs = 2000
    N = 5000
    t = np.arange(N) / fs
    sig = np.sin(2 * np.pi * 400 * t) * np.exp(-((t - 2.5)**2) / 0.2) * 2 + 0.05 * np.random.randn(N)
    salinity = np.full(N, 7.0 + 0.2 * np.sin(2 * np.pi * 0.05 * t))
    flow = np.full(N, 13.0 + 0.5 * np.sin(2 * np.pi * 0.03 * t))
    df = pd.DataFrame({
        'mic1_sig': sig,
        'salinity': salinity,
        'flow': flow,
        'fs': [fs] * N
    })
    return df

# ── Smart Multi-File Loader (NEW: Column Safety + Multi) ──
def load_file_to_df(file):
    filename = file.name
    if filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif filename.endswith('.wav'):
        fs_data, sig = wavfile.read(file)
        if sig.ndim > 1:
            sig = sig[:, 0]
        sig = sig.astype(np.float64) / (np.max(np.abs(sig)) + 1e-9)
        df = pd.DataFrame({
            'mic1_sig': sig,
            'salinity': 7.0,
            'flow': 13.0,
            'fs': [fs_data] * len(sig)
        })
        return df  # WAV always full cols
    else:
        raise ValueError("Unsupported file")

    # Auto-Stub Missing Columns (Fix KeyError!)
    if 'mic1_sig' not in df.columns:
        df['mic1_sig'] = 0.0  # SCADA stub
    if 'fs' not in df.columns:
        # Auto-detect: Acoustic if mic1_sig numeric/non-zero var
        if df['mic1_sig'].std() > 0.1:
            df['fs'] = 2000
        else:
            df['fs'] = 1  # SCADA
    if 'salinity' not in df.columns:
        df['salinity'] = 7.0
    if 'flow' not in df.columns:
        df['flow'] = 13.0
    return df

# ── Main App ──
st.title("🧠 PGPL v2.0 Demo – Multi-File Pulse Gating")
st.markdown("**Multi-upload CSVs/WAVs** or proxy. *Auto-stubs SCADA cols* (no mic1_sig? → fs=1, energy=pressure_z).")

# Multi-File Uploader
uploaded_files = st.file_uploader(
    "📁 Upload Multiple CSVs (SCADA Flows/Salinity) or WAVs", 
    type=['csv', 'wav'], 
    accept_multiple_files=True
)

use_proxy = st.checkbox("🔧 Use Proxy Data (BattLeDIM – overrides files)")

data_mode = st.selectbox("⚙️ Data Mode", ["Auto", "SCADA (fs=1)", "Acoustic (fs=2000)"])

show_preview = st.checkbox("👀 Show Data Preview")

if st.button("🗑️ Clear Data"):
    st.rerun()

# Load Data
df = None
if use_proxy:
    df = load_proxy_data()
    st.success("✅ Proxy (BattLeDIM) loaded")
elif uploaded_files:
    dfs = []
    for f in uploaded_files:
        try:
            this_df = load_file_to_df(f)
            dfs.append(this_df)
            st.success(f"✅ {f.name}: {len(this_df)} rows")
        except Exception as e:
            st.error(f"❌ {f.name}: {e}")
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        st.success(f"✅ Combined: {len(df)} total rows | Columns: {list(df.columns)}")

if df is not None:
    if show_preview:
        st.subheader("📋 Data Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(5), use_container_width=True)
        with col2:
            st.write("**Columns:**", list(df.columns))
            st.write("**Shape:**", df.shape)
            st.write("**FS Sample:**", df['fs'].iloc[0])

    # Override Mode
    if data_mode == "SCADA (fs=1)":
        df['fs'] = 1
        df['mic1_sig'] = 0.0
    elif data_mode == "Acoustic (fs=2000)":
        df['fs'] = 2000

    # Sliders
    col1, col2, col3 = st.columns(3)
    with col1:
        alpha = st.slider("α (Sensitivity)", 0.05, 0.20, 0.10, 0.01)
    with col2:
        psi_th = st.slider("Ψ Threshold", 0.10, 0.50, 0.20, 0.01)
    with col3:
        window_size = st.slider("Window Size", 500, 2000, 1000, 100)
    tide_phase = st.selectbox("🌊 Tide Phase", ["unknown", "high", "low", "ebb", "flood"])

    if st.button("🚀 Process & Gate", type="primary"):
        brain = PULSE_AT_Brain(alpha=alpha, psi_threshold=psi_th)
        results = []
        fs_global = int(df['fs'].iloc[0])  # Assume uniform
        for start in range(0, len(df), window_size):
            end = min(start + window_size, len(df))
            chunk_sig = df['mic1_sig'].iloc[start:end].values  # Safe now!
            sensors = {
                "fs": fs_global,
                "salinity": float(df['salinity'].iloc[start:end].mean()),
                "flow": float(df['flow'].iloc[start:end].mean()),
                "mic1_sig": chunk_sig
            }
            res = brain.process(sensors, tide_phase=tide_phase)
            res['window'] = f"{start}-{end}"
            results.append(res)

        res_df = pd.DataFrame(results)
        res_df.index = range(1, len(res_df) + 1)

        # Results (Unchanged)
        st.subheader("📊 Gating Results")
        st.dataframe(res_df, use_container_width=True)

        st.subheader("🔍 Last Window JSON")
        st.json(res_df.iloc[-1].to_dict())

        # Metrics
        gated_count = res_df['gated'].sum()
        dispatch_pct = (gated_count / len(res_df)) * 100
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("🚨 DISPATCH (Gated)", gated_count, f"{dispatch_pct:.1f}%")
        col_b.metric("📈 Max Score (Z)", res_df['score'].max())
        col_c.metric("🔄 Max Ψ (Drift)", res_df['psi'].max())

        # Charts
        st.subheader("📈 Live Metrics")
        chart_data = res_df[['score', 'p_val', 'psi']]
        st.line_chart(chart_data, use_container_width=True)

        st.markdown("---")
        st.caption("*Your SCADA CSV → Uses flow pressure_z. Multi-files concat. Toggles override.*")

else:
    st.info("👆 Upload files or check proxy to start!")