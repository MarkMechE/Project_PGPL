import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from pgpl_brain import PULSE_AT_Brain

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PGPL: Pulse-AT Gated Piezo System", 
    page_icon="🌊", 
    layout="wide"
)

# --- 2. CACHING & SESSION STATE ---
# Cache the brain to save memory and prevent "Re-initializing" messages
@st.cache_resource
def load_brain():
    return PULSE_AT_Brain()

brain = load_brain()

# Memory for the dashboard to keep results visible
if 'run_active' not in st.session_state:
    st.session_state.run_active = False
    st.session_state.result = None
    st.session_state.chart_data = None

# --- 3. UI SIDEBAR (USER INPUTS) ---
st.sidebar.header("🕹️ Simulation Controls")
st.sidebar.markdown("Adjust parameters to simulate real-world leak conditions.")

true_leak_dist = st.sidebar.slider("True Leak Distance (m)", 1.0, 50.0, 25.0)
current_flow = st.sidebar.slider("Current Flow Rate (L/s)", 10.0, 20.0, 16.5)
ambient_noise = st.sidebar.slider("Ambient Noise Level", 0.0, 1.0, 0.2)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Start Detection Pulse", use_container_width=True)

# --- 4. MAIN APP HEADER ---
st.title("🌊 PGPL: Pulse-AT Gated Piezo Leak Localization")
st.write("**Architecture:** Pulse-AT Filtering + Gated Z-Score Persistence + City2Graph GIS Overlay")
st.markdown("---")

# --- 5. LOGIC PROCESSING ---
if run_btn:
    with st.spinner("Applying PULSE-AT Butterworth Filters..."):
        # Simulate RAW Signals
        fs = 10000
        t = np.linspace(0, 1, fs)
        # Sensor A (Reference)
        mic1 = np.sin(2*np.pi*100*t) + ambient_noise * np.random.randn(fs)
        # Sensor B (Delayed by distance)
        # Note: 1400m/s is speed of sound in water
        delay_s = true_leak_dist / 1400 
        mic2 = np.sin(2*np.pi*100*(t - delay_s)) + ambient_noise * np.random.randn(fs)
        
        sensor_packet = {
            'mic1_sig': mic1, 
            'mic2_sig': mic2, 
            'fs': fs, 
            'flow': current_flow
        }
        
        # Run 3 iterations to satisfy the Gated Persistence (N>=3)
        for _ in range(3):
            final_res = brain.process(sensor_packet)
        
        # Save to Session State
        st.session_state.result = final_res
        st.session_state.chart_data = pd.DataFrame({
            "Sensor A (Reference)": mic1[:400],
            "Sensor B (Delayed)": mic2[:400]
        })
        st.session_state.run_active = True

# --- 6. DATA VISUALIZATION ---
if st.session_state.run_active:
    # Row 1: Metrics
    m_col1, m_col2, m_col3 = st.columns(3)
    res = st.session_state.result
    
    m_col1.metric("Detection Status", res['flag'], delta="ACTIVE" if res['flag']=="DISPATCH" else None)
    m_col2.metric("Calculated Distance", f"{res.get('loc_m', 0)} m", delta_color="inverse")
    m_col3.metric("Gating Score (Z)", res.get('z_score', 0))

    # Row 2: Signal Chart
    st.subheader("📡 PULSE-AT Acoustic Signal Analysis")
    st.line_chart(st.session_state.chart_data)
    st.caption("Signals filtered via Butterworth Bandpass (100Hz-1000Hz) to remove urban noise.")

    # Row 3: GIS Map
    st.subheader("📍 Geographic Localization (Incheon Pipe Network)")
    
    # Map Setup
    incheon_center = [37.4563, 126.7052]
    m = folium.Map(location=incheon_center, zoom_start=15, control_scale=True)
    
    # Add Sensors (Blue)
    folium.Marker([37.4590, 126.7050], popup="Sensor A (Hydrant)", icon=folium.Icon(color='blue', icon='microchip', prefix='fa')).add_to(m)
    folium.Marker([37.4540, 126.7050], popup="Sensor B (Water Meter)", icon=folium.Icon(color='blue', icon='microchip', prefix='fa')).add_to(m)
    
    # Add Leak (Red) if Dispatched
    if res['flag'] == 'DISPATCH':
        folium.Marker(
            incheon_center, 
            popup=f"ALARM: Leak at {res['loc_m']}m", 
            icon=folium.Icon(color='red', icon='bolt', prefix='fa')
        ).add_to(m)
        # Draw a line between sensors representing the pipe
        folium.PolyLine([[37.4590, 126.7050], [37.4540, 126.7050]], color="black", weight=2.5, opacity=0.8).add_to(m)

    st_folium(m, width="100%", height=450, key="pgpl_map")

else:
    st.info("👈 Use the sidebar to set simulation values and click 'Start Detection Pulse' to begin analysis.")
    st.image("https://img.freepik.com/free-vector/network-mesh-wire-digital-technology-background_1017-27428.jpg", caption="System Standby: Awaiting Sensor Input", use_container_width=True)

# --- 7. PROJECT FOOTER ---
st.markdown("---")
st.markdown("© 2026 PGPL Project Team | **Thesis Exhibit A** | Incheon/Busan Pipe Proxy System")