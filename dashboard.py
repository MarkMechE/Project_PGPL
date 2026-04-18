# dashboard.py
# Run: streamlit run dashboard.py

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from pgpl_brain import PULSE_AT_Brain

# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="PGPL — Eco-Delta Leak Localization",
    layout="wide",
    page_icon="🛠️"
)

ECO_DELTA_CENTER = [35.135, 128.970]
EDGES_PATH  = "data/eco_delta_edges.geojson"
RESULTS_CSV = "outputs/simulation_results.csv"

# ------------------------------------------------------------------ #
#  Load brain (cached)                                                 #
# ------------------------------------------------------------------ #
@st.cache_resource
def load_brain():
    return PULSE_AT_Brain()

brain = load_brain()

# ------------------------------------------------------------------ #
#  Sidebar controls                                                    #
# ------------------------------------------------------------------ #
st.sidebar.title("PGPL Controls")
st.sidebar.markdown("**Eco-Delta City, Busan**")

true_dist  = st.sidebar.slider("True leak distance (m)", 5, 120, 45)
flow_val   = st.sidebar.slider("Flow anomaly (L/min)",  13.0, 30.0, 20.0, step=0.5)
pres_val   = st.sidebar.slider("Pressure (bar)",         2.5,  3.5,  2.9, step=0.05)
var_z_val  = st.sidebar.slider("Turbulence var_z",        0.0,  5.0,  3.2, step=0.1)
noise_val  = st.sidebar.slider("Signal noise σ",          0.01, 0.15, 0.05, step=0.01)
num_cycles = st.sidebar.slider("Detection cycles",        1, 8, 5)

run_btn = st.sidebar.button("▶  Run Detection", type="primary")

# ------------------------------------------------------------------ #
#  Tabs                                                                #
# ------------------------------------------------------------------ #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️  Live Map",
    "📡  Signal & Gate",
    "📊  Metrics",
    "📋  Simulation Log",
    "ℹ️  About"
])

# ------------------------------------------------------------------ #
#  Session state                                                       #
# ------------------------------------------------------------------ #
if "results_history" not in st.session_state:
    st.session_state.results_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "cycles_log" not in st.session_state:
    st.session_state.cycles_log = []

# ------------------------------------------------------------------ #
#  Run detection on button press                                       #
# ------------------------------------------------------------------ #
if run_btn:
    mic1, mic2 = brain._sim_piezo(true_dist, noise_sigma=noise_val)
    sensors = {
        "flow":      flow_val,
        "pressure":  pres_val,
        "var_z":     var_z_val,
        "mic1_sig":  mic1,
        "mic2_sig":  mic2,
        "fs":        10000,
        "true_dist": true_dist,
    }

    brain.persist_count = 0
    brain.psi_window.clear()

    cycles_log = []
    final_result = None

    for c in range(num_cycles):
        r = brain.process(sensors, sensor_node_idx=0)
        cycles_log.append({
            "cycle":   c + 1,
            "flag":    r["flag"],
            "score":   r["score"],
            "psi":     r["psi"],
            "persist": r["persist"],
        })
        if r["flag"] == "DISPATCH":
            final_result = r
            break

    if final_result is None:
        final_result = r  # last monitor result

    st.session_state.last_result   = final_result
    st.session_state.cycles_log    = cycles_log
    st.session_state.results_history.append({
        "true_dist_m": true_dist,
        "flag":        final_result["flag"],
        "loc_m":       final_result.get("loc_m"),
        "error_m":     final_result.get("error_m"),
        "score":       final_result["score"],
        "psi":         final_result["psi"],
    })

# ------------------------------------------------------------------ #
#  Tab 1 — Live Map                                                    #
# ------------------------------------------------------------------ #
with tab1:
    st.subheader("Eco-Delta City pipe network — leak pin")

    r = st.session_state.last_result
    dispatched = r is not None and r["flag"] == "DISPATCH"

    m = folium.Map(location=ECO_DELTA_CENTER, zoom_start=14,
                   tiles="OpenStreetMap")

    # Pipe network overlay
    if os.path.exists(EDGES_PATH):
        folium.GeoJson(
            EDGES_PATH,
            name="Pipe network",
            style_function=lambda _: {
                "color": "#3B82F6", "weight": 1.5, "opacity": 0.6
            }
        ).add_to(m)
    else:
        folium.Marker(
            ECO_DELTA_CENTER,
            popup="Eco-Delta City centre",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    # Sensor marker
    folium.CircleMarker(
        ECO_DELTA_CENTER, radius=8,
        color="#10B981", fill=True, fill_opacity=0.9,
        popup="Sensor node 0"
    ).add_to(m)

    # Leak pin (only on DISPATCH)
    if dispatched:
        gps = r["gps"]
        folium.Marker(
            location=gps,
            popup=(f"Leak detected\n"
                   f"Distance: {r['loc_m']} m\n"
                   f"Error: {r['error_m']} m"),
            icon=folium.Icon(color="red", icon="tint", prefix="fa")
        ).add_to(m)
        # Line from sensor to leak
        folium.PolyLine(
            [ECO_DELTA_CENTER, list(gps)],
            color="#EF4444", weight=2, dash_array="6"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=900, height=520)

    # Status badge
    if dispatched:
        st.error(f"🔴 DISPATCH — Leak at **{r['loc_m']} m** "
                 f"| GPS: {r['gps']} "
                 f"| Error: {r['error_m']} m")
    elif r:
        st.success(f"🟢 MONITOR — score={r['score']} "
                   f"| persist={r['persist']} / {brain.min_persist} "
                   f"| PSI={r['psi']}")
    else:
        st.info("Press **Run Detection** to start.")

# ------------------------------------------------------------------ #
#  Tab 2 — Signal & Gate                                               #
# ------------------------------------------------------------------ #
with tab2:
    st.subheader("Acoustic signals and PULSE-AT gate state")

    if st.session_state.last_result:
        mic1, mic2 = brain._sim_piezo(true_dist, noise_sigma=noise_val)
        t = np.linspace(0, 1, len(mic1))

        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(
                pd.DataFrame({"mic1 (ref)": mic1[:500], "mic2 (delayed)": mic2[:500]}),
                height=220
            )
            st.caption("First 500 samples — delay encodes leak distance")

        with col2:
            log_df = pd.DataFrame(st.session_state.cycles_log)
            if not log_df.empty:
                st.dataframe(log_df, use_container_width=True)
                st.bar_chart(log_df.set_index("cycle")["score"], height=160)
    else:
        st.info("Run detection first.")

# ------------------------------------------------------------------ #
#  Tab 3 — Metrics                                                     #
# ------------------------------------------------------------------ #
with tab3:
    st.subheader("Session metrics")

    history = st.session_state.results_history
    if history:
        df_h = pd.DataFrame(history)
        dispatched_h = df_h[df_h["flag"] == "DISPATCH"]
        total_h = len(df_h)
        tp_h = len(dispatched_h)
        f1_h = (2 * tp_h / (2 * tp_h + (total_h - tp_h))
                if total_h > 0 else 0)
        mean_err_h = dispatched_h["error_m"].mean() if len(dispatched_h) > 0 else None

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Runs", total_h)
        col2.metric("Dispatched", tp_h)
        col3.metric("F1 (session)", f"{f1_h:.2f}")
        col4.metric("Mean error", f"{mean_err_h:.1f} m" if mean_err_h else "—")

        st.line_chart(df_h["score"], height=180)
    else:
        st.info("Run several detections to see metrics.")

    # Load pre-run simulation CSV if available
    if os.path.exists(RESULTS_CSV):
        st.markdown("---")
        st.markdown("**Batch simulation results** (`main_pipeline.py`)")
        sim_df = pd.read_csv(RESULTS_CSV)
        tp_s = sim_df["dispatched"].sum()
        total_s = len(sim_df)
        err_s = sim_df["error_m"].dropna().mean()
        f1_s = 2 * tp_s / (2 * tp_s + (total_s - tp_s))

        c1, c2, c3 = st.columns(3)
        c1.metric("Scenarios", total_s)
        c2.metric("F1 score", f"{f1_s:.3f}")
        c3.metric("Mean error", f"{err_s:.2f} m")
        st.dataframe(sim_df.tail(20), use_container_width=True)

# ------------------------------------------------------------------ #
#  Tab 4 — Simulation log                                              #
# ------------------------------------------------------------------ #
with tab4:
    st.subheader("Cycle-by-cycle gate log")
    if st.session_state.cycles_log:
        st.dataframe(pd.DataFrame(st.session_state.cycles_log),
                     use_container_width=True)
    else:
        st.info("Run detection to populate the log.")

# ------------------------------------------------------------------ #
#  Tab 5 — About                                                       #
# ------------------------------------------------------------------ #
with tab5:
    st.markdown("""
## PGPL — Persistence-Gated Piezo Leak Localizer

**Concept**: PULSE-AT brain gates piezo mic anomalies (persist ≥ 3, PSI ≥ 0.20)
→ acoustic cross-correlation Δτ → City2Graph pipe path snap → GPS leak coordinate.

| Layer | Component | Role |
|---|---|---|
| 1 | Piezo mics | Measure dP/dt, Δτ, var_z |
| 2 | PULSE-AT (4 tiers) | Gate anomalies, suppress false alarms |
| 3 | Acoustic corr. | Compute distance from time delay |
| 4 | City2Graph / OSM | Snap distance to real pipe GPS |

**Map**: Eco-Delta City, Myeongji-dong, Gangseo-gu, Busan, South Korea

**Patent claim core**: A method for leak localization comprising multi-variable
z-score gating (persist ≥ 3, PSI ≥ 0.20) as a precondition for acoustic
cross-correlation triangulation, projecting the resulting distance onto a
geospatial pipe graph to produce a GPS coordinate output.
    """)