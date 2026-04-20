# write_dashboard.py
# Run: python write_dashboard.py
# Writes dashboard.py cleanly with no encoding issues

code = open("write_dashboard.py").read()  # self-test only

dashboard_code = """
import os, time
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from pgpl_brain import PULSE_AT_Brain
from pulse_at_bridge import load_or_generate_pulse_at_data, map_columns, row_to_sensors

st.set_page_config(page_title="PGPL Eco-Delta", layout="wide")

ECO_DELTA_CENTER = [35.135, 128.970]
EDGES_PATH = "data/eco_delta_edges.geojson"
BRIDGE_CSV = "outputs/bridge_results.csv"
N_SENSORS  = 4

@st.cache_resource
def load_brain():
    return PULSE_AT_Brain()

@st.cache_data
def load_rows():
    return map_columns(load_or_generate_pulse_at_data())

brain       = load_brain()
df_pulse_at = load_rows()
max_node    = max(1, len(brain.node_list))
snodes      = [int(i * max_node / N_SENSORS) for i in range(N_SENSORS)]
total_rows  = len(df_pulse_at)

def _init():
    for k, v in {"row_idx":0,"playing":False,"cycle_log":[],
                 "dispatch_events":[],"score_history":[],
                 "last_result":None,"last_sensors":None}.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

st.title("PGPL - Eco-Delta City Leak Monitor")
st.caption(f"Persistence-Gated Piezo Leak Localizer  Myeongji-dong, Gangseo-gu, Busan  {N_SENSORS} sensor nodes")

c1,c2,c3,c4,c5 = st.columns([1,1,1,2,3])
with c1:
    if st.button("Play", use_container_width=True, disabled=st.session_state.playing):
        st.session_state.playing = True
with c2:
    if st.button("Pause", use_container_width=True, disabled=not st.session_state.playing):
        st.session_state.playing = False
with c3:
    if st.button("Reset", use_container_width=True):
        brain.persist_count = 0
        brain.psi_window.clear()
        st.session_state.update({"row_idx":0,"playing":False,"cycle_log":[],
            "dispatch_events":[],"score_history":[],"last_result":None,"last_sensors":None})
        st.rerun()
with c4:
    pct = min(1.0, st.session_state.row_idx / max(1, total_rows-1))
    st.progress(pct, text=f"Row {st.session_state.row_idx+1} / {total_rows}")
with c5:
    if st.session_state.row_idx < total_rows:
        reg   = df_pulse_at.iloc[st.session_state.row_idx].get("regime","-")
        label = "LEAK" if reg=="stress" else "NORMAL"
        st.markdown(f"Regime: {label} | Sensors: {N_SENSORS} | Min persist: {brain.min_persist}")

st.divider()
tab1,tab2,tab3,tab4,tab5 = st.tabs(["Live Map","Signal and Gate","Metrics","Event Log","About"])

def build_map(events, last_gps=None):
    m = folium.Map(location=ECO_DELTA_CENTER, zoom_start=14, tiles="OpenStreetMap")
    if os.path.exists(EDGES_PATH):
        folium.GeoJson(EDGES_PATH, style_function=lambda _:
            {"color":"#3B82F6","weight":1.5,"opacity":0.6}).add_to(m)
    for i,nidx in enumerate(snodes):
        nk = brain.node_list[nidx % len(brain.node_list)] if brain.node_list else None
        coords = brain.node_coords.get(nk, ECO_DELTA_CENTER)
        folium.CircleMarker(list(coords), radius=7, color="#10B981",
            fill=True, fill_opacity=0.9, popup=f"Sensor {i}").add_to(m)
    for ev in events[:-1]:
        if ev.get("gps"):
            folium.CircleMarker(list(ev["gps"]), radius=5, color="#FCA5A5",
                fill=True, fill_opacity=0.5, popup=f"Past: {ev.get('loc_m')}m").add_to(m)
    if last_gps:
        folium.Marker(list(last_gps), icon=folium.Icon(color="red",icon="info-sign"),
            popup="Latest leak").add_to(m)
        folium.PolyLine([ECO_DELTA_CENTER,list(last_gps)],
            color="#EF4444",weight=2,dash_array="6").add_to(m)
    folium.LayerControl().add_to(m)
    return m

def advance():
    idx = st.session_state.row_idx
    if idx >= total_rows:
        st.session_state.playing = False
        return
    row     = df_pulse_at.iloc[idx]
    sensors = row_to_sensors(row, brain)
    result  = brain.process(sensors, sensor_node_idx=idx % N_SENSORS)
    st.session_state.last_result  = result
    st.session_state.last_sensors = sensors
    st.session_state.score_history.append({"row":idx,"score":result["score"],
        "psi":result["psi"],"flag":result["flag"],"regime":sensors.get("regime","-")})
    st.session_state.cycle_log.append({"row":idx,
        "timestamp":sensors.get("timestamp",""),
        "regime":sensors.get("regime","-"),
        "flow":round(sensors["flow"],2),
        "pressure":round(sensors["pressure"],3),
        "var_z":round(sensors["var_z"],3),
        "score":result["score"],"psi":result["psi"],
        "persist":result["persist"],"flag":result["flag"],
        "anomaly":result.get("anomaly","--"),
        "loc_m":result.get("loc_m","-"),
        "error_m":result.get("error_m","-")})
    if result["flag"] == "DISPATCH":
        st.session_state.dispatch_events.append({"row":idx,
            "gps":result.get("gps"),"loc_m":result.get("loc_m"),
            "error":result.get("error_m")})
    st.session_state.row_idx += 1

if st.session_state.playing:
    advance()
    time.sleep(0.05)
    st.rerun()

if not st.session_state.playing:
    if st.button("Step one row", disabled=st.session_state.row_idx >= total_rows):
        advance()
        st.rerun()

with tab1:
    r = st.session_state.last_result
    if r is None:
        st.info("Press Play to start.")
    elif r["flag"] == "DISPATCH":
        anom = r.get("anomaly","--")
        st.error(f"DISPATCH: {anom} leak at {r.get('loc_m')}m | GPS:{r.get('gps')} | Err:{r.get('error_m')}m | Score:{r['score']}")
    else:
        st.success(f"MONITOR | Score:{r['score']} | Persist:{r['persist']}/{brain.min_persist} | PSI:{r['psi']}")
    last_gps = st.session_state.dispatch_events[-1].get("gps") if st.session_state.dispatch_events else None
    st_folium(build_map(st.session_state.dispatch_events, last_gps), width=900, height=520, key="live_map")
    if r:
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Score", r["score"])
        k2.metric("PSI", r["psi"])
        k3.metric("Persist", f"{r['persist']}/{brain.min_persist}")
        k4.metric("Epsilon", r["eps"])
        k5.metric("Dispatches", len(st.session_state.dispatch_events))
        k6.metric("Anomaly", r.get("anomaly","--"))

with tab2:
    st.subheader("Acoustic signals from current PULSE-AT row")
    sensors = st.session_state.last_sensors
    if sensors:
        col_a,col_b = st.columns(2)
        with col_a:
            st.line_chart(pd.DataFrame({"mic1":sensors["mic1_sig"][:600],
                                        "mic2":sensors["mic2_sig"][:600]}), height=220)
            st.caption("Delay between curves encodes acoustic time-of-flight to distance")
        with col_b:
            r2 = st.session_state.last_result
            if r2:
                st.dataframe(pd.DataFrame([
                    {"metric":"Score","value":r2["score"]},
                    {"metric":"PSI","value":r2["psi"]},
                    {"metric":"Persist","value":r2["persist"]},
                    {"metric":"p-value","value":r2.get("pvalue","-")},
                    {"metric":"Epsilon","value":r2["eps"]},
                    {"metric":"Flag","value":r2["flag"]},
                    {"metric":"Anomaly type","value":r2.get("anomaly","--")},
                    {"metric":"Anomaly prob","value":r2.get("anom_prob","--")},
                ]), use_container_width=True, hide_index=True)
        if st.session_state.score_history:
            sh = pd.DataFrame(st.session_state.score_history)
            st.line_chart(sh.set_index("row")[["score","psi"]], height=160)
    else:
        st.info("Press Play to start.")

with tab3:
    st.subheader("Live accuracy metrics")
    log = st.session_state.cycle_log
    if log:
        df_log   = pd.DataFrame(log)
        dispatch = df_log[df_log["flag"]=="DISPATCH"]
        stress   = df_log[df_log["regime"]=="stress"]
        tp = len(dispatch[dispatch["regime"]=="stress"])
        fp = len(dispatch[dispatch["regime"]!="stress"])
        fn = len(stress) - tp
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        rc = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*rc/(p+rc) if (p+rc)>0 else 0
        me = pd.to_numeric(dispatch["error_m"],errors="coerce").dropna()
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Rows", len(df_log))
        c2.metric("True pos", tp)
        c3.metric("False pos", fp)
        c4.metric("F1", f"{f1:.3f}")
        c5.metric("Mean error", f"{me.mean():.1f}m" if len(me)>0 else "-")
        chart = df_log[["row","score","regime"]].copy()
        st.line_chart(pd.DataFrame({
            "stress": chart[chart["regime"]=="stress"].set_index("row")["score"],
            "normal": chart[chart["regime"]!="stress"].set_index("row")["score"],
        }), height=180)
        if "anomaly" in df_log.columns:
            st.markdown("**Anomaly type breakdown**")
            st.bar_chart(dispatch["anomaly"].value_counts(), height=140)
    else:
        st.info("No data yet.")
    if os.path.exists(BRIDGE_CSV):
        st.markdown("---")
        s = pd.read_csv(BRIDGE_CSV)
        tp_s = len(s[(s["pgpl_flag"]=="DISPATCH")&(s["regime"]=="stress")])
        f1_s = 2*tp_s/(2*tp_s+(len(s)-tp_s)) if len(s)>0 else 0
        e_s  = pd.to_numeric(s["pgpl_error_m"],errors="coerce").dropna().mean()
        b1,b2,b3 = st.columns(3)
        b1.metric("Batch scenarios", len(s))
        b2.metric("Batch F1", f"{f1_s:.3f}")
        b3.metric("Batch mean error", f"{e_s:.2f}m")

with tab4:
    st.subheader("Row-by-row event log")
    if st.session_state.cycle_log:
        st.dataframe(pd.DataFrame(st.session_state.cycle_log), use_container_width=True)
        st.download_button("Export CSV",
            data=pd.DataFrame(st.session_state.cycle_log).to_csv(index=False),
            file_name="pgpl_session_log.csv", mime="text/csv")
    else:
        st.info("No events yet.")

with tab5:
    st.markdown(f\"\"\"
## PGPL - Persistence-Gated Piezo Leak Localizer

**Location**: Eco-Delta City, Myeongji-dong, Gangseo-gu, Busan, South Korea

**Sensors**: {N_SENSORS} nodes x 2 piezo mics spaced 50 m apart

| Layer | File | Role |
|---|---|---|
| 1 | sensors (physical) | Flow, pressure, turbulence |
| 2 | pgpl_brain.py Tier 1 | Bidirectional z-score |
| 2 | pgpl_brain.py Tier 2 | Conformal p-value |
| 2 | pgpl_brain.py Tier 3 | Persist >= {brain.min_persist}, PSI >= 0.20 |
| 2 | pgpl_brain.py Tier 4 | Adaptive epsilon |
| 3 | pgpl_brain.py | Acoustic cross-correlation to distance |
| 3 | anomaly_classifier.py | CNN-MLP leak/burst/drop/normal |
| 4 | pgpl_brain.py | City2Graph OSM GPS snap |
| - | pulse_at_bridge.py | PULSE-AT rows to sensor dict |
| - | dashboard.py | Streamlit live monitor |
\"\"\")
"""

with open("dashboard.py", "w", encoding="utf-8") as f:
    f.write(dashboard_code.lstrip("\n"))

import ast
try:
    ast.parse(open("dashboard.py").read())
    print("dashboard.py written OK - no syntax errors")
except SyntaxError as e:
    print(f"SYNTAX ERROR line {e.lineno}: {e.msg}")
    print(f"  {e.text}")