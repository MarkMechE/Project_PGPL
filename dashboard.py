import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from pgpl_brain import PULSE_AT_Brain
from pulse_at_bridge import (
    load_or_generate_pulse_at_data,
    map_columns,
    row_to_sensors,
)

st.set_page_config(
    page_title="PGPL Eco-Delta Leak Monitor",
    layout="wide",
)

ECO_DELTA_CENTER = [35.135, 128.970]
EDGES_PATH       = "data/eco_delta_edges.geojson"
BRIDGE_CSV       = "outputs/bridge_results.csv"
N_SENSORS        = 4

PRIORITY_COLOR = {
    "P1-CRITICAL": "#E24B4A",
    "P2-HIGH":     "#EF9F27",
    "P3-MODERATE": "#1D9E75",
    "P4-LOW":      "#378ADD",
}


@st.cache_resource
def load_brain():
    return PULSE_AT_Brain()


@st.cache_data
def load_pulse_at_rows():
    return map_columns(load_or_generate_pulse_at_data())


brain        = load_brain()
df_pulse_at  = load_pulse_at_rows()
max_node     = max(1, len(brain.node_list))
sensor_nodes = [int(i * max_node / N_SENSORS) for i in range(N_SENSORS)]
total_rows   = len(df_pulse_at)


def _init_state():
    defaults = {
        "row_idx":         0,
        "playing":         False,
        "cycle_log":       [],
        "dispatch_events": [],
        "score_history":   [],
        "last_result":     None,
        "last_sensors":    None,
        "priority_queue":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

st.title("PGPL - Eco-Delta City Leak Monitor")
st.caption(
    f"Persistence-Gated Piezo Leak Localizer  |  "
    f"Myeongji-dong, Gangseo-gu, Busan  |  "
    f"{N_SENSORS} sensor nodes  |  "
    f"{N_SENSORS * 2} piezo mics"
)

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 2, 3])

with c1:
    if st.button("Play", use_container_width=True,
                 disabled=st.session_state.playing):
        st.session_state.playing = True

with c2:
    if st.button("Pause", use_container_width=True,
                 disabled=not st.session_state.playing):
        st.session_state.playing = False

with c3:
    if st.button("Reset", use_container_width=True):
        brain.persist_count = 0
        brain.psi_window.clear()
        st.session_state.update({
            "row_idx":         0,
            "playing":         False,
            "cycle_log":       [],
            "dispatch_events": [],
            "score_history":   [],
            "last_result":     None,
            "last_sensors":    None,
            "priority_queue":  [],
        })
        st.rerun()

with c4:
    pct = min(1.0, st.session_state.row_idx / max(1, total_rows - 1))
    st.progress(pct, text=f"Row {st.session_state.row_idx + 1} / {total_rows}")

with c5:
    if st.session_state.row_idx < total_rows:
        reg   = df_pulse_at.iloc[st.session_state.row_idx].get("regime", "-")
        label = "LEAK" if reg == "stress" else "NORMAL"
        st.markdown(
            f"**Regime:** {label} | "
            f"**Sensors:** {N_SENSORS} | "
            f"**Min persist:** {brain.min_persist}"
        )

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Map",
    "Signal and Gate",
    "Metrics",
    "Event Log",
    "About",
])


def priority_color(priority):
    return PRIORITY_COLOR.get(priority, "#888780")


def build_map(dispatch_events, last_gps=None):
    m = folium.Map(
        location=ECO_DELTA_CENTER,
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    if os.path.exists(EDGES_PATH):
        folium.GeoJson(
            EDGES_PATH,
            name="Pipe network",
            style_function=lambda _: {
                "color": "#3B82F6", "weight": 1.5, "opacity": 0.6
            },
        ).add_to(m)

    for i, nidx in enumerate(sensor_nodes):
        nk     = brain.node_list[nidx % len(brain.node_list)] if brain.node_list else None
        coords = brain.node_coords.get(nk, ECO_DELTA_CENTER)
        folium.CircleMarker(
            location=list(coords),
            radius=7,
            color="#10B981",
            fill=True,
            fill_opacity=0.9,
            popup=f"Sensor node {i}",
        ).add_to(m)

    for ev in dispatch_events[:-1]:
        if ev.get("gps"):
            color = priority_color(ev.get("priority", ""))
            folium.CircleMarker(
                location=list(ev["gps"]),
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=(
                    f"Priority: {ev.get('priority','—')}<br>"
                    f"Type: {ev.get('anomaly','—')}<br>"
                    f"Distance: {ev.get('loc_m')} m<br>"
                    f"Severity: {ev.get('severity','—')}"
                ),
            ).add_to(m)

    if last_gps and dispatch_events:
        last = dispatch_events[-1]
        color = priority_color(last.get("priority", ""))
        folium.Marker(
            location=list(last_gps),
            icon=folium.Icon(color="red", icon="info-sign"),
            popup=(
                f"LATEST — {last.get('priority','—')}<br>"
                f"Type: {last.get('anomaly','—')}<br>"
                f"Distance: {last.get('loc_m')} m"
            ),
        ).add_to(m)
        folium.PolyLine(
            [ECO_DELTA_CENTER, list(last_gps)],
            color=color,
            weight=2,
            dash_array="6",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def advance_one_row():
    idx = st.session_state.row_idx
    if idx >= total_rows:
        st.session_state.playing = False
        return

    row     = df_pulse_at.iloc[idx]
    sensors = row_to_sensors(row, brain)
    result  = brain.process(sensors, sensor_node_idx=idx % N_SENSORS)

    st.session_state.last_result  = result
    st.session_state.last_sensors = sensors

    st.session_state.score_history.append({
        "row":    idx,
        "score":  result["score"],
        "psi":    result["psi"],
        "flag":   result["flag"],
        "regime": sensors.get("regime", "-"),
    })

    log_row = {
        "row":       idx,
        "timestamp": sensors.get("timestamp", ""),
        "regime":    sensors.get("regime", "-"),
        "flow":      round(sensors["flow"], 2),
        "pressure":  round(sensors["pressure"], 3),
        "var_z":     round(sensors["var_z"], 3),
        "score":     result["score"],
        "psi":       result["psi"],
        "persist":   result["persist"],
        "flag":      result["flag"],
        "anomaly":   result.get("anomaly", "-"),
        "priority":  result.get("priority", "-"),
        "severity":  result.get("severity", "-"),
        "loc_m":     result.get("loc_m", "-"),
        "error_m":   result.get("error_m", "-"),
    }
    st.session_state.cycle_log.append(log_row)

    if result["flag"] == "DISPATCH":
        ev = {
            "row":      idx,
            "gps":      result.get("gps"),
            "loc_m":    result.get("loc_m"),
            "error":    result.get("error_m"),
            "anomaly":  result.get("anomaly", "-"),
            "priority": result.get("priority", "-"),
            "severity": result.get("severity", 0),
        }
        st.session_state.dispatch_events.append(ev)

        pq = st.session_state.priority_queue
        pq.append(ev)
        pq.sort(key=lambda x: x.get("severity", 0), reverse=True)
        st.session_state.priority_queue = pq

    st.session_state.row_idx += 1


if st.session_state.playing:
    advance_one_row()
    time.sleep(0.05)
    st.rerun()

if not st.session_state.playing:
    if st.button(
        "Step one row",
        disabled=st.session_state.row_idx >= total_rows,
    ):
        advance_one_row()
        st.rerun()


with tab1:
    r = st.session_state.last_result

    if r is None:
        st.info("Press Play to start processing PULSE-AT data.")
    elif r["flag"] == "DISPATCH":
        pri = r.get("priority", "")
        if "P1" in pri:
            st.error(
                f"{pri} — {r.get('anomaly')} at {r.get('loc_m')} m  |  "
                f"Severity: {r.get('severity')}  |  "
                f"GPS: {r.get('gps')}  |  "
                f"Score: {r['score']}"
            )
        elif "P2" in pri:
            st.warning(
                f"{pri} — {r.get('anomaly')} at {r.get('loc_m')} m  |  "
                f"Severity: {r.get('severity')}  |  "
                f"GPS: {r.get('gps')}  |  "
                f"Score: {r['score']}"
            )
        else:
            st.info(
                f"{pri} — {r.get('anomaly')} at {r.get('loc_m')} m  |  "
                f"Severity: {r.get('severity')}  |  "
                f"Score: {r['score']}"
            )
    else:
        st.success(
            f"MONITOR  |  Score: {r['score']}  |  "
            f"Persist: {r['persist']}/{brain.min_persist}  |  "
            f"PSI: {r['psi']}"
        )

    last_gps = (
        st.session_state.dispatch_events[-1].get("gps")
        if st.session_state.dispatch_events
        else None
    )
    st_folium(
        build_map(st.session_state.dispatch_events, last_gps),
        width=900,
        height=500,
        key="live_map",
    )

    if r:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Score",      r["score"])
        k2.metric("PSI",        r["psi"])
        k3.metric("Persist",    f"{r['persist']}/{brain.min_persist}")
        k4.metric("Epsilon",    r["eps"])
        k5.metric("Dispatches", len(st.session_state.dispatch_events))
        k6.metric("Priority",   r.get("priority", "-"))

    if st.session_state.priority_queue:
        st.markdown("**Active priority queue — ranked by severity**")
        pq_df = pd.DataFrame(st.session_state.priority_queue)[
            ["row", "priority", "anomaly", "severity", "loc_m", "error"]
        ]
        st.dataframe(pq_df, use_container_width=True, hide_index=True)


with tab2:
    st.subheader("Acoustic signals from current PULSE-AT row")
    sensors = st.session_state.last_sensors

    if sensors:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Mic 1 vs Mic 2 — first 600 samples**")
            st.line_chart(
                pd.DataFrame({
                    "mic1 reference": sensors["mic1_sig"][:600],
                    "mic2 delayed":   sensors["mic2_sig"][:600],
                }),
                height=220,
            )
            st.caption(
                "Delay between curves encodes acoustic time-of-flight to distance"
            )

        with col_b:
            st.markdown("**Gate state this row**")
            r2 = st.session_state.last_result
            if r2:
                st.dataframe(
                    pd.DataFrame([
                        {"metric": "Score",        "value": r2["score"]},
                        {"metric": "PSI",          "value": r2["psi"]},
                        {"metric": "Persist",      "value": r2["persist"]},
                        {"metric": "p-value",      "value": r2.get("pvalue", "-")},
                        {"metric": "Epsilon",      "value": r2["eps"]},
                        {"metric": "Flag",         "value": r2["flag"]},
                        {"metric": "Anomaly type", "value": r2.get("anomaly", "-")},
                        {"metric": "Anom prob",    "value": r2.get("anom_prob", "-")},
                        {"metric": "Priority",     "value": r2.get("priority", "-")},
                        {"metric": "Severity",     "value": r2.get("severity", "-")},
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

        if st.session_state.score_history:
            st.markdown("**Score and PSI history**")
            sh = pd.DataFrame(st.session_state.score_history)
            st.line_chart(
                sh.set_index("row")[["score", "psi"]],
                height=160,
            )
    else:
        st.info("Press Play to start.")


with tab3:
    st.subheader("Live accuracy metrics")
    log = st.session_state.cycle_log

    if log:
        df_log   = pd.DataFrame(log)
        dispatch = df_log[df_log["flag"] == "DISPATCH"]
        stress   = df_log[df_log["regime"] == "stress"]
        tp       = len(dispatch[dispatch["regime"] == "stress"])
        fp       = len(dispatch[dispatch["regime"] != "stress"])
        fn       = len(stress) - tp
        prec     = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec      = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1       = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
        err_vals = pd.to_numeric(dispatch["error_m"], errors="coerce").dropna()
        mean_err = err_vals.mean() if len(err_vals) > 0 else None

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows processed",  len(df_log))
        c2.metric("True positives",  tp)
        c3.metric("False positives", fp)
        c4.metric("F1 score",        f"{f1:.3f}")
        c5.metric("Mean error",      f"{mean_err:.1f} m" if mean_err else "-")

        st.markdown("**Score by regime**")
        chart_df = df_log[["row", "score", "regime"]].copy()
        st.line_chart(
            pd.DataFrame({
                "stress": chart_df[
                    chart_df["regime"] == "stress"
                ].set_index("row")["score"],
                "normal": chart_df[
                    chart_df["regime"] != "stress"
                ].set_index("row")["score"],
            }),
            height=180,
        )

        if "priority" in df_log.columns:
            st.markdown("**Dispatch breakdown by priority**")
            pri_counts = (
                dispatch["priority"]
                .value_counts()
                .reindex(["P1-CRITICAL", "P2-HIGH", "P3-MODERATE", "P4-LOW"])
                .fillna(0)
            )
            st.bar_chart(pri_counts, height=140)

        if "anomaly" in df_log.columns:
            st.markdown("**Dispatch breakdown by anomaly type**")
            st.bar_chart(dispatch["anomaly"].value_counts(), height=140)

    else:
        st.info("No data yet — press Play.")

    if os.path.exists(BRIDGE_CSV):
        st.markdown("---")
        st.markdown("**Pre-computed batch results**")
        sim_df  = pd.read_csv(BRIDGE_CSV)
        tp_s    = len(sim_df[
            (sim_df["pgpl_flag"] == "DISPATCH") &
            (sim_df["regime"] == "stress")
        ])
        total_s = len(sim_df)
        err_s   = pd.to_numeric(
            sim_df["pgpl_error_m"], errors="coerce"
        ).dropna().mean()
        f1_s    = (2 * tp_s / (2 * tp_s + (total_s - tp_s))
                   if total_s > 0 else 0)
        b1, b2, b3 = st.columns(3)
        b1.metric("Scenarios",  total_s)
        b2.metric("F1",         f"{f1_s:.3f}")
        b3.metric("Mean error", f"{err_s:.2f} m")


with tab4:
    st.subheader("Row-by-row event log")
    if st.session_state.cycle_log:
        st.dataframe(
            pd.DataFrame(st.session_state.cycle_log),
            use_container_width=True,
        )
        st.download_button(
            "Export CSV",
            data=pd.DataFrame(
                st.session_state.cycle_log
            ).to_csv(index=False),
            file_name="pgpl_session_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No events yet.")


with tab5:
    st.markdown(f"""
## PGPL - Persistence-Gated Piezo Leak Localizer

**Location**: Eco-Delta City, Myeongji-dong, Gangseo-gu, Busan, South Korea

**Sensor deployment**: {N_SENSORS} nodes x 2 piezo mics = {N_SENSORS * 2} mics total, spaced 50 m apart

| Layer | File | Role |
|---|---|---|
| 1 | Physical sensors | Flow, pressure, turbulence |
| 2 | pgpl_brain Tier 1 | Bidirectional z-score |
| 2 | pgpl_brain Tier 2 | Conformal p-value calibration |
| 2 | pgpl_brain Tier 3 | Persist >= {brain.min_persist}, PSI >= 0.20 |
| 2 | pgpl_brain Tier 4 | Adaptive epsilon feedback |
| 3 | pgpl_brain | Acoustic cross-correlation to distance |
| 3 | anomaly_classifier | CNN-MLP: leak / burst / drop / normal |
| 3 | pgpl_brain | Severity scoring P1-P4 |
| 4 | pgpl_brain | City2Graph OSM GPS snap |

**Priority levels**

| Level | Trigger | Action |
|---|---|---|
| P1-CRITICAL | Burst, severity >= 0.80 | Immediate dispatch + zone shutoff |
| P2-HIGH | Leak, severity 0.50-0.80 | Repair within 4 hours |
| P3-MODERATE | Pressure drop, severity 0.20-0.50 | Log and monitor |
| P4-LOW | Slug or noise, severity < 0.20 | Record only |

**Patent claim core**: Multi-variable z-score persistence gating (>= {brain.min_persist}
consecutive windows, PSI >= 0.20) as precondition for acoustic cross-correlation
triangulation, projecting distance onto a geospatial pipe graph to produce a
GPS coordinate, with severity-ranked dispatch output.
""")