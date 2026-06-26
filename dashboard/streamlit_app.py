import time
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from collections import deque

# =====================================================================
# --- CONFIG
# =====================================================================

import os
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MachineLearningCVE"

ALERT_COLORS = {
    "GREEN":  "#00c853",
    "YELLOW": "#ffd600",
    "ORANGE": "#ff6d00",
    "RED":    "#d50000"
}

ACTION_COLORS = {
    "ALLOW":    "#00c853",
    "THROTTLE": "#ffd600",
    "DROP":     "#ff6d00",
    "HONEYPOT": "#aa00ff"
}

FEATURE_COLUMNS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# API feature names (underscore format)
API_FEATURE_MAP = {col: col.replace(" ", "_").replace("/", "/") for col in FEATURE_COLUMNS}

# =====================================================================
# --- DATA LOADER
# =====================================================================

@st.cache_data
def load_sample_packets(n=500):
    """Load attack-heavy sample from CICIDS2017 for simulation."""
    dfs = []
    # Prioritise attack-rich files
    priority_files = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    ]

    for fname in priority_files:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip", nrows=3000)
            df.columns = df.columns.str.strip()
            if "Label" not in df.columns:
                continue
            # Take 70% attack, 30% benign
            attack = df[df["Label"] != "BENIGN"].head(int(n * 0.7 / len(priority_files)))
            benign = df[df["Label"] == "BENIGN"].head(int(n * 0.3 / len(priority_files)))
            dfs.append(pd.concat([attack, benign]))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], 0).fillna(0)
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


def row_to_api_payload(row: pd.Series) -> dict:
    """Convert a CSV row to the API feature dict (underscore keys)."""
    features = {}
    for col in FEATURE_COLUMNS:
        api_key = col.replace(" ", "_")
        val = row.get(col, 0.0)
        try:
            val = float(val)
            if np.isnan(val) or np.isinf(val):
                val = 0.0
        except Exception:
            val = 0.0
        features[api_key] = val
    return {"features": features, "include_shap": False}


# =====================================================================
# --- API CALLS
# =====================================================================

def analyze_packet(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/analyze", json=payload, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# =====================================================================
# --- SESSION STATE INIT
# =====================================================================

def init_state():
    defaults = {
        "running":        False,
        "packet_log":     [],
        "alert_counts":   {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0},
        "action_counts":  {"ALLOW": 0, "THROTTLE": 0, "DROP": 0, "HONEYPOT": 0},
        "class_counts":   {},
        "packets_sent":   0,
        "threats_caught": 0,
        "total_ms":       0.0,
        "sample_data":    None,
        "sample_idx":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def color_alert(val):
    colors = {
        "GREEN":  "background-color:#1b5e20;color:white",
        "YELLOW": "background-color:#f57f17;color:black",
        "ORANGE": "background-color:#e65100;color:white",
        "RED":    "background-color:#b71c1c;color:white"
    }
    return colors.get(val, "")


def color_action(val):
    colors = {
        "ALLOW":    "color:#00c853;font-weight:bold",
        "THROTTLE": "color:#ffd600;font-weight:bold",
        "DROP":     "color:#ff6d00;font-weight:bold",
        "HONEYPOT": "color:#aa00ff;font-weight:bold"
    }
    return colors.get(val, "")


# =====================================================================
# --- MAIN DASHBOARD
# =====================================================================

def main():
    st.set_page_config(
        page_title="NIDS — Live Threat Monitor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_state()


    # ---- Sidebar ----
    with st.sidebar:
        st.title("🛡️ NIDS Control Panel")
        st.divider()

        api_ok = check_api_health()
        if api_ok:
            st.success("API Online", icon="✅")
        else:
            st.error("API Offline", icon="🔴")
            st.info(f"Expected at: {API_URL}")

        st.divider()
        st.subheader("Simulation Settings")
        speed = st.slider("Packets/second", min_value=1, max_value=20, value=5)
        burst_size = st.slider("Packets per burst", min_value=1, max_value=10, value=3)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start", use_container_width=True, disabled=not api_ok):
                if st.session_state.sample_data is None:
                    with st.spinner("Loading packet data..."):
                        st.session_state.sample_data = load_sample_packets(500)
                st.session_state.running = True

        with col2:
            if st.button("⏹ Stop", use_container_width=True):
                st.session_state.running = False

        if st.button("🔄 Reset", use_container_width=True):
            for key in ["packet_log", "alert_counts", "action_counts",
                        "class_counts", "packets_sent", "threats_caught",
                        "total_ms", "sample_idx"]:
                st.session_state[key] = [] if key == "packet_log" else (
                    {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0} if key == "alert_counts" else
                    {"ALLOW": 0, "THROTTLE": 0, "DROP": 0, "HONEYPOT": 0} if key == "action_counts" else
                    {} if key == "class_counts" else
                    0
                )
            st.session_state.running = False

        st.divider()
        st.subheader("Recent Alerts")
        try:
            r = requests.get(f"{API_URL}/alerts?limit=5", timeout=2)
            if r.status_code == 200:
                data = r.json()
                counts = data.get("counts", {})
                st.metric("Total Alerts", counts.get("total", 0))
                a1, a2 = st.columns(2)
                a1.metric("RED", counts.get("RED", 0), delta=None)
                a2.metric("ORANGE", counts.get("ORANGE", 0), delta=None)
                alerts = data.get("alerts", [])
                for alert in alerts[:3]:
                    color = "#d50000" if alert["alert_level"] == "RED" else "#ff6d00"
                    st.markdown(
                        f"<div style='border-left:3px solid {color};padding:4px 8px;margin:4px 0;font-size:12px'>"
                        f"<b>{alert['alert_level']}</b> — {alert['threat_class']}<br>"
                        f"<span style='color:gray'>{alert['timestamp']}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        except Exception:
            st.caption("Alert feed unavailable")

        st.divider()
        st.caption("CICIDS;lll2017 Dataset Simulation")
        st.caption("Three-Tier Autonomous Defense")

    # ---- Header ----
    st.title("🛡️ Autonomous Network Intrusion Detection System")
    st.caption("Live Threat Monitor — LightGBM · Isolation Forest · PPO RL Agent · SHAP")

    # ---- KPI Row ----
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Packets Analyzed", f"{st.session_state.packets_sent:,}")
    k2.metric("Threats Detected", f"{st.session_state.threats_caught:,}")
    detection_rate = (
        st.session_state.threats_caught / st.session_state.packets_sent * 100
        if st.session_state.packets_sent > 0 else 0
    )
    k3.metric("Detection Rate", f"{detection_rate:.1f}%")
    avg_ms = (
        st.session_state.total_ms / st.session_state.packets_sent
        if st.session_state.packets_sent > 0 else 0
    )
    k4.metric("Avg Latency", f"{avg_ms:.1f} ms")
    k5.metric("Status", "🟢 RUNNING" if st.session_state.running else "⚪ IDLE")

    st.divider()

    # ---- Charts Row ----
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        st.subheader("Alert Level Timeline")
        if st.session_state.packet_log:
            log_df = pd.DataFrame(st.session_state.packet_log[-100:])
            fig = go.Figure()
            level_map = {"GREEN": 1, "YELLOW": 2, "ORANGE": 3, "RED": 4}
            colors = [ALERT_COLORS.get(l, "#888") for l in log_df["alert_level"]]
            fig.add_trace(go.Scatter(
                x=list(range(len(log_df))),
                y=[level_map.get(l, 0) for l in log_df["alert_level"]],
                mode="markers+lines",
                marker=dict(color=colors, size=8),
                line=dict(color="#444", width=1)
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis=dict(
                    tickvals=[1, 2, 3, 4],
                    ticktext=["GREEN", "YELLOW", "ORANGE", "RED"],
                    range=[0, 5]
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Start simulation to see live alert timeline")

    with c2:
        st.subheader("Actions Taken")
        counts = st.session_state.action_counts
        if sum(counts.values()) > 0:
            fig = go.Figure(go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                marker_colors=[ACTION_COLORS[a] for a in counts.keys()],
                hole=0.4
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No actions yet")

    with c3:
        st.subheader("Threat Classes")
        if st.session_state.class_counts:
            cc = dict(sorted(
                st.session_state.class_counts.items(),
                key=lambda x: x[1], reverse=True
            )[:8])
            fig = go.Figure(go.Bar(
                x=list(cc.values()),
                y=list(cc.keys()),
                orientation="h",
                marker_color="#ef5350"
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No threats yet")

    st.divider()

    # ---- Live Packet Feed ----
    st.subheader("📡 Live Packet Feed")
    feed_placeholder = st.empty()

    # ---- Render packet table ----
    if st.session_state.packet_log:
        log_df = pd.DataFrame(st.session_state.packet_log[-100:]).iloc[::-1]

        styled = log_df.style\
            .map(color_alert, subset=["alert_level"])\
            .map(color_action, subset=["action"])

        feed_placeholder.dataframe(styled, use_container_width=True, height=400)
    else:
        feed_placeholder.info("Waiting for packets...")

    # ---- Simulation Loop ----
    if st.session_state.running and st.session_state.sample_data is not None:
        data = st.session_state.sample_data

        for _ in range(burst_size):
            idx = st.session_state.sample_idx % len(data)
            row = data.iloc[idx]
            st.session_state.sample_idx += 1

            payload = row_to_api_payload(row)
            result = analyze_packet(payload)

            if result:
                true_label = str(row.get("Label", "UNKNOWN"))
                entry = {
                    "packet_#":    st.session_state.packets_sent + 1,
                    "true_label":  true_label,
                    "t1_class":    result["tier1_predicted_class"],
                    "confidence":  f"{result['tier1_confidence']:.0%}",
                    "t2_anomaly":  "⚠️" if result["tier2_is_anomalous"] else "✅",
                    "t2_score":    f"{result['tier2_anomaly_score']:.3f}",
                    "action":      result["tier3_action"],
                    "alert_level": result["system_alert_level"],
                    "latency_ms":  result["processing_ms"],
                }

                st.session_state.packet_log.append(entry)
                st.session_state.packets_sent += 1
                st.session_state.total_ms += result["processing_ms"]

                alert = result["system_alert_level"]
                st.session_state.alert_counts[alert] = st.session_state.alert_counts.get(alert, 0) + 1

                action = result["tier3_action"]
                st.session_state.action_counts[action] = st.session_state.action_counts.get(action, 0) + 1

                t1 = result["tier1_predicted_class"]
                if t1 != "BENIGN":
                    st.session_state.threats_caught += 1
                    st.session_state.class_counts[t1] = st.session_state.class_counts.get(t1, 0) + 1

                if len(st.session_state.packet_log) > 200:
                    st.session_state.packet_log = st.session_state.packet_log[-200:]

        # Update table after burst
        if st.session_state.packet_log:
            log_df = pd.DataFrame(st.session_state.packet_log[-30:]).iloc[::-1]
            styled = log_df.style\
                .map(color_alert, subset=["alert_level"])\
                .map(color_action, subset=["action"])
            feed_placeholder.dataframe(styled, use_container_width=True, height=400)

        time.sleep(1.0 / max(speed, 1))
        st.rerun()


if __name__ == "__main__":
    main()