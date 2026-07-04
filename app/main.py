from dotenv import load_dotenv
load_dotenv()
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from stable_baselines3 import PPO
from app.shap_explainer import ThreatExplainer
from app.alerts import AlertManager

# Exact 78 feature names in training order — used to reindex every incoming packet
FEATURE_COLUMNS = [
    'Destination_Port', 'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets',
    'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean', 'Fwd_Packet_Length_Std',
    'Bwd_Packet_Length_Max', 'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean', 'Bwd_Packet_Length_Std',
    'Flow_Bytes/s', 'Flow_Packets/s',
    'Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min',
    'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min',
    'Bwd_IAT_Total', 'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max', 'Bwd_IAT_Min',
    'Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Fwd_URG_Flags', 'Bwd_URG_Flags',
    'Fwd_Header_Length', 'Bwd_Header_Length',
    'Fwd_Packets/s', 'Bwd_Packets/s',
    'Min_Packet_Length', 'Max_Packet_Length', 'Packet_Length_Mean', 'Packet_Length_Std', 'Packet_Length_Variance',
    'FIN_Flag_Count', 'SYN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count',
    'ACK_Flag_Count', 'URG_Flag_Count', 'CWE_Flag_Count', 'ECE_Flag_Count',
    'Down/Up_Ratio', 'Average_Packet_Size', 'Avg_Fwd_Segment_Size', 'Avg_Bwd_Segment_Size',
    'Fwd_Header_Length.1',
    'Fwd_Avg_Bytes/Bulk', 'Fwd_Avg_Packets/Bulk', 'Fwd_Avg_Bulk_Rate',
    'Bwd_Avg_Bytes/Bulk', 'Bwd_Avg_Packets/Bulk', 'Bwd_Avg_Bulk_Rate',
    'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes', 'Subflow_Bwd_Packets', 'Subflow_Bwd_Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min',
    'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min'
]
# =====================================================================
# --- APP INIT
# =====================================================================

app = FastAPI(
    title="Autonomous Network Intrusion Detection System",
    description=(
        "Three-tier autonomous cyber defense pipeline: "
        "LightGBM (Tier 1) + Isolation Forest (Tier 2) + PPO RL Agent (Tier 3) "
        "with SHAP explainability on every flagged packet."
    ),
    version="1.0.0"
)

# Global model registry — loaded once at startup, reused for every request
MODELS = {}
ALERT_MANAGER = None

# =====================================================================
# --- SCHEMAS
# =====================================================================

class PacketInput(BaseModel):
    """
    Incoming network packet with 78 flow features.
    Feature names match the CIC-IDS2017 dataset column headers exactly.
    Values should be raw (unscaled) — the API handles scaling internally.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of 78 network flow features (raw, unscaled values)",
        example={
            "Flow Duration": 42000.0,
            "Total Fwd Packets": 4.0,
            "Total Backward Packets": 2.0,
            "Total Length of Fwd Packets": 880.0,
            "Total Length of Bwd Packets": 240.0,
            "Fwd Packet Length Max": 220.0,
            "Fwd Packet Length Min": 0.0,
            "Fwd Packet Length Mean": 45.3,
            "Fwd Packet Length Std": 12.1,
            "Bwd Packet Length Max": 120.0,
            "Bwd Packet Length Min": 0.0,
            "Bwd Packet Length Mean": 30.1,
            "Bwd Packet Length Std": 8.4,
            "Flow Bytes/s": 8095.2,
            "Flow Packets/s": 142.8,
            "Flow IAT Mean": 8400.0,
            "Flow IAT Std": 1200.0,
            "Flow IAT Max": 12000.0,
            "Flow IAT Min": 50.0,
            "Fwd IAT Total": 33600.0,
            "Fwd IAT Mean": 11200.0,
            "Fwd IAT Std": 800.0,
            "Fwd IAT Max": 12000.0,
            "Fwd IAT Min": 50.0,
            "Bwd IAT Total": 8400.0,
            "Bwd IAT Mean": 8400.0,
            "Bwd IAT Std": 0.0,
            "Bwd IAT Max": 8400.0,
            "Bwd IAT Min": 8400.0,
            "Fwd PSH Flags": 0.0,
            "Bwd PSH Flags": 0.0,
            "Fwd URG Flags": 0.0,
            "Bwd URG Flags": 0.0,
            "Fwd Header Length": 80.0,
            "Bwd Header Length": 40.0,
            "Fwd Packets/s": 95.2,
            "Bwd Packets/s": 47.6,
            "Min Packet Length": 0.0,
            "Max Packet Length": 220.0,
            "Packet Length Mean": 37.7,
            "Packet Length Std": 10.2,
            "Packet Length Variance": 1250.5,
            "FIN Flag Count": 0.0,
            "SYN Flag Count": 1.0,
            "RST Flag Count": 0.0,
            "PSH Flag Count": 0.0,
            "ACK Flag Count": 1.0,
            "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0,
            "ECE Flag Count": 0.0,
            "Down/Up Ratio": 0.5,
            "Average Packet Size": 35.0,
            "Avg Fwd Segment Size": 45.3,
            "Avg Bwd Segment Size": 30.1,
            "Fwd Header Length.1": 80.0,
            "Fwd Avg Bytes/Bulk": 0.0,
            "Fwd Avg Packets/Bulk": 0.0,
            "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0,
            "Bwd Avg Packets/Bulk": 0.0,
            "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": 4.0,
            "Subflow Fwd Bytes": 880.0,
            "Subflow Bwd Packets": 2.0,
            "Subflow Bwd Bytes": 240.0,
            "Init_Win_bytes_forward": 8192.0,
            "Init_Win_bytes_backward": 8192.0,
            "act_data_pkt_fwd": 3.0,
            "min_seg_size_forward": 20.0,
            "Active Mean": 0.0,
            "Active Std": 0.0,
            "Active Max": 0.0,
            "Active Min": 0.0,
            "Idle Mean": 0.0,
            "Idle Std": 0.0,
            "Idle Max": 0.0,
            "Idle Min": 0.0
        }
    )
    include_shap: bool = Field(
        default=True,
        description="Whether to compute SHAP explanation. Set False for faster inference."
    )


class SHAPFeature(BaseModel):
    feature: str
    value: float
    shap_contribution: float
    direction: str


class AnalysisResponse(BaseModel):
    """Full three-tier analysis response."""
    # Tier 1
    tier1_predicted_class: str
    tier1_confidence: float
    tier1_top_classes: Dict[str, float]

    # Tier 2
    tier2_anomaly_score: float
    tier2_is_anomalous: bool

    # Tier 3
    tier3_action: str
    tier3_action_code: int

    # SHAP
    shap_verdict: Optional[str]
    shap_top_features: Optional[List[SHAPFeature]]

    # Meta
    processing_ms: float
    system_alert_level: str


# =====================================================================
# --- STARTUP: LOAD ALL MODELS ONCE
# =====================================================================

@app.on_event("startup")
def load_models():
    """
    Loads all artifacts into memory at startup.
    FastAPI keeps these alive for the lifetime of the server process —
    no reloading per request.
    """
    print("\n[STARTUP] Loading all model artifacts...")

    BASE_DIR = Path(__file__).resolve().parent.parent
    ARTIFACTS_DIR = BASE_DIR / "artifacts"

    try:
        MODELS["label_encoder"] = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
        MODELS["lgbm"]          = joblib.load(ARTIFACTS_DIR / "tier1_lightgbm.pkl")
        MODELS["iforest"]       = joblib.load(ARTIFACTS_DIR / "tier2_isolation_forest.pkl")
        MODELS["ppo"]           = PPO.load(ARTIFACTS_DIR / "tier3_ppo_agent_scaled")
        MODELS["scaler"] = joblib.load(ARTIFACTS_DIR / "feature_scaler.pkl")
        MODELS["explainer"]     = ThreatExplainer(ARTIFACTS_DIR)

        # Load the scaler — we fit a new one on startup using saved artifacts
        # In production you would save/load the scaler object too
        # For now we flag that raw features need scaling
        MODELS["class_names"]   = list(MODELS["label_encoder"].classes_)
        MODELS["benign_idx"]    = int(
            np.where(np.array(MODELS["class_names"]) == "BENIGN")[0][0]
        )

        # Read system config
        config = {}
        with open(ARTIFACTS_DIR / "system_config.txt", "r") as f:
            for line in f:
                key, val = line.strip().split("=")
                config[key] = float(val)
        MODELS["anomaly_threshold"] = config["SECURITY_THRESHOLD"]

        print("[STARTUP] All models loaded successfully.")
        print(f"  Classes     : {MODELS['class_names']}")
        print(f"  BENIGN idx  : {MODELS['benign_idx']}")
        print(f"  Anomaly thr : {MODELS['anomaly_threshold']:.4f}\n")
        global ALERT_MANAGER
        ALERT_MANAGER = AlertManager()

    except Exception as e:
        print(f"[CRITICAL] Model loading failed: {e}")
        raise RuntimeError(e)


# =====================================================================
# --- ANOMALY FEATURES (same list used in training)
# =====================================================================

ANOMALY_FEATURES = [
    'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean',
    'Bwd_Packet_Length_Max', 'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean',
    'Flow_Bytes/s', 'Flow_Packets/s', 'Flow_IAT_Mean', 'Flow_IAT_Max', 'Flow_IAT_Min',
    'Fwd_Header_Length', 'Bwd_Header_Length', 'Packet_Length_Variance',
    'Average_Packet_Size', 'Avg_Fwd_Segment_Size', 'Avg_Bwd_Segment_Size'
]

WINDOW_SIZE = 5  # Must match what train.py used


# =====================================================================
# --- HELPER: BUILD OBSERVATION VECTOR FOR RL AGENT
# =====================================================================

def build_observation(t1_probs: np.ndarray, t2_score: float) -> np.ndarray:
    """
    Builds the 85-dimensional sliding window observation vector.
    For single-packet API inference, we pad all 5 window slots with
    the same current packet — this is the correct approach for
    real-time single-packet inference where history isn't available.
    """
    # Single packet observation (17 signals)
    single_obs = np.concatenate([
        t1_probs,                          # 15 class probabilities
        np.array([t2_score]),              # 1 anomaly score
        np.array([0.0])                    # rolling threat rate (unknown for single packet)
    ]).astype(np.float32)

    # Replicate across window size → 85 signals total
    windowed = np.tile(single_obs, WINDOW_SIZE).astype(np.float32)
    return windowed


# =====================================================================
# --- HELPER: ALERT LEVEL
# =====================================================================

def compute_alert_level(
    predicted_class: str,
    confidence: float,
    t2_anomalous: bool,
    action: str
) -> str:
    if predicted_class == "BENIGN" and not t2_anomalous:
        return "GREEN"
    # Tier 2 caught something even if Tier 1 missed it
    if t2_anomalous and predicted_class == "BENIGN":
        return "ORANGE"
    if predicted_class in ("DDoS", "DoS GoldenEye", "DoS Hulk", "Heartbleed"):
        return "RED"
    if action in ("DROP", "HONEYPOT"):
        return "ORANGE"
    if action == "ALLOW":
        return "YELLOW"
    return "YELLOW"


# =====================================================================
# --- ROUTES
# =====================================================================

@app.get("/health", status_code=200)
def health_check():
    """Confirms the service is running and all models are loaded."""
    loaded = [k for k, v in MODELS.items() if v is not None]
    return {
        "status": "online",
        "models_loaded": loaded,
        "tier_count": 3
    }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_packet(payload: PacketInput):
    """
    Full three-tier analysis of a single network packet.

    Pipeline:
        1. Tier 1 — LightGBM classifies threat type and confidence
        2. Tier 2 — Isolation Forest scores anomaly level
        3. Tier 3 — PPO RL agent decides optimal response action
        4. SHAP   — Explains which features drove the Tier 1 decision
    """
    start = time.perf_counter()

    # --- Parse and align input ---
    raw = payload.features

    # Tier 1 needs underscore names (how LightGBM was trained)
    packet_df = pd.DataFrame(
        [{col: raw.get(col, 0.0) for col in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS
    )

  # Scale using space-named columns (how scaler was fitted in preprocess.py)
    # then rename back to underscore format for LightGBM
    scaler_cols = list(MODELS["scaler"].feature_names_in_)
    packet_df_scaler = packet_df.rename(
        columns={col: scaler_cols[i] for i, col in enumerate(FEATURE_COLUMNS)}
    )
    packet_df = pd.DataFrame(
        MODELS["scaler"].transform(packet_df_scaler),
        columns=FEATURE_COLUMNS
    )

    # Tier 2 (Isolation Forest) was trained with space-separated names
    tier2_df = packet_df.rename(columns=lambda c: c.replace("_", " "))
    ANOMALY_FEATURES_SPACED = [f.replace("_", " ") for f in ANOMALY_FEATURES]

    try:
        # ---- TIER 1: LightGBM ----
        t1_probs = MODELS["lgbm"].predict(packet_df)[0]
        predicted_idx = int(np.argmax(t1_probs))
        predicted_class = MODELS["class_names"][predicted_idx]
        confidence = float(t1_probs[predicted_idx])

        sorted_indices = np.argsort(t1_probs)[::-1][:5]
        top_classes = {
            MODELS["class_names"][i]: round(float(t1_probs[i]), 4)
            for i in sorted_indices
        }

        # ---- TIER 2: Isolation Forest ----
        t2_score = float(
            MODELS["iforest"].decision_function(tier2_df[ANOMALY_FEATURES_SPACED])[0]
        )
        t2_anomalous = t2_score < MODELS["anomaly_threshold"]

        # ---- TIER 3: PPO RL Agent ----
        obs = build_observation(t1_probs, t2_score)
        action_raw, _ = MODELS["ppo"].predict(obs, deterministic=True)
        action_code = int(action_raw)
        action_map = {0: "ALLOW", 1: "THROTTLE", 2: "DROP", 3: "HONEYPOT"}
        action = action_map.get(action_code, "DROP")

        # ---- SHAP Explainability ----
        shap_verdict = None
        shap_top_features = None

        if payload.include_shap and (predicted_class != "BENIGN" or t2_anomalous):
            explanation = MODELS["explainer"].explain_packet(packet_df, top_n=5)
            shap_verdict = explanation["verdict"]
            shap_top_features = [
                SHAPFeature(**f) for f in explanation["top_features"]
            ]

        # ---- Alert Level ----
        alert_level = compute_alert_level(
            predicted_class, confidence, t2_anomalous, action
        )

        # ---- Fire Alert (ORANGE / RED only) ----
        ALERT_MANAGER.process(
            alert_level=alert_level,
            threat_class=predicted_class,
            confidence=confidence,
            action=action,
            anomaly_score=t2_score,
            shap_verdict=shap_verdict,
            shap_top_features=shap_top_features,
        )

        processing_ms = round((time.perf_counter() - start) * 1000, 2)

        return AnalysisResponse(
            tier1_predicted_class=predicted_class,
            tier1_confidence=round(confidence, 4),
            tier1_top_classes=top_classes,
            tier2_anomaly_score=round(t2_score, 4),
            tier2_is_anomalous=t2_anomalous,
            tier3_action=action,
            tier3_action_code=action_code,
            shap_verdict=shap_verdict,
            shap_top_features=shap_top_features,
            processing_ms=processing_ms,
            system_alert_level=alert_level
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference pipeline error: {str(e)}"
        )

@app.post("/analyze/batch", response_model=List[AnalysisResponse])
def analyze_batch(packets: List[PacketInput]):
    """
    Analyzes multiple packets in one request.
    SHAP is skipped for batch requests to keep latency reasonable.
    """
    if len(packets) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 100 packets per request."
        )

    results = []
    for packet in packets:
        # Force SHAP off for batch
        packet.include_shap = False
        results.append(analyze_packet(packet))

    return results

@app.get("/alerts")
def get_recent_alerts(limit: int = 50):
    """
    Returns the most recent ORANGE/RED alerts fired by the RL agent.
    Used by the dashboard to populate the live alert feed.
    """
    return {
        "alerts": ALERT_MANAGER.get_recent(limit=limit),
        "counts": ALERT_MANAGER.get_counts(),
    }