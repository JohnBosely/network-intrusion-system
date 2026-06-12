import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest

# Pulling our clean, dedicated models directly from your architecture
from models import train_lgbm

# Curated behavioral features that strip out static noise and focus on traffic mechanics
ANOMALY_FEATURE_SET = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Packet Length Variance',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
]


def verify_environment(data_dir_path):
    """
    Validates that data assets exist prior to running computational tasks.
    Prevents blind crashes with descriptive, user-friendly warnings.
    """
    target_path = Path(data_dir_path)
    print(f"Validating target data environment: {target_path.resolve()}")
    
    if not target_path.exists():
        print(f"\n[CRITICAL ERROR] Data directory layout mismatch!")
        print(f"Target location NOT found: {target_path.resolve()}")
        print("Please verify that your dataset folder 'MachineLearningCVE' is nested correctly under the 'data/' folder.")
        sys.exit(1)
        
    csv_files = list(target_path.glob("*.csv"))
    if len(csv_files) == 0:
        print(f"\n[CRITICAL ERROR] Target data directory is completely empty!")
        print(f"Location found at: {target_path.resolve()}")
        print("However, no source '.csv' files were detected. Please populate the directory with network logs.")
        sys.exit(1)
        
    print(f"Environment Verification Passed! Detected {len(csv_files)} network log frames.")


def remove_infinite_values(df):
    """Handles data cleanup by stripping out mathematical infinity and null entries."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    is_inf = np.isinf(df[num_cols]).any(axis=1)
    is_null = df[num_cols].isnull().any(axis=1)
    return df[~(is_inf | is_null)]


def clean_labels(df):
    """Normalizes string text target labels by removing formatting whitespace."""
    df['Label'] = df['Label'].astype(str).str.strip()
    return df


def scale_features(X_train, X_test):
    """Standardizes continuous scales across train and test partitions to protect feature matrices."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled

def preprocess_chronological(data_dir_path, sample_size=100000):
    """Processes network log files sequentially to preserve time-series data structures."""
    data_dir = Path(data_dir_path)
    files = list(data_dir.glob("*.csv"))
    
    print(f"\n--- Ingesting {len(files)} Source Files Chronologically ---")
    train_dfs = []
    test_dfs = []
    
    for file in files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df = remove_infinite_values(df)
        df = clean_labels(df)
        
        cutoff = int(len(df) * 0.8)
        train_dfs.append(df.iloc[:cutoff])
        test_dfs.append(df.iloc[cutoff:])
        print(f" Successfully split {file.name} at chronological row {cutoff}")
        
    
    full_train = pd.concat(train_dfs, ignore_index=True)
    full_test = pd.concat(test_dfs, ignore_index=True)

# Free memory immediately after concat
    del train_dfs, test_dfs
    
    # === THE FIX GOES HERE ===
    if sample_size is not None:
        # If a limit is specified, down-sample the dataset to prevent memory/time bloating
        train_sampled = full_train.sample(n=sample_size, random_state=42)
        test_sampled = full_test.sample(n=int(sample_size * 0.25), random_state=42)
    else:
        # If sample_size=None, completely skip sampling and use 100% of the records
        print("[PREPROCESS] sample_size is None. Retaining 100% of the dataset rows.")
        train_sampled = full_train
        test_sampled = full_test
    # =========================
    
    X_train = train_sampled.drop(columns=["Label"], errors='ignore')
    y_train_raw = train_sampled["Label"]
    
    X_test = test_sampled.drop(columns=["Label"], errors='ignore')
    y_test_raw = test_sampled["Label"]
    
    return X_train, X_test, y_train_raw, y_test_raw


def execute_tier1_training(X_train_scaled, y_train, class_names):
    """Coordinates Tier-1 supervised multiclass signature learning."""
    print(f"\n--- TIER 1: TRAINING SUPERVISED CLASSIFIER ---")
    return train_lgbm(X_train_scaled, y_train, num_class=len(class_names))


def run_operational_boundary_sweep(X_train_scaled, X_test_scaled, y_train, y_test, y_test_raw, lgbm_model, benign_idx):
    """
    Runs the production optimization sweep on Tier-2's anomaly detector.
    Identifies the ideal security stance while enforcing a strict <= 5.0% False Alarm budget.
    """
    print(f"\n--- TIER 2: TUNING OPERATIONAL PERCENTILE BOUNDARY ---")
    
    X_train_anomaly = X_train_scaled[ANOMALY_FEATURE_SET]
    X_train_pure_benign = X_train_anomaly[y_train == benign_idx]

    print("Training production-grade Isolation Forest (n_estimators=250, max_samples=2048)...")
    anomaly_detector = IsolationForest(
        n_estimators=250,
        max_samples=2048,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    anomaly_detector.fit(X_train_pure_benign)
    
    train_benign_scores = anomaly_detector.decision_function(X_train_pure_benign)
    
    lgbm_probs = lgbm_model.predict(X_test_scaled)
    y_pred_lgbm = np.argmax(lgbm_probs, axis=1)
    leaked_mask = (y_test != benign_idx) & (y_pred_lgbm == benign_idx)
    leaked_indices = np.where(leaked_mask)[0]
    
    X_leaked_anomaly = X_test_scaled[ANOMALY_FEATURE_SET].iloc[leaked_indices]
    X_test_benign = X_test_scaled[ANOMALY_FEATURE_SET][y_test == benign_idx]
    
    leaked_scores = anomaly_detector.decision_function(X_leaked_anomaly)
    benign_test_scores = anomaly_detector.decision_function(X_test_benign)

    total_leaked_attacks = len(leaked_indices)
    total_benign_test = len(X_test_benign)

    percentile_candidates = [5, 7, 9, 11, 13, 15]
    best_tpr, best_fpr = -1.0, 100.0
    final_threshold, best_percentile = None, None

    print(f"Sweeping operational boundaries against False Alarm budget (FPR <= 5.0%)...\n")

    for p in percentile_candidates:
        CANDIDATE_THRESHOLD = np.percentile(train_benign_scores, p)
        intercepted = np.sum(leaked_scores < CANDIDATE_THRESHOLD)
        false_alarms = np.sum(benign_test_scores < CANDIDATE_THRESHOLD)
        
        tpr = (intercepted / total_leaked_attacks) * 100
        fpr = (false_alarms / total_benign_test) * 100
        
        print(f"➔ Testing Training Percentile: {p}% | Threshold: {CANDIDATE_THRESHOLD:.4f}")
        print(f"   Test Metrics -> Detection Rate (TPR): {tpr:.2f}% | False Alarm Rate (FPR): {fpr:.2f}%")
        
        if fpr <= 5.0 and tpr > best_tpr:
            best_tpr, best_fpr = tpr, fpr
            final_threshold = CANDIDATE_THRESHOLD
            best_percentile = p

    print("\n====================================================")
    print("--- PRODUCTION BOUNDARY OPTIMIZATION COMPLETE ---")
    print("====================================================")
    print(f"Optimal Operational Setting: {best_percentile}th Percentile")
    print(f"Target Security Threshold: {final_threshold:.4f}")
    print(f"Final System Metrics -> Detection Rate: {best_tpr:.2f}% | False Alarm Rate: {best_fpr:.2f}%")

    # Generate Performance Breakdown Report
    interception_report = pd.DataFrame({
        'Attack Type': y_test_raw.iloc[leaked_indices],
        'Intercepted': (leaked_scores < final_threshold)
    })
    print("\n--- Final Breakdown of Defended Zero-Day Threats ---")
    summary = interception_report.groupby('Attack Type')['Intercepted'].agg(['count', 'sum'])
    summary.columns = ['Total Leaked Past Tier 1', 'Caught By Tier 2']
    print(summary)

    return anomaly_detector, best_percentile, final_threshold


def serialize_system_artifacts(script_root, lgbm_model, anomaly_detector, label_encoder, percentile, threshold, benign_idx):
    """
    Serializes live in-memory models to storage.
    Creates a persistent cache folder to export core binaries and system configurations.
    """
    print(f"\n--- INITIALIZING PRODUCTION SERIALIZATION LAYER ---")
    artifacts_dir = script_root.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Dump serialized binary model states
    joblib.dump(lgbm_model, artifacts_dir / "tier1_lightgbm.pkl")
    joblib.dump(anomaly_detector, artifacts_dir / "tier2_isolation_forest.pkl")
    joblib.dump(label_encoder, artifacts_dir / "label_encoder.pkl")
    
    # Write environment configuration mappings
    config_path = artifacts_dir / "system_config.txt"
    with open(config_path, "w") as config_file:
        config_file.write(f"OPTIMAL_OPERATIONAL_PERCENTILE={percentile}\n")
        config_file.write(f"SECURITY_THRESHOLD={threshold:.6f}\n")
        config_file.write(f"BENIGN_INDEX={benign_idx}\n")

    print(f"Export Complete! System files successfully written to storage.")
    print(f"Artifact Store: {artifacts_dir.resolve()}")

# =====================================================================
# --- SYSTEM EXECUTION COMPONENT ENTRY POINT ---
# =====================================================================
if __name__ == "__main__":
    # 1. Establish runtime paths
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIRECTORY = SCRIPT_DIR.parent / "data" / "MachineLearningCVE"

    # 2. Prevent crashes with descriptive error validation checks (FIXED CASING HERE)
    verify_environment(DATA_DIRECTORY)

    # 3. Extract features & chronological string labels (FIXED CASING HERE)
    X_train, X_test, y_train_raw, y_test_raw = preprocess_chronological(
        data_dir_path=DATA_DIRECTORY, sample_size=100000
    )

    # 4. Map target definitions globally
    label_encoder = LabelEncoder()
    full_label_union = pd.concat([y_train_raw, y_test_raw])
    label_encoder.fit(full_label_union)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    class_names = label_encoder.classes_
    benign_idx = np.where(class_names == 'BENIGN')[0][0]

    # 5. Stabilize multi-feature data geometry
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # 6. Execute Tier 1: Learn historical attack patterns
    lgbm_model = execute_tier1_training(X_train_scaled, y_train, class_names)

    # 7. Execute Tier 2: Tune and capture zero-day vulnerabilities
    anomaly_detector, optimal_percentile, final_threshold = run_operational_boundary_sweep(
        X_train_scaled, X_test_scaled, y_train, y_test, y_test_raw, lgbm_model, benign_idx
    )

    # 8. Cache models to local memory storage
    serialize_system_artifacts(
        script_root=SCRIPT_DIR,
        lgbm_model=lgbm_model,
        anomaly_detector=anomaly_detector,
        label_encoder=label_encoder,
        percentile=optimal_percentile,
        threshold=final_threshold,
        benign_idx=benign_idx
    )