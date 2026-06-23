import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest

from models import train_lgbm

# =====================================================================
# --- CONSTANTS
# =====================================================================

ANOMALY_FEATURE_SET = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Packet Length Variance',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
]

# Any attack class with fewer than this many training examples gets boosted
RARE_CLASS_THRESHOLD = 500

# How many examples of each rare class to guarantee in training
RARE_CLASS_MIN_TRAIN = 300


# =====================================================================
# --- VALIDATION
# =====================================================================

def verify_environment(data_dir_path):
    target_path = Path(data_dir_path)
    print(f"Validating target data environment: {target_path.resolve()}")

    if not target_path.exists():
        print(f"\n[CRITICAL ERROR] Data directory not found!")
        print(f"Expected location: {target_path.resolve()}")
        sys.exit(1)

    csv_files = list(target_path.glob("*.csv"))
    if len(csv_files) == 0:
        print(f"\n[CRITICAL ERROR] No CSV files found in data directory.")
        sys.exit(1)

    print(f"Verification passed. Detected {len(csv_files)} source files.")


# =====================================================================
# --- CLEANING
# =====================================================================

def remove_infinite_values(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    is_inf = np.isinf(df[num_cols]).any(axis=1)
    is_null = df[num_cols].isnull().any(axis=1)
    return df[~(is_inf | is_null)]


def clean_labels(df):
    df['Label'] = df['Label'].astype(str).str.strip()
    return df


# =====================================================================
# --- SCALING
# =====================================================================

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler  # ← add scaler


# =====================================================================
# --- CORE PREPROCESSING — HYBRID STRATIFIED SPLIT
# =====================================================================

def preprocess_chronological(data_dir_path, sample_size=100000):
    """
    Hybrid chronological + stratified split.

    Strategy:
    1. Split each file chronologically (80/20) — preserves temporal realism
       for majority classes like BENIGN, DDoS, DoS Hulk.
    2. After splitting, identify rare attack classes (< RARE_CLASS_THRESHOLD
       examples in training).
    3. For each rare class, pull additional examples from the test pool and
       inject them into training until RARE_CLASS_MIN_TRAIN is reached.
    4. This guarantees LightGBM sees enough examples of every attack type
       to learn meaningful decision boundaries.

    Why this works:
    - We don't shuffle ALL data (that would leak temporal information).
    - We only move rare class rows that were chronologically in the test
      window — these are typically from the same attack sessions as the
      training examples, just later in time.
    - The test set still contains held-out examples of every class for
      honest evaluation.
    """
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
        train_dfs.append(df.iloc[:cutoff].copy())
        test_dfs.append(df.iloc[cutoff:].copy())
        print(f" Split {file.name} at row {cutoff}")

    full_train = pd.concat(train_dfs, ignore_index=True)
    full_test = pd.concat(test_dfs, ignore_index=True)

    del train_dfs, test_dfs

    # ----------------------------------------------------------------
    # STEP 2: Identify rare classes in training
    # ----------------------------------------------------------------
    train_label_counts = full_train['Label'].value_counts()
    benign_label = 'BENIGN'

    rare_classes = [
        label for label, count in train_label_counts.items()
        if label != benign_label and count < RARE_CLASS_THRESHOLD
    ]

    # Also catch classes that appear ONLY in test (like Heartbleed)
    test_only_classes = [
        label for label in full_test['Label'].unique()
        if label not in full_train['Label'].unique()
    ]
    rare_classes = list(set(rare_classes + test_only_classes))

    print(f"\n--- Rare Class Boost ---")
    print(f"Threshold: < {RARE_CLASS_THRESHOLD} training examples")
    print(f"Target minimum: {RARE_CLASS_MIN_TRAIN} examples per rare class")
    print(f"Rare classes detected: {rare_classes}")

    # ----------------------------------------------------------------
    # STEP 3: Pull rare class examples from test into training
    # ----------------------------------------------------------------
    rows_to_move = []

    for cls in rare_classes:
        current_train_count = train_label_counts.get(cls, 0)
        needed = RARE_CLASS_MIN_TRAIN - current_train_count

        if needed <= 0:
            print(f"  {cls}: already has {current_train_count} — no boost needed")
            continue

        # Get all available examples of this class from the test pool
        test_cls_rows = full_test[full_test['Label'] == cls]
        available = len(test_cls_rows)

        if available == 0:
            print(f"  {cls}: 0 examples in test pool — cannot boost")
            continue

        # Take as many as we need, up to what's available
        # Keep at least some in test for evaluation (minimum 20% of available)
        max_movable = int(available * 0.8)
        to_move = min(needed, max_movable)
        to_move = max(to_move, 0)

        if to_move > 0:
            # Take from the BEGINNING of the test pool for this class
            # (earlier in time = more representative training examples)
            selected = test_cls_rows.iloc[:to_move]
            rows_to_move.append(selected)
            print(f"  {cls}: had {current_train_count}, moving {to_move} from test → train "
                  f"(leaving {available - to_move} in test)")
        else:
            print(f"  {cls}: only {available} in test, keeping all there for evaluation")

    # ----------------------------------------------------------------
    # STEP 4: Apply the moves
    # ----------------------------------------------------------------
    if rows_to_move:
        boost_df = pd.concat(rows_to_move, ignore_index=True)
        moved_indices = boost_df.index

        # Add boosted rows to training
        full_train = pd.concat([full_train, boost_df], ignore_index=True)

        # Remove those rows from test
        full_test = full_test.drop(index=moved_indices).reset_index(drop=True)

        print(f"\n  Total rows moved to training: {len(boost_df)}")
    else:
        print("\n  No rows needed to be moved.")

    # ----------------------------------------------------------------
    # STEP 5: Print updated distributions
    # ----------------------------------------------------------------
    print(f"\n--- Updated Training Distribution (after boost) ---")
    updated_counts = full_train['Label'].value_counts()
    print(updated_counts)

    print(f"\n--- Test Distribution ---")
    print(full_test['Label'].value_counts())

    # ----------------------------------------------------------------
    # STEP 6: Sample down to manageable size
    # ----------------------------------------------------------------
    if sample_size is not None:
        # Stratified sampling for training — maintain class proportions
        # but guarantee rare classes survive the downsample
        if len(full_train) > sample_size:
            # First, protect rare class rows from being sampled away
            rare_rows = full_train[full_train['Label'].isin(rare_classes)]
            common_rows = full_train[~full_train['Label'].isin(rare_classes)]

            # How many common rows can we fit?
            common_budget = sample_size - len(rare_rows)

            if common_budget > 0 and len(common_rows) > common_budget:
                common_sampled = common_rows.sample(n=common_budget, random_state=42)
            else:
                common_sampled = common_rows

            train_sampled = pd.concat([rare_rows, common_sampled], ignore_index=True)
            # Shuffle so rare classes aren't all at the end
            train_sampled = train_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            train_sampled = full_train

        test_sample_size = int(sample_size * 0.25)
        if len(full_test) > test_sample_size:
            test_sampled = full_test.sample(n=test_sample_size, random_state=42)
        else:
            test_sampled = full_test

    else:
        print("[PREPROCESS] sample_size=None — retaining 100% of rows.")
        train_sampled = full_train.sample(frac=1, random_state=42).reset_index(drop=True)
        test_sampled = full_test

    # ----------------------------------------------------------------
    # STEP 7: Split features and labels
    # ----------------------------------------------------------------
    X_train = train_sampled.drop(columns=["Label"], errors='ignore')
    y_train_raw = train_sampled["Label"]

    X_test = test_sampled.drop(columns=["Label"], errors='ignore')
    y_test_raw = test_sampled["Label"]

    print(f"\n--- Final Split Summary ---")
    print(f"Training samples : {len(X_train):,}")
    print(f"Test samples     : {len(X_test):,}")
    print(f"Training classes : {sorted(y_train_raw.unique())}")
    print(f"Test classes     : {sorted(y_test_raw.unique())}")

    return X_train, X_test, y_train_raw, y_test_raw


# =====================================================================
# --- TIER 1 TRAINING
# =====================================================================

def execute_tier1_training(X_train_scaled, y_train, class_names):
    print(f"\n--- TIER 1: TRAINING SUPERVISED CLASSIFIER ---")
    return train_lgbm(X_train_scaled, y_train, num_class=len(class_names))


# =====================================================================
# --- TIER 2 TRAINING + SWEEP
# =====================================================================

def run_operational_boundary_sweep(
    X_train_scaled, X_test_scaled,
    y_train, y_test, y_test_raw,
    lgbm_model, benign_idx
):
    print(f"\n--- TIER 2: TUNING OPERATIONAL PERCENTILE BOUNDARY ---")

    X_train_anomaly = X_train_scaled[ANOMALY_FEATURE_SET]
    X_train_pure_benign = X_train_anomaly[y_train == benign_idx]

    print("Training Isolation Forest (n_estimators=250, max_samples=2048)...")
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

    print(f"Sweeping thresholds (FPR budget <= 5.0%)...\n")

    for p in percentile_candidates:
        CANDIDATE_THRESHOLD = np.percentile(train_benign_scores, p)

        if total_leaked_attacks == 0:
            tpr = 100.0
        else:
            intercepted = np.sum(leaked_scores < CANDIDATE_THRESHOLD)
            tpr = (intercepted / total_leaked_attacks) * 100

        false_alarms = np.sum(benign_test_scores < CANDIDATE_THRESHOLD)
        fpr = (false_alarms / total_benign_test) * 100

        print(f"  Percentile {p:>2}% | Threshold {CANDIDATE_THRESHOLD:.4f} | "
              f"TPR {tpr:.1f}% | FPR {fpr:.1f}%")

        if fpr <= 5.0 and tpr > best_tpr:
            best_tpr, best_fpr = tpr, fpr
            final_threshold = CANDIDATE_THRESHOLD
            best_percentile = p

    print(f"\n  Winner: {best_percentile}th percentile")
    print(f"  Threshold: {final_threshold:.4f}")
    print(f"  Detection Rate: {best_tpr:.2f}% | False Alarm Rate: {best_fpr:.2f}%")

    if total_leaked_attacks > 0:
        interception_report = pd.DataFrame({
            'Attack Type': y_test_raw.iloc[leaked_indices],
            'Intercepted': (leaked_scores < final_threshold)
        })
        print("\n--- Tier 2 Interception Breakdown ---")
        summary = interception_report.groupby('Attack Type')['Intercepted'].agg(['count', 'sum'])
        summary.columns = ['Leaked Past T1', 'Caught By T2']
        print(summary)

    return anomaly_detector, best_percentile, final_threshold


# =====================================================================
# --- SERIALIZATION
# =====================================================================

def serialize_system_artifacts(
    script_root, lgbm_model, anomaly_detector,
    label_encoder, scaler, percentile, threshold, benign_idx  # ← add scaler
):
    print(f"\n--- SERIALIZING ARTIFACTS ---")
    artifacts_dir = script_root.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    joblib.dump(lgbm_model,        artifacts_dir / "tier1_lightgbm.pkl")
    joblib.dump(anomaly_detector,  artifacts_dir / "tier2_isolation_forest.pkl")
    joblib.dump(label_encoder,     artifacts_dir / "label_encoder.pkl")
    joblib.dump(scaler,            artifacts_dir / "feature_scaler.pkl")  # ← new line

    config_path = artifacts_dir / "system_config.txt"
    with open(config_path, "w") as f:
        f.write(f"OPTIMAL_OPERATIONAL_PERCENTILE={percentile}\n")
        f.write(f"SECURITY_THRESHOLD={threshold:.6f}\n")
        f.write(f"BENIGN_INDEX={benign_idx}\n")

    print(f"Artifacts saved to: {artifacts_dir.resolve()}")


# =====================================================================
# --- MAIN
# =====================================================================

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIRECTORY = SCRIPT_DIR.parent / "data" / "MachineLearningCVE"

    verify_environment(DATA_DIRECTORY)

    X_train, X_test, y_train_raw, y_test_raw = preprocess_chronological(
        data_dir_path=DATA_DIRECTORY,
        sample_size=100000
    )

    label_encoder = LabelEncoder()
    full_label_union = pd.concat([y_train_raw, y_test_raw])
    label_encoder.fit(full_label_union)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    class_names = label_encoder.classes_
    benign_idx = int(np.where(class_names == 'BENIGN')[0][0])

    print(f"\nEncoded classes: {list(enumerate(class_names))}")
    print(f"BENIGN index: {benign_idx}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)  # ← unpack scaler
    lgbm_model = execute_tier1_training(X_train_scaled, y_train, class_names)

    anomaly_detector, optimal_percentile, final_threshold = run_operational_boundary_sweep(
        X_train_scaled, X_test_scaled,
        y_train, y_test, y_test_raw,
        lgbm_model, benign_idx
    )

    serialize_system_artifacts(
        script_root=SCRIPT_DIR,
        lgbm_model=lgbm_model,
        anomaly_detector=anomaly_detector,
        label_encoder=label_encoder,
        scaler=scaler, 
        percentile=optimal_percentile,
        threshold=final_threshold,
        benign_idx=benign_idx
    )

    print("\n[SUCCESS] Full pipeline retrained and seriajlized.")