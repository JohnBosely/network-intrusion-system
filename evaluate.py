"""
evaluate.py — Full pipeline evaluation on the held-out test set.

Runs every test-set row through the EXACT inference path main.py uses
(raw features -> scaler.transform -> LightGBM -> IsolationForest -> PPO)
and reports:

  1. Tier 1 (LightGBM) per-class precision / recall / F1 + confusion matrix
  2. Tier 2 (Isolation Forest) catch-rate: of the attacks Tier 1 missed,
     how many did Tier 2 flag as anomalous?
  3. Tier 3 (PPO) action distribution: what does the RL agent actually do
     when it sees a real attack vs real benign traffic?
  4. Overall system alert-level breakdown (GREEN/YELLOW/ORANGE/RED), using
     the same compute_alert_level() logic as main.py

This is the script that produces real numbers for the README instead of
a 10-row anecdotal spot check.

USAGE
-----
Save in the project ROOT (same place as retrain.py), run from anywhere:

    python evaluate.py

Uses the SAME test split logic as retrain.py (chronological + rare-class
boost) so results reflect held-out data the model never trained on.
"""

import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent / "app"
if not _APP_DIR.exists():
    _APP_DIR = _THIS_FILE.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from preprocess import preprocess_chronological, verify_environment

# =====================================================================
# --- CONSTANTS (must match main.py exactly)
# =====================================================================

ANOMALY_FEATURE_SET = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Packet Length Variance',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
]

WINDOW_SIZE = 5

# How many test rows to evaluate. None = use the full held-out test split.
# Start with a number for a fast sanity check; rerun with None for the
# real README numbers once you're confident the pipeline is solid.
EVAL_SAMPLE_SIZE = None

# Same SAMPLE_SIZE used to build the train/test split in retrain.py —
# MUST match what you actually trained with, or the test split won't
# line up with what the model has/hasn't seen.
TRAIN_SAMPLE_SIZE = 200000


def compute_alert_level(predicted_class, confidence, t2_anomalous, action):
    """Mirrors main.py's compute_alert_level() exactly."""
    if predicted_class == "BENIGN" and not t2_anomalous:
        return "GREEN"
    if t2_anomalous and predicted_class == "BENIGN":
        return "ORANGE"
    if predicted_class in ("DDoS", "DoS GoldenEye", "DoS Hulk", "Heartbleed"):
        return "RED"
    if action in ("DROP", "HONEYPOT"):
        return "ORANGE"
    if action == "ALLOW":
        return "YELLOW"
    return "YELLOW"


def build_observation(t1_probs_row, t2_score):
    """Mirrors main.py's build_observation() — single packet, no history,
    so all 5 window slots get the same current-packet signal."""
    single_obs = np.concatenate([
        t1_probs_row,
        np.array([t2_score]),
        np.array([0.0])
    ]).astype(np.float32)
    return np.tile(single_obs, WINDOW_SIZE).astype(np.float32)


def main():
    t_start = time.perf_counter()

    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "app" else SCRIPT_DIR
    DATA_DIR = ROOT_DIR / "data" / "MachineLearningCVE"
    ARTIFACTS_DIR = ROOT_DIR / "artifacts"

    print("=" * 70)
    print("EVALUATE.PY — Full system evaluation on held-out test set")
    print("=" * 70)

    verify_environment(DATA_DIR)

    # =================================================================
    # Load artifacts — exactly what main.py loads at startup
    # =================================================================
    print("\nLoading artifacts...")
    scaler = joblib.load(ARTIFACTS_DIR / "feature_scaler.pkl")
    lgbm = joblib.load(ARTIFACTS_DIR / "tier1_lightgbm.pkl")
    iforest = joblib.load(ARTIFACTS_DIR / "tier2_isolation_forest.pkl")
    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    class_names = list(label_encoder.classes_)
    benign_idx = int(np.where(np.array(class_names) == "BENIGN")[0][0])

    config = {}
    with open(ARTIFACTS_DIR / "system_config.txt", "r") as f:
        for line in f:
            key, val = line.strip().split("=")
            config[key] = val
    anomaly_threshold = float(config["SECURITY_THRESHOLD"])

    ppo = None
    try:
        from stable_baselines3 import PPO
        ppo = PPO.load(ARTIFACTS_DIR / "tier3_ppo_agent_scaled")
        print("PPO agent loaded — Tier 3 evaluation enabled.")
    except Exception as e:
        print(f"[WARNING] Could not load PPO agent ({e}). "
              f"Tier 3 / alert-level breakdown will be skipped.")

    print(f"Classes: {class_names}")
    print(f"BENIGN index: {benign_idx}")
    print(f"Anomaly threshold: {anomaly_threshold:.4f}")

    # =================================================================
    # Rebuild the SAME train/test split used during training
    # =================================================================
    print("\n" + "=" * 70)
    print("Rebuilding held-out test split (same logic as retrain.py)")
    print("=" * 70)

    X_train, X_test, y_train_raw, y_test_raw = preprocess_chronological(
        data_dir_path=DATA_DIR,
        sample_size=TRAIN_SAMPLE_SIZE
    )

    if EVAL_SAMPLE_SIZE is not None and len(X_test) > EVAL_SAMPLE_SIZE:
        idx = X_test.sample(n=EVAL_SAMPLE_SIZE, random_state=42).index
        X_test = X_test.loc[idx]
        y_test_raw = y_test_raw.loc[idx]

    print(f"\nEvaluating on {len(X_test):,} held-out test rows.")
    print(f"Test set class distribution:")
    print(y_test_raw.value_counts())

    # =================================================================
    # Run the FULL pipeline, row by row, exactly as main.py does
    # =================================================================
    print("\n" + "=" * 70)
    print("Running full 3-tier pipeline on test set...")
    print("=" * 70)

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Tier 1 — batch predict is fine, LightGBM doesn't need row-by-row
    t1_probs_all = lgbm.predict(X_test_scaled)
    t1_pred_idx = np.argmax(t1_probs_all, axis=1)
    t1_pred_labels = np.array([class_names[i] for i in t1_pred_idx])
    t1_confidence = np.max(t1_probs_all, axis=1)

    # Tier 2 — batch predict
    t2_scores_all = iforest.decision_function(X_test_scaled[ANOMALY_FEATURE_SET])
    t2_anomalous_all = t2_scores_all < anomaly_threshold

    y_true = y_test_raw.values

    # =================================================================
    # SECTION 1 — Tier 1 classification report
    # =================================================================
    print("\n" + "=" * 70)
    print("SECTION 1 — Tier 1 (LightGBM) Classification Report")
    print("=" * 70)

    try:
        from sklearn.metrics import (
            classification_report, confusion_matrix, accuracy_score
        )
        report = classification_report(
            y_true, t1_pred_labels, zero_division=0, digits=4
        )
        print(report)

        overall_acc = accuracy_score(y_true, t1_pred_labels)
        print(f"Overall accuracy: {overall_acc:.4f}")

        print("\n--- Confusion Matrix (rows=true, cols=predicted) ---")
        labels_present = sorted(set(y_true) | set(t1_pred_labels))
        cm = confusion_matrix(y_true, t1_pred_labels, labels=labels_present)
        cm_df = pd.DataFrame(cm, index=labels_present, columns=labels_present)
        with pd.option_context('display.max_columns', None, 'display.width', 200):
            print(cm_df)
    except ImportError:
        print("[WARNING] sklearn.metrics not available, skipping detailed report.")
        overall_acc = float(np.mean(t1_pred_labels == y_true))
        print(f"Overall accuracy: {overall_acc:.4f}")

    # =================================================================
    # SECTION 2 — Tier 2 catch-rate (the whole point of the architecture)
    # =================================================================
    print("\n" + "=" * 70)
    print("SECTION 2 — Tier 2 (Isolation Forest) Catch Rate")
    print("=" * 70)

    is_attack = y_true != "BENIGN"
    t1_missed = is_attack & (t1_pred_labels == "BENIGN")
    n_total_attacks = int(is_attack.sum())
    n_missed_by_t1 = int(t1_missed.sum())

    print(f"Total real attacks in test set : {n_total_attacks}")
    print(f"Missed by Tier 1 (said BENIGN) : {n_missed_by_t1} "
          f"({n_missed_by_t1 / n_total_attacks * 100:.2f}% of attacks)" if n_total_attacks else "")

    if n_missed_by_t1 > 0:
        caught_by_t2 = int(t2_anomalous_all[t1_missed].sum())
        print(f"Of those Tier-1 misses, caught by Tier 2: {caught_by_t2} "
              f"({caught_by_t2 / n_missed_by_t1 * 100:.2f}%)")
        print(f"Slipped past BOTH tiers (true blind spot): "
              f"{n_missed_by_t1 - caught_by_t2} "
              f"({(n_missed_by_t1 - caught_by_t2) / n_total_attacks * 100:.2f}% of all attacks)")

        print("\n--- Breakdown of Tier-1-missed attacks by type ---")
        missed_types = pd.Series(y_true[t1_missed]).value_counts()
        caught_mask_within_missed = t2_anomalous_all[t1_missed]
        breakdown = pd.DataFrame({
            'Missed_by_T1': missed_types,
        })
        for cls in missed_types.index:
            cls_mask = (y_true == cls) & t1_missed
            breakdown.loc[cls, 'Caught_by_T2'] = int(t2_anomalous_all[cls_mask].sum())
        breakdown['Caught_by_T2'] = breakdown['Caught_by_T2'].astype(int)
        breakdown['T2_catch_rate_%'] = (
            breakdown['Caught_by_T2'] / breakdown['Missed_by_T1'] * 100
        ).round(1)
        print(breakdown)
    else:
        print("Tier 1 missed zero attacks in this sample — Tier 2 had nothing to catch.")

    # False alarm rate — pure BENIGN traffic Tier 2 flags as anomalous
    benign_mask = y_true == "BENIGN"
    if benign_mask.sum() > 0:
        t2_false_alarms = int(t2_anomalous_all[benign_mask].sum())
        fpr = t2_false_alarms / benign_mask.sum() * 100
        print(f"\nTier 2 false alarm rate on real BENIGN traffic: "
              f"{t2_false_alarms}/{benign_mask.sum()} ({fpr:.2f}%)")

    # =================================================================
    # SECTION 3 — Overall system effective detection rate
    # =================================================================
    print("\n" + "=" * 70)
    print("SECTION 3 — Overall System Detection Rate (Tier 1 OR Tier 2)")
    print("=" * 70)

    system_caught = (t1_pred_labels != "BENIGN") | t2_anomalous_all
    if n_total_attacks > 0:
        overall_caught = int((system_caught & is_attack).sum())
        print(f"Attacks flagged by the SYSTEM (T1 correct OR T2 anomalous): "
              f"{overall_caught}/{n_total_attacks} "
              f"({overall_caught / n_total_attacks * 100:.2f}%)")
    if benign_mask.sum() > 0:
        system_false_alarms = int((system_caught & benign_mask).sum())
        print(f"System false alarm rate on BENIGN traffic: "
              f"{system_false_alarms}/{benign_mask.sum()} "
              f"({system_false_alarms / benign_mask.sum() * 100:.2f}%)")

    # =================================================================
    # SECTION 4 — Tier 3 (PPO) action distribution + alert levels
    # =================================================================
    if ppo is not None:
        print("\n" + "=" * 70)
        print("SECTION 4 — Tier 3 (PPO) Action Distribution + Alert Levels")
        print("=" * 70)
        print("(Running PPO row-by-row — this is the slow part, sampling "
              "if test set is large)")

        ppo_sample_size = min(5000, len(X_test_scaled))
        sample_idx = np.random.RandomState(42).choice(
            len(X_test_scaled), size=ppo_sample_size, replace=False
        )

        action_map = {0: "ALLOW", 1: "THROTTLE", 2: "DROP", 3: "HONEYPOT"}
        actions = []
        alert_levels = []

        for i in sample_idx:
            t1_row = t1_probs_all[i]
            t2_score = t2_scores_all[i]
            obs = build_observation(t1_row, t2_score)
            action_code, _ = ppo.predict(obs, deterministic=True)
            action = action_map.get(int(action_code), "DROP")
            actions.append(action)

            level = compute_alert_level(
                predicted_class=t1_pred_labels[i],
                confidence=t1_confidence[i],
                t2_anomalous=bool(t2_anomalous_all[i]),
                action=action
            )
            alert_levels.append(level)

        actions = np.array(actions)
        alert_levels = np.array(alert_levels)
        sample_true = y_true[sample_idx]
        sample_is_attack = sample_true != "BENIGN"

        print(f"\nSampled {ppo_sample_size} rows for Tier 3 evaluation.\n")

        print("--- PPO action distribution overall ---")
        print(pd.Series(actions).value_counts())

        print("\n--- PPO action distribution, real attacks only ---")
        if sample_is_attack.sum() > 0:
            print(pd.Series(actions[sample_is_attack]).value_counts())
        else:
            print("(no attack rows in this sample)")

        print("\n--- PPO action distribution, real BENIGN only ---")
        print(pd.Series(actions[~sample_is_attack]).value_counts())

        print("\n--- System alert level distribution ---")
        print(pd.Series(alert_levels).value_counts())

        print("\n--- Alert level vs ground truth (rows=true class, cols=alert) ---")
        alert_cross = pd.crosstab(sample_true, alert_levels)
        print(alert_cross)
    else:
        print("\n[SKIPPED] Section 4 — PPO agent not loaded.")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"[DONE] Evaluation completed in {elapsed / 60:.1f} minutes.")
    print("=" * 70)


if __name__ == "__main__":
    main()
