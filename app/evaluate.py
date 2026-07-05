import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from stable_baselines3 import PPO

from preprocess import preprocess_chronological, scale_features
from sklearn.preprocessing import LabelEncoder
from env import FastNetworkDefenseEnv


# =====================================================================
# --- ACTION LABELS (for readable output)
# =====================================================================
ACTION_LABELS = {
    0: "ALLOW",
    1: "THROTTLE",
    2: "DROkPi",
    3: "HONEoYPOT"
}


def load_artifacts(artifacts_dir: Path):
    """Loads all saved models and config from the artifacts folder."""
    print("[EVAL] Loading saved artifacts...")

    label_encoder = joblib.load(artifacts_dir / "label_encoder.pkl")
    lgbm          = joblib.load(artifacts_dir / "tier1_lightgbm.pkl")
    iforest       = joblib.load(artifacts_dir / "tier2_isolation_forest.pkl")
    agent         = PPO.load(artifacts_dir / "tier3_ppo_agent_scaled")

    config = {}
    with open(artifacts_dir / "system_config.txt", "r") as f:
        for line in f:
            key, val = line.strip().split("=")
            config[key] = float(val) if "." in val else int(val)

    benign_idx = int(np.where(label_encoder.classes_ == "BENIGN")[0][0])

    print(f"  Classes detected : {list(label_encoder.classes_)}")
    print(f"  BENIGN index     : {benign_idx}")
    print(f"  System config    : {config}")
    print("[EVAL] All artifacts loaded successfully.\n")

    return label_encoder, lgbm, iforest, agent, benign_idx, config


def build_observation_matrix(lgbm, iforest, label_encoder, X_test_scaled, anomaly_features):
    """
    Pre-computes the full observation matrix for the test set.
    Same approach used in train.py — runs inference once up front
    instead of once per step inside the environment.
    """
    print("[EVAL] Running batch inference on test set...")

    t1_probs  = lgbm.predict(X_test_scaled)
    t2_scores = iforest.decision_function(X_test_scaled[anomaly_features])

    y_pred_test = np.argmax(t1_probs, axis=1)
    benign_idx_local = int(np.where(label_encoder.classes_ == "BENIGN")[0][0])
    is_predicted_attack = (y_pred_test != benign_idx_local).astype(float)

    window = 100
    rolling_threat = np.zeros(len(is_predicted_attack))
    for i in range(len(is_predicted_attack)):
        start = max(0, i - window)
        rolling_threat[i] = is_predicted_attack[start:i+1].mean()

    base_obs = np.hstack([
    t1_probs,
    t2_scores.reshape(-1, 1),
    rolling_threat.reshape(-1, 1)
]).astype(np.float32)

    WINDOW_SIZE = 5
    n_samples, n_features = base_obs.shape
    windowed_obs = np.zeros((n_samples, n_features * WINDOW_SIZE), dtype=np.float32)

    for i in range(n_samples):
        for w in range(WINDOW_SIZE):
            src_idx = max(0, i - (WINDOW_SIZE - 1 - w))
            windowed_obs[i, w * n_features:(w + 1) * n_features] = base_obs[src_idx]

    obs_matrix = windowed_obs

    print(f"  Observation matrix shape: {obs_matrix.shape}")
    print("[EVAL] Batch inference complete.\n")

    return obs_matrix


def run_agent_evaluation(agent, obs_matrix, y_test, benign_idx, class_names):
    """
    Runs the trained PPO agent across every packet in the test set.
    Records every action taken and compares it to the true label.
    Returns a full results DataFrame.
    """
    print("[EVAL] Running agent across test set...")

    total_packets = len(y_test)
    results = []

    for i in range(total_packets):
        obs        = obs_matrix[i].reshape(1, -1)
        action, _  = agent.predict(obs, deterministic=True)
        action     = int(action.item())

        true_label_idx = y_test[i]
        is_attack      = (true_label_idx != benign_idx)
        true_label_str = class_names[true_label_idx]

        # Determine if the action was correct
        # ALLOW on benign = correct, ALLOW on attack = miss
        # Any non-ALLOW on attack = some form of interception
        # Any non-ALLOW on benign = false alarm
        if not is_attack:
            correct = (action == 0)   # Correct only if ALLOW
        else:
            correct = (action != 0)   # Correct if any defensive action taken

        results.append({
            "true_label":  true_label_str,
            "is_attack":   is_attack,
            "action":      action,
            "action_name": ACTION_LABELS[action],
            "correct":     correct
        })

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1:,} / {total_packets:,} packets...")

    print(f"[EVAL] Agent evaluation complete. {total_packets:,} packets processed.\n")
    return pd.DataFrame(results)


def print_overall_metrics(results_df):
    """Prints top-level accuracy, attack recall, and false alarm rate."""
    total    = len(results_df)
    correct  = results_df["correct"].sum()
    accuracy = (correct / total) * 100

    attack_rows  = results_df[results_df["is_attack"]]
    benign_rows  = results_df[~results_df["is_attack"]]

    attack_recall = (attack_rows["correct"].sum() / len(attack_rows)) * 100 if len(attack_rows) > 0 else 0
    false_alarm   = ((~benign_rows["correct"]).sum() / len(benign_rows)) * 100 if len(benign_rows) > 0 else 0

    print("=" * 54)
    print("  OVERALL AGENT PERFORMANCE")
    print("=" * 54)
    print(f"  Total packets evaluated : {total:,}")
    print(f"  Overall accuracy        : {accuracy:.2f}%")
    print(f"  Attack recall           : {attack_recall:.2f}%  (caught attacks / total attacks)")
    print(f"  False alarm rate        : {false_alarm:.2f}%  (benign wrongly flagged)")
    print("=" * 54)
    print()


def print_action_distribution(results_df):
    """
    Prints how often each action was chosen overall.
    This is the most important diagnostic — if the agent is
    always picking the same action, the policy is degenerate.
    """
    total = len(results_df)

    print("=" * 54)
    print("  GLOBAL ACTION DISTRIBUTION")
    print("  (If one action dominates >90%, policy is degenerate)")
    print("=" * 54)

    counts = results_df["action"].value_counts().sort_index()
    for action_idx, count in counts.items():
        pct  = (count / total) * 100
        label = ACTION_LABELS[action_idx]
        bar  = "#" * int(pct / 2)
        print(f"  Action {action_idx} ({label:<8}) : {count:>6,}  ({pct:5.1f}%)  {bar}")

    print("=" * 54)
    print()


def print_action_by_attack_type(results_df):
    """
    Breaks down which actions the agent chose for each specific attack type.
    This reveals whether the agent learned attack-specific responses
    or just uses a blanket strategy for everything.
    """
    print("=" * 54)
    print("  ACTION BREAKDOWN BY ATTACK TYPE")
    print("  (Healthy: different attack types get different actions)")
    print("  (Unhealthy: all attack types get the same action)")
    print("=" * 54)

    attack_df = results_df[results_df["is_attack"]]

    for label in sorted(attack_df["true_label"].unique()):
        group = attack_df[attack_df["true_label"] == label]
        total = len(group)
        print(f"\n  [{label}]  ({total} packets)")

        action_counts = group["action"].value_counts().sort_index()
        for action_idx in range(4):
            count = action_counts.get(action_idx, 0)
            pct   = (count / total) * 100
            name  = ACTION_LABELS[action_idx]
            bar   = "#" * int(pct / 2)
            print(f"    {name:<10} : {count:>5,}  ({pct:5.1f}%)  {bar}")

    print()


def print_benign_action_distribution(results_df):
    """
    Shows what the agent does with benign traffic specifically.
    Ideally >90% ALLOW. If DROP or HONEYPOT dominate, the agent
    is being too aggressive and will disrupt real users.
    """
    benign_df = results_df[~results_df["is_attack"]]
    total     = len(benign_df)

    print("=" * 54)
    print("  BENIGN TRAFFIC HANDLING")
    print("  (Healthy: >90% ALLOW)")
    print("  (Unhealthy: large DROP or HONEYPOT on benign packets)")
    print("=" * 54)

    action_counts = benign_df["action"].value_counts().sort_index()
    for action_idx in range(4):
        count = action_counts.get(action_idx, 0)
        pct   = (count / total) * 100
        name  = ACTION_LABELS[action_idx]
        bar   = "#" * int(pct / 2)
        print(f"  {name:<10} : {count:>6,}  ({pct:5.1f}%)  {bar}")

    print("=" * 54)
    print()


def print_missed_attacks(results_df):
    """
    Shows which attack types the agent completely missed (ALLOWed through).
    These are your false negatives — the most dangerous failures.
    """
    attack_df = results_df[results_df["is_attack"]]
    missed_df = attack_df[attack_df["action"] == 0]   # ALLOW on an attack = miss

    print("=" * 54)
    print("  MISSED ATTACKS (agent chose ALLOW on real attacks)")
    print("=" * 54)

    if len(missed_df) == 0:
        print("  Zero missed attacks. Agent flagged every attack.\n")
        return

    missed_counts = missed_df["true_label"].value_counts()
    total_attacks = len(attack_df)
    total_missed  = len(missed_df)

    print(f"  Total missed: {total_missed:,} / {total_attacks:,}  ({(total_missed/total_attacks)*100:.1f}%)\n")

    for label, count in missed_counts.items():
        group_total = len(attack_df[attack_df["true_label"] == label])
        pct = (count / group_total) * 100
        print(f"  {label:<30} : {count:>4,} missed  ({pct:.1f}% of that type)")

    print()


def print_policy_verdict(results_df):
    """
    Prints a plain-English verdict on whether the agent's policy
    is degenerate, partially learned, or meaningfully learned.
    """
    total          = len(results_df)
    action_counts  = results_df["action"].value_counts()
    top_action_pct = (action_counts.iloc[0] / total) * 100

    attack_df     = results_df[results_df["is_attack"]]
    attack_recall = (attack_df["correct"].sum() / len(attack_df)) * 100 if len(attack_df) > 0 else 0

    benign_df   = results_df[~results_df["is_attack"]]
    false_alarm = ((~benign_df["correct"]).sum() / len(benign_df)) * 100 if len(benign_df) > 0 else 0

    print("=" * 54)
    print("  POLICY VERDICT")
    print("=" * 54)

    if top_action_pct > 90:
        dominant = ACTION_LABELS[action_counts.index[0]]
        print(f"  DEGENERATE POLICY DETECTED")
        print(f"  The agent chose {dominant} {top_action_pct:.1f}% of the time.")
        print(f"  It learned a lazy default rather than a real policy.")
        print(f"  Fix: retrain with adjusted reward function.")

    elif attack_recall > 70 and false_alarm < 10:
        print(f"  HEALTHY POLICY")
        print(f"  Attack recall {attack_recall:.1f}% with {false_alarm:.1f}% false alarms.")
        print(f"  The agent learned a meaningful defense strategy.")
        print(f"  Next step: SHAP explainability + FastAPI deployment.")

    elif attack_recall > 50:
        print(f"  PARTIALLY LEARNED POLICY")
        print(f"  Attack recall {attack_recall:.1f}% — catching most threats.")
        print(f"  False alarms at {false_alarm:.1f}% — may need tuning.")
        print(f"  Consider retraining with more steps or adjusted rewards.")

    else:
        print(f"  WEAK POLICY")
        print(f"  Attack recall only {attack_recall:.1f}%.")
        print(f"  The agent is not reliably detecting threats.")
        print(f"  Fix: check reward scaling and class imbalance handling.")

    print("=" * 54)
    print()


# =====================================================================
# --- MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":

    ANOMALY_FEATURES = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd Header Length', 'Bwd Header Length', 'Packet Length Variance',
        'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
    ]

    SCRIPT_DIR    = Path(__file__).resolve().parent
    ROOT_DIR      = SCRIPT_DIR.parent
    DATA_DIR      = ROOT_DIR / "data" / "MachineLearningCVE"
    ARTIFACTS_DIR = ROOT_DIR / "artifacts"

    print("=" * 54)
    print("  TIER 3 AGENT EVALUATION PIPELINE")
    print("=" * 54)
    print()

    # 1. Load everything from artifacts
    label_encoder, lgbm, iforest, agent, benign_idx, config = load_artifacts(ARTIFACTS_DIR)
    class_names = label_encoder.classes_

    # 2. Load and preprocess the TEST set (same split as training, different rows)
    print("[EVAL] Loading test partition...")
    X_train, X_test, y_train_raw, y_test_raw = preprocess_chronological(
        data_dir_path=DATA_DIR,
        sample_size=100000
    )

    # 3. Encode labels using the saved encoder
    y_test = label_encoder.transform(y_test_raw)
    print(f"  Test set size : {len(y_test):,} packets")
    print(f"  Attack packets: {(y_test != benign_idx).sum():,}")
    print(f"  Benign packets: {(y_test == benign_idx).sum():,}\n")

    # 4. Scale features using the same scaler as training
    _, X_test_scaled = scale_features(X_train, X_test)

    # 5. Build the observation matrix for the test set
    obs_matrix = build_observation_matrix(lgbm, iforest, label_encoder, X_test_scaled, ANOMALY_FEATURES)
    # 6. Run the agent across every test packet
    results_df = run_agent_evaluation(agent, obs_matrix, y_test, benign_idx, class_names)

    # 7. Print all diagnostic reports
    print_overall_metrics(results_df)
    print_action_distribution(results_df)
    print_benign_action_distribution(results_df)
    print_action_by_attack_type(results_df)
    print_missed_attacks(results_df)
    print_policy_verdict(results_df)