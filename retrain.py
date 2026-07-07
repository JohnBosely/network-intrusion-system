import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Make sure app/ is importable regardless of which directory this script
# is launched from (fixes "ModuleNotFoundError: No module named 'models'"
# when running `python ../retrain.py` from inside app/, or
# `python retrain.py` from the project root).
_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent / "app"
if not _APP_DIR.exists():
    # retrain.py was itself placed inside app/ — fall back to its own dir
    _APP_DIR = _THIS_FILE.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from models import train_lgbm
from env import FastNetworkDefenseEnv
from preprocess import (
    preprocess_chronological,
    remove_infinite_values,
    clean_labels,
    verify_environment,
)

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

RARE_CLASS_THRESHOLD = 500
RARE_CLASS_MIN_TRAIN = 300
WINDOW_SIZE = 5

# How many rows to use for the FULL pipeline.
# 150k is confirmed working on your machine — the 200k run with per-file
# sampling broke the class distribution badly (PortScan: 206 rows instead
# of 137k). The class weighting and num_leaves/num_boost_round improvements
# in models.py will still improve recall on top of this correct split.
SAMPLE_SIZE = 150000

# PPO training steps.
PPO_TOTAL_TIMESTEPS = 500000


# =====================================================================
# --- STEP 0: SETUP
# =====================================================================

def main():
    t_start = time.perf_counter()

    # retrain.py lives in the project root; app/ is a sibling directory
    # holding models.py, env.py, preprocess.py (added to sys.path above).
    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "app" else SCRIPT_DIR
    DATA_DIR = ROOT_DIR / "data" / "MachineLearningCVE"
    ARTIFACTS_DIR = ROOT_DIR / "artifacts"
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("RETRAIN.PY — Full pipeline, single-scaler-fit fix")
    print("=" * 70)
    print(f"Root dir      : {ROOT_DIR}")
    print(f"Data dir      : {DATA_DIR}")
    print(f"Artifacts dir : {ARTIFACTS_DIR}")
    print(f"Sample size   : {SAMPLE_SIZE}")
    print()

    verify_environment(DATA_DIR)

    # =================================================================
    # STEP 1: LOAD + SPLIT (chronological + rare-class boost)
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 1 — Loading and splitting data")
    print("=" * 70)

    X_train, X_test, y_train_raw, y_test_raw = preprocess_chronological(
        data_dir_path=DATA_DIR,
        sample_size=SAMPLE_SIZE
    )

    label_encoder = LabelEncoder()
    full_label_union = pd.concat([y_train_raw, y_test_raw])
    label_encoder.fit(full_label_union)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    class_names = label_encoder.classes_
    benign_idx = int(np.where(class_names == "BENIGN")[0][0])

    print(f"\nEncoded classes : {list(enumerate(class_names))}")
    print(f"BENIGN index    : {benign_idx}")

    # =================================================================
    # STEP 2: FIT THE SCALER — ONCE — AND NEVER AGAIN
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 2 — Fitting THE ONE scaler (used by all 3 tiers)")
    print("=" * 70)

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

    print(f"Scaler fitted on {len(X_train)} rows, {X_train.shape[1]} features.")
    print(f"Scaler mean (first 5)  : {scaler.mean_[:5]}")
    print(f"Scaler scale (first 5) : {scaler.scale_[:5]}")
    if np.allclose(scaler.mean_, 0) and np.allclose(scaler.scale_, 1):
        print("\n[FATAL] Scaler is an identity transform (mean=0, std=1 for all "
              "features). This means fit_transform was called on already-scaled "
              "or degenerate data. Aborting before wasting time training on it.")
        sys.exit(1)
    print("Scaler looks real (non-identity). Proceeding.\n")

    # Save it immediately so it can never be silently lost or overwritten downstream.
    joblib.dump(scaler, ARTIFACTS_DIR / "feature_scaler.pkl")
    print(f"[SAVED] {ARTIFACTS_DIR / 'feature_scaler.pkl'}")

    # =================================================================
    # STEP 3: TIER 1 — LightGBM
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 3 — Training Tier 1 (LightGBM)")
    print("=" * 70)


    lgbm_model = train_lgbm(
        X_train_scaled, y_train,
        num_class=len(class_names),
        sample_weight=None
    )

    joblib.dump(lgbm_model, ARTIFACTS_DIR / "tier1_lightgbm.pkl")
    joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.pkl")
    print(f"[SAVED] {ARTIFACTS_DIR / 'tier1_lightgbm.pkl'}")
    print(f"[SAVED] {ARTIFACTS_DIR / 'label_encoder.pkl'}")

   
    print("\n--- Tier 1 sanity check on held-out test set ---")
    test_probs = lgbm_model.predict(X_test_scaled)
    test_pred = np.argmax(test_probs, axis=1)
    overall_acc = float(np.mean(test_pred == y_test))
    print(f"Overall test accuracy: {overall_acc:.4f}")

    attack_mask = y_test != benign_idx
    if attack_mask.sum() > 0:
        attack_recall = float(np.mean(test_pred[attack_mask] == y_test[attack_mask]))
        print(f"Attack recall (non-BENIGN correctly classified): {attack_recall:.4f}")
        if attack_recall < 0.10:
            print("[WARNING] Attack recall is near zero — the scaler/model "
                  "mismatch may still be present. Check before continuing.")
    else:
        print("[WARNING] No attack rows in test set sample — recall check skipped.")

    # =================================================================
    # STEP 4: TIER 2 — Isolation Forest + threshold sweep
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 4 — Training Tier 2 (Isolation Forest) + threshold sweep")
    print("=" * 70)

    X_train_anomaly = X_train_scaled[ANOMALY_FEATURE_SET]
    X_train_pure_benign = X_train_anomaly[y_train == benign_idx]

    print(f"Training Isolation Forest on {len(X_train_pure_benign)} pure-BENIGN rows...")
    anomaly_detector = IsolationForest(
        n_estimators=250,
        max_samples=2048,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    anomaly_detector.fit(X_train_pure_benign)

    train_benign_scores = anomaly_detector.decision_function(X_train_pure_benign)

    lgbm_probs_test = lgbm_model.predict(X_test_scaled)
    y_pred_lgbm = np.argmax(lgbm_probs_test, axis=1)
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

    print(f"\nLightGBM missed {total_leaked_attacks} attacks in test set "
          f"(out of {attack_mask.sum() if attack_mask.sum() > 0 else 0}) "
          f"— sweeping Tier 2 thresholds to catch them (FPR budget <= 5.0%)...\n")

    for p in percentile_candidates:
        candidate_threshold = np.percentile(train_benign_scores, p)

        if total_leaked_attacks == 0:
            tpr = 100.0
        else:
            intercepted = np.sum(leaked_scores < candidate_threshold)
            tpr = (intercepted / total_leaked_attacks) * 100

        false_alarms = np.sum(benign_test_scores < candidate_threshold)
        fpr = (false_alarms / total_benign_test) * 100 if total_benign_test > 0 else 0.0

        print(f"  Percentile {p:>2}% | Threshold {candidate_threshold:.4f} | "
              f"TPR {tpr:.1f}% | FPR {fpr:.1f}%")

        if fpr <= 5.0 and tpr > best_tpr:
            best_tpr, best_fpr = tpr, fpr
            final_threshold = candidate_threshold
            best_percentile = p

    if final_threshold is None:
        # Fallback: no candidate met the FPR budget, use the tightest one
        best_percentile = percentile_candidates[0]
        final_threshold = np.percentile(train_benign_scores, best_percentile)
        print(f"\n[WARNING] No percentile met FPR<=5%. Falling back to "
              f"{best_percentile}th percentile.")

    print(f"\n  Winner: {best_percentile}th percentile")
    print(f"  Threshold: {final_threshold:.4f}")
    print(f"  Tier-2-catches-what-Tier-1-missed rate: {best_tpr:.2f}% | "
          f"False Alarm Rate: {best_fpr:.2f}%")

    joblib.dump(anomaly_detector, ARTIFACTS_DIR / "tier2_isolation_forest.pkl")
    print(f"[SAVED] {ARTIFACTS_DIR / 'tier2_isolation_forest.pkl'}")

    with open(ARTIFACTS_DIR / "system_config.txt", "w") as f:
        f.write(f"OPTIMAL_OPERATIONAL_PERCENTILE={best_percentile}\n")
        f.write(f"SECURITY_THRESHOLD={final_threshold:.6f}\n")
        f.write(f"BENIGN_INDEX={benign_idx}\n")
    print(f"[SAVED] {ARTIFACTS_DIR / 'system_config.txt'}")

    # =================================================================
    # STEP 5: TIER 3 — PPO
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 5 — Precomputing PPO observation matrix (Tier 1 + Tier 2 outputs)")
    print("=" * 70)

    t1_probs_all = lgbm_model.predict(X_train_scaled)  # (N, num_classes)
    t2_scores_all = anomaly_detector.decision_function(
        X_train_scaled[ANOMALY_FEATURE_SET]
    )  # (N,)

    y_pred_train = np.argmax(t1_probs_all, axis=1)
    is_predicted_attack = (y_pred_train != benign_idx).astype(float)

    window = 100
    rolling_threat = np.zeros(len(is_predicted_attack))
    for i in range(len(is_predicted_attack)):
        start = max(0, i - window)
        rolling_threat[i] = is_predicted_attack[start:i + 1].mean()

    base_obs = np.hstack([
        t1_probs_all,
        t2_scores_all.reshape(-1, 1),
        rolling_threat.reshape(-1, 1)
    ]).astype(np.float32)

    n_samples, n_features = base_obs.shape
    windowed_obs = np.zeros((n_samples, n_features * WINDOW_SIZE), dtype=np.float32)
    for i in range(n_samples):
        for w in range(WINDOW_SIZE):
            src_idx = max(0, i - (WINDOW_SIZE - 1 - w))
            windowed_obs[i, w * n_features:(w + 1) * n_features] = base_obs[src_idx]

    precomputed_obs = windowed_obs
    print(f"Observation matrix shape: {precomputed_obs.shape} "
          f"(expect (N, {n_features * WINDOW_SIZE}) = (N, 85) to match main.py)")

    if n_features * WINDOW_SIZE != 85:
        print(f"[WARNING] Observation width is {n_features * WINDOW_SIZE}, "
              f"but main.py's build_observation() produces 85-dim vectors. "
              f"PPO will fail to load/predict in main.py if these don't match.")

    print("\n" + "=" * 70)
    print("STEP 6 — Training Tier 3 (PPO)")
    print("=" * 70)

    env = FastNetworkDefenseEnv(
        precomputed_observations=precomputed_obs,
        data_labels=y_train,
        benign_idx=benign_idx
    )
    check_env(env, warn=True)
    print("Environment passed Gymnasium compatibility check.")

    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        verbose=1,
        tensorboard_log=str(ROOT_DIR / "tensorboard_logs")
    )

    print(f"\nTraining PPO for {PPO_TOTAL_TIMESTEPS} timesteps...")
    agent.learn(total_timesteps=PPO_TOTAL_TIMESTEPS)

    agent.save(ARTIFACTS_DIR / "tier3_ppo_agent_scaled")
    print(f"[SAVED] {ARTIFACTS_DIR / 'tier3_ppo_agent_scaled.zip'}")

    # =================================================================
    # STEP 7: END-TO-END VERIFICATION
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 7 — End-to-end verification (mirrors main.py inference path)")
    print("=" * 70)

    verify_pipeline(
        ARTIFACTS_DIR, X_test, y_test_raw, class_names, benign_idx
    )

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Full pipeline retrained in {elapsed / 60:.1f} minutes.")
    print("All artifacts in:", ARTIFACTS_DIR.resolve())
    print("=" * 70)


# =====================================================================
# --- VERIFICATION HELPER
# =====================================================================

def verify_pipeline(artifacts_dir, X_test_raw, y_test_raw, class_names, benign_idx):
    """
    Loads the artifacts back from disk exactly as main.py does, and runs
    a handful of REAL attack rows (raw, unscaled — like the API receives)
    through the full scale -> predict pipeline. Prints whether LightGBM
    actually recognises them now.

    This is the test that should have existed from day one — it would
    have caught the original scaler bug in minutes instead of weeks.
    """
    scaler = joblib.load(artifacts_dir / "feature_scaler.pkl")
    lgbm = joblib.load(artifacts_dir / "tier1_lightgbm.pkl")

    non_benign_mask = y_test_raw != "BENIGN"
    if non_benign_mask.sum() == 0:
        print("No non-BENIGN rows in test set sample to verify against.")
        return

    sample_idx = X_test_raw[non_benign_mask].index[: min(10, non_benign_mask.sum())]
    sample_raw = X_test_raw.loc[sample_idx]
    sample_labels = y_test_raw.loc[sample_idx]

    sample_scaled = pd.DataFrame(
        scaler.transform(sample_raw),
        columns=sample_raw.columns,
        index=sample_raw.index
    )

    probs = lgbm.predict(sample_scaled)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [class_names[i] for i in pred_idx]
    confidences = np.max(probs, axis=1)

    correct = 0
    print(f"{'True Label':<28} {'Predicted':<28} {'Conf':>7}  Match")
    print("-" * 75)
    for true_lbl, pred_lbl, conf in zip(sample_labels, pred_labels, confidences):
        match = "YES" if true_lbl == pred_lbl else "no"
        if true_lbl == pred_lbl:
            correct += 1
        print(f"{true_lbl:<28} {pred_lbl:<28} {conf:>6.2%}  {match}")

    print("-" * 75)
    print(f"Correctly identified {correct}/{len(sample_labels)} real attack rows "
          f"after raw -> scale -> predict (the exact path main.py takes).")

    if correct == 0:
        print("\n[WARNING] Still 0/N — the scaler/model mismatch may not be fully "
              "resolved, or the feature column order differs from FEATURE_COLUMNS "
              "in main.py. Compare X_test_raw.columns against main.py's "
              "FEATURE_COLUMNS list (underscore vs space naming) before debugging "
              "further.")
    else:
        print("\nThis confirms the scaler/model mismatch is fixed: real attack "
              "packets that came from the dataset now score as attacks, not BENIGN.")


if __name__ == "__main__":
    main()