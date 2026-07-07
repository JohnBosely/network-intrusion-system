import joblib
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import FastNetworkDefenseEnv
from preprocess import preprocess_chronological, scale_features

def execute_mass_training():
    print("=============================================")
    print("--- LAUNCHING FULL-SCALE DATASET TRAINING ---")
    print("=============================================\n")

    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR.parent
    DATA_DIRECTORY = ROOT_DIR / "data" / "MachineLearningCVE"
    ARTIFACTS_FOLDER = ROOT_DIR / "artifacts"

    # 1. Load the Entire Dataset (Removing sample constraints)
    print("[PRE-COMPUTE] Ingesting all source network PCAP logs...")
    X_train, X_test, y_train_raw, y_test_raw = preprocess_chronological(
        data_dir_path=DATA_DIRECTORY, sample_size=None # None loads EVERYTHING
    )

    # 2. Re-align and Scale
    label_encoder = joblib.load(ARTIFACTS_FOLDER / "label_encoder.pkl")
    y_train = label_encoder.transform(y_train_raw)
    X_train_scaled, _ = scale_features(X_train, X_test)

    # 3. Load Trained Passive Analytical Components
    print("[PRE-COMPUTE] Loading Tier 1 and Tier 2 pipeline models...")
    lgbm = joblib.load(ARTIFACTS_FOLDER / "tier1_lightgbm.pkl")
    iforest = joblib.load(ARTIFACTS_FOLDER / "tier2_isolation_forest.pkl")
    
    # 4. Mass Batch Inference Vectorization
    print("[PRE-COMPUTE] Running mass batch inference across all rows (This saves hours!)...")
    t1_probs_all = lgbm.predict(X_train_scaled) # Shape: (NumPackets, NumClasses)
    
    anomaly_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd Header Length', 'Bwd Header Length', 'Packet Length Variance',
        'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
    ]
    t2_scores_all = iforest.decision_function(X_train_scaled[anomaly_features]) # Shape: (NumPackets,)
    
    # Rolling threat rate: what fraction of the last 100 packets were attacks
    # Uses LightGBM predictions to estimate threat rate without needing true labels
    y_pred_train = np.argmax(t1_probs_all, axis=1)
    benign_idx_local = int(np.where(label_encoder.classes_ == "BENIGN")[0][0])
    is_predicted_attack = (y_pred_train != benign_idx_local).astype(float)

# Rolling window of 100 packets
    window = 100
    rolling_threat = np.zeros(len(is_predicted_attack))
    for i in range(len(is_predicted_attack)):
        start = max(0, i - window)
        rolling_threat[i] = is_predicted_attack[start:i+1].mean()
    
    # Stacking everything horizontally into a single array block
    print("[PRE-COMPUTE] Assembling vectorized observation matrix...")
    # Build base observation matrix first
    base_obs = np.hstack([
        t1_probs_all,
        t2_scores_all.reshape(-1, 1),
        rolling_threat.reshape(-1, 1)
    ]).astype(np.float32)

    # Sliding window of 5 packets — agent sees current + 4 previous packets
    WINDOW_SIZE = 5
    n_samples, n_features = base_obs.shape
    windowed_obs = np.zeros((n_samples, n_features * WINDOW_SIZE), dtype=np.float32)

    for i in range(n_samples):
        for w in range(WINDOW_SIZE):
            src_idx = max(0, i - (WINDOW_SIZE - 1 - w))
            windowed_obs[i, w * n_features:(w + 1) * n_features] = base_obs[src_idx]

    precomputed_obs = windowed_obs
    print(f"[PRE-COMPUTE] Sliding window applied. Observation shape: {precomputed_obs.shape}")

    # 5. Spin Up the High-Speed Board
    benign_idx = int(np.where(label_encoder.classes_ == "BENIGN")[0][0])
    env = FastNetworkDefenseEnv(
        precomputed_observations=precomputed_obs,
        data_labels=y_train,
        benign_idx=benign_idx
    )

    check_env(env, warn=True)

    # 6. Initialize PPO
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

    TOTAL_TRAINING_STEPS = 1000000
    print(f"\n[TRAIN] Launching high-speed loop for {TOTAL_TRAINING_STEPS} steps...")
    agent.learn(total_timesteps=TOTAL_TRAINING_STEPS)

    # 7. Save The Final Brain Blueprint
    agent.save(ARTIFACTS_FOLDER / "tier3_ppo_agent_scaled")
    print("\n[SUCCESS] Scaled Agent fully optimized and saved successfully!")

if __name__ == "__main__":
    execute_mass_training()