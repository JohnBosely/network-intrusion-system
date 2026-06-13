# import numpy as np
# import pandas as pd
# from pathlib import Path
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env

# # Import your custom assets and preprocessing modules
# from env import NetworkDefenseEnv
# from preprocess import preprocess_chronological, scale_features, LabelEncoder

# def execute_agent_training():
#     print("=============================================")
#     print("--- INITIALIZING TIER 3 POLICIES TRAINING ---")
#     print("=============================================\n")

#     # 1. Resolve Paths
#     SCRIPT_DIR = Path(__file__).resolve().parent
#     ROOT_DIR = SCRIPT_DIR.parent
#     DATA_DIRECTORY = ROOT_DIR / "data" / "MachineLearningCVE"
#     ARTIFACTS_FOLDER = ROOT_DIR / "artifacts"

#     # 2. Ingest Data Stream (Sampling a chunk for reinforcement loops)
#     print("[TRAIN] Streaming and formatting training partitions...")
#     X_train, X_test, y_train_raw, y_test_raw = preprocess_chronological(
#         data_dir_path=DATA_DIRECTORY, sample_size=50000
#     )

#     # 3. Align Label Indices with Saved Encoders
#     print("[TRAIN] Aligning categorical target encoders...")
#     label_encoder = LabelEncoder()
#     full_union = pd.concat([y_train_raw, y_test_raw])
#     label_encoder.fit(full_union)
    
#     y_train = label_encoder.transform(y_train_raw)

#     # 4. Scale Features to Stabilize Neural Networks Inputs
#     print("[TRAIN] Standardizing feature geometry...")
#     X_train_scaled, _ = scale_features(X_train, X_test)

#     # 5. Instantiate the Custom Gymnasium Environment
#     print("[TRAIN] Spinning up the network defense environment game board...")
#     env = NetworkDefenseEnv(
#         data_features=X_train_scaled, 
#         data_labels=y_train, 
#         artifacts_dir_path=ARTIFACTS_FOLDER
#     )

#     # 6. Enterprise Sanity Check
#     print("[TRAIN] Running Gymnasium compatibility audit...")
#     check_env(env, warn=True)
#     print("✓ Environment passed all structure checks successfully!")

#     # 7. Configure the PPO Reinforcement Learning Agent
#     print("\n[TRAIN] Initializing PPO Agent with a Multi-Layer Perceptron (MlpPolicy)...")
#     # We use a standard 2-layer network [64, 64] to map the 17 inputs to our 4 discrete actions
#     agent = PPO(
#         policy="MlpPolicy",
#         env=env,
#         learning_rate=3e-4,     # Standard stable learning speed
#         n_steps=2048,           # Number of steps to run per optimization update
#         batch_size=64,          # Mini-batch sizes for gradient updates
#         n_epochs=10,            # Number of times to pass over the collected experiences
#         verbose=1,              # Prints training progression tables automatically
#         tensorboard_log=str(ROOT_DIR / "tensorboard_logs")
#     )

#     # 8. Run the Training Loop
#     # Let the agent process 20,000 packets to figure out the optimal defense matrices
#     TRAINING_STEPS = 20000
#     print(f"\n[TRAIN] Launching automation loop for {TRAINING_STEPS} environmental steps...")
#     agent.learn(total_timesteps=TRAINING_STEPS)

#     # 9. Save the Trained Agent Brain
#     models_dir = ROOT_DIR / "artifacts"
#     agent_save_path = models_dir / "tier3_ppo_agent"
#     agent.save(agent_save_path)
    
#     print(f"\n=============================================")
#     print(f"[SUCCESS] Tier 3 RL Agent successfully optimized!")
#     print(f"Saved model binary to: {agent_save_path}.zip")
#     print("=============================================")

# if __name__ == "__main__":
#     execute_agent_training()


import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import FastNetworkDefenseEnv
from preprocess import preprocess_chronological, scale_features, LabelEncoder

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
    
    # Generate mock network congestion timelines for the matrix
    mock_loads_all = (np.arange(len(X_train_scaled)) % 100) / 100.0
    
    # Stacking everything horizontally into a single array block
    print("[PRE-COMPUTE] Assembling vectorized observation matrix...")
    precomputed_obs = np.hstack([
        t1_probs_all, 
        t2_scores_all.reshape(-1, 1), 
        mock_loads_all.reshape(-1, 1)
    ]).astype(np.float32)

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
        # Remove policy_kwargs entirely — back to default [64, 64]
        tensorboard_log=str(ROOT_DIR / "tensorboard_logs")
    )

    TOTAL_TRAINING_STEPS = 2000000  # keep this

    print(f"\n[TRAIN] Launching high-speed loop for {TOTAL_TRAINING_STEPS} steps...")
    agent.learn(total_timesteps=TOTAL_TRAINING_STEPS)

    # 7. Save Final Brain Blueprint
    agent.save(ARTIFACTS_FOLDER / "tier3_ppo_agent_scaled")
    print("\n[SUCCESS] Scaled Agent fully optimized and saved successfully!")

if __name__ == "__main__":
    execute_mass_training()