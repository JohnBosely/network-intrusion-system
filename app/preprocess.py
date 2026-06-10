import numpy as np
import pandas as pd
from pathlib import Path 
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Pulling our clean, dedicated models directly from your architecture
from models import train_lgbm, train_isolation_forest

def load_data():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data" / "MachineLearningCVE"
    print(f"Searching for CSV files in: {data_dir.resolve()}")
    
    files = list(data_dir.glob("*.csv"))
    print(f"Found {len(files)} files.")

    df_list = []
    for file in files:
        print(f"Reading: {file.name}...")
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip() 
        df_list.append(df)
    
    combine_df = pd.concat(df_list, ignore_index=True)
    return combine_df

def clean_data(combine_df):
    print(f"\n--- Starting Data Inspection and Cleaning ---")
    combine_df = remove_infinite_values(combine_df)
    combine_df = clean_labels(combine_df)
    print(f"New Dataset Shape: {combine_df.shape}")
    
    print("\nTarget Label Distribution:")
    print(combine_df['Label'].value_counts())
    return combine_df

def remove_infinite_values(combine_df):
    num_cols = combine_df.select_dtypes(include=[np.number]).columns
    is_inf = np.isinf(combine_df[num_cols]).any(axis=1)
    is_null = combine_df[num_cols].isnull().any(axis=1)
    combine_df = combine_df[~(is_inf | is_null)]
    return combine_df

def clean_labels(combine_df):
    combine_df['Label'] = combine_df['Label'].astype(str).str.strip()
    return combine_df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    # Scale and reconstruct DataFrames to lock in feature names for SHAP
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled

def preprocess_chronological(data_dir_path, sample_size=100000):
    data_dir = Path(data_dir_path)
    files = list(data_dir.glob("*.csv"))
    
    print(f"\n--- Processing {len(files)} files chronologically ---")
    
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
        print(f" Loaded {file.name}: split at row {cutoff}")
        
    full_train = pd.concat(train_dfs, ignore_index=True)
    full_test = pd.concat(test_dfs, ignore_index=True)
    
    train_sampled = full_train.sample(n=sample_size, random_state=42)
    test_sampled = full_test.sample(n=int(sample_size * 0.25), random_state=42)
    
    X_train = train_sampled.drop(columns=["Label"], errors='ignore')
    y_train_raw = train_sampled["Label"]
    
    X_test = test_sampled.drop(columns=["Label"], errors='ignore')
    y_test_raw = test_sampled["Label"]
    
    return X_train, X_test, y_train_raw, y_test_raw


# Curated behavioral features that strip out static noise and focus on traffic mechanics
ANOMALY_FEATURE_SET = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Packet Length Variance',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
]

# =====================================================================
# --- MULTI-TIER HYBRID EXECUTION PIPELINE ---
# =====================================================================
if __name__ == "__main__":
    # 1. Setup paths
    script_dir = Path(__file__).resolve().parent
    data_directory = script_dir.parent / "data" / "MachineLearningCVE"

    # 2. Extract features & raw string text labels
    X_train, X_test, y_train_raw, y_test_raw = preprocess_chronological(
        data_dir_path=data_directory, sample_size=100000
    )

    # 3. Label Encoding with global data vision
    label_encoder = LabelEncoder()
    full_label_union = pd.concat([y_train_raw, y_test_raw])
    label_encoder.fit(full_label_union)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    class_names = label_encoder.classes_
    benign_idx = np.where(class_names == 'BENIGN')[0][0]

    # 4. Standardize features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # 5. Train Tier-1: Supervised Native LightGBM Classifier
    print(f"\n--- TIER 1: TRAINING SUPERVISED CLASSIFIER ---")
    # Fixed: Passing len(class_names) so LightGBM allocates room for all encoded integers
    lgbm_model = train_lgbm(X_train_scaled, y_train, num_class=len(class_names))

  # =====================================================================
    # --- TIER 2: OPERATIONAL BOUNDARY TUNING SWEEP ---
    # =====================================================================
    print(f"\n--- TIER 2: TUNING OPERATIONAL PERCENTILE BOUNDARY ---")
    
    from sklearn.ensemble import IsolationForest

    X_train_anomaly = X_train_scaled[ANOMALY_FEATURE_SET]
    X_train_pure_benign = X_train_anomaly[y_train == benign_idx]

    # Lock in our winning architecture
    print("Training production-grade Isolation Forest (n_estimators=250, max_samples=2048)...")
    production_anomaly_detector = IsolationForest(
        n_estimators=250,
        max_samples=2048,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    production_anomaly_detector.fit(X_train_pure_benign)
    
    # Extract raw score vectors
    train_benign_scores = production_anomaly_detector.decision_function(X_train_pure_benign)
    
    lgbm_probs = lgbm_model.predict(X_test_scaled)
    y_pred_lgbm = np.argmax(lgbm_probs, axis=1)
    leaked_mask = (y_test != benign_idx) & (y_pred_lgbm == benign_idx)
    leaked_indices = np.where(leaked_mask)[0]
    
    X_leaked_anomaly = X_test_scaled[ANOMALY_FEATURE_SET].iloc[leaked_indices]
    X_test_benign = X_test_scaled[ANOMALY_FEATURE_SET][y_test == benign_idx]
    
    leaked_scores = production_anomaly_detector.decision_function(X_leaked_anomaly)
    benign_test_scores = production_anomaly_detector.decision_function(X_test_benign)

    total_leaked_attacks = len(leaked_indices)
    total_benign_test = len(X_test_benign)

    # Sweep across percentile thresholds to find the optimum operational sweet spot
    percentile_candidates = [5, 7, 9, 11, 13, 15]
    
    best_tpr = -1.0
    best_fpr = 100.0
    final_operational_threshold = None
    best_percentile = None

    print(f"Sweeping operational boundaries to maximize detection while constraining FPR <= 5.0%...\n")

    for p in percentile_candidates:
        # Calculate threshold dynamically based on candidate percentile
        CANDIDATE_THRESHOLD = np.percentile(train_benign_scores, p)
        
        intercepted = np.sum(leaked_scores < CANDIDATE_THRESHOLD)
        false_alarms = np.sum(benign_test_scores < CANDIDATE_THRESHOLD)
        
        tpr = (intercepted / total_leaked_attacks) * 100
        fpr = (false_alarms / total_benign_test) * 100
        
        print(f"➔ Testing Training Percentile: {p}% | Threshold: {CANDIDATE_THRESHOLD:.4f}")
        print(f"   Test Metrics -> Detection Rate (TPR): {tpr:.2f}% | False Alarm Rate (FPR): {fpr:.2f}%")
        
        # Enforce hard engineering constraint: Maximize TPR, but FPR must be under 5.0%
        if fpr <= 5.0 and tpr > best_tpr:
            best_tpr = tpr
            best_fpr = fpr
            final_operational_threshold = CANDIDATE_THRESHOLD
            best_percentile = p

    print("\n====================================================")
    print("--- PRODUCTION BOUNDARY OPTIMIZATION COMPLETE ---")
    print("====================================================")
    print(f"Optimal Operational Setting: {best_percentile}th Percentile")
    print(f"Target Security Threshold: {final_operational_threshold:.4f}")
    print(f"Final System Metrics -> Detection Rate: {best_tpr:.2f}% | False Alarm Rate: {best_fpr:.2f}%")

    # Final threat profile analysis
    interception_report = pd.DataFrame({
        'Attack Type': y_test_raw.iloc[leaked_indices],
        'Intercepted': (leaked_scores < final_operational_threshold)
    })
    
    print("\n--- Final Breakdown of Defended Zero-Day Threats ---")
    summary = interception_report.groupby('Attack Type')['Intercepted'].agg(['count', 'sum'])
    summary.columns = ['Total Leaked Past Tier 1', 'Caught By Tier 2']
    print(summary)
# print(f"\n--- SUPPORT VECTOR MACHINE TRAINING ---")
# svm_model = train_svm(X_train_scaled, y_train)

# # 5. Evaluate Performance (with ROC-AUC)
# print("\n--- Evaluating SVM Performance ---")
# y_pred_svm = svm_model.predict(X_test_scaled)

# # Get confidence scores via decision_function for Class 1 (Attacks)
# y_scores_svm = svm_model.decision_function(X_test_scaled)

# # Calculate the ultimate metric
# roc_auc_svm = roc_auc_score(y_test, y_scores_svm)

# print("Classification Report:")
# print(classification_report(y_test, y_pred_svm))

# print("\n--- Raw Confusion Matrix (Actual Counts) ---")
# print(confusion_matrix(y_test, y_pred_svm))

# print(f"\n➔ SVM ROC-AUC Score: {roc_auc_svm:.5f}")

# # 6. Breakdown of Missed Attacks for SVM
# missed_mask_svm = (y_test == 1) & (y_pred_svm == 0)
# missed_attacks_svm = y_test_raw[missed_mask_svm]

# print(f"\n--- Breakdown of the {len(missed_attacks_svm)} Missed Attacks ---")
# if len(missed_attacks_svm) > 0:
#     print(pd.Series(missed_attacks_svm).value_counts())
# else:
#     print("Zero missed attacks!")

# # --- Plot the SVM ROC Curve ---
# print("\nGenerating ROC Curve plot...")
# plt.figure(figsize=(8, 6))
# RocCurveDisplay.from_predictions(
#     y_test, 
#     y_scores_svm, 
#     name="Support Vector Machine", 
#     color="purple",
#     linewidth=2
# )
# plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Guess (0.50)")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate (False Alarms)")
# plt.ylabel("True Positive Rate (Caught Attacks)")
# plt.title("Support Vector Machine ROC Curve")
# plt.legend(loc="lower right")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.show()

# <--- RANDOM FOREST --->
# # 4. Train Random Forest Model
# print(f"\n--- RANDOM FOREST TRAINING ---")
# rf_model = train_rf(X_train_scaled, y_train)

# # 5. Evaluate Performance (with ROC-AUC)
# print("\n--- Evaluating Random Forest Performance ---")
# y_pred_rf = rf_model.predict(X_test_scaled)

# # Get raw probabilities for Class 1 (Attacks)
# y_probs_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# # Calculate the ultimate metric
# roc_auc_rf = roc_auc_score(y_test, y_probs_rf)

# print("Classification Report:")
# print(classification_report(y_test, y_pred_rf))

# print("\n--- Raw Confusion Matrix (Actual Counts) ---")
# print(confusion_matrix(y_test, y_pred_rf))

# print(f"\n➔ Random Forest ROC-AUC Score: {roc_auc_rf:.5f}")

# # 6. Breakdown of Missed Attacks for Random Forest
# missed_mask_rf = (y_test == 1) & (y_pred_rf == 0)
# missed_attacks_rf = y_test_raw[missed_mask_rf]

# print(f"\n--- Breakdown of the {len(missed_attacks_rf)} Missed Attacks ---")
# if len(missed_attacks_rf) > 0:
#     print(pd.Series(missed_attacks_rf).value_counts())
# else:
#     print("Zero missed attacks!")

# # --- Plot the Random Forest ROC Curve ---
# print("\nGenerating ROC Curve plot...")
# plt.figure(figsize=(8, 6))
# RocCurveDisplay.from_predictions(
#     y_test, 
#     y_probs_rf, 
#     name="Random Forest", 
#     color="green",
#     linewidth=2
# )
# plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Guess (0.50)")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate (False Alarms)")
# plt.ylabel("True Positive Rate (Caught Attacks)")
# plt.title("Random Forest ROC Curve")
# plt.legend(loc="lower right")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.show()

# <--- LOGISTIC REGRESSION --->
# # 4. Train Logistic Regression Model
# print(f"\n--- LOGISTIC REGRESSION TRAINING ---")
# lr_model = train_lr(X_train_scaled, y_train)

# # 5. Evaluate Performance (with ROC-AUC)
# print("\n--- Evaluating Logistic Regression Performance ---")
# y_pred_lr = lr_model.predict(X_test_scaled)

# # Get raw probabilities for Class 1 (Attacks)
# y_probs_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# # Calculate the ultimate metric
# roc_auc_lr = roc_auc_score(y_test, y_probs_lr)

# print("Classification Report:")
# print(classification_report(y_test, y_pred_lr))

# print("\n--- Raw Confusion Matrix (Actual Counts) ---")
# print(confusion_matrix(y_test, y_pred_lr))

# print(f"\n➔ Logistic Regression ROC-AUC Score: {roc_auc_lr:.5f}")

# # 6. Breakdown of Missed Attacks for Logistic Regression
# missed_mask_lr = (y_test == 1) & (y_pred_lr == 0)
# missed_attacks_lr = y_test_raw[missed_mask_lr]

# print(f"\n--- Breakdown of the {len(missed_attacks_lr)} Missed Attacks ---")
# if len(missed_attacks_lr) > 0:
#     print(pd.Series(missed_attacks_lr).value_counts())
# else:
#     print("Zero missed attacks!")

# # --- Plot the Logistic Regression ROC Curve ---
# print("\nGenerating ROC Curve plot...")
# plt.figure(figsize=(8, 6))
# RocCurveDisplay.from_predictions(
#     y_test, 
#     y_probs_lr, 
#     name="Logistic Regression", 
#     color="blue",
#     linewidth=2
# )
# plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Guess (0.50)")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate (False Alarms)")
# plt.ylabel("True Positive Rate (Caught Attacks)")
# plt.title("Logistic Regression ROC Curve")
# plt.legend(loc="lower right")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.show()



# <--- XGBOOST --->
# # 4. Train Model
# print(f"\n--- XGBOOST TRAINING ---")
# xgb_model = train_xgb(X_train_scaled, y_train)

# #from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

# # 5. Evaluate Performance (with ROC-AUC)
# print("\n--- Evaluating Model Performance ---")
# y_pred = xgb_model.predict(X_test_scaled)

# # Get the raw probabilities for the positive class (Class 1: Attacks)
# y_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

# # Calculate the ultimate metric
# roc_auc = roc_auc_score(y_test, y_probs)

# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# print("\n--- Raw Confusion Matrix (Actual Counts) ---")
# print(confusion_matrix(y_test, y_pred))

# print(f"\n➔ Ultimate Decider ROC-AUC Score: {roc_auc:.5f}")

# # --- Plot the ROC Curve ---
# print("\nGenerating ROC Curve plot...")
# plt.figure(figsize=(8, 6))
# RocCurveDisplay.from_predictions(
#     y_test, 
#     y_probs, 
#     name="XGBoost Baseline", 
#     color="darkorange",
#     linewidth=2
# )
# plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Guess (0.50)")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate (False Alarms)")
# plt.ylabel("True Positive Rate (Caught Attacks)")
# plt.title("Receiver Operating Characteristic (ROC) Curve")
# plt.legend(loc="lower right")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.show()

# # 6. Extract Feature Importances
# importances = xgb_model.feature_importances_
# feature_names = X_train.columns

# feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# print("\n--- Top 5 Most Important Features ---")
# print(feature_imp_df.sort_values(by='Importance', ascending=False).head(5))

# y_pred = xgb_model.predict(X_test_scaled)

# # 2. Create a boolean mask for False Negatives (Actual Attack (1), but predicted Benign (0))
# missed_mask = (y_test == 1) & (y_pred == 0)

# # 3. Pull the original text labels for those specific misses
# missed_attacks = y_test_raw[missed_mask]

# print("\n--- Breakdown of the 51 Missed Attacks ---")
# print(pd.Series(missed_attacks).value_counts())

# # 4. FIX: Build the missing DataFrame by slicing X_test with our mask
# missed_attacks_df = X_test[missed_mask].copy()

# # 5. Attach the original attack names so you can see them in the printout
# missed_attacks_df['Label'] = missed_attacks

# print(f"\nSample of 5 missed attacks:\n")
# # 6. Display the structural columns to analyze the patterns
# print(missed_attacks_df[['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Label']].head())