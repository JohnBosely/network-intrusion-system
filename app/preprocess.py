from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path 


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
    print(f"Head of Dataset: {combine_df.head(5)}")
    
    null_count = combine_df.isnull().sum()
    print("\nColumns with missing values:")
    print(null_count[null_count > 0])

    num_cols = combine_df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(combine_df[num_cols]).sum()
    print("\nColumns with infinite values:")
    print(inf_counts[inf_counts > 0])

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
    combine_df['Label'] = combine_df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    return combine_df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def split_features_target(combine_df):
    X = combine_df.drop(columns=['Label'])
    y = combine_df['Label']
    return X, y

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
        
        # <--- ADD THIS LINE: Keep a backup of the original text names --->
        df['Raw_Label'] = df['Label'].astype(str).str.strip()
        
        df = clean_labels(df)
        
        cutoff = int(len(df) * 0.8)
        train_dfs.append(df.iloc[:cutoff])
        test_dfs.append(df.iloc[cutoff:])
        print(f" Loaded {file.name}: split at row {cutoff}")
        
    full_train = pd.concat(train_dfs, ignore_index=True)
    full_test = pd.concat(test_dfs, ignore_index=True)
    
    train_sampled = full_train.sample(n=sample_size, random_state=42)
    test_sampled = full_test.sample(n=int(sample_size * 0.25), random_state=42)
    
    X_train = train_sampled.drop(columns=["Label", "Raw_Label"], errors='ignore')
    y_train = train_sampled["Label"]
    
    X_test = test_sampled.drop(columns=["Label", "Raw_Label"], errors='ignore')
    y_test = test_sampled["Label"]
    
    # --- ADD THIS LINE: Pull the text names for the test set ---
    y_test_raw = test_sampled["Raw_Label"].values
    
    # Return 5 things now instead of 4
    return X_train, X_test, y_train, y_test, y_test_raw

def train_xgb(X_train_scaled, y_train):
    print("\n--- Training XGBoost Classifier ---")
    model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("Training Complete!")
    return model

def train_lr(X_train_scaled, y_train):
    print("\n--- Training Logistic Regression Classifier ---")
    # max_iter=1000 ensures the solver has enough steps to converge on 78 features
    model = LogisticRegression(max_iter=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("Training Complete!")
    return model

def train_rf(X_train_scaled, y_train):
    print("\n--- Training Random Forest Classifier ---")
    # n_jobs=-1 uses all CPU cores for parallel tree building
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("Training Complete!")
    return model

def train_svm(X_train_scaled, y_train):
    print("\n--- Training Support Vector Machine Classifier (RBF Kernel) ---")
    print("Note: SVM training can take a couple of minutes depending on your CPU...")
    # We leave probability=False and use decision_function later for speed
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_scaled, y_train)
    print("Training Complete!")
    return model

from lightgbm import LGBMClassifier

def train_lgbm(X_train_scaled, y_train):
    print("\n--- Training LightGBM Classifier ---")
    # n_jobs=-1 utilizes all available CPU cores
    # verbose=-1 silences unnecessary internal logging messages
    model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train_scaled, y_train)
    print("Training Complete!")
    return model

def evaluate_model(model, X_test_scaled, y_test):
    print("\n--- Evaluating Model Performance ---")
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    return report


# --- MAIN EXECUTION PIPELINE --

# 1. Setup paths
script_dir = Path(__file__).resolve().parent
data_directory = script_dir.parent / "data" / "MachineLearningCVE"

# 2. Extract features & targets using the chronological strategy
# Catch the new y_test_raw variable
X_train, X_test, y_train, y_test, y_test_raw = preprocess_chronological(
    data_dir_path=data_directory, sample_size=100000
)

print(f"\n<--- Train-Test Split Complete --->")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# 3. Standardize features
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
print(f"--- Feature Scaling Complete ---")

# 4. Train LightGBM Model
print(f"\n--- LIGHTGBM TRAINING ---")
lgbm_model = train_lgbm(X_train_scaled, y_train)

# 5. Evaluate Performance (with ROC-AUC)
print("\n--- Evaluating LightGBM Performance ---")
y_pred_lgbm = lgbm_model.predict(X_test_scaled)

# Get raw probabilities for Class 1 (Attacks)
y_probs_lgbm = lgbm_model.predict_proba(X_test_scaled)[:, 1]

# Calculate the ultimate metric
roc_auc_lgbm = roc_auc_score(y_test, y_probs_lgbm)

print("Classification Report:")
print(classification_report(y_test, y_pred_lgbm))

print("\n--- Raw Confusion Matrix (Actual Counts) ---")
print(confusion_matrix(y_test, y_pred_lgbm))

print(f"\n➔ LightGBM ROC-AUC Score: {roc_auc_lgbm:.5f}")

# 6. Breakdown of Missed Attacks for LightGBM
missed_mask_lgbm = (y_test == 1) & (y_pred_lgbm == 0)
missed_attacks_lgbm = y_test_raw[missed_mask_lgbm]

print(f"\n--- Breakdown of the {len(missed_attacks_lgbm)} Missed Attacks ---")
if len(missed_attacks_lgbm) > 0:
    print(pd.Series(missed_attacks_lgbm).value_counts())
else:
    print("Zero missed attacks!")

# --- Plot the LightGBM ROC Curve ---
print("\nGenerating ROC Curve plot...")
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_predictions(
    y_test, 
    y_probs_lgbm, 
    name="LightGBM", 
    color="teal",
    linewidth=2
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Guess (0.50)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (False Alarms)")
plt.ylabel("True Positive Rate (Caught Attacks)")
plt.title("LightGBM ROC Curve")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# <--- 4. Train Support Vector Machine Model --->
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