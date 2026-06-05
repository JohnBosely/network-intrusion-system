from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
        df.columns = df.columns.str.strip() # This was added due to trailing spaces in the column
        df_list.append(df)
    
    combine_df = pd.concat(df_list, ignore_index=True)
    return combine_df


def clean_data(combine_df):
    print(f"\n--- Starting Data Inspection and Cleaning ---")

    # Call the remove infinite values function
    combine_df = remove_infinite_values(combine_df)

    # Call the clean labels function
    combine_df = clean_labels(combine_df)
    print(f"New Dataset Shape: {combine_df.shape}")

    # 1. Look at the shape and head
    print(f"Head of Dataset: {combine_df.head(5)}")
    print(f"Shape of Dataset: {combine_df.shape}")
    
    # 2. Check for missing values (Nulls)
    null_count = combine_df.isnull().sum()
    print("\nColumns with missing values:")
    print(null_count[null_count > 0])

    # 3. Check for infinite values (only in numerical columns)
    num_cols = combine_df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(combine_df[num_cols]).sum()
    print("\nColumns with infinite values:")
    print(inf_counts[inf_counts > 0])

    # 4. Check the Target Class Distribution
    print("\nTarget Label Distribution:")
    print(combine_df['Label'].value_counts())

    return combine_df

def remove_infinite_values(combine_df):
    print("Filtering out infinite and null rows efficiently...")
    
    # 1. Find numerical columns
    num_cols = combine_df.select_dtypes(include=[np.number]).columns
    
    # 2. Check for bad values
    is_inf = np.isinf(combine_df[num_cols]).any(axis=1)
    is_null = combine_df[num_cols].isnull().any(axis=1)
    
    # 3. FIX: Slice the dataframe directly without using .copy()
    combine_df = combine_df[~(is_inf | is_null)]
    
    return combine_df

def clean_labels(combine_df):
    """
    Converts the multi-class labels into binary labels.
    0 = BENIGN (Normal traffic)
    1 = ATTACK (Any type of malicious traffic)
    """
    # Force the column to string type and strip any accidental whitespace
    combine_df['Label'] = combine_df['Label'].astype(str).str.strip()
    
    # Map 'BENIGN' to 0, and everything else to 1
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

    return X,y

def preprocess_data(data_dir_path, sample_size=100000):
    data_dir = Path(data_dir_path)

    # 1. Dynamically search for files matching the days of the week
    train_files = []
    for day in ["Tuesday", "Wednesday", "Thursday"]:
        # Looks for files like *Tuesday*.csv, *Wednesday*.csv, etc.
        matched_files = list(data_dir.glob(f"*{day}*.csv"))
        train_files.extend(matched_files)

    # Do the same dynamic look up for Friday
    test_files = list(data_dir.glob("*Friday*.csv"))

    # Quick sanity check printout so you can see what Python found
    print(f" Found {len(train_files)} training files.")
    print(f" Found {len(test_files)} testing files.")

    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Could not find day-based files in {data_dir_path}. "
            f"Check your folder names!"
        )

    # 2. Build the Training DataFrame (Tue + Wed + Thu)
    train_dfs = [pd.read_csv(f) for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df.columns = train_df.columns.str.strip()

    # 3. Build the Testing DataFrame (Friday only)
    test_dfs = [pd.read_csv(f) for f in test_files]
    test_df = pd.concat(test_dfs, ignore_index=True)
    test_df.columns = test_df.columns.str.strip()
    
    # ... (the rest of your cleaning and sampling code stays exactly the same!)
    # 4. Clean up missing/infinite values (using your existing functions)
    train_df = remove_infinite_values(train_df)
    train_df = clean_labels(train_df)

    test_df = remove_infinite_values(test_df)
    test_df = clean_labels(test_df)

    # 5. Take your controlled sample sizes so your PC doesn't freeze
    # We sample 100k from training, and a proportional 25k from testing
    train_df = train_df.sample(n=sample_size, random_state=42)
    test_df = test_df.sample(n=int(sample_size * 0.25), random_state=42)

    # 6. Split into X and y directly by day! No random train_test_split needed.
    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]

    return X_train, X_test, y_train, y_test

def encode_labels(y):
    return y

def train_xgb(X_train_scaled, y_train):
    print("\n--- Training XGBoost Classifier ---")
    
    model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    model.fit(X_train_scaled, y_train)
    
    print("Training Complete!")
    return model


def evaluate_model(model, X_test_scaled, y_test):
    print("\n--- Evaluating Model Performance ---")
    # 1. Ask the model to predict the classes for the test features
    y_pred = model.predict(X_test_scaled)
    
    # 2. Generate a comprehensive metrics report
    report = classification_report(y_test, y_pred)
    
    print("Classification Report:")
    print(report)
    return report

# combine_df = load_data()
# cleaned_df = clean_data(combine_df)

# cleaned_df = cleaned_df.sample(
#     n=100_000,
#     random_state=42
# )

# X, y = split_features_target(cleaned_df)
# print(f"\n--- Features and Target Split Complete ---")
# print(f"Features shape (X): {X.shape}")
# print(f"Target shape (y): {y.shape}")

# y_encoded = encode_labels(y)
# print(f"\n--- Label Encoding Complete ---")
# print(f"Original text sample: {y.head(5).values}")
# print(f"Encoded numeric sample: {y_encoded[:5]}")

# X_train, X_test, y_train, y_test = split_train_test(X, y_encoded)

# --- ADD THIS NEW LINE TO REPLACE THE BLOCK ABOVE ---

# 1. Define the path to your data folder (using your existing path logic)
script_dir = Path(__file__).resolve().parent
data_directory = script_dir.parent / "data" / "MachineLearningCVE"

# 2. Call the function to get your clean, day-split data directly
X_train, X_test, y_train, y_test = preprocess_data(
    data_dir_path=data_directory, sample_size=100000
)

print(f"\n--- Train-Test Split Complete ---")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

print(f"\n--- Feature Scaling Complete ---")

print(f"\n--- XGBOOST TRAINING ---")
xgb_model = train_xgb(X_train_scaled, y_train)
evaluation = evaluate_model(xgb_model, X_test_scaled, y_test)
print(evaluation)

# Get feature importances from your trained XGBoost model
importances = xgb_model.feature_importances_
feature_names = X_train.columns

# Pair them up, sort them, and look at the top 5
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(feature_imp_df.sort_values(by='Importance', ascending=False).head(5))