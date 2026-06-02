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
    combine_df = combine_df.replace([np.inf, -np.inf], np.nan)
    combine_df = combine_df.dropna()
    return combine_df

def clean_labels(combine_df):
    # 1. Fix the web attacks by searching for any label containing "Web Attack"
    # This automatically catches the broken characters and names them beautifully!
    combine_df.loc[combine_df['Label'].str.contains('Web Attack', na=False, case=False), 'Label'] = 'Web Attack'
    
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

def split_train_test(X, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    return X_train, X_test, y_train, y_test

def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded


combine_df = load_data()
cleaned_df = clean_data(combine_df)

X, y = split_features_target(cleaned_df)
print(f"\n--- Features and Target Split Complete ---")
print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")

y_encoded = encode_labels(y)
print(f"\n--- Label Encoding Complete ---")
print(f"Original text sample: {y.head(5).values}")
print(f"Encoded numeric sample: {y_encoded[:5]}")

X_train, X_test, y_train, y_test = split_train_test(X, y_encoded)

print(f"\n--- Train-Test Split Complete ---")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

print(f"\n--- Feature Scaling Complete ---")
print(f"Scaled training sample (first row):\n{X_train_scaled[0][:5]}") 

