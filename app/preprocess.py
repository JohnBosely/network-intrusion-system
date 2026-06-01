from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

# def handle_missing_values():
#     pass

# def remove_infinite_values():
#     pass

# def encode_labels():
#     pass

# def split_features_target():
#     pass

combine_df = load_data()
cleaned_df = clean_data(combine_df)