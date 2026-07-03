import pandas as pd
import numpy as np
from pathlib import Path
from preprocess import preprocess_chronological

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "MachineLearningCVE"

X_train, X_test, y_train_raw, y_test_raw = preprocess_chronological(
    data_dir_path=DATA_DIR,
    sample_size=100000
)

print("\n=== TRAINING SET LABEL DISTRIBUTION ===")
train_counts = y_train_raw.value_counts()
print(train_counts)

print("\n=== TEST SET LABEL DISTRIBUTION ===")
test_counts = y_test_raw.value_counts()
print(test_counts)

print("\n=== CLASSES MISSING FROM TRAINING ===")
train_classes = set(y_train_raw.unique())
test_classes = set(y_test_raw.unique())
missing = test_classes - train_classes
print(missing if missing else "None")

print("\n=== CLASSES WITH VERY FEW TRAINING EXAMPLES (<100) ===")
rare = train_counts[train_counts < 100]
print(rare if len(rare) > 0 else "None")

print("\n=== RATIO: TEST vs TRAIN for each class ===")
all_classes = sorted(set(y_train_raw.unique()) | set(y_test_raw.unique()))
print(f"{'Class':<35} {'Train':>8} {'Test':>8} {'Test%':>8}")
print("-" * 62)
for cls in all_classes:
    train_n = train_counts.get(cls, 0)
    test_n = test_counts.get(cls, 0)
    total = train_n + test_n
    test_pct = (test_n / total * 100) if total > 0 else 0
    print(f"{cls:<35} {train_n:>8} {test_n:>8} {test_pct:>7.1f}%")