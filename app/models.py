import numpy as np
import lightgbm as lgb
from sklearn.ensemble import IsolationForest


def train_lgbm(X_train, y_train, num_class, sample_weight=None):
    """
    Trains the multi-class LightGBM classifier on the known universe of attacks.

    sample_weight : array-like of shape (n_samples,), optional
        Per-sample weights passed to lgb.Dataset. Use this to compensate
        for class imbalance — rare attack classes get higher weights so
        LightGBM penalises missing them more than missing BENIGN rows.
        If None, all samples are treated equally (original behaviour).
    """
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)

    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1,
    }

    print("Fitting LightGBM Classifier...")
    model = lgb.train(params, train_data, num_boost_round=100)
    return model


def train_isolation_forest(X_train, y_train, benign_idx, contamination=0.05):
    """
    Trains an unsupervised Isolation Forest exclusively on BENIGN data
    to establish a secure network baseline.
    """
    print("Filtering training data for pure BENIGN traffic...")
    normal_traffic = X_train[y_train == benign_idx]

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    print(f"Fitting Isolation Forest baseline on {normal_traffic.shape[0]} normal packets...")
    iso_forest.fit(normal_traffic)
    return iso_forest