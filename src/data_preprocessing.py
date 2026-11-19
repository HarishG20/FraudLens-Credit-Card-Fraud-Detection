"""
Data Preprocessing Module for FraudLens
Handles dataset ingestion, structural inspection, stratified splitting,
leak-free normalization using StandardScaler fitted solely on legitimate records,
and artifact serialization.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def load_dataset(filepath: Path | str = config.DATASET_PATH) -> pd.DataFrame:
    """
    Loads creditcard.csv dataset from disk.
    Provides clear troubleshooting guidance if the file is absent.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at expected location: '{path.resolve()}'.\n"
            f"Please download 'creditcard.csv' from Kaggle:\n"
            f"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place it inside the 'dataset/' folder."
        )
    print(f"[INFO] Loading dataset from: {path.resolve()}")
    df = pd.read_csv(path)
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects shape, missing values, duplicate records, and target class distribution.
    """
    total_records = len(df)
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    
    class_counts = df[config.TARGET_COLUMN].value_counts().to_dict()
    normal_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    
    fraud_percentage = (fraud_count / total_records) * 100.0 if total_records > 0 else 0.0
    normal_percentage = (normal_count / total_records) * 100.0 if total_records > 0 else 0.0

    stats = {
        "total_records": total_records,
        "total_features": df.shape[1],
        "missing_values": missing_count,
        "duplicate_rows": duplicate_count,
        "normal_count": normal_count,
        "normal_percentage": normal_percentage,
        "fraud_count": fraud_count,
        "fraud_percentage": fraud_percentage,
    }

    print("=" * 60)
    print("           FRAUDLENS DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"Total Transactions : {total_records:,}")
    print(f"Total Columns      : {df.shape[1]} ({config.TARGET_COLUMN} + {len(config.FEATURE_COLUMNS)} features)")
    print(f"Missing Values     : {missing_count}")
    print(f"Duplicate Rows     : {duplicate_count}")
    print(f"Legitimate (0)     : {normal_count:,} ({normal_percentage:.4f}%)")
    print(f"Fraudulent (1)     : {fraud_count:,} ({fraud_percentage:.4f}%)")
    print("=" * 60)

    return stats


def split_data(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Performs stratified 70% / 15% / 15% train/val/test splitting to preserve
    the extreme class imbalance across all partitions without data leakage.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    # Separate input features and target labels
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].copy()

    # First split off the test set (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )

    # Next split remaining 85% into train (70%) and val (15%)
    # val_relative_ratio = 0.15 / (0.70 + 0.15) = 0.15 / 0.85 ~ 0.17647
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_ratio,
        stratify=y_train_val,
        random_state=random_state,
    )

    print(f"[INFO] Stratified Splitting Complete:")
    print(f"       Train set : {len(X_train):,} samples (Normal: {(y_train == 0).sum():,}, Fraud: {(y_train == 1).sum():,})")
    print(f"       Val set   : {len(X_val):,} samples (Normal: {(y_val == 0).sum():,}, Fraud: {(y_val == 1).sum():,})")
    print(f"       Test set  : {len(X_test):,} samples (Normal: {(y_test == 0).sum():,}, Fraud: {(y_test == 1).sum():,})")

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_autoencoder_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler_save_path: Path | str = config.SCALER_PATH,
) -> Dict[str, Any]:
    """
    Prepares data exclusively for semi-supervised / unsupervised deep autoencoder:
    1. Filters training data strictly to legitimate transactions (Class == 0).
    2. Fits StandardScaler strictly on normal training transactions (prevents leakage).
    3. Transforms validation and test partitions using fitted scaler.
    4. Serializes the scaler using Joblib.
    """
    # Filter training to normal transactions only
    X_train_normal = X_train[y_train == 0].copy()
    print(f"[INFO] Autoencoder Training Pool: {len(X_train_normal):,} purely legitimate transactions.")

    # Filter validation normal transactions for autoencoder reconstruction loss monitoring
    X_val_normal = X_val[y_val == 0].copy()

    # Fit scaler only on normal training data
    print("[INFO] Fitting StandardScaler strictly on normal training transactions...")
    scaler = StandardScaler()
    scaler.fit(X_train_normal)

    # Save fitted scaler
    if scaler_save_path:
        save_path = Path(scaler_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"[INFO] Scaler saved successfully to: {save_path.resolve()}")

    # Transform all subsets
    X_train_normal_scaled = scaler.transform(X_train_normal)
    X_val_normal_scaled = scaler.transform(X_val_normal)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train_normal_scaled": X_train_normal_scaled,
        "X_val_normal_scaled": X_val_normal_scaled,
        "X_val_scaled": X_val_scaled,
        "y_val": y_val.to_numpy(),
        "X_test_scaled": X_test_scaled,
        "y_test": y_test.to_numpy(),
        "scaler": scaler,
    }


def run_preprocessing_pipeline() -> Dict[str, Any]:
    """
    Runs the full preprocessing pipeline end-to-end and returns prepared data dictionaries.
    """
    df = load_dataset()
    inspect_dataset(df)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    data = prepare_autoencoder_data(X_train, y_train, X_val, y_val, X_test, y_test)
    return data


if __name__ == "__main__":
    try:
        run_preprocessing_pipeline()
        print("\n[SUCCESS] Data preprocessing module executed without errors.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)
