"""
Anomaly Detection Engine for FraudLens
Computes reconstruction error (MSE per sample), calibrates thresholds using
Percentile-based (unsupervised) and F1-optimal (supervised calibration on validation set) strategies,
and assigns anomaly classifications ('Normal' vs 'Potential Fraud').
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import config


def compute_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    """
    Computes sample-wise Mean Squared Reconstruction Error:
      Error_i = mean_over_features((x_ij - x_hat_ij)^2)
    """
    assert original.shape == reconstructed.shape, (
        f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
    )
    mse = np.mean(np.square(original - reconstructed), axis=1)
    return mse


def calculate_percentile_threshold(
    normal_errors: np.ndarray,
    percentile: float = config.DEFAULT_PERCENTILE_THRESHOLD,
) -> float:
    """
    Unsupervised Strategy A:
    Selects threshold as a specific percentile (e.g. 99th) of reconstruction errors
    computed strictly on legitimate (normal) validation transactions.
    """
    threshold = float(np.percentile(normal_errors, percentile))
    return threshold


def find_optimal_f1_threshold(
    val_errors: np.ndarray,
    y_val: np.ndarray,
    num_steps: int = 500,
) -> Dict[str, Any]:
    """
    Supervised Calibration Strategy B:
    Sweeps candidate thresholds across validation errors to find the threshold
    that maximizes the F1-Score for detecting fraudulent transactions.
    Note: Threshold tuning is strictly performed on validation data, NEVER on the test set.
    """
    min_err, max_err = np.min(val_errors), np.percentile(val_errors, 99.9)
    candidates = np.linspace(min_err, max_err, num_steps)

    best_threshold = float(candidates[0])
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for thresh in candidates:
        preds = (val_errors > thresh).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(thresh)
            best_precision = float(precision_score(y_val, preds, zero_division=0))
            best_recall = float(recall_score(y_val, preds, zero_division=0))

    return {
        "optimal_threshold": best_threshold,
        "f1_score": best_f1,
        "precision": best_precision,
        "recall": best_recall,
    }


def compare_threshold_strategies(
    val_errors: np.ndarray,
    y_val: np.ndarray,
    normal_val_errors: np.ndarray,
    percentile: float = config.DEFAULT_PERCENTILE_THRESHOLD,
) -> Dict[str, Dict[str, Any]]:
    """
    Compares the Percentile-based approach and F1-optimal calibration on validation data.
    """
    # Percentile strategy
    p_thresh = calculate_percentile_threshold(normal_val_errors, percentile=percentile)
    p_preds = (val_errors > p_thresh).astype(int)
    percentile_metrics = {
        "strategy": f"Percentile ({percentile}th)",
        "threshold": p_thresh,
        "precision": float(precision_score(y_val, p_preds, zero_division=0)),
        "recall": float(recall_score(y_val, p_preds, zero_division=0)),
        "f1_score": float(f1_score(y_val, p_preds, zero_division=0)),
    }

    # F1-optimal strategy
    f1_result = find_optimal_f1_threshold(val_errors, y_val)
    f1_metrics = {
        "strategy": "F1-Score Maximization",
        "threshold": f1_result["optimal_threshold"],
        "precision": f1_result["precision"],
        "recall": f1_result["recall"],
        "f1_score": f1_result["f1_score"],
    }

    return {
        "percentile_strategy": percentile_metrics,
        "f1_optimal_strategy": f1_metrics,
    }


def save_threshold_artifact(
    threshold_value: float,
    metadata: Optional[Dict[str, Any]] = None,
    save_path: Path | str = config.THRESHOLD_PATH,
) -> None:
    """
    Saves the selected anomaly threshold and associated metadata using Joblib.
    """
    artifact = {
        "threshold": float(threshold_value),
        "metadata": metadata or {},
    }
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    print(f"[INFO] Anomaly threshold ({threshold_value:.6f}) saved to: {path.resolve()}")


def load_threshold_artifact(save_path: Path | str = config.THRESHOLD_PATH) -> Dict[str, Any]:
    """
    Loads saved threshold artifact.
    """
    path = Path(save_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved threshold not found at: '{path.resolve()}'.\n"
            f"Please run model training first: `python -m src.train`"
        )
    artifact = joblib.load(path)
    # Backward compatibility if a raw float was saved
    if isinstance(artifact, (float, int)):
        return {"threshold": float(artifact), "metadata": {}}
    return artifact


def classify_anomalies(
    reconstruction_errors: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies decision boundary:
      reconstruction_error > threshold -> 1 ('Potential Fraud')
      reconstruction_error <= threshold -> 0 ('Normal')
    Returns:
      binary_predictions (0 or 1), text_labels ('Normal' or 'Potential Fraud')
    """
    predictions = (reconstruction_errors > threshold).astype(int)
    labels = np.where(predictions == 1, "Potential Fraud", "Normal")
    return predictions, labels
