"""
Model Evaluation and Benchmark Module for FraudLens
Evaluates the trained Deep Autoencoder on the held-out, untouched test dataset.
Computes imbalance-aware metrics (Precision, Recall, F1, Average Precision, ROC-AUC),
generates diagnostic plots, and compares against an Isolation Forest baseline.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

import config
from src.data_preprocessing import load_dataset, split_data, prepare_autoencoder_data
from src.autoencoder import load_trained_autoencoder
from src.anomaly_detection import (
    compute_reconstruction_error,
    load_threshold_artifact,
    classify_anomalies,
)
from src.visualization import (
    plot_reconstruction_error,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)

EVALUATION_RESULTS_PATH = config.OUTPUTS_DIR / "evaluation_metrics.json"


def evaluate_autoencoder_on_test(
    autoencoder=None,
    scaler=None,
    threshold: float = None,
    test_data: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Evaluates the trained Autoencoder on the untouched test partition.
    """
    # Load components if not passed
    if autoencoder is None:
        print("[INFO] Loading trained autoencoder from disk...")
        autoencoder = load_trained_autoencoder()

    if threshold is None:
        threshold_artifact = load_threshold_artifact()
        threshold = threshold_artifact["threshold"]

    if test_data is None:
        print("[INFO] Loading and preprocessing test dataset...")
        df = load_dataset()
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
        prep = prepare_autoencoder_data(X_train, y_train, X_val, y_val, X_test, y_test)
        X_test_scaled = prep["X_test_scaled"]
        y_test = prep["y_test"]
    else:
        X_test_scaled = test_data["X_test_scaled"]
        y_test = test_data["y_test"]

    print(f"\n[INFO] Running Autoencoder reconstruction on {len(X_test_scaled):,} test transactions...")
    reconstructed = autoencoder.predict(X_test_scaled, batch_size=config.BATCH_SIZE, verbose=0)
    errors = compute_reconstruction_error(X_test_scaled, reconstructed)

    # Classify based on the fixed threshold
    y_pred, _ = classify_anomalies(errors, threshold=threshold)

    # Calculate metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, errors))
    avg_precision = float(average_precision_score(y_test, errors))
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=["Normal", "Fraud"], output_dict=True)

    normal_errors = errors[y_test == 0]
    fraud_errors = errors[y_test == 1]

    # Generate visual outputs
    plot_confusion_matrix(cm, save_path=config.CONFUSION_MATRIX_PLOT)
    plot_precision_recall_curve(y_test, errors, avg_precision, save_path=config.PRECISION_RECALL_PLOT)
    plot_roc_curve(y_test, errors, roc_auc, save_path=config.ROC_CURVE_PLOT)
    plot_reconstruction_error(normal_errors, fraud_errors, threshold, save_path=config.RECONSTRUCTION_ERROR_PLOT)

    results = {
        "model": "Deep Autoencoder",
        "threshold": float(threshold),
        "test_samples": int(len(y_test)),
        "test_frauds": int(np.sum(y_test)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "error_stats": {
            "normal_mean": float(np.mean(normal_errors)),
            "normal_std": float(np.std(normal_errors)),
            "normal_max": float(np.max(normal_errors)),
            "fraud_mean": float(np.mean(fraud_errors)),
            "fraud_std": float(np.std(fraud_errors)),
            "fraud_max": float(np.max(fraud_errors)),
        },
    }

    print("\n" + "=" * 65)
    print("         FRAUDLENS AUTOENCODER TEST EVALUATION RESULTS")
    print("=" * 65)
    print(f"Test Partition Size   : {len(y_test):,} transactions (74 frauds)")
    print(f"Applied Anomaly Thresh: {threshold:.6f}")
    print(f"Accuracy              : {accuracy * 100:.2f}% (Caution: Imbalanced metric)")
    print(f"Precision (Fraud)     : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall (Fraud)        : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score              : {f1:.4f}")
    print(f"ROC-AUC               : {roc_auc:.4f}")
    print(f"Average Precision (AP): {avg_precision:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0, 0]:,} | FP: {cm[0, 1]:,}")
    print(f"  FN: {cm[1, 0]:,} | TP: {cm[1, 1]:,}")
    print("=" * 65)

    return results, errors


def evaluate_isolation_forest_baseline(
    X_train_normal_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Trains and evaluates an Isolation Forest model as a baseline comparison.
    Trained on normal data with contamination tuned to validation fraud rate.
    """
    print("\n[INFO] Training Isolation Forest baseline...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.0017,  # Approximate fraud proportion in dataset
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_normal_scaled)

    # In isolation forest: -1 is anomaly (fraud), +1 is inlier (normal)
    raw_scores = -iso_forest.score_samples(X_test_scaled)  # Invert so higher means more anomalous
    preds = iso_forest.predict(X_test_scaled)
    y_pred_iso = (preds == -1).astype(int)

    precision = float(precision_score(y_test, y_pred_iso, zero_division=0))
    recall = float(recall_score(y_test, y_pred_iso, zero_division=0))
    f1 = float(f1_score(y_test, y_pred_iso, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, raw_scores))
    avg_precision = float(average_precision_score(y_test, raw_scores))

    baseline_results = {
        "model": "Isolation Forest (Baseline)",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
    }

    print("\n[INFO] Isolation Forest Baseline Results:")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | AP: {avg_precision:.4f}")

    return baseline_results


def run_full_evaluation():
    """
    Executes full evaluation on test set and stores combined metrics.
    """
    df = load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    prep = prepare_autoencoder_data(X_train, y_train, X_val, y_val, X_test, y_test)

    autoencoder = load_trained_autoencoder()
    threshold_artifact = load_threshold_artifact()
    threshold = threshold_artifact["threshold"]

    ae_results, _ = evaluate_autoencoder_on_test(
        autoencoder=autoencoder,
        threshold=threshold,
        test_data=prep,
    )

    # Isolation Forest comparison
    iso_results = evaluate_isolation_forest_baseline(
        X_train_normal_scaled=prep["X_train_normal_scaled"],
        X_test_scaled=prep["X_test_scaled"],
        y_test=prep["y_test"],
    )

    combined_eval = {
        "autoencoder": ae_results,
        "isolation_forest_baseline": iso_results,
    }

    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(combined_eval, f, indent=4)
    print(f"\n[INFO] Comprehensive evaluation metrics exported to: {EVALUATION_RESULTS_PATH.resolve()}")

    return combined_eval


if __name__ == "__main__":
    try:
        run_full_evaluation()
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
