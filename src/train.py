"""
Training Pipeline for FraudLens Deep Autoencoder
Executes unsupervised training on legitimate transactions, validates reconstruction
fidelity on unseen normal transactions, applies EarlyStopping with weight restoration,
saves the model to models/autoencoder.keras, and calibrates the anomaly threshold.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

import keras
from keras import callbacks
import numpy as np

import config
from src.data_preprocessing import load_dataset, inspect_dataset, split_data, prepare_autoencoder_data
from src.autoencoder import build_autoencoder, get_model_summary
from src.anomaly_detection import (
    compute_reconstruction_error,
    compare_threshold_strategies,
    save_threshold_artifact,
)
from src.visualization import plot_training_loss


def train_fraudlens_autoencoder() -> Dict[str, Any]:
    """
    Coordinates end-to-end training:
    - Preprocessing & data preparation
    - Autoencoder instantiation
    - Model fitting on normal transactions with EarlyStopping
    - Model artifact serialization (.keras)
    - Threshold calibration on validation set
    - Visualization of training dynamics
    """
    start_time = time.time()
    print("=" * 65)
    print("        STARTING FRAUDLENS AUTOENCODER TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load and Preprocess Data
    df = load_dataset()
    inspect_dataset(df)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    prep_data = prepare_autoencoder_data(X_train, y_train, X_val, y_val, X_test, y_test)

    X_train_normal = prep_data["X_train_normal_scaled"]
    X_val_normal = prep_data["X_val_normal_scaled"]
    X_val_all = prep_data["X_val_scaled"]
    y_val = prep_data["y_val"]

    # 2. Build Autoencoder Architecture
    print("\n[INFO] Building custom Deep Autoencoder...")
    autoencoder = build_autoencoder(
        input_dim=config.INPUT_DIM,
        learning_rate=config.LEARNING_RATE,
    )
    print(get_model_summary(autoencoder))

    # 3. Callbacks: EarlyStopping with patience=5 and best weight restoration
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    # 4. Train Autoencoder
    # Notice: Input and Target are both X_train_normal (unsupervised self-reconstruction)
    print(f"\n[INFO] Commencing training on {len(X_train_normal):,} normal transactions...")
    print(f"       Batch Size: {config.BATCH_SIZE} | Max Epochs: {config.MAX_EPOCHS}")

    history = autoencoder.fit(
        X_train_normal,
        X_train_normal,
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        validation_data=(X_val_normal, X_val_normal),
        callbacks=[early_stopping],
        verbose=1,
    )

    # 5. Save Trained Model to .keras format
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    autoencoder.save(config.MODEL_PATH)
    print(f"\n[SUCCESS] Trained model saved to: {config.MODEL_PATH.resolve()}")

    # 6. Generate Training Loss Visualization
    plot_training_loss(history.history, save_path=config.TRAINING_LOSS_PLOT)

    # 7. Anomaly Threshold Calibration (Validation Set)
    print("\n[INFO] Calibrating anomaly detection thresholds on validation set...")
    val_reconstructions = autoencoder.predict(X_val_all, batch_size=config.BATCH_SIZE, verbose=0)
    val_errors = compute_reconstruction_error(X_val_all, val_reconstructions)

    val_normal_reconstructions = autoencoder.predict(X_val_normal, batch_size=config.BATCH_SIZE, verbose=0)
    val_normal_errors = compute_reconstruction_error(X_val_normal, val_normal_reconstructions)

    comparison = compare_threshold_strategies(
        val_errors=val_errors,
        y_val=y_val,
        normal_val_errors=val_normal_errors,
        percentile=config.DEFAULT_PERCENTILE_THRESHOLD,
    )

    print("\n" + "=" * 65)
    print("          THRESHOLD CALIBRATION COMPARISON (VALIDATION SET)")
    print("=" * 65)
    for key, strat in comparison.items():
        print(f"Strategy: {strat['strategy']}")
        print(f"  - Calibrated Threshold : {strat['threshold']:.6f}")
        print(f"  - Precision            : {strat['precision']:.4f}")
        print(f"  - Recall               : {strat['recall']:.4f}")
        print(f"  - F1-Score             : {strat['f1_score']:.4f}")
        print("-" * 65)

    # We select the F1-optimal threshold as primary default (or percentile as alternative)
    # Save the F1-optimal threshold with comparison metadata
    chosen_threshold = comparison["f1_optimal_strategy"]["threshold"]
    save_threshold_artifact(
        threshold_value=chosen_threshold,
        metadata={
            "comparison": comparison,
            "chosen_strategy": "f1_optimal_strategy",
            "percentile_threshold": comparison["percentile_strategy"]["threshold"],
            "f1_optimal_threshold": comparison["f1_optimal_strategy"]["threshold"],
        },
        save_path=config.THRESHOLD_PATH,
    )

    elapsed = time.time() - start_time
    print(f"\n[COMPLETED] Pipeline execution finished in {elapsed:.2f} seconds.")
    print("=" * 65)

    return {
        "autoencoder": autoencoder,
        "history": history.history,
        "comparison": comparison,
        "chosen_threshold": chosen_threshold,
        "prep_data": prep_data,
    }


if __name__ == "__main__":
    try:
        train_fraudlens_autoencoder()
    except Exception as e:
        print(f"\n[FATAL] Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
