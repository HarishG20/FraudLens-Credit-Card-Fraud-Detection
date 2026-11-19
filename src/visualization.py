"""
Visualization Module for FraudLens
Generates high-resolution publication-quality figures:
- Training and Validation Loss Curves
- Reconstruction Error Distributions (Normal vs Fraud with Threshold)
- Confusion Matrix Heatmaps
- Precision-Recall Curves with AP / PR-AUC
- ROC Curves with AUC
"""

from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

import config

# Aesthetics setup
sns.set_theme(style="whitegrid", font="sans-serif")
PALETTE = {
    "normal": "#10B981",    # Emerald Green
    "fraud": "#EF4444",     # Crimson Red
    "accent": "#6366F1",    # Indigo / Electric Purple
    "primary": "#3B82F6",   # Electric Blue
    "dark": "#0F172A",      # Slate Navy
}


def plot_training_loss(
    history: Dict[str, Any],
    save_path: Path | str = config.TRAINING_LOSS_PLOT,
) -> None:
    """
    Plots training vs validation loss over training epochs.
    """
    epochs = range(1, len(history["loss"]) + 1)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(epochs, history["loss"], label="Training Loss (MSE)", color="#3B82F6", linewidth=2.5)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Validation Loss (Normal Only)", color="#8B5CF6", linewidth=2.5, linestyle="--")

    ax.set_title("FraudLens Deep Autoencoder - Training Dynamics", fontsize=14, fontweight="bold", pad=12, color=PALETTE["dark"])
    ax.set_xlabel("Epoch", fontsize=11, fontweight="medium")
    ax.set_ylabel("Mean Squared Error (Reconstruction Loss)", fontsize=11, fontweight="medium")
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Training loss curve saved to: {Path(save_path).resolve()}")


def plot_reconstruction_error(
    normal_errors: np.ndarray,
    fraud_errors: np.ndarray,
    threshold: float,
    save_path: Path | str = config.RECONSTRUCTION_ERROR_PLOT,
) -> None:
    """
    Plots reconstruction error distribution comparing normal vs fraudulent transactions
    with decision threshold marked.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. KDE / Histogram Comparison (Log scale for visualization)
    bins = np.linspace(0, min(float(np.percentile(fraud_errors, 95)), 10.0), 80)
    
    ax1.hist(
        normal_errors,
        bins=bins,
        alpha=0.65,
        color=PALETTE["normal"],
        label=f"Normal (n={len(normal_errors):,})",
        density=True,
    )
    ax1.hist(
        fraud_errors,
        bins=bins,
        alpha=0.65,
        color=PALETTE["fraud"],
        label=f"Fraudulent (n={len(fraud_errors):,})",
        density=True,
    )
    ax1.axvline(
        threshold,
        color=PALETTE["dark"],
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold:.4f})",
    )
    ax1.set_title("Reconstruction Error Density Distribution", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Reconstruction Error (MSE)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Scatter / Point view across index
    sample_size = min(3000, len(normal_errors))
    idx_normal = np.random.choice(len(normal_errors), sample_size, replace=False)
    
    ax2.scatter(
        np.arange(sample_size),
        normal_errors[idx_normal],
        c=PALETTE["normal"],
        alpha=0.4,
        s=12,
        label="Normal (Sampled)",
    )
    ax2.scatter(
        np.arange(len(fraud_errors)),
        fraud_errors,
        c=PALETTE["fraud"],
        alpha=0.85,
        s=28,
        edgecolors="black",
        linewidth=0.5,
        label=f"Fraud (All {len(fraud_errors)})",
    )
    ax2.axhline(
        threshold,
        color=PALETTE["dark"],
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold:.4f})",
    )
    ax2.set_title("Transaction Anomaly Scores vs Decision Boundary", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Sample Index", fontsize=11)
    ax2.set_ylabel("Reconstruction Error (MSE)", fontsize=11)
    ax2.set_ylim(0, min(float(np.percentile(fraud_errors, 98)), 15.0))
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Reconstruction error distribution saved to: {Path(save_path).resolve()}")


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path | str = config.CONFUSION_MATRIX_PLOT,
) -> None:
    """
    Plots styled confusion matrix with counts and percentages.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    labels = ["Normal (0)", "Fraud (1)"]
    annot = np.array([
        [f"{cm[0, 0]:,}\n({cm[0, 0]/cm.sum()*100:.2f}%)", f"{cm[0, 1]:,}\n({cm[0, 1]/cm.sum()*100:.2f}%)"],
        [f"{cm[1, 0]:,}\n({cm[1, 0]/cm.sum()*100:.2f}%)", f"{cm[1, 1]:,}\n({cm[1, 1]/cm.sum()*100:.2f}%)"],
    ])

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": 11, "fontweight": "bold"},
    )

    ax.set_title("FraudLens Test Confusion Matrix", fontsize=13, fontweight="bold", pad=12, color=PALETTE["dark"])
    ax.set_xlabel("Predicted Transaction Label", fontsize=11, fontweight="medium")
    ax.set_ylabel("Actual Ground Truth", fontsize=11, fontweight="medium")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to: {Path(save_path).resolve()}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    reconstruction_scores: np.ndarray,
    average_precision: float,
    save_path: Path | str = config.PRECISION_RECALL_PLOT,
) -> None:
    """
    Plots the Precision-Recall curve highlighting Average Precision.
    """
    precision, recall, _ = precision_recall_curve(y_true, reconstruction_scores)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(recall, precision, color="#6366F1", linewidth=2.5, label=f"Autoencoder (AP = {average_precision:.4f})")
    
    # Baseline for random guessing is positive prevalence
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(baseline, color="#94A3B8", linestyle=":", label=f"Random Chance ({baseline:.4f})")

    ax.set_title("Precision-Recall Curve (Fraud Class)", fontsize=13, fontweight="bold", pad=12, color=PALETTE["dark"])
    ax.set_xlabel("Recall (Detection Rate)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="medium")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Precision-Recall curve saved to: {Path(save_path).resolve()}")


def plot_roc_curve(
    y_true: np.ndarray,
    reconstruction_scores: np.ndarray,
    roc_auc: float,
    save_path: Path | str = config.ROC_CURVE_PLOT,
) -> None:
    """
    Plots the Receiver Operating Characteristic (ROC) curve with AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, reconstruction_scores)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(fpr, tpr, color="#0EA5E9", linewidth=2.5, label=f"Autoencoder (ROC-AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", label="Random Classifier (AUC = 0.5000)")

    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12, color=PALETTE["dark"])
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="medium")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="medium")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] ROC curve saved to: {Path(save_path).resolve()}")
