"""
FraudLens Configuration Module
Contains all file paths, hyperparameter configurations, and feature definitions.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "creditcard.csv"
SAMPLE_TEST_PATH = DATASET_DIR / "sample_test.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "autoencoder.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.pkl"

OUTPUTS_DIR = BASE_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Output Visualization Paths
TRAINING_LOSS_PLOT = OUTPUTS_DIR / "training_loss.png"
RECONSTRUCTION_ERROR_PLOT = OUTPUTS_DIR / "reconstruction_error.png"
CONFUSION_MATRIX_PLOT = OUTPUTS_DIR / "confusion_matrix.png"
PRECISION_RECALL_PLOT = OUTPUTS_DIR / "precision_recall_curve.png"
ROC_CURVE_PLOT = OUTPUTS_DIR / "roc_curve.png"

# Ensure directories exist
for directory in [DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Dataset Configuration
TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
EXPECTED_FEATURE_COUNT = len(FEATURE_COLUMNS)  # 30

# Stratified Splitting Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Deep Autoencoder Architecture & Hyperparameters
INPUT_DIM = 30
ENCODER_DIM_1 = 16
ENCODER_DIM_2 = 8
BOTTLENECK_DIM = 4
DECODER_DIM_1 = 8
DECODER_DIM_2 = 16
OUTPUT_DIM = 30

ACTIVATION_HIDDEN = "relu"
ACTIVATION_OUTPUT = "linear"

OPTIMIZER = "adam"
LEARNING_RATE = 0.001
LOSS_FUNCTION = "mse"
BATCH_SIZE = 256
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Anomaly Detection Settings
DEFAULT_PERCENTILE_THRESHOLD = 99.0
