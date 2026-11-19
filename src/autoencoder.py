"""
Deep Autoencoder Architecture for FraudLens
Implements a symmetric fully-connected deep neural network in TensorFlow/Keras
to compress 30-dimensional transaction feature vectors into a 4-dimensional latent space
and reconstruct legitimate patterns with minimal reconstruction error.
"""

import io
from pathlib import Path
from typing import Optional, Tuple

import keras
from keras import layers, models, optimizers

import config


def build_autoencoder(
    input_dim: int = config.INPUT_DIM,
    learning_rate: float = config.LEARNING_RATE,
) -> models.Model:
    """
    Constructs the Deep Autoencoder architecture:
      Input (30) -> Dense(16, relu) -> Dense(8, relu)
                 -> Bottleneck(4, relu)
                 -> Dense(8, relu) -> Dense(16, relu) -> Output(30, linear)
    """
    # Input Layer
    inputs = layers.Input(shape=(input_dim,), name="transaction_input")

    # Encoder
    encoded_1 = layers.Dense(
        config.ENCODER_DIM_1,
        activation=config.ACTIVATION_HIDDEN,
        name="encoder_dense_1",
    )(inputs)

    encoded_2 = layers.Dense(
        config.ENCODER_DIM_2,
        activation=config.ACTIVATION_HIDDEN,
        name="encoder_dense_2",
    )(encoded_1)

    # Bottleneck / Latent Representation
    bottleneck = layers.Dense(
        config.BOTTLENECK_DIM,
        activation=config.ACTIVATION_HIDDEN,
        name="latent_bottleneck",
    )(encoded_2)

    # Decoder
    decoded_1 = layers.Dense(
        config.DECODER_DIM_1,
        activation=config.ACTIVATION_HIDDEN,
        name="decoder_dense_1",
    )(bottleneck)

    decoded_2 = layers.Dense(
        config.DECODER_DIM_2,
        activation=config.ACTIVATION_HIDDEN,
        name="decoder_dense_2",
    )(decoded_1)

    # Output Layer (Reconstructed Transaction)
    outputs = layers.Dense(
        config.OUTPUT_DIM,
        activation=config.ACTIVATION_OUTPUT,
        name="reconstructed_output",
    )(decoded_2)

    autoencoder = models.Model(inputs=inputs, outputs=outputs, name="FraudLens_Autoencoder")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(
        optimizer=optimizer,
        loss=config.LOSS_FUNCTION,
        metrics=["mae"],
    )

    return autoencoder


def get_model_summary(model: models.Model) -> str:
    """
    Captures model.summary() as a formatted string.
    """
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


def load_trained_autoencoder(model_path: Path | str = config.MODEL_PATH) -> models.Model:
    """
    Loads serialized Keras autoencoder model.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained autoencoder model not found at: '{path.resolve()}'.\n"
            f"Please run the training script first: `python -m src.train`"
        )
    model = keras.models.load_model(path)
    return model


if __name__ == "__main__":
    print("[INFO] Initializing FraudLens Deep Autoencoder architecture...")
    model = build_autoencoder()
    print(get_model_summary(model))
    print("[SUCCESS] Autoencoder initialized and compiled successfully.")
