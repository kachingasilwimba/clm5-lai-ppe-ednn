#!/usr/bin/env python3
# ----------------------------
"""
Module to train an Evidential Deep Neural Network (EDNN) regressor with
regularization, noise, dropout, and callbacks.

Functions:
    train_ednn_regressor: Builds, compiles, trains, and evaluates the EDNN model.
"""

# ----------------------------
# Imports
# ----------------------------
import logging
from pathlib import Path
from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from mlguess.keras.losses import evidential_reg_loss
from mlguess.keras.models import EvidentialRegressorDNN

# ----------------------------
# Function: train_ednn_regressor
# ----------------------------
def ednn_regressor_model(
    x_train, y_train,
    x_val,   y_val,
    model_path,
    hidden_layers: int = 4,
    hidden_neurons: int = 64,
    activation: str = "relu",
    l2_weight: float = 1e-3,
    use_dropout: bool = True,
    dropout_alpha: float = 0.2,
    use_noise: bool = False,
    noise_sd: float = 0.01,
    loss_weight: float = 1e-4,
    learning_rate: float = 1e-3,
    batch_size: int = 1024,
    epochs: int = 100,
    patience: int = 10,
):
    """
    Train an EvidentialRegressorDNN with L2 regularization, optional dropout or noise,
    and callbacks for early stopping, learning rate reduction, and checkpointing.

    Args:
        x_train: Training input features.
        y_train: Training target values.
        x_val: Validation input features.
        y_val: Validation target values.
        model_path: File path to save the best model.

    Keyword Args:
        hidden_layers: Number of hidden layers in the network.
        hidden_neurons: Number of neurons per hidden layer.
        activation: Activation function for hidden layers.
        l2_weight: L2 regularization factor for weights.
        use_dropout: Flag to include dropout layers.
        dropout_rate: Dropout rate between layers.
        use_noise: Flag to include Gaussian noise layers.
        noise_stddev: Standard deviation of the Gaussian noise.
        loss_weight: Weighting factor for the evidential loss term.
        learning_rate: Initial learning rate for RMSprop optimizer.
        batch_size: Batch size for training.
        epochs: Maximum number of training epochs.
        patience: Number of epochs with no improvement before stopping.

    Returns:
        Tuple containing:
            - (pred_with_uncertainty_val, uncert_val): Predictions and uncertainties on validation set.
            - pred_val: Predictions without uncertainties on validation set.
            - (pred_with_uncertainty_train, uncert_train): Predictions and uncertainties on training set.
            - pred_train: Predictions without uncertainties on training set.
            - best_model: The best model loaded from disk.
            - history: History object from model.fit.
    """
    #---------------------------- Ensure output directory exists
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    logging.debug("Verified model directory at %s", model_dir)

    #---------------------------- Initialize the Evidential DNN model
    model = EvidentialRegressorDNN(
        hidden_layers=hidden_layers,
        hidden_neurons=hidden_neurons,
        activation=activation,
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        use_dropout=use_dropout,
        dropout_alpha=dropout_alpha,
        use_noise=use_noise,
        noise_sd=noise_sd,
        kernel_reg=tf.keras.regularizers.l2(l2_weight),
        verbose=1,
    )
    logging.info(
        "Initialized EDNN: %d layers x %d neurons, activation=%s",
        hidden_layers,
        hidden_neurons,
        activation,
    )

    #---------------------------- Compile the model
    model.compile(
        loss=evidential_reg_loss(loss_weight),
        metrics=["mse"],
    )
    logging.info("Compiled model with evidential loss (weight=%f)", loss_weight)

    #---------------------------- Setup training callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    logging.debug("Callbacks configured: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint")

    #---------------------------- Train the model
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    logging.info("Training completed; best val_loss=%.4f", min(history.history["val_loss"]))

    #---------------------------- Predict on validation set
    p_wu_val = model.predict(x_val, return_uncertainties=True)  # shape (n_val, 3)
    p_wo_val = model.predict(x_val, return_uncertainties=False) # shape (n_val, 4)
    logging.debug("Validation predictions generated")

    #---------------------------- Predict on training set
    p_with_uncertainty  = model.predict(x_train, return_uncertainties=True)
    p_without_uncertainty = model.predict(x_train, return_uncertainties=False)
    logging.debug("Training predictions generated")

    #---------------------------- Load and compile best model
    best_model = tf.keras.models.load_model(model_path, compile=False)
    best_model.compile(
        loss=evidential_reg_loss(loss_weight),
        metrics=["mse"],
    )
    logging.info("Loaded best model from %s", model_path)

    return p_wu_val, p_wo_val, p_with_uncertainty, p_without_uncertainty, model, history

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage (replace with real data loader)
    # X_train, y_train, X_val, y_val = load_dataset()
    # train_ednn_regressor(X_train, y_train, X_val, y_val, "models/best_ednn.h5")