#!/usr/bin/env python3
#----------------
"""
Module: ednn_model_utils.py

Utility functions to load, compile, set up, and generate predictions
for an EvidentialRegressorDNN model.

Functions:
    load_ednn_model: Load and compile a saved EDNN model with custom objects.
    set_training_variance: Compute and assign training variance to the model.
    predict_ednn: Generate predictions with and without uncertainties.
"""

#----------------
# Imports
#----------------
import logging
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import tensorflow as tf
from mlguess.keras.models import EvidentialRegressorDNN
from mlguess.keras.losses import evidential_reg_loss

#----------------
# Function: load_ednn_model
#----------------

def load_ednn_model(
    model_path: Union[str, Path],
    loss_weight: float
) -> tf.keras.Model:
    """
    Load a saved EvidentialRegressorDNN model, include custom objects,
    and compile it to enable uncertainty predictions.

    Args:
        model_path: Path to the saved EDNN model file.
        loss_weight: Weight parameter for the evidential loss.

    Returns:
        A compiled tf.keras.Model ready for inference.
    """
    model_path = Path(model_path)
    logging.info("Loading EDNN model from %s", model_path)

    #---------------- Load without compilation to include custom objects
    custom_objects = {
        'EvidentialRegressorDNN': EvidentialRegressorDNN,
        'evidential_reg_loss': evidential_reg_loss
    }
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        custom_objects=custom_objects
    )

    #---------------- Compile to enable return_uncertainties in predict
    model.compile(
        loss=evidential_reg_loss(loss_weight),
        metrics=['mse']
    )
    logging.info("Compiled EDNN model with loss_weight=%f", loss_weight)
    return model

#----------------
# Function: set_training_variance
#----------------

def set_training_variance(
    model: tf.keras.Model,
    y_train: np.ndarray
) -> None:
    """
    Compute the variance of training targets and assign to model.training_var,
    required for uncertainty calibration in EDNN.

    Args:
        model: The compiled EDNN model.
        y_train: Training target array of shape (n_samples, ...).
    """
    var = np.var(y_train, axis=0)
    model.training_var = var
    logging.info("Set model.training_var to %s", var)

#----------------
# Function: predict_ednn
#----------------

def predict_ednn(
    model: tf.keras.Model,
    X: Any,
    batch_size: int = 1200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions from an EDNN model with and without uncertainty.

    Args:
        model: A compiled EDNN model.
        X: Input features for prediction.
        batch_size: Batch size for model.predict.

    Returns:
        p_wu: Predictions with uncertainties (shape: [n_samples, 3]).
        p_wo: Predictions without uncertainties (shape: [n_samples, 2]).
    """
    #---------------- Predict mean, aleatoric, epistemic
    p_wu = model.predict(X, batch_size=batch_size, return_uncertainties=True)
    logging.debug("Predicted with uncertainties: %s", p_wu.shape)

    #---------------- Predict mean and variance
    p_wo = model.predict(X, batch_size=batch_size, return_uncertainties=False)
    logging.debug("Predicted without uncertainties: %s", p_wo.shape)

    return p_wu, p_wo

#----------------
# Example Usage
#----------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Define paths and parameters
    MODEL_DIR = "/bsuhome/ksilwimba/scratch/clm5-lai-ppe-ednn/temporal_ppe_emulation/saved_model"
    MODEL_FILE = "ednn_reg_model.keras"
    LOSS_WEIGHT = 1e-10
    # Load and prepare model
    model = load_ednn_model(Path(MODEL_DIR) / MODEL_FILE, LOSS_WEIGHT)
    # Assume y_train is defined elsewhere
    # set_training_variance(model, y_train)
    # Generate predictions
    # p_wu_val, p_wo_val = predict_ednn(model, X_val)
