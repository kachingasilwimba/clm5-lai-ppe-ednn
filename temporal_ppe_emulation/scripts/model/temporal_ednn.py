#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EDNN Model Training Script
--------------------------
This script defines a function to train an Evidential Regression model using an Evidential Deep Neural Network (EDNN)
for predicting Leaf Area Index (LAI) along with uncertainty quantification. The model is trained on provided features 
and labels, and it saves the trained model while returning predictions with and without uncertainty.

Dependencies:
    - numpy
    - evidential_deep_learning (providing EvidentialRegressorDNN, evidential_reg_loss, load_model)

Author: Kachinga Silwimba et al.
Date: 2025-04-07
License: Apache License
"""

from mlguess.keras.models import EvidentialRegressorDNN
import keras
import numpy as np
from mlguess.keras.callbacks import ReportEpoch
from mlguess.keras.losses import evidential_cat_loss, evidential_reg_loss


def ednn_reg_model(x_train, y_train, model_path, hidden_layers=6, epochs=10, loss_weight=0.01):
    """
    Trains an Evidential Regression model and returns predictions with and without uncertainty.

    Parameters:
        x_train (np.array): Training features.
        y_train (np.array): Training labels.
        model_path (str): Path to save the trained model.
        hidden_layers (int, optional): Number of hidden layers in the EvidentialRegressorDNN model. Default is 6.
        epochs (int, optional): Number of epochs to train the model. Default is 10.
        loss_weight (float, optional): Weight for the evidential regression loss. Default is 0.01.

    Returns:
        tuple:
            p_with_uncertainty (np.array): Predictions with uncertainty (last dimension size should be 3).
            p_without_uncertainty (np.array): Predictions without uncertainty (last dimension size should be 4).
            model (EvidentialRegressorDNN): The trained Evidential Regressor model.
            history: Training history object.
    """
    try:
        #---------- Initialize the model
        model = EvidentialRegressorDNN(hidden_layers=hidden_layers)
        model.compile(
            loss=evidential_reg_loss(loss_weight),
            optimizer="adam",
            metrics=['mse']
        )

        #---------- Train the model
        history = model.fit(x_train, y_train, epochs=epochs, verbose=2)

        #---------- Get predictions with and without uncertainty
        p_with_uncertainty = model.predict(x_train, return_uncertainties=True)
        p_without_uncertainty = model.predict(x_train, return_uncertainties=False)

        #---------- Validate the output shapes
        if p_with_uncertainty.shape[-1] != 3:
            raise ValueError(
                f"Expected 3 outputs for uncertainty predictions; got {p_with_uncertainty.shape[-1]}."
            )
        if p_without_uncertainty.shape[-1] != 4:
            raise ValueError(
                f"Expected 4 outputs for non-uncertainty predictions; got {p_without_uncertainty.shape[-1]}."
            )

        #---------- Save the trained model
        model.save(model_path)
        print(f"Model saved at: {model_path}")

        #---------- Load the saved model to verify successful saving
        loaded_model = load_model(model_path)
        print(f"Model successfully loaded from: {model_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

    return p_with_uncertainty, p_without_uncertainty, model, history