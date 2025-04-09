#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EDNN Model Training Function
----------------------------
This module defines a function that trains an Evidential Regression model using an 
Evidential Deep Neural Network (EDNN) for predicting Leaf Area Index (LAI) with uncertainty 
quantification. The function returns predictions (with and without uncertainty), the trained 
model, and the training history. It also saves the trained model to disk and reloads it to 
verify the saving process.

Parameters:
    x_train (np.ndarray): Training features.
    y_train (np.ndarray): Training labels.
    batch_size (int): Batch size for training.
    model_path (str): Path to save the trained model.
    hidden_layers (int, optional): Number of hidden layers in the EvidentialRegressorDNN model. Default is 6.
    epochs (int, optional): Number of epochs to train the model. Default is 10.
    loss_weight (float, optional): Weight for the evidential regression loss. Default is 0.01.

Returns:
    tuple:
        p_with_uncertainty (np.ndarray): Predictions with uncertainty (last dimension size == 3).
        p_without_uncertainty (np.ndarray): Predictions without uncertainty (last dimension size == 4).
        model (EvidentialRegressorDNN): The trained evidential regression model.
        history: Training history object.
"""

from mlguess.keras.models import EvidentialRegressorDNN
import keras
import numpy as np
from mlguess.keras.losses import evidential_reg_loss
from keras.models import load_model


def ednn_reg_model(x_train: np.ndarray, y_train: np.ndarray, batch_size: int, model_path: str,
                   hidden_layers: int = 6, epochs: int = 10, loss_weight: float = 0.01):
    """
    Trains an Evidential Regression model and returns predictions with and without uncertainty.

    Parameters:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        batch_size (int): Number of samples per training batch.
        model_path (str): Path where the trained model will be saved.
        hidden_layers (int, optional): Number of hidden layers in the EvidentialRegressorDNN. Default is 6.
        epochs (int, optional): Number of training epochs. Default is 10.
        loss_weight (float, optional): Weight for the evidential regression loss. Default is 0.01.

    Returns:
        tuple:
            p_with_uncertainty (np.ndarray): Predictions with uncertainty (last dimension size == 3).
            p_without_uncertainty (np.ndarray): Predictions without uncertainty (last dimension size == 4).
            model (EvidentialRegressorDNN): The trained evidential regression model.
            history: Training history object.
    """
    try:
        #---------- Initialize the Evidential Regression Model
        model = EvidentialRegressorDNN(hidden_layers=hidden_layers)
        
        #---------- Compile the model with evidential regression loss and the Adam optimizer
        model.compile(loss=evidential_reg_loss(loss_weight), optimizer="adam", metrics=['mse'])
        
        #---------- Train the model with specified epochs and batch size
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        
        #---------- Generate predictions with and without uncertainty
        p_with_uncertainty = model.predict(x_train, return_uncertainties=True)
        p_without_uncertainty = model.predict(x_train, return_uncertainties=False)
        
        #---------- Validate prediction shapes
        assert p_with_uncertainty.shape[-1] == 3, "Expected 3 outputs for uncertainty predictions."
        assert p_without_uncertainty.shape[-1] == 4, "Expected 4 outputs for non-uncertainty predictions."
        
        #---------- Save the trained model to the specified path
        model.save(model_path)
        print(f"Model saved at: {model_path}")
        
        #---------- Load the saved model to verify successful saving
        loaded_model = load_model(model_path)
        print(f"Model successfully loaded from: {model_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None
    
    #---------- Return predictions, the model, and the training history
    return p_with_uncertainty, p_without_uncertainty, model, history


# Example usage:
# (Uncomment and modify the following lines as needed)
# x_train = np.random.rand(100, 10)  # Dummy training features
# y_train = np.random.rand(100, 4)   # Dummy training labels
# batch_size = 32
# model_path = "evi_regression_model.keras"
# results = ednn_reg_model(x_train, y_train, batch_size, model_path)
# if results[0] is not None:
#     p_with_uncertainty, p_without_uncertainty, trained_model, history = results
