#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evidential_regression.py
-------------------------
This script defines a function to train an evidential regression model using an Evidential 
Deep Neural Network (EDNN) for predicting Leaf Area Index (LAI) with uncertainty quantification.
The function saves the trained model to disk, and returns predictions (with and without uncertainty),
the trained model, and the Keras History object. 

Usage:
    from evidential_regression import evidential_regression_model
    p_with_uncertainty, p_without_uncertainty, model, history = evidential_regression_model(
        x_train, y_train, model_path, hidden_layers=3, batch_size=500,
        n_output_tasks=400, epochs=10, loss_weight=0.01
    )

Assumptions:
    - x_train and y_train are provided as array-like objects (e.g., numpy arrays).
    - The module 'mlguess.keras.models' contains the class EvidentialRegressorDNN.
    - The module 'mlguess.keras.losses' contains the function evidential_reg_loss.
    - The Keras function load_model is available from keras.models.

Dependencies:
    - numpy
    - mlguess.keras.models
    - mlguess.keras.losses
    - keras.models
"""

import numpy as np
from mlguess.keras.models import EvidentialRegressorDNN
from mlguess.keras.losses import evidential_reg_loss
from keras.models import load_model

# ---------- Define the evidential regression model training function
def evidential_regression_model(x_train: np.ndarray, 
                                y_train: np.ndarray, 
                                model_path: str, 
                                hidden_layers: int = 3, 
                                batch_size: int = 500,
                                n_output_tasks: int = 400, 
                                epochs: int = 10, 
                                loss_weight: float = 0.01):
    """
    Trains an evidential regression model and saves it to the specified path.

    Parameters:
        x_train (np.ndarray): Training input samples, shape (n_samples, n_features).
        y_train (np.ndarray): Target values, shape (n_samples,) or (n_samples, n_output_tasks).
        model_path (str): File path where the trained model will be saved.
        hidden_layers (int, optional): Number of hidden layers in the EvidentialRegressorDNN. Default is 3.
        batch_size (int, optional): Number of samples per gradient update. Default is 500.
        n_output_tasks (int, optional): Number of output tasks. Default is 400.
        epochs (int, optional): Number of epochs for training. Default is 10.
        loss_weight (float, optional): Weight for the evidential regularization loss. Default is 0.01.

    Returns:
        tuple:
            p_with_uncertainty (np.ndarray): Predictions with uncertainty estimates (last dimension size == 3).
            p_without_uncertainty (np.ndarray): Predictions without uncertainty estimates (last dimension size == 4).
            model (EvidentialRegressorDNN): The trained evidential regression model.
            history: Keras History object containing training history.
    """
    try:
        # ---------- Initialize the Evidential Regression Model
        model = EvidentialRegressorDNN(hidden_layers=hidden_layers, n_output_tasks=n_output_tasks)
        
        # ---------- Compile the model with evidential regression loss and the Adam optimizer
        model.compile(loss=evidential_reg_loss(loss_weight), optimizer="adam")
        
        # ---------- Train the model with specified epochs and batch size
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        
        # ---------- Generate predictions with and without uncertainty
        p_with_uncertainty = model.predict(x_train, return_uncertainties=True)
        p_without_uncertainty = model.predict(x_train, return_uncertainties=False)
        
        # ---------- Validate prediction shapes
        assert p_with_uncertainty.shape[-1] == 3, "Expected 3 outputs for uncertainty predictions."
        assert p_without_uncertainty.shape[-1] == 4, "Expected 4 outputs for non-uncertainty predictions."
        
        # ---------- Save the trained model to the specified path
        model.save(model_path)
        print(f"Model successfully saved to: {model_path}")
        
        # ---------- Reload the saved model to verify successful saving (optional)
        loaded_model = load_model(model_path)
        print(f"Model successfully loaded from: {model_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None
    
    # ---------- Return the predictions, the trained model, and the history
    return p_with_uncertainty, p_without_uncertainty, model, history

# ---------- Example usage (uncomment for testing)
# if __name__ == "__main__":
#     # Generate dummy data for testing
#     x_train = np.random.rand(100, 10)  # 100 samples, 10 features
#     y_train = np.random.rand(100, 400)  # 100 samples, 400 output tasks
#     model_path = "evi_regression_model.keras"
#     
#     results = evidential_regression_model(x_train, y_train, model_path)
#     if results[0] is not None:
#         p_with_uncertainty, p_without_uncertainty, trained_model, history = results
#         print("Predictions with uncertainty shape:", p_with_uncertainty.shape)
#         print("Predictions without uncertainty shape:", p_without_uncertainty.shape)
