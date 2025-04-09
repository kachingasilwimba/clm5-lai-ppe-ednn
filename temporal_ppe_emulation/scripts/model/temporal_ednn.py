#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EDNN Model Training Function
----------------------------
This function trains an Evidential Regression model using an Evidential Deep Neural Network (EDNN) 
for predicting Leaf Area Index (LAI) with uncertainty quantification. It returns predictions (with 
and without uncertainty), the trained model, and the training history. The function saves the trained 
model to disk and then reloads it to verify the saving process.

Parameters:
    x_train (np.array): Training features.
    y_train (np.array): Training labels.
    model_path (str): Path to save the trained model.
    hidden_layers (int, optional): Number of hidden layers in the EvidentialRegressorDNN model. Default is 6.
    epochs (int, optional): Number of epochs to train the model. Default is 10.
    loss_weight (float, optional): Weight for the evidential regression loss. Default is 0.01.

Returns:
    tuple:
        p_with_uncertainty (np.array): Predictions with uncertainty (last dimension size == 3).
        p_without_uncertainty (np.array): Predictions without uncertainty (last dimension size == 4).
        model (EvidentialRegressorDNN): The trained evidential regression model.
        history: Training history object.
"""

from mlguess.keras.models import EvidentialRegressorDNN
import keras
import numpy as np
from mlguess.keras.losses import evidential_reg_loss
from keras.models import load_model

def ednn_reg_model(x_train, y_train, batch_size, model_path, hidden_layers=6, epochs=10, loss_weight=0.01):
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
    
    #---------- Return predictions, the model, and training history
    return p_with_uncertainty, p_without_uncertainty, model, history


