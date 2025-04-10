#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fast_model.py
-------------
This script defines a function to perform a Fourier Amplitude Sensitivity Test (FAST) analysis using a trained model and 
the SALib library. The function generates FAST samples based on ensemble parameter data, obtains model predictions on these 
samples, and then performs sensitivity analysis on the specified output using FAST. The analysis can be used to determine 
the sensitivity of the model output (e.g., mean predictions, aleatoric uncertainty, epistemic uncertainty) with respect to 
the input parameters.

Usage:
    from fast_model import fast_model
    sensitivity_results = fast_model(x_data, model, pred_index, M=1, num_resamples=100, conf_level=0.95)
    print(sensitivity_results)

Parameters:
    x_data (pandas.DataFrame): DataFrame containing ensemble parameters (each column representing a parameter).
    model: A trained model with a .predict() method that accepts a sample array and returns outputs.
    pred (int): The column index of the model output to analyze (e.g., 0 for the predicted mean).
    M (int, optional): The interference parameter (number of harmonics to sum) for the Fourier series decomposition. Default is 1.
    num_resamples (int, optional): Number of resamples for bootstrap analysis. Default is 100.
    conf_level (float, optional): Confidence level for the bootstrap analysis. Default is 0.95.

Returns:
    TWS_FAST: The sensitivity analysis results as computed by the SALib FAST analysis.

Dependencies:
    - numpy
    - SALib (for fast_sampler and fast analysis)
    - pandas
"""

import numpy as np
import pandas as pd
from SALib.sample import fast_sampler
from SALib.analyze import fast as fast_analyze

# ---------- Define the FAST analysis function
def fast_model(x_data: pd.DataFrame, model, pred: int, M: int = 1, num_resamples: int = 100, conf_level: float = 0.95):
    """
    Performs a Fourier Amplitude Sensitivity Test (FAST) analysis on the given model output.

    Parameters:
        x_data (pd.DataFrame): DataFrame of ensemble parameters used to define the FAST problem.
        model: A trained model with a .predict() method to compute model output given input samples.
        pred (int): The index of the output to analyze (e.g., 0 for the predicted mean).
        M (int, optional): The interference parameter (number of harmonics to sum) for the Fourier series decomposition. Default is 1.
        num_resamples (int, optional): The number of bootstrap resamples for the FAST analysis. Default is 100.
        conf_level (float, optional): The confidence level for the bootstrap analysis. Default is 0.95.

    Returns:
        TWS_FAST: The sensitivity analysis results as computed by the FAST analysis.
    """
    # ---------- Define the FAST problem using the ensemble parameter names and uniform bounds
    problem = {
        'names': list(x_data.columns),
        'num_vars': len(x_data.columns),
        'bounds': [[-5.2, 5.2] for _ in range(35)],  # 35 is hard-coded; adjust if needed.
    }
    
    # ---------- Generate FAST samples using SALib's fast_sampler
    # Here, 100000 samples are generated with an interference parameter M=4.
    sample = fast_sampler.sample(problem, 100000, M=4, seed=None)
    
    # ---------- Obtain predictions for the generated sample from the model
    Y = model.predict(sample)
    
    # ---------- Perform FAST sensitivity analysis on the selected output column (flattened)
    TWS_FAST = fast_analyze.analyze(problem, Y[:, pred].flatten(), M=4, num_resamples=num_resamples,
                                    conf_level=conf_level, print_to_console=True)
    
    return TWS_FAST

# ---------- Example usage (uncomment for testing)
# if __name__ == "__main__":
#     # Create dummy ensemble parameter DataFrame
#     x_data = pd.DataFrame(np.random.rand(100, 35), columns=[f'param_{i+1}' for i in range(35)])
#     
#     # Define a dummy model with a predict method for testing
#     class DummyModel:
#         def predict(self, sample):
#             # Return dummy outputs with 3 output columns (e.g., [mu, aleatoric, epistemic])
#             return np.random.rand(sample.shape[0], 3)
#     
#     model = DummyModel()
#     pred_index = 0  # Analyze the first output (mu)
#     
#     TWS_FAST = fast_model(x_data, model, pred_index, M=1, num_resamples=100, conf_level=0.95)
#     print("FAST sensitivity analysis results:")
#     print(TWS_FAST)
