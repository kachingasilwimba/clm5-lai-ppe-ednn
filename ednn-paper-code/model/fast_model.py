#!/usr/bin/env python3
"""
fast_sensitivity.py

Perform FAST global sensitivity analysis on an Evidential DNN (EDNN) emulator's
predictive mean, varying only the PPE parameter inputs and holding forcing
dimensions at their climatological means.

Usage:
    python fast_sensitivity.py \
        --features features.npy \
        --model ednn_model.keras \
        --scaler scaler.pkl \
        --output santa_sensitivity.csv
"""
import argparse
import numpy as np
import pandas as pd
from SALib.sample.fast_sampler import sample as fast_sample
from SALib.analyze.fast import analyze as fast_analyze


def fast_model_ednn(
    X: np.ndarray,
    feature_names: list,
    ednn_model,
    scaler,
    pred_index: int = 0,
    M: int = 4,
    num_resamples: int = 10000,
    conf_level: float = 0.95,
):
    """
    Run FAST sensitivity analysis on an evidential DNN emulator.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix (samples × features).
    feature_names : list of str
        Column names matching X's features.
    ednn_model : keras.Model
        Trained EDNN model.
    scaler : sklearn.BaseEstimator
        Pre-fit scaler for X.
    pred_index : int, default=0
        Index of the predictive mean in the model's output vector.
    M : int, default=4
        FAST Fourier frequency parameter.
    num_resamples : int, default=10000
        Number of FAST samples to draw.
    conf_level : float, default=0.95
        Confidence level for bootstrap.

    Returns
    -------
    dict
        SALib FAST sensitivity indices.
    """
    #-------------------- DataFrame for slicing by name
    df = pd.DataFrame(X, columns=feature_names)

    #-------------------- Separate forcing vs PPE parameters
    forcing_names = ["TSA", "PRECIP", "m_cos", "m_sin", "year"]
    param_names = [n for n in feature_names if n not in forcing_names]

    #-------------------- Climatological forcing baseline
    baseline_forc = df[forcing_names].mean(axis=0).values

    #-------------------- Define SALib problem
    bounds = [[df[p].min(), df[p].max()] for p in param_names]
    problem = {
        "names": param_names,
        "num_vars": len(param_names),
        "bounds": bounds,
    }

    #-------------------- Sample PPE subspace via FAST
    Xp = fast_sample(problem, num_resamples, M=M, seed=0)
    n_fast = Xp.shape[0]
    #-------------------- Tile forcings to match PPE samples
    forc_mat = np.tile(baseline_forc, (n_fast, 1))

    #-------------------- Reassemble full input matrix
    X_full = np.hstack([Xp, forc_mat])
    df_full = pd.DataFrame(X_full, columns=param_names + forcing_names)
    X_ordered = df_full[feature_names].values

    #-------------------- Scale and predict
    X_scaled = scaler.transform(X_ordered)
    Y = ednn_model.predict(X_scaled, return_uncertainties=False)
    Y_mu = Y[:, pred_index]

    #-------------------- FAST sensitivity analysis
    sens = fast_analyze(
        problem,
        Y_mu,
        M=M,
        num_resamples=500,
        conf_level=conf_level,
        print_to_console=True,
        seed=0,
    )
    return sens


if __name__ == "__main__":
    main()
