#!/usr/bin/env python3
#----------------
"""
Script: scripts/train.py

Orchestrates the end-to-end training pipeline:
1. Build feature matrix and target arrays.
2. Split and scale data.
3. Train an Evidential DNN model with specified hyperparameters.
"""

#----------------
# Imports & Setup
#----------------
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

#---------------- Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    logging.debug("Added project root to sys.path: %s", PROJECT_ROOT)

from model.pipeline import build_feature_matrix_and_target
from data_processing.split import split_member_time
from model.ednn_model import ednn_reg_model

#----------------
# Constants & Configuration
#----------------
# Default model directory (can be overridden via env var)
DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "saved_models"))
MODEL_FILENAME = "ednn_reg_model.keras"

# Hyperparameters for EDNN training
HYPERPARAMS = {
    "hidden_layers": 4,
    "hidden_neurons": 64,
    "activation": "leaky_relu",
    "l2_weight": 7.1073e-04,
    "dropout_alpha": 0.1,
    "use_dropout": False,
    "use_noise": False,
    "noise_sd": 0.1,
    "loss_weight": 1e-10,
    "learning_rate": 1.8102e-04,
    "batch_size": 1200,
    "epochs": 200,
    "patience": 140,
}

#----------------
# Function: main
#----------------
def main():
    """
    Execute the training pipeline:
      1) Load features and target
      2) Split and scale
      3) Train EDNN and save best model
    """
    logging.info("Starting training pipeline")

    #---------------- 1) Build feature matrix and target arrays
    X, y = build_feature_matrix_and_target()
    logging.info("Built feature matrix X%s and target y%s", X.shape, y.shape)

    #---------------- 2) Split and scale data
    (
        X_train_xr, X_val_xr,
        y_train_xr, y_val_xr,
        X_train,    X_val,
        y_train,    y_val
    ) = split_member_time(X, y)
    logging.info(
        "Split data: X_train %s, X_val %s; y_train %s, y_val %s",
        X_train.shape, X_val.shape, y_train.shape, y_val.shape
    )

    #---------------- 3) Prepare output directory
    MODEL_DIR = DEFAULT_MODEL_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILENAME
    logging.info("Model will be saved to %s", model_path)

    #---------------- 4) Train Evidential DNN
    results = ednn_regressor_model(
        x_train=X_train,
        y_train=y_train,
        x_val=X_val,
        y_val=y_val,
        model_path=str(model_path),
        **HYPERPARAMS
    )
    p_wu_val, p_wo_val, p_wu_train, p_wo_train, model, hist = results
    logging.info("Training complete")

    #---------------- 5) Summary
    logging.info(
        "Validation preds w/uncertainty: %s; w/o uncertainty: %s",
        p_wu_val.shape, p_wo_val.shape
    )
    print(f"Model saved at: {model_path}")


#----------------
# Entry Point
#----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()



























# #!/usr/bin/env python3

# # =============================================================================
# # scripts/train.py
# # =============================================================================
# import os, sys
# PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

    
    
# import os
# from model.pipeline        import build_feature_matrix_and_target
# from data_processing.split import split_member_time
# from model.ednn_model    import ednn_reg_model
# import numpy as np


# def main():
#     # 1) Build the full feature & target arrays
#     X, y = build_feature_matrix_and_target()

#     # 2) Split & scale (returns 8 objects)
#     (
#         X_train_xr, X_val_xr,
#         y_train_xr, y_val_xr,
#         X_train,    X_val,
#         y_train,    y_val
#     ) = split_member_time(X, y)

#     # 3) Sanity‐check shapes
#     print("Full X, y shapes:         ", X.shape, y.shape)
#     print("X_train_xr, X_val_xr:     ", X_train_xr.shape, X_val_xr.shape)
#     print("X_train, X_val:           ", X_train.shape,    X_val.shape)
#     print("y_train_xr, y_val_xr:     ", y_train_xr.shape, y_val_xr.shape)
#     print("y_train, y_val:           ", y_train.shape,    y_val.shape)

#     # 4) Prepare model output folder
#     model_folder = "/bsuhome/ksilwimba/scratch/clm5-lai-ppe-ednn/temporal_ppe_emulation/saved_model"
#     os.makedirs(model_folder, exist_ok=True)
#     model_path = os.path.join(model_folder, "ednn_reg_model.keras")

#     # 5) Train the Evidential DNN
#     p_wu_val, p_wo_val, p_wu_train, p_wo_train, model, hist = ednn_reg_model(
#         x_train=X_train, y_train=y_train,
#         x_val=  X_val,   y_val=y_val,
#         model_path=model_path,
#         hidden_layers=4,
#         hidden_neurons=64,
#         activation='leaky_relu',
#         l2_weight=0.00071073,
#         dropout_alpha=0.1,
#         use_dropout=False,
#         use_noise=False,
#         noise_sd=0.1,
#         loss_weight=1e-10,
#         learning_rate=0.00018102,
#         batch_size=1200,
#         epochs=200,
#         patience=140,
#     )

#     # 6) Final summary
#     print(f"Validation preds w/uncertainty: {p_wu_val.shape}")
#     print(f"Validation preds w/o uncertainty: {p_wo_val.shape}")
#     print(f"Model saved to: {model_path}")

# if __name__ == "__main__":
#     main()
