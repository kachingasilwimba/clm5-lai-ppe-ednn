#!/usr/bin/env python3
#----------------
"""
Module: data_processing/split.py

Provides functionality to split PPE member and time-indexed datasets
into training and validation sets, and perform quantile-based scaling.

Functions:
    split_member_time: Split X, y by member and time, then scale features.
"""

#----------------
# Imports
#----------------
import logging
from typing import Tuple

import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

#----------------
# Default Settings
#----------------
DEFAULT_TRAIN_MEMBERS = 400
DEFAULT_VAL_MEMBERS = 100
DEFAULT_VAL_YEARS = (2000, 2014)
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_QUANTILES = 1000
DEFAULT_OUTPUT_DISTRIBUTION = 'uniform'
DEFAULT_SUBSAMPLE = 100_000

#----------------
# Function: split_member_time
#----------------

def split_member_time(
    X: xr.DataArray,
    y: xr.DataArray,
    train_members: int = DEFAULT_TRAIN_MEMBERS,
    val_members: int = DEFAULT_VAL_MEMBERS,
    val_years: Tuple[int, int] = DEFAULT_VAL_YEARS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    output_distribution: str = DEFAULT_OUTPUT_DISTRIBUTION,
    subsample: int = DEFAULT_SUBSAMPLE,
) -> Tuple[
    xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Split PPE-membered and time-indexed DataArrays into training and validation sets,
    then apply QuantileTransformer scaling to the features.

    Args:
        X: Feature DataArray with 'member' and 'time' coordinates.
        y: Target DataArray aligned with X.
        train_members: Number of ensemble members for training split.
        val_members: Number of ensemble members for validation split.
        val_years: Year range (start, end) for validation period inclusive.
        random_state: Seed for reproducibility of member split and transformer.
        n_quantiles: Number of quantiles for QuantileTransformer.
        output_distribution: Desired distribution for transformed data.
        subsample: Maximum number of samples to use for quantile estimation.

    Returns:
        Tuple containing:
            - X_train_xr, X_val_xr: xarray DataArrays for features.
            - y_train_xr, y_val_xr: xarray DataArrays for targets.
            - X_train, X_val: numpy arrays of scaled features.
            - y_train, y_val: numpy arrays of targets.
    """
    logging.debug("Starting split_member_time")

    #---------------- 1) Member-based split indices
    members = X.coords['member'].values
    unique_mems = np.unique(members)
    train_mems, val_mems = train_test_split(
        unique_mems,
        train_size=train_members,
        test_size=val_members,
        random_state=random_state,
        shuffle=False
    )
    is_train_mem = np.isin(members, train_mems)
    is_val_mem = np.isin(members, val_mems)
    logging.info(
        "Selected %d train members and %d validation members",
        len(train_mems), len(val_mems)
    )

    #---------------- 2) Time-based split mask
    times = X.coords['time'].values
    years = np.array([t.year for t in times])
    val_time_mask = (years >= val_years[0]) & (years <= val_years[1])
    train_time_mask = ~val_time_mask
    logging.info(
        "Time split: validation years %d to %d",
        val_years[0], val_years[1]
    )

    #---------------- 3) Combine masks and compute indices
    train_mask = is_train_mem & train_time_mask
    val_mask = is_val_mem & val_time_mask
    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    logging.debug(
        "Computed train indices (%d) and val indices (%d)",
        train_idx.size, val_idx.size
    )

    #---------------- 4) Slice xarray DataArrays
    X_train_xr = X.isel(sample=train_idx)
    X_val_xr = X.isel(sample=val_idx)
    y_train_xr = y.isel(sample=train_idx)
    y_val_xr = y.isel(sample=val_idx)
    logging.debug("Sliced xarray datasets")

    #---------------- 5) Sanity check time coverage
    train_years = np.unique(years[train_mask])
    val_years_u = np.unique(years[val_mask])
    logging.info(
        "Train spans: %d–%d; Val spans: %d–%d",
        train_years.min(), train_years.max(),
        val_years_u.min(), val_years_u.max()
    )

    #---------------- 6) Materialize and scale feature arrays
    X_train = X_train_xr.data.compute().astype(np.float32)
    X_val = X_val_xr.data.compute().astype(np.float32)
    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        subsample=subsample,
        random_state=random_state
    )
    X_train = qt.fit_transform(X_train)
    X_val = qt.transform(X_val)
    logging.info("Applied QuantileTransformer to feature arrays")

    #---------------- 7) Materialize target arrays
    y_train = y_train_xr.data.compute().astype(np.float32).reshape(-1, 1)
    y_val = y_val_xr.data.compute().astype(np.float32).reshape(-1, 1)
    logging.info(
        "Final shapes: X_train %s, X_val %s; y_train %s, y_val %s",
        X_train.shape, X_val.shape, y_train.shape, y_val.shape
    )

    return (
        X_train_xr, X_val_xr,
        y_train_xr, y_val_xr,
        X_train, X_val,
        y_train, y_val
    )

#----------------
# Module Test / CLI
#----------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Running split_member_time example for verification")
    # Example usage (requires X, y definitions):
    # X, y = load_example_data()
    # results = split_member_time(X, y)
    # print(results)
