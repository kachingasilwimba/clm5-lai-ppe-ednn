#!/usr/bin/env python3
#----------------
"""
Module: data_processing/annual_data_loader.py

Builds an xarray.Dataset of PPE parameters and temporal features,
without materializing large arrays at construction time.

Functions:
    x_input_xr: Load PPE parameters, add cyclical month and year features,
                and chunk for efficient dask processing.
"""

#----------------
# Imports
#----------------
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml

#----------------
# Configuration Loading
#----------------
MODULE_PATH = Path(__file__).resolve()
PROJECT_ROOT = MODULE_PATH.parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

try:
    with open(CONFIG_PATH, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    logging.info("Loaded configuration from %s", CONFIG_PATH)
except FileNotFoundError:
    logging.error("Configuration file not found: %s", CONFIG_PATH)
    raise

#---------------- Extract configuration entries
PPE_CSV: str = cfg.get("file_paths", {}).get("ppe_csv")
CHUNKS: dict = cfg.get("chunk_sizes", {})
if not PPE_CSV:
    logging.error("Missing 'ppe_csv' in configuration under 'file_paths'")
    raise KeyError("Configuration key 'file_paths.ppe_csv' is required")
if not CHUNKS or not all(k in CHUNKS for k in ("member", "time")):
    logging.error("Missing or incomplete 'chunk_sizes' in configuration")
    raise KeyError("Configuration key 'chunk_sizes' with 'member' and 'time' is required")

#----------------
# Function: x_input_xr
#----------------

def x_input_xr(
    param_file_path: Optional[str] = None,
    times: np.ndarray = None
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Create an xarray.Dataset of PPE parameters and time-based features.

    Args:
        param_file_path: Path to PPE CSV file (defaults to config PPE_CSV).
        times: Array of datetime-like objects for the time coordinate.

    Returns:
        X_features: xr.Dataset containing broadcasted PPE params and temporal features.
        params_raw: xr.Dataset of raw PPE parameters indexed by 'member'.
    """
    if times is None:
        logging.error("Argument 'times' must be provided as an array of datetime objects")
        raise ValueError("Parameter 'times' is required")

    #---------------- 1) Load PPE parameter CSV into xarray.Dataset
    path = Path(param_file_path) if param_file_path else Path(PPE_CSV)
    logging.debug("Reading PPE parameters from %s", path)
    ppe_df = pd.read_csv(path)
    params_raw = xr.Dataset.from_dataframe(ppe_df.set_index("member"))
    members = ppe_df["member"].values

    #---------------- 2) Compute cyclical month and year features
    logging.debug("Computing time-based features for %d timestamps", len(times))
    years = np.array([t.year for t in times], dtype="int16")
    months = np.array([t.month for t in times])
    m_sin = np.sin(2 * np.pi * months / 12.0).astype("float32")
    m_cos = np.cos(2 * np.pi * months / 12.0).astype("float32")

    sin_da = xr.DataArray(m_sin, dims=("time",), coords={"time": times})
    cos_da = xr.DataArray(m_cos, dims=("time",), coords={"time": times})
    yr_da = xr.DataArray(years, dims=("time",), coords={"time": times})

    #---------------- 3) Assemble X_features by broadcasting and combining
    X_features = xr.Dataset()
    #---------------- 3a) Broadcast parameters across time
    for var in params_raw.data_vars:
        X_features[var] = (
            params_raw[var]
            .expand_dims(time=times)
            .transpose("member", "time")
        )
    logging.info("Broadcasted PPE parameters across time dimension")

    #---------------- 3b) Broadcast cyclical and year features across members
    for name, da in [("m_sin", sin_da), ("m_cos", cos_da), ("year", yr_da)]:
        X_features[name] = (
            da
            .expand_dims(member=members)
            .transpose("member", "time")
        )
    logging.info("added cyclical month and year features")

    #---------------- 4) Chunk dataset for dask
    logging.debug("Chunking dataset with sizes %s", CHUNKS)
    X_features = X_features.chunk({"member": CHUNKS["member"], "time": CHUNKS["time"]})

    return X_features, params_raw

#----------------
# Module Test / CLI
#----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import pandas as pd
    # Example: generate a range of monthly timestamps
    times = pd.date_range("2000-01-01", "2000-12-01", freq="MS").values
    X, params = x_input_xr(times=times)
    print(X)
    print(params)

