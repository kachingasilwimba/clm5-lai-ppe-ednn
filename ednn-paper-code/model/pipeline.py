#!/usr/bin/env python3

# =============================================================================
# model/pipeline.py
# =============================================================================
import xarray as xr
from yaml import safe_load

from data_processing.meteo        import load_meteorological_forcing
from data_processing.lai          import load_lai_anomalies, stack_target
from data_processing.annual_data_loader import x_input_xr

#---------- Load configuration ----------

import os
import yaml

# -----------------------------------------------------------------------------
# locate the project-root, two levels above this file
# -----------------------------------------------------------------------------
MODULE_PATH  = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODULE_PATH))
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

cfg = yaml.safe_load(open(CONFIG_PATH))



# cfg    = safe_load(open("configs/default.yaml"))
chunks = cfg["chunk_sizes"]

def build_feature_matrix_and_target():
    """
    Orchestrates loading forcing, LAI anomalies, stacking targets,
    generating PPE features, and combining into X (features) and y (target).
    Returns (X, y).
    """
    #---------- 1) Meteorological forcing
    met = load_meteorological_forcing()

    #---------- 2) LAI anomalies & stack target
    lai = load_lai_anomalies()
    y   = stack_target(lai)

    #---------- 3) PPE-derived features
    ppe_ds, _ = x_input_xr(times=met.time.values)

    #---------- 4) Build a combined xarray.Dataset
    ds = xr.Dataset(
        coords={
            "member":   ppe_ds.member.values,
            "time":     met.time.values,
            "gridcell": met.gridcell.values
        }
    ).chunk(chunks)

    #---------- 4a) Broadcast PPE params
    for var in ppe_ds.data_vars:
        ds[var] = ppe_ds[var].broadcast_like(ds)

    #---------- 4b) Broadcast meteorological vars
    for var in ["TSA","PRECIP"]:
        ds[var] = met[var].broadcast_like(ds)

    #---------- 5) Stack into 2D feature matrix
    features = list(ppe_ds.data_vars) + ["TSA","PRECIP"]
    X = (
        ds[features]
          .to_array(dim="feature")
          .stack(sample=("member","time","gridcell"))
          .transpose("sample","feature")
          .chunk({"sample": 2_000_000})
    )

    return X, y
