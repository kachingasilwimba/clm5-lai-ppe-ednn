#!/usr/bin/env python3
#----------------
"""
Module: data_processing/lai.py

Provides functionality to load LAI netCDF files, compute anomalies,
and stack target DataArrays for machine learning workflows.

Functions:
    load_lai_anomalies: Load and compute TLAI anomalies for the CONUS domain.
    stack_target: Stack DataArray dimensions into a 1D 'sample' axis.
"""

#----------------
# Imports
#----------------
import logging
from pathlib import Path
from typing import Any

import xarray as xr
import yaml

from .grid import load_conus_gridcells

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

LAI_GLOB = cfg.get("file_paths", {}).get("lai_glob")
if not LAI_GLOB:
    logging.error("Missing 'lai_glob' entry under 'file_paths' in configuration")
    raise KeyError("Configuration key 'file_paths.lai_glob' is required")

#----------------
# Function: load_lai_anomalies
#----------------

def load_lai_anomalies(
    start: str = "1901",
    end:   str = "2014"
) -> xr.DataArray:
    """
    Load LAI NetCDF files, subset to CONUS and time window, compute anomalies.

    Args:
        start: Inclusive start time (year or timestamp).
        end:   Inclusive end time (year or timestamp).

    Returns:
        xr.DataArray of TLAI anomalies over CONUS gridcells.
    """
    logging.debug("Loading CONUS gridcell indices")
    gridcells = load_conus_gridcells()

    logging.debug("Opening LAI datasets with pattern %s", LAI_GLOB)
    ds = (
        xr.open_mfdataset(LAI_GLOB)
          .sel(time=slice(start, end))
          .sel(gridcell=gridcells)
          .astype("float32")
          .persist()
    )
    logging.info(
        "Loaded LAI data from %s to %s for %d gridcells",
        start, end, len(gridcells)
    )

    mean_ds = ds.mean(dim="time")
    anomalies = ds - mean_ds
    logging.info("Computed LAI anomalies (TLAI)")

    return anomalies["TLAI"]

#----------------
# Function: stack_target
#----------------

def stack_target(
    da: xr.DataArray,
    chunk: int = 500_000
) -> xr.DataArray:
    """
    Stack DataArray along member, time, and gridcell into 1D 'sample' axis,
    then rechunk for efficient training.

    Args:
        da: Input DataArray with dims ['member','time','gridcell',...].
        chunk: Desired chunk size for the 'sample' dimension.

    Returns:
        xr.DataArray with a single 'sample' dimension.
    """
    logging.debug(
        "Stacking DataArray into 'sample' dimension with chunk size %d", chunk
    )
    stacked = da.stack(sample=("member", "time", "gridcell")).chunk(sample=chunk)
    logging.info(
        "Stacked target DataArray; new sample size = %d",
        stacked.sizes.get("sample", 0)
    )
    return stacked

#----------------
# Module Test / CLI
#----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running data_processing/lai module as script for testing")
    anomalies = load_lai_anomalies("2000-01-01", "2000-12-31")
    stacked = stack_target(anomalies)
    print(stacked)

