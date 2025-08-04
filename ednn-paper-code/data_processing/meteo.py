#!/usr/bin/env python3
#----------------
"""
Module: data_processing/meteo.py

Provides functionality to load and preprocess CLM5-PPE
meteorological forcing data for the CONUS domain.

Functions:
    load_meteorological_forcing: Load variables TSA, RAIN, SNOW;
      subset by time and gridcells; compute total PRECIP.
"""

#----------------
# Imports
#----------------
import os
import logging
from pathlib import Path
from typing import Optional

import xarray as xr
import yaml

from .grid import load_conus_gridcells

#----------------
# Configuration Loading
#----------------
# Determine project root two levels up
MODULE_PATH = Path(__file__).resolve()
PROJECT_ROOT = MODULE_PATH.parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

#---------------- Load YAML configuration
try:
    with open(CONFIG_PATH, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    logging.info("Loaded configuration from %s", CONFIG_PATH)
except FileNotFoundError:
    logging.error("Configuration file not found: %s", CONFIG_PATH)
    raise

#---------------- Extract meteorology file path
MET_PATH: str = cfg.get("file_paths", {}).get("met_nc")
if not MET_PATH:
    logging.error("Missing 'met_nc' path in configuration under 'file_paths'")
    raise KeyError("Configuration key file_paths.met_nc is required")

#----------------
# Function: load_meteorological_forcing
#----------------
def load_meteorological_forcing(
    start: str = "1901",
    end:   str = "2014"
) -> xr.Dataset:
    """
    Load and preprocess meteorological forcing data (TSA, RAIN, SNOW) from NetCDF.

    Args:
        start: Start year or timestamp slice (inclusive).
        end:   End year or timestamp slice (inclusive).

    Returns:
        xr.Dataset containing TSA and computed PRECIP for CONUS gridcells,
        cast to float32.
    """
    logging.debug("Loading CONUS gridcell indices")
    #---------------- Get CONUS gridcell indices from grid module
    gridcells = load_conus_gridcells()

    logging.debug(
        "Opening dataset from %s, slicing time %s to %s",
        MET_PATH, start, end
    )
    #---------------- Open dataset, select variables, subset in time and space, cast types
    ds = (
        xr.open_dataset(MET_PATH)
          .sel(time=slice(start, end))
          .sel(gridcell=gridcells)[["TSA", "RAIN", "SNOW"]]
          .astype("float32")
          .persist()
    )
    logging.info(
        "Dataset loaded: %s to %s for %d gridcells",
        start, end, len(gridcells)
    )

    #---------------- Compute combined precipitation and drop original components
    ds = ds.assign(
        PRECIP = ds["RAIN"] + ds["SNOW"]
    ).drop_vars(["RAIN", "SNOW"]);
    logging.info("Computed PRECIP and dropped RAIN/SNOW variables")

    return ds

#----------------
# Module Test / CLI
#----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running module as script: loading sample data")
    ds_sample = load_meteorological_forcing("2000-01-01", "2000-12-31")
    print(ds_sample)
