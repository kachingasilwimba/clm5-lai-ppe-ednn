#!/usr/bin/env python3
#----------------
"""
Module: data_processing/grid.py

Loads CLM5 grid definitions and extracts gridcell IDs within the
CONUS bounding box for downstream processing.

Functions:
    load_conus_gridcells: Return list of gridcell IDs within CONUS bounds.
"""

#----------------
# Imports
#----------------
import logging
from pathlib import Path
from typing import List

import pandas as pd
import xarray as xr
import yaml

#----------------
# Configuration Loading
#----------------
# Determine project root two levels up
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

# Extract required configuration entries
GRID_PATH = cfg.get("file_paths", {}).get("grid_nc")
BOUNDS = cfg.get("conus")
if not GRID_PATH:
    logging.error("Missing 'grid_nc' path under 'file_paths' in config")
    raise KeyError("Configuration key 'file_paths.grid_nc' is required")
if not BOUNDS or not all(k in BOUNDS for k in ("min_lon", "max_lon", "min_lat", "max_lat")):
    logging.error("Missing or incomplete 'conus' bounds in config")
    raise KeyError("Configuration key 'conus' with min/max lat/lon is required")

#----------------
# Function: load_conus_gridcells
#----------------

def load_conus_gridcells() -> List[int]:
    """
    Load the CLM5 grid dataset and filter gridcells within the CONUS bounding box.

    Returns:
        List[int]: Gridcell IDs falling inside the configured CONUS bounds.
    """
    logging.debug("Opening grid dataset at %s", GRID_PATH)
    ds = xr.open_dataset(GRID_PATH)

    # Build DataFrame of cell coordinates
    logging.debug("Constructing DataFrame of gridcell coordinates")
    df = pd.DataFrame({
        "gridcell": ds["gridcell"].values,
        "longitude": ds["grid1d_lon"].values,
        "latitude": ds["grid1d_lat"].values,
    })

    # Filter to CONUS bounds
    query_str = (
        f"longitude >= {BOUNDS['min_lon']} and longitude <= {BOUNDS['max_lon']} and "
        f"latitude >= {BOUNDS['min_lat']} and latitude <= {BOUNDS['max_lat']}"
    )
    logging.info(
        "Filtering gridcells with bounds: lon [%s, %s], lat [%s, %s]",
        BOUNDS['min_lon'], BOUNDS['max_lon'], BOUNDS['min_lat'], BOUNDS['max_lat']
    )
    conus_df = df.query(query_str)

    gridcell_list = conus_df["gridcell"].tolist()
    logging.info("Selected %d CONUS gridcells", len(gridcell_list))

    return gridcell_list

#----------------
# Module Test / CLI
#----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running data_processing/grid module as script for testing")
    cells = load_conus_gridcells()
    print(f"Loaded {len(cells)} CONUS gridcells: {cells[:10]}...")
