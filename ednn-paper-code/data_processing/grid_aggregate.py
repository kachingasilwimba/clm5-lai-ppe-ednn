#!/usr/bin/env python3
#----------------
"""
Module: utils/global_mean.py

Compute area-weighted global means for xarray objects using land area weights.

Constants:
    LAI_GLOB: glob pattern for LAI NetCDF files.
    LAND_AREA_PATH: path to sparsegrid land area NetCDF.

Functions:
    global_mean: Compute weighted mean for DataArray or Dataset recursively.
"""

#----------------
# Imports
#----------------
import logging
from pathlib import Path
from typing import Union

import xarray as xr

#----------------
# Constants
#----------------
LAI_GLOB = Path(
    "/bsuhome/ksilwimba/scratch/NCAR/Data/LAI/TLAI"
).as_posix() + "/*.nc"

LAND_AREA_PATH = Path(
    "/bsuhome/ksilwimba/scratch/NCAR/Data/helpers/sparsegrid_landarea.nc"
)

#---------------- Load land area weights once
try:
    land_area = xr.open_dataset(LAND_AREA_PATH).landarea
    logging.info("Loaded land area weights from %s", LAND_AREA_PATH)
except FileNotFoundError:
    logging.error("Land area file not found: %s", LAND_AREA_PATH)
    raise

#----------------
# Function: global_mean
#----------------

def conus_mean(
    data: Union[xr.DataArray, xr.Dataset],
    land_area: xr.DataArray,
    scale_factor: float = 1.0
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute the area-weighted global mean of an xarray DataArray or Dataset.

    This function handles Dataset inputs by recursively computing the weighted mean
    for each variable. For DataArray inputs, it selects matching gridcells,
    broadcasts weights, and computes (sum(data * weight) / sum(weight)).

    Args:
        data: Input DataArray or Dataset with a 'gridcell' dimension.
        land_area: DataArray of land area weights indexed by 'gridcell'.
        scale_factor: Multiplier applied to the computed mean (default: 1.0).

    Returns:
        Weighted mean over 'gridcell', as a computed DataArray or Dataset.
    """
    #---------------- If Dataset, recurse over each variable
    if isinstance(data, xr.Dataset):
        return xr.Dataset({
            var: conus_mean(data[var], land_area, scale_factor)
            for var in data.data_vars
        })

    #---------------- Handle DataArray: align weights to data
    weights = land_area.sel(gridcell=data.gridcell)
    weights = weights.broadcast_like(data)

    #---------------- Compute weighted global mean
    weighted_sum = (data * weights).sum(dim="gridcell")
    total_weight = weights.sum(dim="gridcell")
    mean = scale_factor * (weighted_sum / total_weight)

    #---------------- Trigger computation if using lazy arrays
    return mean.compute()
