#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global Mean Calculation Module
-------------------------------
This module extracts grid cell information for the CONUS region from a sample CLM output file,
and defines a function to compute the global mean of an xarray.DataArray weighted by land area.

Functions:
    - global_mean(data: xr.DataArray, land_area: xr.DataArray, cf: float = 1.0) -> xr.DataArray
          Computes the global mean of the input data, weighted by the land area.
"""

import xarray as xr
import pandas as pd
import numpy as np

#---------- Define CONUS longitude boundaries
conus_min_lon = 235
conus_max_lon = 294

#---------- Load a sample CLM output file to extract grid cell information
file_path = "/bsuhome/ksilwimba/scratch/NCAR/Data/output_v4/PPEn11_transient_LHC0000.clm2.h0.2005-02-01-00000.nc"
file = xr.open_dataset(file_path)

#---------- Extract grid cell and longitude values from the sample file
grid_cells = file.gridcell.values
longitudes = file.grid1d_lon.values

#---------- Create a DataFrame mapping grid cells to longitudes
grid_lon_df = pd.DataFrame({'GridCell': grid_cells, 'Longitude': longitudes})

#---------- Subset grid cells that fall within the CONUS region
conus_grid_cells = grid_lon_df[
    (grid_lon_df['Longitude'] >= conus_min_lon) & 
    (grid_lon_df['Longitude'] <= conus_max_lon)
]

#---------- Global Mean Function Definition
def global_mean(data: xr.DataArray, land_area: xr.DataArray, cf: float = 1.0) -> xr.DataArray:
    """
    Compute the global mean of data weighted by land area.

    Parameters:
        data (xr.DataArray): The input data array for which the global mean needs to be computed.
            Must have a 'gridcell' dimension.
        land_area (xr.DataArray): Land area weights for each gridcell. Must have the same 'gridcell' dimension as data.
        cf (float, optional): A scaling factor to apply to the global mean. Default is 1.0.

    Returns:
        xr.DataArray:
            The global mean of the data, weighted by land area. Ensure that the data and land area 
            are aligned along the 'gridcell' dimension.
    """
    #---------- Apply the land area weighting
    weighted_data = land_area.sel(gridcell=list(conus_grid_cells.GridCell)) * data

    #---------- Compute the global mean: sum weighted data over 'gridcell' and normalize by the sum of land area
    global_mean_value = cf * (weighted_data.sum(dim='gridcell') / land_area.sum(dim='gridcell')).compute()

    return global_mean_value
