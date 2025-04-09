#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
regrid_to_lat_lon.py
--------------------
This script defines a function to regrid a given xarray.DataArray from a sparse grid to a standard 
latitude/longitude grid. The function loads sparse grid information from a specified NetCDF file and 
determines the output dimensions by including all dimensions of the input data array except the "gridcell" 
dimension (assumed to be of size 400). It then adds latitude and longitude coordinates from the sparse grid 
to the output and fills the output array with values from the corresponding grid cells. Finally, the function 
returns the regridded data as a new xarray.DataArray with the original attributes preserved.

Usage:
    from regrid_to_lat_lon import regrid_to_lat_lon
    regridded_data = regrid_to_lat_lon(data_array)
    print(regridded_data)

Assumptions:
    - The input data array contains a dimension "gridcell" with 400 entries.
    - Sparse grid information is available in the file:
          /bsuhome/ksilwimba/scratch/NCAR/Data/output_v4/clusters.clm51_PPEn02ctsm51d021_2deg_GSWP3V1_leafbiomassesai_PPE3_hist.annual+sd.400.nc
    - A secondary dataset is used to obtain grid cell latitude and longitude values from:
          /bsuhome/ksilwimba/scratch/NCAR/Data/output_v4/PPEn11_transient_LHC0000.clm2.h0.2005-02-01-00000.nc

Dependencies:
    - xarray, numpy, glob, os.path, and optionally pandas.
"""

import glob
import numpy as np
import xarray as xr
from os.path import join

# ---------- Function definition for regridding to a latitude/longitude grid
def regrid_to_lat_lon(data_array: xr.DataArray) -> xr.DataArray:
    """
    Regrids a given data array from a sparse grid to a standard latitude/longitude grid.
    
    It is recommended to perform any dimension-reducing calculations before calling this function to improve performance.
    
    Parameters:
        data_array (xr.DataArray): The data array to regrid.
    
    Returns:
        xr.DataArray: The data array regridded to a standard latitude/longitude grid.
    """
    # ---------- Load the sparse grid information
    base_directory = '/bsuhome/ksilwimba/scratch/NCAR/Data/output_v4/'
    filename = 'clusters.clm51_PPEn02ctsm51d021_2deg_GSWP3V1_leafbiomassesai_PPE3_hist.annual+sd.400.nc'
    sparse_grid = xr.open_dataset(f'{base_directory}{filename}')
    
    # ---------- Determine the output shape and assemble coordinates
    output_shape = []
    coordinates = [] 
    
    # ---------- Include dimensions (other than gridcell) from the input data array
    for coord, size in zip(data_array.coords, data_array.shape):
        if size != 400:  # Assuming 400 is the gridcell count for the sparse grid
            output_shape.append(size)
            coordinates.append((coord, data_array[coord].values))
            
    # ---------- Include latitude and longitude from the sparse grid
    for coord in ['lat', 'lon']:
        size = len(sparse_grid[coord])
        output_shape.append(size)
        coordinates.append((coord, sparse_grid[coord].values))

    # ---------- Initialize the output array with NaNs
    output_array = np.full(output_shape, np.nan)
    
    # ---------- Load a dataset to obtain grid cell coordinate information
    ds = xr.open_dataset('/bsuhome/ksilwimba/scratch/NCAR/Data/output_v4/PPEn11_transient_LHC0000.clm2.h0.2005-02-01-00000.nc')
    
    # ---------- Fill the output array with values from the input data array for each grid cell
    for i in range(400):  # Iterate over grid cells
        lat = ds.grid1d_lat[i]
        lon = ds.grid1d_lon[i]
        cluster_center = sparse_grid.rcent.sel(lat=lat, lon=lon, method='nearest')
        mask = sparse_grid.cclass == cluster_center
        
        if output_array.ndim == 2:
            output_array[mask] = data_array.isel(gridcell=i)
        else:
            num_matches = mask.sum().values
            output_array[:, mask] = np.tile(data_array.isel(gridcell=i).values[:, np.newaxis], [1, num_matches])
    
    # ---------- Create the output DataArray with the computed data and coordinates
    output_data_array = xr.DataArray(data=output_array, name=data_array.name, coords=coordinates)
    output_data_array.attrs = data_array.attrs

    return output_data_array

# ---------- Example usage (uncomment the following lines to test the function)
# if __name__ == "__main__":
#     # Load your input data array (update the file path and variable name accordingly)
#     data_array = xr.open_dataset("path_to_input_data.nc")["variable_name"]
#     regridded_data = regrid_to_lat_lon(data_array)
#     print(regridded_data)
