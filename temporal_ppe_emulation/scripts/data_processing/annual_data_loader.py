#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Augment CLM PPE Data Script
---------------------------
This module provides functions to process Community Land Model (CLM) output data
and to generate machine learning ingestible datasets from both model output and ensemble
parameters. The functions perform the following tasks:

1. y_input:
   - Loads intermediate processed CLM output files (by year) for all ensemble members.
   - Subsets the data to the CONUS region based on longitude bounds.
   - Stacks the CLM output along the member and time dimensions into a single 'sample' dimension.

2. x_input:
   - Generates an augmented dataset that combines ensemble parameters with cyclical time features.
   - Reads ensemble parameters from a CSV file (excluding the 'member' column).
   - Duplicates ensemble parameters to match the number of time steps in the processed output.
   - Adds cyclical encodings for the month and includes the year to yield unique samples.

Dependencies:
    - numpy
    - pandas
    - xarray
    - glob
    - os.path.join

Usage Example:
    Uncomment and modify the code block in the __main__ section to test the functions.
    
Author: [Your Name]
Date: 2025-04-07
License: [Insert your chosen license, e.g., MIT License]
"""

import glob
import numpy as np
import pandas as pd
import xarray as xr
from os.path import join


def y_input(clm_ppe_path, start_year=1901, end_year=2000):
    """
    Loads intermediate processed CLM output data from multiple NetCDF files, subsets the data
    by the CONUS region, and stacks the data into a machine learning ingestible format.

    This function depends on a file path to intermediate processed files of CLM output (by year)
    that each contain variable information for all 500 ensemble members. See the 'intermediate_data_gen.ipynb'
    notebook for details on data generation.

    Parameters:
        clm_ppe_path (str): Path to the intermediate processed files of CLM output.
        start_year (int, optional): The starting year for data selection. Default is 1901.
        end_year (int, optional): The ending year for data selection. Default is 2000.

    Returns:
        xarray.DataArray: Stacked CLM data with a new 'sample' dimension created from 'member' and 'time'.
    """
    # Define longitude boundaries for the CONUS region
    conus_min_lon = 235
    conus_max_lon = 294

    # Load a sample file to extract grid cell and longitude information
    file_path = "/bsuhome/ksilwimba/scratch/NCAR/Data/output_v4/PPEn11_transient_LHC0000.clm2.h0.2005-02-01-00000.nc"
    file = xr.open_dataset(file_path)
    grid_cells = file.gridcell.values
    longitudes = file.grid1d_lon.values
    grid_lon_df = pd.DataFrame({'GridCell': grid_cells, 'Longitude': longitudes})
    
    # Subset grid cells in the CONUS region
    conus_grid_cells = grid_lon_df[
        (grid_lon_df['Longitude'] >= conus_min_lon) & 
        (grid_lon_df['Longitude'] <= conus_max_lon)
    ]
    
    # Load all PPE files and subset the data by time and grid cells
    clm_ppe_files = sorted(glob.glob(join(clm_ppe_path, "*.nc")))
    clm_ppe_data = xr.open_mfdataset(
        clm_ppe_files,
        parallel=True
    ).sel(
        time=slice(f"{start_year}", f"{end_year}")
    ).sel(
        gridcell=list(conus_grid_cells.GridCell)
    )
    
    # Stack the data along the 'member' and 'time' dimensions into a single dimension 'sample'
    clm_ppe_data = clm_ppe_data.stack({'sample': ('member', 'time')})
    
    return clm_ppe_data


def x_input(data, param_file_path):
    """
    Generates an augmented dataset for machine learning by combining ensemble parameters with 
    cyclical encodings of time features.

    This function processes the given data by:
      - Unstacking the data to access the 'time' dimension.
      - Extracting the year and month from the time coordinate.
      - Creating cyclical encodings (sine and cosine) for the month.
      - Repeating the ensemble parameters (loaded from a CSV file) for each time step.
      - Adding the cyclical features (m_sin, m_cos) and year columns to the resulting DataFrame.

    Parameters:
        data (xarray.DataArray): The data containing a 'time' dimension (e.g., output from y_input).
        param_file_path (str): Path to the CSV file containing ensemble parameters (excluding the 'member' column).

    Returns:
        tuple:
            x_data (pandas.DataFrame): DataFrame containing ensemble parameters with added cyclical encodings and year.
            x_data_no_cyclic (pandas.DataFrame): A duplicate of the original repeated ensemble parameters without cyclical features.
    """
    # Unstack the data to restore the 'time' coordinate
    data = data.unstack()

    n_time_steps = len(data.time)
    # Extract years and months from the 'time' dimension
    years = data.time.dt.year.values
    months = data.time.dt.month.values

    # Apply cyclical encoding to the month feature
    period = 12
    m_sin = np.sin((months / period) * 2 * np.pi)
    m_cos = np.cos((months / period) * 2 * np.pi)

    # Load ensemble parameters from CSV and remove the 'member' column
    ensemble_parameters = pd.read_csv(param_file_path).drop(columns='member')

    # Repeat ensemble parameters to match the number of time steps
    x_data = ensemble_parameters.iloc[
        np.repeat(np.arange(len(ensemble_parameters)), n_time_steps)
    ].copy()
    
    # Also create a duplicate DataFrame without cyclical features (if needed)
    x_data_no_cyclic = ensemble_parameters.iloc[
        np.repeat(np.arange(len(ensemble_parameters)), n_time_steps)
    ].copy()

    # Add cyclical month features and year to the data
    x_data['m_sin'] = np.tile(m_sin, len(ensemble_parameters))
    x_data['m_cos'] = np.tile(m_cos, len(ensemble_parameters))
    x_data['year'] = np.tile(years, len(ensemble_parameters))

    return x_data, x_data_no_cyclic

