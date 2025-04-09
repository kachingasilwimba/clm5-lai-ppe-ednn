#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
augment_ppe_clm.py
------------------
This script provides functions to process and augment PPE CLM data for machine learning.
It contains two main functions:

1. y_input(clm_ppe_path, start_year, end_year):
   - Loads intermediate processed NetCDF files containing CLM output for all ensemble members.
   - Subsets the data by a given time range and groups the data by season.
   - Stacks the data along the 'member' and 'time' dimensions.
   - Returns four xarray.DataArray objects corresponding to the seasons: DJF, MAM, JJA, and SON.
   
2. x_input(season_data, param_file_path):
   - Generates a machine learning input DataFrame by combining ensemble parameters with cyclical
     encodings of time features.
   - Reads ensemble parameters from a CSV file (which should exclude the 'member' column),
     duplicates the parameters to match the size of the seasonal data, and adds cyclical encoded features
     for the month (using sine and cosine transformations) as well as the year.
   - Returns two pandas DataFrames: one with cyclic features and one without.
   
Usage:
    Import these functions into your project or run the script directly to process your PPE CLM data.
    
Dependencies:
    - glob
    - os.path.join from os
    - xarray
    - pandas
    - numpy
"""

import glob
from os.path import join
import xarray as xr
import pandas as pd
import numpy as np

# ---------- Function to load and stack CLM PPE data for each season
def y_input(clm_ppe_path: str, start_year: int = 1901, end_year: int = 2000) -> tuple:
    """
    Loads intermediate processed CLM output files from the specified directory and stacks the data 
    into a machine learning ingestible format by grouping data by season.
    
    This function loads all available data, subsets it by the provided year range, and groups by season. 
    It then stacks the data along the 'member' and 'time' dimensions.
    
    Parameters:
        clm_ppe_path (str): Directory path containing intermediate processed NetCDF files of CLM output.
        start_year (int, optional): The starting year for data selection. Default is 1901.
        end_year (int, optional): The ending year for data selection. Default is 2000.
        
    Returns:
        tuple: (djf_seas, mam_seas, jja_seas, son_seas) where each element is an xarray.DataArray for the respective season.
    """
    # ---------- Load PPE data from all NetCDF files in the directory
    clm_ppe_files = sorted(glob.glob(join(clm_ppe_path, "*.nc")))
    clm_ppe_data = xr.open_mfdataset(clm_ppe_files, parallel=False).sel(time=slice(f"{start_year}", f"{end_year}"))
    
    # ---------- Group the data by season
    seasonal_data = clm_ppe_data.groupby('time.season')
    
    # ---------- Compute seasonal means (optional reference)
    djf_mean = seasonal_data.mean(dim='time').sel(season='DJF')
    mam_mean = seasonal_data.mean(dim='time').sel(season='MAM')
    jja_mean = seasonal_data.mean(dim='time').sel(season='JJA')
    son_mean = seasonal_data.mean(dim='time').sel(season='SON')
    
    # ---------- Organize seasonal data into a dictionary
    seasons_dict = {season: group for season, group in seasonal_data}
    djf = seasons_dict['DJF']
    mam = seasons_dict['MAM']
    jja = seasons_dict['JJA']
    son = seasons_dict['SON']
    
    # ---------- Stack each seasonal dataset along the ('member', 'time') dimensions
    djf_seas = djf.stack({'sample': ('member', 'time')})
    mam_seas = mam.stack({'sample': ('member', 'time')})
    jja_seas = jja.stack({'sample': ('member', 'time')})
    son_seas = son.stack({'sample': ('member', 'time')})
    
    return djf_seas, mam_seas, jja_seas, son_seas

# ---------- Function to generate machine learning input DataFrames with cyclical encodings
def x_input(season_data: xr.DataArray, param_file_path: str) -> tuple:
    """
    Generates a dataset for machine learning by combining ensemble parameters with cyclical encodings of time features.
    
    This function takes seasonal data (with a 'time' dimension), extracts the year and month from the time coordinate, 
    applies cyclical encoding (sine and cosine) to the month, and duplicates the ensemble parameters (loaded from a CSV
    file, excluding the 'member' column) to match the number of time steps in the seasonal data. The resulting DataFrame 
    includes the ensemble parameters along with the cyclical features (`m_sin`, `m_cos`) and the year.
    
    Parameters:
        season_data (xr.DataArray): Seasonal data (e.g., DJF, MAM, JJA, SON) containing a 'time' dimension.
        param_file_path (str): Path to the CSV file with ensemble parameters (excluding the 'member' column).
        
    Returns:
        tuple:
            x_data (pd.DataFrame): DataFrame containing the ensemble parameters along with cyclical features (m_sin, m_cos, year).
            x_data_no_cyclic (pd.DataFrame): DataFrame containing the repeated ensemble parameters without cyclical features.
    """
    # ---------- Unstack the seasonal data to expose the 'time' dimension
    season_data = season_data.unstack()
    
    # ---------- Determine the number of time steps
    n_time_steps = len(season_data.time)
    
    # ---------- Extract years and months from the 'time' dimension
    years = season_data.time.dt.year.values
    months = season_data.time.dt.month.values
    
    # ---------- Compute cyclical encodings for the month feature
    period = 12
    m_sin = np.sin(months / period * 2 * np.pi)
    m_cos = np.cos(months / period * 2 * np.pi)
    
    # ---------- Load ensemble parameters from the CSV file and drop the 'member' column
    ensemble_parameters = pd.read_csv(param_file_path).drop(columns='member')
    
    # ---------- Repeat the ensemble parameters for each time step of the seasonal data
    x_data = ensemble_parameters.iloc[np.repeat(np.arange(len(ensemble_parameters)), n_time_steps)].copy()
    x_data_no_cyclic = ensemble_parameters.iloc[np.repeat(np.arange(len(ensemble_parameters)), n_time_steps)].copy()
    
    # ---------- Add cyclical encoded month features and the year to the DataFrame
    x_data['m_sin'] = np.tile(m_sin, len(ensemble_parameters))
    x_data['m_cos'] = np.tile(m_cos, len(ensemble_parameters))
    x_data['year'] = np.tile(years, len(ensemble_parameters))
    
    return x_data, x_data_no_cyclic

