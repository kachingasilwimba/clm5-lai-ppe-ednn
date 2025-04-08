#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Augment CLM PPE Seasonal Data Script
-------------------------------------
This module provides functions to process Community Land Model (CLM) output data and generate
machine learning ingestible datasets from both model output and ensemble parameters.
It specifically extracts seasonal means (DJF, MAM, JJA, SON) from the CLM PPE output and stacks 
the data, and it augments ensemble parameters with cyclical encodings of time features for each season.

Functions:
    - y_input(clm_ppe_path, start_year=1901, end_year=2000)
          Loads processed CLM output files, subsets the data to the CONUS region, groups by season,
          and stacks the data into a machine learning format.
    - x_input(season_data, param_file_path)
          Generates a dataset for machine learning that combines ensemble parameters with cyclical 
          encodings for the month (using sine and cosine transformations) and the year.
          
Dependencies:
    - numpy
    - pandas
    - xarray
    - glob
    - os.path.join

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
    Loads intermediate processed CLM output data for all ensemble members,
    subsets the data to the CONUS region, groups the data by season, computes
    seasonal means, and stacks the CLM data into a machine learning ingestible format.

    This function depends on a file path to intermediate processed files of CLM output
    (by year) that each contain variable information for all 500 ensemble members.
    See 'intermediate_data_gen.ipynb' for details on data generation.

    Parameters:
        clm_ppe_path (str): Path to the intermediate processed files (NetCDF) of CLM output.
        start_year (int, optional): Starting year for data selection. Default is 1901.
        end_year (int, optional): Ending year for data selection. Default is 2000.

    Returns:
        tuple: Four xarray.DataArray objects corresponding to the stacked data for each season:
               (djf_seas, mam_seas, jja_seas, son_seas)
    """
    #---------- Define longitude boundaries for the CONUS region
    conus_min_lon = 235
    conus_max_lon = 294

    #---------- Load a sample file to extract gridcell and longitude information
    sample_file_path = "/bsuhome/ksilwimba/scratch/NCAR/Data/output_v4/PPEn11_transient_LHC0000.clm2.h0.2005-02-01-00000.nc"
    sample_file = xr.open_dataset(sample_file_path)

    grid_cells = sample_file.gridcell.values
    longitudes = sample_file.grid1d_lon.values
    grid_lon_df = pd.DataFrame({'GridCell': grid_cells, 'Longitude': longitudes})
    
    #---------- Subset grid cells within the CONUS region
    conus_grid_cells = grid_lon_df[
        (grid_lon_df['Longitude'] >= conus_min_lon) &
        (grid_lon_df['Longitude'] <= conus_max_lon)
    ]

    #---------- Load PPE data from all NetCDF files in the specified path
    clm_ppe_files = sorted(glob.glob(join(clm_ppe_path, "*.nc")))
    clm_ppe_data = xr.open_mfdataset(
        clm_ppe_files,
        parallel=True
    ).sel(
        time=slice(f"{start_year}", f"{end_year}")
    ).sel(
        gridcell=list(conus_grid_cells.GridCell)
    )

    #---------- Group the data by season
    seasonal_data = clm_ppe_data.groupby('time.season')
    
    #---------- (Optional) Compute seasonal means for reference
    djf_mean = seasonal_data.mean(dim='time').sel(season='DJF')
    mam_mean = seasonal_data.mean(dim='time').sel(season='MAM')
    jja_mean = seasonal_data.mean(dim='time').sel(season='JJA')
    son_mean = seasonal_data.mean(dim='time').sel(season='SON')

    #---------- Create a dictionary mapping season names to their respective groups
    seasons_dict = {season: group for season, group in seasonal_data}

    djf = seasons_dict['DJF']
    mam = seasons_dict['MAM']
    jja = seasons_dict['JJA']
    son = seasons_dict['SON']

    #---------- Stack each seasonal dataset along ('member', 'time') into a new dimension 'sample'
    djf_seas = djf.stack({'sample': ('member', 'time')})
    mam_seas = mam.stack({'sample': ('member', 'time')})
    jja_seas = jja.stack({'sample': ('member', 'time')})
    son_seas = son.stack({'sample': ('member', 'time')})

    return djf_seas, mam_seas, jja_seas, son_seas


def x_input(season_data, param_file_path):
    """
    Generates a machine learning dataset that combines ensemble parameters with 
    cyclical encodings of time features from seasonal data.

    This function processes seasonal data by applying cyclical encoding to the month feature 
    (to preserve the cyclical nature of months), and repeats ensemble parameters for each time step 
    in the seasonal data. It returns a DataFrame that contains ensemble parameters, cyclical encoded 
    month features ('m_sin', 'm_cos'), and the year.

    Parameters:
        season_data (xarray.DataArray): Seasonal data (e.g., DJF, MAM, JJA, SON) containing a 'time' dimension.
        param_file_path (str): Path to the CSV file with ensemble parameters (excluding the 'member' column).

    Returns:
        tuple:
            x_data (pandas.DataFrame): DataFrame with ensemble parameters plus cyclical encoded month and year information.
            x_data_no_cyclic (pandas.DataFrame): Copy of the repeated ensemble parameters without the added cyclic features.
    """
    #---------- Unstack to restore the 'time' dimension
    season_data = season_data.unstack()

    n_time_steps = len(season_data.time)
    #---------- Extract years and months from the 'time' dimension
    years = season_data.time.dt.year.values
    months = season_data.time.dt.month.values

    #---------- Apply cyclical encoding to the month feature
    period = 12
    m_sin = np.sin((months / period) * 2 * np.pi)
    m_cos = np.cos((months / period) * 2 * np.pi)

    #---------- Load ensemble parameters from CSV and drop the 'member' column
    ensemble_parameters = pd.read_csv(param_file_path).drop(columns='member')

    #---------- Repeat ensemble parameters for each time step
    x_data = ensemble_parameters.iloc[
        np.repeat(np.arange(len(ensemble_parameters)), n_time_steps)
    ].copy()
    x_data_no_cyclic = ensemble_parameters.iloc[
        np.repeat(np.arange(len(ensemble_parameters)), n_time_steps)
    ].copy()

    #---------- Add cyclical month features and year information
    x_data['m_sin'] = np.tile(m_sin, len(ensemble_parameters))
    x_data['m_cos'] = np.tile(m_cos, len(ensemble_parameters))
    x_data['year'] = np.tile(years, len(ensemble_parameters))

    return x_data, x_data_no_cyclic
