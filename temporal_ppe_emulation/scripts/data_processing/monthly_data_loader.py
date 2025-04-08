#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Processing Script for CLM PPE and Ensemble Augmentation
------------------------------------------------------------
This module contains functions to process Community Land Model (CLM) output data and to 
generate a machine learning input dataset that combines ensemble parameters with cyclical 
encoded time features.

Functions:
    - y_input(clm_ppe_path, start_year=1901, end_year=2000):
          Processes CLM output data by grouping by month and stacking it into an ML-ready format.
    - x_input(month_data, param_file_path):
          Generates an augmented dataset by repeating ensemble parameters for each time step, and adding
          cyclical month encodings and year information.
          
Dependencies:
    - numpy
    - pandas
    - xarray
    - glob
    - os.path.join
"""

import glob
import numpy as np
import pandas as pd
import xarray as xr
from os.path import join


def y_input(clm_ppe_path, start_year=1901, end_year=2000):
    """
    Processes CLM output data, grouping it by month and stacking it into a machine learning ingestible format.

    Parameters:
        clm_ppe_path (str): Path to the intermediate processed files of CLM output.
        start_year (int, optional): The starting year for data selection. Default is 1901.
        end_year (int, optional): The ending year for data selection. Default is 2000.

    Returns:
        list: A list of stacked data arrays for each month (from January to December).
    """
    #---------- Define CONUS longitude boundaries
    conus_min_lon = 235
    conus_max_lon = 294

    #---------- Load a sample CLM output file to extract grid cell information
    file_path = "/bsuhome/ksilwimba/scratch/NCAR/Data/output_v4/PPEn11_transient_LHC0000.clm2.h0.2005-02-01-00000.nc"
    file = xr.open_dataset(file_path)

    grid_cells = file.gridcell.values
    longitudes = file.grid1d_lon.values
    grid_lon_df = pd.DataFrame({'GridCell': grid_cells, 'Longitude': longitudes})

    #---------- Select grid cells that fall within the CONUS region
    conus_grid_cells = grid_lon_df[
        (grid_lon_df['Longitude'] >= conus_min_lon) &
        (grid_lon_df['Longitude'] <= conus_max_lon)
    ]

    #---------- Load PPE data from all NetCDF files
    clm_ppe_files = sorted(glob.glob(join(clm_ppe_path, "*.nc")))
    clm_ppe_data = xr.open_mfdataset(
        clm_ppe_files,
        parallel=False
    ).sel(
        time=slice(f"{start_year}", f"{end_year}")
    ).sel(
        gridcell=list(conus_grid_cells.GridCell)
    )

    #---------- Group data by month
    monthly_data = clm_ppe_data.groupby("time.month")
    #---------- Create a dictionary for months
    months_dict = {month: group for month, group in monthly_data}

    #---------- Stack data for each month
    monthly_stacked = {}
    for month in range(1, 13):
        #---------- Extract data for the current month and stack along ('member', 'time')
        month_data = months_dict[month]
        month_stacked = month_data.stack({"sample": ("member", "time")})
        monthly_stacked[month] = month_stacked

    #---------- Return the stacked data arrays for each month
    return [monthly_stacked[month] for month in range(1, 13)]


def x_input(month_data, param_file_path):
    """
    Generates a dataset for machine learning models that combines ensemble parameters with cyclical encodings of time features.

    This function processes seasonal data by applying cyclical encoding to the month feature (to preserve the cyclical nature 
    of months), and then repeats ensemble parameters for each time step of the seasonal data. It returns a DataFrame 
    that combines ensemble parameters, cyclical encoded month features ('m_sin', 'm_cos'), and the year.

    Parameters:
        month_data (xarray.DataArray): The seasonal data (e.g., DJF, MAM, JJA, SON) containing a 'time' dimension.
        param_file_path (str): Path to the CSV file containing ensemble parameters. This CSV should include ensemble member 
            parameters, excluding the 'member' column.

    Returns:
        pandas.DataFrame: A DataFrame containing ensemble parameters, cyclical month encodings, and year information repeated 
        for each ensemble member and time step.
    """
    #---------- Unstack the monthly data to expose the 'time' dimension
    month_data = month_data.unstack()

    #---------- Determine the number of time steps in the data
    n_time_steps = len(month_data.time)
    #---------- Extract years and months from the 'time' dimension
    years = month_data.time.dt.year.values
    months = month_data.time.dt.month.values

    #---------- Apply cyclical encoding to the month feature
    period = 12
    m_sin = np.sin((months / period) * 2 * np.pi)
    m_cos = np.cos((months / period) * 2 * np.pi)

    #---------- Read ensemble parameters from CSV and drop the 'member' column
    ensemble_parameters = pd.read_csv(param_file_path).drop(columns="member")

    #---------- Repeat ensemble parameters for each time step
    x_data = ensemble_parameters.iloc[
        np.repeat(np.arange(len(ensemble_parameters)), n_time_steps)
    ].copy()

    #---------- Add cyclical encoded month features and year information to the data
    x_data["m_sin"] = np.tile(m_sin, len(ensemble_parameters))
    x_data["m_cos"] = np.tile(m_cos, len(ensemble_parameters))
    x_data["year"] = np.tile(years, len(ensemble_parameters))

    return x_data

