#!/usr/bin/env python3
#----------------
"""
Module: data_processing/modis_lai.py

Load, preprocess, and summarize MODIS LAI monthly data for the CONUS region.

Functions:
    load_modis_lai: Open monthly NetCDF files, fix time coordinate.
    normalize_longitudes: Convert longitude to [0,360) and sort.
    subset_conus: Crop dataset to CONUS lat/lon bounds.
    compute_lai_anomalies: Compute per-gridcell anomalies from climatology.
    compute_spatial_mean_anomaly: Compute spatial (lat,lon) mean of anomaly.
"""
#----------------
# Imports
#----------------
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

#----------------
# Logging
#----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

#----------------
# Functions
#----------------

def fix_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Assign correct time coordinates to each monthly file based on its filename.

    The filename is expected to end with '_YYYY.nc'.
    """
    src = ds.encoding.get('source', '')
    year = int(Path(src).stem.split('_')[-1])
    times = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    ds = ds.assign_coords(time=times)
    logging.debug("Fixed time for %s: %s to %s", src, times[0], times[-1])
    return ds


def load_modis_lai(
    path_pattern: str,
    concat_dim: str = 'time'
) -> xr.Dataset:
    """
    Load and concatenate multiple MODIS LAI NetCDF files along time.

    Args:
        path_pattern: Glob pattern for .nc files (e.g. '/path/to/lai_*.nc').
        concat_dim: Dimension name to concatenate along (default: 'time').

    Returns:
        Concatenated xarray.Dataset with fixed time coordinates.
    """
    files = sorted(Path().glob(path_pattern)) if '**' in path_pattern else sorted(Path(path_pattern).parent.glob(Path(path_pattern).name))
    file_paths = [str(p) for p in files]
    logging.info("Found %d files for pattern %s", len(file_paths), path_pattern)

    ds = xr.open_mfdataset(
        file_paths,
        preprocess=fix_time,
        concat_dim=concat_dim,
        combine='nested',
        combine_attrs='override'
    )
    logging.info("Loaded dataset with dimensions %s", dict(ds.sizes))
    return ds


def normalize_longitudes(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert longitude to the [0,360) range and sort coordinate.
    """
    ds = ds.assign_coords(lon=(ds.lon % 360)).sortby('lon')
    logging.debug("Normalized and sorted longitudes")
    return ds


def subset_conus(
    ds: xr.Dataset,
    bounds: Dict[str, float]
) -> xr.Dataset:
    """
    Subset dataset to CONUS using provided bounds.

    Args:
        ds: Input dataset with 'lat' and 'lon' coords.
        bounds: Dict with 'min_lon','max_lon','min_lat','max_lat'.
    """
    subset = ds.sel(
        lon=slice(bounds['min_lon'], bounds['max_lon']),
        lat=slice(bounds['min_lat'], bounds['max_lat'])
    )
    logging.info(
        "Subset to CONUS: lon[%s,%s], lat[%s,%s] -> %s grid points",
        bounds['min_lon'], bounds['max_lon'],
        bounds['min_lat'], bounds['max_lat'],
        subset.sizes.get('lon', 0) * subset.sizes.get('lat', 0)
    )
    return subset


def compute_lai_anomalies(ds: xr.Dataset) -> xr.DataArray:
    """
    Compute anomalies by subtracting the time-mean from each gridcell's LAI time series.

    Returns:
        DataArray of LAI anomalies (T x lat x lon).
    """
    lai = ds['lai']
    climatology = lai.mean(dim='time')
    anomalies = lai - climatology
    logging.info("Computed LAI anomalies")
    return anomalies


def compute_spatial_mean_anomaly(
    anomalies: xr.DataArray
) -> xr.DataArray:
    """
    Compute spatial average (lat, lon) of the anomaly time series.

    Returns:
        1D DataArray of mean anomaly over time.
    """
    mean_ts = anomalies.mean(dim=['lat', 'lon'])
    logging.info("Computed spatial mean anomaly time series of length %d", mean_ts.size)
    return mean_ts

#----------------
# Example Usage / CLI
#----------------
if __name__ == '__main__':
    #---------------- Define paths and bounds
    PATTERN = "/bsuhome/ksilwimba/scratch/emulation_with_forcing/observation_data/lai_monthly_0.1_20*.nc"
    BOUNDS = {'min_lon': 235, 'max_lon': 294, 'min_lat': 24.5, 'max_lat': 49.5}

    #---------------- Load and preprocess
    ds = load_modis_lai(PATTERN)
    ds = normalize_longitudes(ds)
    ds_conus = subset_conus(ds, BOUNDS)

    #---------------- Compute anomalies and mean
    anom = compute_lai_anomalies(ds_conus)
    ts = compute_spatial_mean_anomaly(anom)

    print(ts)
