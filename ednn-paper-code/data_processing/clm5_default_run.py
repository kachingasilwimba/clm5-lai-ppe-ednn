#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: data_processing/lai.py

Load and preprocess CLM5-PPE TLAI for the CONUS domain.

Functions
---------
load_tlai : Load TLAI from a saved NetCDF, subset by time and gridcells,
            cast to float32, and optionally compute monthly anomalies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import xarray as xr
import yaml

from .grid import load_conus_gridcells

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODULE_PATH = Path(__file__).resolve()
PROJECT_ROOT = MODULE_PATH.parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

try:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    logging.info("Loaded configuration from %s", CONFIG_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

# path to the saved TLAI file, e.g.,
# /glade/work/ksilwimba/PPE-Optimization/clm5-default-run/TLAI_PPE11_transient.nc
TLAI_PATH: str = cfg.get("file_paths", {}).get("tlai_nc", "")
if not TLAI_PATH:
    raise KeyError("Configuration key 'file_paths.tlai_nc' is required")

# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------
def load_tlai(
    start: str = "2000-01-01",
    end: str = "2014-12-31",
    *,
    anomaly: Optional[str] = None,                 # {"clim", "zscore"} or None
    climatology_period: Tuple[str, str] = ("2000-01-01", "2014-12-31"),
    chunks: Optional[dict] = None
) -> xr.Dataset:
    """
    Load the saved TLAI file and subset to CONUS/time window.

    Parameters
    ----------
    start, end : str
        Inclusive time slice for the returned data.
    anomaly : {"clim", "zscore", None}, optional
        If "clim": return monthly anomalies (value - monthly climatology).
        If "zscore": (value - monthly mean) / monthly std.
        If None: return raw TLAI.
    climatology_period : (str, str)
        Period used to compute the monthly climatology for anomalies.
    chunks : dict, optional
        dask chunking to apply when opening (e.g., {"time": 60}).

    Returns
    -------
    xr.Dataset
        Dataset with variable:
          - "TLAI" (float32) if anomaly is None,
          - "TLAI_anom" if anomaly in {"clim", "zscore"}.
        Dimensions are typically ("time", "gridcell").
    """
    gridcells = load_conus_gridcells()

    logging.info("Opening TLAI: %s", TLAI_PATH)
    ds = xr.open_dataset(TLAI_PATH)#, chunks=chunks or {})  # lazy open

    if "TLAI" not in ds:
        raise KeyError(f"'TLAI' not found in {TLAI_PATH}")

    # subset to time + CONUS gridcells and cast
    da = (
        ds["TLAI"]
        .sel(time=slice(start, end))
        .sel(gridcell=gridcells)
        .astype("float32")
        .persist()
    )

    if anomaly is None:
        return da.to_dataset(name="TLAI")

    # compute monthly anomalies on a chosen baseline period
    base = (
        ds["TLAI"]
        .sel(time=slice(*climatology_period))
        .sel(gridcell=gridcells)
        .astype("float32")
    )

    gb = base.groupby("time.month")
    clim_mean = gb.mean("time")

    if anomaly.lower() == "clim":
        anom = da.groupby("time.month") - clim_mean
    elif anomaly.lower() == "zscore":
        clim_std = gb.std("time")
        anom = (da.groupby("time.month") - clim_mean) / (clim_std + 1e-9)
    else:
        raise ValueError("anomaly must be one of {None, 'clim', 'zscore'}")

    anom = anom.rename("TLAI_anom")
    return anom.to_dataset()

# ---------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds_raw = load_tlai("2000-01-01", "2014-12-31")
    print(ds_raw)

    ds_anom = load_tlai("2000-01-01", "2014-12-31", anomaly="clim")
    print(ds_anom)
