#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: data_processing/save_split_masks.py

Purpose
-------
Recreate the member/time masks used by `split_member_time` and persist:
  • Masked X,y subsets for TRAIN and VAL
  • Boolean masks (aligned to original 'sample')
  • Index arrays and chosen member lists

Works with xarray DataArrays that have:
  - dimensions: ('sample', ...)
  - coordinates: 'member' and 'time' (1D arrays aligned to 'sample')

Outputs
-------
out_dir/
  train/
    X_train_masked.nc
    y_train_masked.nc
  val/
    X_val_masked.nc
    y_val_masked.nc
  split_artifacts.npz   # masks, indices, member lists, years, counts

CLI Examples
------------
python -m data_processing.save_split_masks \
  --x-nc /path/to/X.nc --x-var X \
  --y-nc /path/to/y.nc --y-var y \
  --out-dir ./artifacts/splits \
  --train-members 400 --val-members 100 \
  --val-start 2000 --val-end 2014 --random-state 42

If you already have X,y in memory, import and call `save_split_masks(...)`.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import xarray as xr


# ----------------------------
# Default settings (kept in sync with split.py)
# ----------------------------
DEFAULT_TRAIN_MEMBERS = 400
DEFAULT_VAL_MEMBERS = 100
DEFAULT_VAL_YEARS = (2000, 2014)
DEFAULT_RANDOM_STATE = 42


# ----------------------------
# Core helpers
# ----------------------------
def _compute_masks(
    X: xr.DataArray,
    train_members: int,
    val_members: int,
    val_years: Tuple[int, int],
    random_state: int,
):
    """
    Reproduce member/time masks exactly like split_member_time.
    """
    assert "sample" in X.dims, "Expected a 'sample' dimension in X."
    assert "member" in X.coords, "X is missing 'member' coordinate."
    assert "time" in X.coords, "X is missing 'time' coordinate."

    members = X.coords["member"].values
    unique_mems = np.unique(members)

    # NOTE: shuffle=False as in split.py (deterministic order)
    # We do not import sklearn here to avoid the dependency just for masks;
    # split.py used train_test_split(..., shuffle=False). With shuffle=False,
    # the split is deterministic: first train_members, next val_members.
    if train_members + val_members > unique_mems.size:
        raise ValueError("train_members + val_members exceeds total unique members.")

    train_mems = unique_mems[:train_members]
    val_mems = unique_mems[train_members : train_members + val_members]

    is_train_mem = np.isin(members, train_mems)
    is_val_mem = np.isin(members, val_mems)

    times = xr.conventions.decode_cf_datetime(X.coords["time"]).values
    years = np.array([t.astype("datetime64[Y]").astype(int) + 1970 for t in times])
    val_time_mask = (years >= val_years[0]) & (years <= val_years[1])
    train_time_mask = ~val_time_mask

    train_mask = is_train_mem & train_time_mask
    val_mask = is_val_mem & val_time_mask

    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)

    return train_mask, val_mask, train_idx, val_idx, train_mems, val_mems


def _to_da_mask(mask: np.ndarray, X: xr.DataArray, name: str) -> xr.DataArray:
    return xr.DataArray(
        mask.astype(bool),
        dims=("sample",),
        coords={"sample": X.coords.get("sample", np.arange(X.sizes["sample"]))},
        name=name,
    )


def _save_xr(da: xr.DataArray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # NetCDF is a safe default; change to .to_zarr if needed.
    da.to_netcdf(path)


def save_split_masks(
    X: xr.DataArray,
    y: xr.DataArray,
    out_dir: str | Path,
    train_members: int = DEFAULT_TRAIN_MEMBERS,
    val_members: int = DEFAULT_VAL_MEMBERS,
    val_years: Tuple[int, int] = DEFAULT_VAL_YEARS,
    random_state: int = DEFAULT_RANDOM_STATE,
    x_name_train: str = "X_train_masked",
    x_name_val: str = "X_val_masked",
    y_name_train: str = "y_train_masked",
    y_name_val: str = "y_val_masked",
) -> dict:
    """
    Compute masks like `split_member_time` and persist masked datasets + artifacts.
    Returns a small dict with file paths and basic metadata.
    """
    out_dir = Path(out_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    (
        train_mask,
        val_mask,
        train_idx,
        val_idx,
        train_mems,
        val_mems,
    ) = _compute_masks(X, train_members, val_members, val_years, random_state)

    # Masked xarray subsets
    X_train_xr = X.isel(sample=train_idx)
    X_val_xr = X.isel(sample=val_idx)
    y_train_xr = y.isel(sample=train_idx)
    y_val_xr = y.isel(sample=val_idx)

    # Save masked subsets
    _save_xr(X_train_xr.rename(x_name_train), train_dir / f"{x_name_train}.nc")
    _save_xr(y_train_xr.rename(y_name_train), train_dir / f"{y_name_train}.nc")
    _save_xr(X_val_xr.rename(x_name_val), val_dir / f"{x_name_val}.nc")
    _save_xr(y_val_xr.rename(y_name_val), val_dir / f"{y_name_val}.nc")

    # Save boolean masks aligned to original 'sample'
    train_mask_da = _to_da_mask(train_mask, X, "train_mask")
    val_mask_da = _to_da_mask(val_mask, X, "val_mask")
    _save_xr(train_mask_da, out_dir / "train_mask.nc")
    _save_xr(val_mask_da, out_dir / "val_mask.nc")

    # Save indices and metadata (npz)
    np.savez(
        out_dir / "split_artifacts.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        train_mems=train_mems,
        val_mems=val_mems,
        val_years=np.array(val_years, dtype=int),
        total_samples=X.sizes["sample"],
    )

    return {
        "X_train_nc": str(train_dir / f"{x_name_train}.nc"),
        "y_train_nc": str(train_dir / f"{y_name_train}.nc"),
        "X_val_nc": str(val_dir / f"{x_name_val}.nc"),
        "y_val_nc": str(val_dir / f"{y_name_val}.nc"),
        "train_mask_nc": str(out_dir / "train_mask.nc"),
        "val_mask_nc": str(out_dir / "val_mask.nc"),
        "artifacts_npz": str(out_dir / "split_artifacts.npz"),
    }


# ----------------------------
# CLI
# ----------------------------
def _open_da_from_nc(path: str, var: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in {path}. Available: {list(ds.data_vars)}")
    return ds[var]


def main(
    x_nc: Optional[str],
    x_var: Optional[str],
    y_nc: Optional[str],
    y_var: Optional[str],
    out_dir: str,
    train_members: int,
    val_members: int,
    val_start: int,
    val_end: int,
    random_state: int,
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if x_nc and y_nc:
        logging.info("Loading X and y from NetCDF")
        X = _open_da_from_nc(x_nc, x_var or "X")
        y = _open_da_from_nc(y_nc, y_var or "y")
    else:
        raise SystemExit(
            "Provide --x-nc/--x-var and --y-nc/--y-var to load DataArrays from NetCDF."
        )

    meta = save_split_masks(
        X=X,
        y=y,
        out_dir=out_dir,
        train_members=train_members,
        val_members=val_members,
        val_years=(val_start, val_end),
        random_state=random_state,
    )
    logging.info("Saved split artifacts:\n%s", "\n".join(f"- {k}: {v}" for k, v in meta.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Persist masked train/val subsets and masks.")
    p.add_argument("--x-nc", type=str, required=True, help="Path to NetCDF containing X")
    p.add_argument("--x-var", type=str, default="X", help="Variable name for X in the NetCDF")
    p.add_argument("--y-nc", type=str, required=True, help="Path to NetCDF containing y")
    p.add_argument("--y-var", type=str, default="y", help="Variable name for y in the NetCDF")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory")

    p.add_argument("--train-members", type=int, default=DEFAULT_TRAIN_MEMBERS)
    p.add_argument("--val-members", type=int, default=DEFAULT_VAL_MEMBERS)
    p.add_argument("--val-start", type=int, default=DEFAULT_VAL_YEARS[0])
    p.add_argument("--val-end", type=int, default=DEFAULT_VAL_YEARS[1])
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)

    args = p.parse_args()
    main(
        x_nc=args.x_nc,
        x_var=args.x_var,
        y_nc=args.y_nc,
        y_var=args.y_var,
        out_dir=args.out_dir,
        train_members=args.train_members,
        val_members=args.val_members,
        val_start=args.val_start,
        val_end=args.val_end,
        random_state=args.random_state,
    )

    
    
    
#!/usr/bin/env python3
#----------------
"""
Module: data_processing/split.py

Provides functionality to split PPE member and time-indexed datasets
into training and validation sets, and perform quantile-based scaling.

Functions:
    split_member_time: Split X, y by member and time, then scale features.
"""

#----------------
# Imports
#----------------
import logging
from typing import Tuple

import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

#----------------
# Default Settings
#----------------
DEFAULT_TRAIN_MEMBERS = 400
DEFAULT_VAL_MEMBERS = 100
DEFAULT_VAL_YEARS = (2000, 2014)
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_QUANTILES = 1000
DEFAULT_OUTPUT_DISTRIBUTION = 'uniform'
DEFAULT_SUBSAMPLE = 100_000

#----------------
# Function: split_member_time
#----------------

def split_member_time(
    X: xr.DataArray,
    y: xr.DataArray,
    train_members: int = DEFAULT_TRAIN_MEMBERS,
    val_members: int = DEFAULT_VAL_MEMBERS,
    val_years: Tuple[int, int] = DEFAULT_VAL_YEARS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    output_distribution: str = DEFAULT_OUTPUT_DISTRIBUTION,
    subsample: int = DEFAULT_SUBSAMPLE,
) -> Tuple[
    xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Split PPE-membered and time-indexed DataArrays into training and validation sets,
    then apply QuantileTransformer scaling to the features.

    Args:
        X: Feature DataArray with 'member' and 'time' coordinates.
        y: Target DataArray aligned with X.
        train_members: Number of ensemble members for training split.
        val_members: Number of ensemble members for validation split.
        val_years: Year range (start, end) for validation period inclusive.
        random_state: Seed for reproducibility of member split and transformer.
        n_quantiles: Number of quantiles for QuantileTransformer.
        output_distribution: Desired distribution for transformed data.
        subsample: Maximum number of samples to use for quantile estimation.

    Returns:
        Tuple containing:
            - X_train_xr, X_val_xr: xarray DataArrays for features.
            - y_train_xr, y_val_xr: xarray DataArrays for targets.
            - X_train, X_val: numpy arrays of scaled features.
            - y_train, y_val: numpy arrays of targets.
    """
    logging.debug("Starting split_member_time")

    #---------------- 1) Member-based split indices
    members = X.coords['member'].values
    unique_mems = np.unique(members)
    train_mems, val_mems = train_test_split(
        unique_mems,
        train_size=train_members,
        test_size=val_members,
        random_state=random_state,
        shuffle=False
    )
    is_train_mem = np.isin(members, train_mems)
    is_val_mem = np.isin(members, val_mems)
    logging.info(
        "Selected %d train members and %d validation members",
        len(train_mems), len(val_mems)
    )

    #---------------- 2) Time-based split mask
    times = X.coords['time'].values
    years = np.array([t.year for t in times])
    val_time_mask = (years >= val_years[0]) & (years <= val_years[1])
    train_time_mask = ~val_time_mask
    logging.info(
        "Time split: validation years %d to %d",
        val_years[0], val_years[1]
    )

    #---------------- 3) Combine masks and compute indices
    train_mask = is_train_mem & train_time_mask
    val_mask = is_val_mem & val_time_mask
    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    logging.debug(
        "Computed train indices (%d) and val indices (%d)",
        train_idx.size, val_idx.size
    )

    #---------------- 4) Slice xarray DataArrays
    X_train_xr = X.isel(sample=train_idx)
    X_val_xr = X.isel(sample=val_idx)
    y_train_xr = y.isel(sample=train_idx)
    y_val_xr = y.isel(sample=val_idx)
    logging.debug("Sliced xarray datasets")

    #---------------- 5) Sanity check time coverage
    train_years = np.unique(years[train_mask])
    val_years_u = np.unique(years[val_mask])
    logging.info(
        "Train spans: %d–%d; Val spans: %d–%d",
        train_years.min(), train_years.max(),
        val_years_u.min(), val_years_u.max()
    )

    #---------------- 6) Materialize and scale feature arrays
    X_train = X_train_xr.data.compute().astype(np.float32)
    X_val = X_val_xr.data.compute().astype(np.float32)
    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        subsample=subsample,
        random_state=random_state
    )
    X_train = qt.fit_transform(X_train)
    X_val = qt.transform(X_val)
    logging.info("Applied QuantileTransformer to feature arrays")

    #---------------- 7) Materialize target arrays
    y_train = y_train_xr.data.compute().astype(np.float32).reshape(-1, 1)
    y_val = y_val_xr.data.compute().astype(np.float32).reshape(-1, 1)
    logging.info(
        "Final shapes: X_train %s, X_val %s; y_train %s, y_val %s",
        X_train.shape, X_val.shape, y_train.shape, y_val.shape
    )

    return (
        X_train_xr, X_val_xr,
        y_train_xr, y_val_xr,
        X_train, X_val,
        y_train, y_val
    )

#----------------
# Module Test / CLI
#----------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Running split_member_time example for verification")
    # Example usage (requires X, y definitions):
    # X, y = load_example_data()
    # results = split_member_time(X, y)
    # print(results)
