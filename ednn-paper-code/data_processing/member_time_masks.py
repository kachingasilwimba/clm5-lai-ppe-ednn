# -*- coding: utf-8 -*-
"""Utilities to build train/validation masks for EDNN experiments."""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split


# ---------- helpers ----------

def _years_from_time(time_coord: xr.DataArray) -> np.ndarray:
    """Convert a time coordinate to integer calendar years.

    Handles CFTime/DatetimeIndex, numpy datetime64, and generic datetime-like arrays.

    Args:
      time_coord: 1D xarray DataArray time coordinate aligned with the target dimension.

    Returns:
      A 1D NumPy array of `int` years with the same length/order as `time_coord`.

    Raises:
      ValueError: If years cannot be inferred from the coordinate values.
    """
    # Fast path: xarray index with .year
    try:
        return np.asarray(time_coord.to_index().year, dtype=int)
    except Exception:
        pass

    # Fallbacks
    v = time_coord.values
    try:
        if np.issubdtype(v.dtype, np.datetime64):
            # Convert to years since 1970 and shift to calendar years
            return (v.astype("datetime64[Y]").astype(int) + 1970).astype(int)
        # Last resort: let pandas figure it out (handles strings/objects/CFTime-like)
        return pd.to_datetime(v).year.to_numpy(dtype=int)
    except Exception as exc:
        raise ValueError("Unable to convert time coordinate to years.") from exc


def _to_member_id(arr: Sequence) -> np.ndarray:
    """Normalize member identifiers to integers.

    Examples:
      - ["LHC0001", "LHC0500"] -> [1, 500]
      - [1, 2, 3] -> [1, 2, 3]

    Args:
      arr: Sequence of member identifiers (strings like 'LHC0001' or integers).

    Returns:
      A 1D NumPy array of `int` member IDs.

    Raises:
      ValueError: If a string member ID contains no digits.
    """
    arr_np = np.asarray(arr)
    if arr_np.dtype.kind in ("U", "O"):  # strings/objects
        out = []
        for a in arr_np:
            s = "".join(ch for ch in str(a) if ch.isdigit())
            if not s:
                raise ValueError(f"Member label {a!r} contains no digits.")
            out.append(int(s))
        return np.asarray(out, dtype=int)
    return arr_np.astype(int)


# ---------- main ----------

def make_masks_for_X_and_y(
    X: xr.DataArray,
    y: xr.DataArray,
    train_members: int = 400,
    val_members: int = 100,
    val_years: Tuple[int, int] = (2000, 2014),
    random_state: int = 42,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Build boolean masks for train/validation splits on X and y.

    This mirrors a member/time split commonly used in CLM5 PPE experiments.
    The member split is determined from `X` (deterministic order; `shuffle=False`),
    then applied to `y` after normalizing member labels to integers.

    Assumptions:
      * `X` and `y` each have a `"sample"` dimension.
      * Each sample has `"member"` and `"time"` coordinates aligned with `"sample"`.
      * `X.member` can be strings like 'LHC0001'; `y.member` may be ints or strings.
      * Validation years are **inclusive** of both endpoints.

    Args:
      X: Features as an xarray DataArray with dims including `"sample"`, and coords
        `"member"` and `"time"` aligned to `"sample"`.
      y: Targets as an xarray DataArray with dims including `"sample"`, and coords
        `"member"` and `"time"` aligned to `"sample"`.
      train_members: Number of unique members to assign to training (from `X`).
      val_members: Number of unique members to assign to validation (from `X`).
      val_years: Inclusive `(start_year, end_year)` for the validation period.
      random_state: Seed for reproducible `train_test_split` (order is preserved by `shuffle=False`).

    Returns:
      A 4-tuple of boolean masks as xarray.DataArray, each over the `"sample"` dim:
        (train_mask_X, val_mask_X, train_mask_y, val_mask_y)

    Raises:
      ValueError: If required coords/dims are missing, or the member split is inconsistent.
    """
    # ---- basic validation ----
    for name, da in (("X", X), ("y", y)):
        if "sample" not in da.dims:
            raise ValueError(f"{name} must have a 'sample' dimension.")
        for c in ("member", "time"):
            if c not in da.coords:
                raise ValueError(f"{name} must have a '{c}' coordinate aligned to 'sample'.")
            if da.coords[c].sizes.get("sample", None) != da.sizes["sample"]:
                raise ValueError(f"{name}.{c} must be a 1D coordinate aligned with 'sample'.")

    y0, y1 = int(val_years[0]), int(val_years[1])
    if y1 < y0:
        raise ValueError("val_years must be (start_year, end_year) with start <= end.")

    # ---- member split chosen from X (deterministic order, shuffle=False) ----
    unique_mems_X = np.unique(np.asarray(X.coords["member"].values))
    if train_members + val_members != unique_mems_X.size:
        raise ValueError(
            f"train_members + val_members ({train_members}+{val_members}) "
            f"!= number of unique members in X ({unique_mems_X.size})."
        )

    train_mems_X, val_mems_X = train_test_split(
        unique_mems_X,
        train_size=train_members,
        test_size=val_members,
        random_state=random_state,
        shuffle=False,  # preserve original order (e.g., 'LHC0001'..'LHC0500')
    )

    # ---- masks for X (string/any members) ----
    years_X = _years_from_time(X.coords["time"])
    is_train_mem_X = np.isin(X.coords["member"].values, train_mems_X)
    is_val_mem_X = np.isin(X.coords["member"].values, val_mems_X)
    val_time_X = (years_X >= y0) & (years_X <= y1)
    train_time_X = ~val_time_X

    train_mask_X = (is_train_mem_X & train_time_X).astype(bool)
    val_mask_X = (is_val_mem_X & val_time_X).astype(bool)

    train_mask_da_X = xr.DataArray(
        train_mask_X,
        dims=("sample",),
        coords={"sample": X.coords["sample"]},
        name="train_mask_X",
    )
    val_mask_da_X = xr.DataArray(
        val_mask_X,
        dims=("sample",),
        coords={"sample": X.coords["sample"]},
        name="val_mask_X",
    )

    # ---- masks for y (normalize members to integers) ----
    train_ids = _to_member_id(train_mems_X)  # typically 1..400
    val_ids = _to_member_id(val_mems_X)      # typically 401..500

    years_y = _years_from_time(y.coords["time"])
    y_member_ids = _to_member_id(y.coords["member"].values)

    is_train_mem_y = np.isin(y_member_ids, train_ids)
    is_val_mem_y = np.isin(y_member_ids, val_ids)
    val_time_y = (years_y >= y0) & (years_y <= y1)
    train_time_y = ~val_time_y

    train_mask_y = (is_train_mem_y & train_time_y).astype(bool)
    val_mask_y = (is_val_mem_y & val_time_y).astype(bool)

    train_mask_da_y = xr.DataArray(
        train_mask_y,
        dims=("sample",),
        coords={"sample": y.coords["sample"]},
        name="train_mask_y",
    )
    val_mask_da_y = xr.DataArray(
        val_mask_y,
        dims=("sample",),
        coords={"sample": y.coords["sample"]},
        name="val_mask_y",
    )

    return train_mask_da_X, val_mask_da_X, train_mask_da_y, val_mask_da_y


# -------------------- Example (Notebook) --------------------
# train_mask_X, val_mask_X, train_mask_y, val_mask_y = make_masks_for_X_and_y(
#     X_da, y_da, train_members=400, val_members=100, val_years=(2000, 2014), random_state=42
# )
# X_train = X_da.sel(sample=train_mask_X)
# X_val   = X_da.sel(sample=val_mask_X)
# y_train = y_da.sel(sample=train_mask_y)
# y_val   = y_da.sel(sample=val_mask_y)
