#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
euclidean_distance.py
---------------------
This module defines a function to compute the pointwise Euclidean distance 
between two 2D fields. Since the Euclidean distance for each grid cell is 
√((x - y)^2) and this is equivalent to the absolute difference (|x - y|),
the function returns the absolute difference for each grid cell.

Usage:
    from euclidean_distance import EucDistance
    distance = EucDistance(field1, field2)

Parameters:
    field1 : np.ndarray or xarray.DataArray
        First 2D array (lat, lon) representing EOF spatial patterns.
    field2 : np.ndarray or xarray.DataArray
        Second 2D array (lat, lon) to compare against field1.

Returns:
    distance : np.ndarray or xarray.DataArray
        A 2D array of shape (lat, lon) containing the pointwise Euclidean distance 
        (i.e., absolute difference) for each grid cell.

Dependencies:
    - numpy
"""

import numpy as np

# ---------- Compute the pointwise Euclidean distance
def EucDistance(field1, field2):
    """
    Compute the pointwise Euclidean distance between two 2D fields of shape (lat, lon).

    Since √((x - y)^2) = |x - y|, each grid cell returns the absolute difference.

    Parameters
    ----------
    field1 : np.ndarray or xarray.DataArray
        First 2D array (lat, lon) representing EOF spatial patterns.
    field2 : np.ndarray or xarray.DataArray
        Second 2D array (lat, lon) to compare against field1.

    Returns
    -------
    distance : np.ndarray or xarray.DataArray
        A 2D array of shape (lat, lon) containing the pointwise Euclidean distance (absolute difference)
        for each grid cell.
    """
    # ---------- Return the absolute difference between the two fields
    return np.abs(field1 - field2)

# ---------- Example usage (uncomment below to test the function)
# if __name__ == "__main__":
#     # Create sample data for testing
#     field1 = np.array([[1, 2], [3, 4]])
#     field2 = np.array([[4, 3], [2, 1]])
#     distance = EucDistance(field1, field2)
#     print("Euclidean Distance:\n", distance)
