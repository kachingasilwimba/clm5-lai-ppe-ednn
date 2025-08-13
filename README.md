# CLM5 LAI PPE EDNN Emulation

This repository contains code for training an Evidential Deep Neural Network (EDNN) model for simulating Leaf Area Index (LAI) using perturbed parameter ensemble (PPE) data from the Community Land Model (CLM)  and the historical Global Soil Wetness Project Phase 3 (GSWP3) meteorological forcing. In addition, the repository includes scripts for data augmentation, regridding, FAST sensitivity analysis, and visualization of uncertainty maps and scatter plots for the paper "Efficient Emulation and Uncertainty Quantification of CLM5 Leaf Area Index Dynamics Using Evidential Deep Neural Networks".

---

## Repository Structure

- **`evidential_regression.py`**  
  Contains the `evidential_regression_model` function for training an evidential regression model that predicts LAI along with uncertainty quantification. The function saves the trained model, returns predictions (with and without uncertainty), the model, and the training history.

- **`regrid_to_lat_lon.py`**  
  Contains the `regrid_to_lat_lon` function that regrids an xarray.DataArray from a sparse grid to a standard latitude/longitude grid by combining additional grid information.

- **`fast_model.py`**  
  Implements the `fast_model` function to perform a Fourier Amplitude Sensitivity Test (FAST) analysis on model outputs using the SALib library.

- **`uncertainty_maps.py`**  
  Contains code to generate and visualize uncertainty maps for EDNN LAI predictions on a standard latitude/longitude grid. The plots use Cartopy for geospatial visualization and are arranged in a grid with horizontal colorbars.

- **Notebook Cells**  
  A collection of Jupyter Notebook cells are included to perform data augmentation, regridding, model training, sensitivity analysis, and plotting (e.g., scatter plots, histograms with Gaussian fits, and cyclic representations).

- **`environment.yml`**  
  A Conda environment file that defines the necessary dependencies for the project, including installation of the `mlguess` package directly from GitHub and additional packages like SALib.

---

## Installation

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed. To create the Conda environment, run the following command in your terminal from the project root:

```bash
conda env create -f environment.yml
conda activate ednn_env
