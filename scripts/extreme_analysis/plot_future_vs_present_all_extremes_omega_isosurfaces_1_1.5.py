"""
author: Fiona Pfäffli
description: This file plots present-day vs. future extremes for different thresholds (Isosurface version).
"""

#%% Enable module visibility for imports
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean as cmo
from get_study_regions import GetRegions as GetRegions
from get_model_datasets_omega import ModelCarbonateChemistryGetter as ModelCarbonateChemistryGetter
from get_model_datasets_omega import ModelGetter as ModelGetter

from importlib import reload
import get_obs_datasets 
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter

from func_for_clim_thresh import ThreshClimFuncs
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs

from plotting_functions_general import PlotFuncs as PlotFuncs
from matplotlib.lines import Line2D

#%% Define threshold parameters
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.omega_arag_instance()  # 5th percentile threshold for isosurface

#%% Define variables
model_temp_resolution = 'daily'
scenarios = ['present', 'ssp585', 'ssp245']
configs = ['romsoc_fully_coupled']
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'

#%%

isosurface_varias = [
    'depth_of_omega_arag_offl_1_isosurface',
    'depth_of_omega_arag_offl_1.5_isosurface'
]

#%% Get the model datasets for isosurface variables
ocean_ds = {}
for config in configs:
    ocean_ds[config] = {}
    for scenario in scenarios:
        print(f'--{config}, {scenario}--')
        try:
            model_path, file_name = ModelCarbonateChemistryGetter.get_isosurface_paths_and_files(
                config=config,
                scenario=scenario,
                simulation_type=simulation_type,
                ensemble_run=ensemble_run,
                temp_resolution=model_temp_resolution,
                parent_model=parent_model
            )
            print('Loading ocean dataset...')
            ocean_ds[config][scenario] = ModelCarbonateChemistryGetter.open_isosurface_files(model_path, file_name)
            print(f'Dataset loaded for config {config} and scenario {scenario}.')
        except Exception as e:
            print(f"Failed to load dataset for config '{config}' and scenario '{scenario}': {e}")



#%% Load isosurface data

variables = {}

for config in configs:
    variables[config] = {}
    for scenario in scenarios:
        print(f'Extracting variables for {config}, {scenario}')
        variables[config][scenario] = {}

        # Ensure dataset exists before extracting
        if scenario in ocean_ds[config]:
            dataset = ocean_ds[config][scenario]

            for varia in isosurface_varias:
                if varia in dataset:
                    variables[config][scenario][varia] = dataset[varia].load()
                    print(f"Loaded {varia} for {config}, {scenario}")
                else:
                    print(f"Variable '{varia}' not found in ocean_ds[{config}][{scenario}]")
        

#%% 
# Get the climatology for each variable
print('Get the climatology')
clims = dict()
for config in configs:
     clims[config] = dict()
     for scenario in scenarios:
          clims[config][scenario] = dict()
          for varia in isosurface_varias:
               print(f'Calculate the climatology for: {config}, {scenario}, {varia}.')
               dummy_clim = ThreshClimFuncs.calc_clim(params,variables[config][scenario][varia])
               smoothed_clim = ThreshClimFuncs.smooth_array(params,dummy_clim)
               smoothed_clim = smoothed_clim.rename({'day_of_year_adjusted': 'time'})
               smoothed_clim = smoothed_clim.assign_coords({'time': pd.date_range('2015-01-01','2015-12-31')})
               clims[config][scenario][varia] = ThreshClimFuncs.repeat_array_multiple_years(smoothed_clim)


#%% LOAD in the threshold for omega

def load_thresholds_from_file(filepath, varia, include_leap_years=True):
    print(f"Loading threshold data from: {filepath}")

    # Load the NetCDF file
    ds = xr.open_dataset(filepath)

    # Ensure the variable exists in the dataset
    if varia not in ds:
        raise KeyError(f"Variable '{varia}' not found in {filepath}.")

    # Extract the threshold data
    threshold = ds[varia]

    # Rename time dimension if needed
    if 'day_of_year_adjusted' in threshold.dims:
        threshold = threshold.rename({'day_of_year_adjusted': 'time'})
    elif 'day_of_year_adjusted' in threshold.coords:
        threshold = threshold.rename_coords({'day_of_year_adjusted': 'time'})

    # Include Feb 29 if needed
    if include_leap_years:
        threshold_366 = include_feb29(threshold)
    else:
        threshold_366 = threshold

    # Store thresholds in a dictionary
    thresholds_mult = {
        'present': threshold,
        'present_plus_meandelta': threshold_366  # Modify as needed
    }

    return thresholds_mult

def include_feb29(data_365):
    """ Adjusts time dimension to include Feb 29 by interpolating between Feb 28 and Mar 1. """
    if np.size(data_365.time) == 365:
        data_365['time'] = pd.date_range('2001-01-01', '2001-12-31')

        # Compute Feb 29 as mean of Feb 28 and Mar 1
        feb29 = (data_365.sel(time="2001-02-28") + data_365.sel(time="2001-03-01")) / 2
        feb29['time'] = pd.to_datetime("2004-02-29")

        # Split dataset into two parts
        part1 = data_365.sel(time=slice(None, "2001-02-28"))
        part1["time"] = pd.date_range("2004-01-01", "2004-02-28")

        part2 = data_365.sel(time=slice("2001-03-01", None))
        part2["time"] = pd.date_range("2004-03-01", "2004-12-31")

        # Concatenate parts with interpolated Feb 29
        data_366 = xr.concat([part1, feb29, part2], dim="time")
        data_366['time'] = np.arange(366)

        return data_366
    else:
        return data_365  # If already 366 days, return as is

# Filepath to NetCDF file
file_path = "/nfs/sea/work/fpfaeffli/future_sim/thresholds_and_climatologies/romsoc_fully_coupled/hobday2016_threshold_and_climatology_5th_percentile_isosurface_1_and_1.5.nc"

# Load the thresholds for both variables
thresholds_1 = load_thresholds_from_file(file_path, "thresh_smoothed_romsoc_fully_coupled_present_depth_of_omega_arag_offl_1_isosurface")
thresholds_1_5 = load_thresholds_from_file(file_path, "thresh_smoothed_romsoc_fully_coupled_present_depth_of_omega_arag_offl_1.5_isosurface")


# %% COMPUTE THE THRESHOLD EXCEEDANCES (EXTREMES MASK)

# Combine both thresholds into a single dictionary
thresholds_mult = {
    'romsoc_fully_coupled': {
        'present': {
            'present': {
                'depth_of_omega_arag_offl_1_isosurface': thresholds_1['present'],
                'depth_of_omega_arag_offl_1.5_isosurface': thresholds_1_5['present']
            },
            'present_plus_meandelta': {
                'depth_of_omega_arag_offl_1_isosurface': thresholds_1['present_plus_meandelta'],
                'depth_of_omega_arag_offl_1.5_isosurface': thresholds_1_5['present_plus_meandelta']
            }
        },
        'ssp245': {
            'present': {
                'depth_of_omega_arag_offl_1_isosurface': thresholds_1['present'],
                'depth_of_omega_arag_offl_1.5_isosurface': thresholds_1_5['present']
            },
            'present_plus_meandelta': {
                'depth_of_omega_arag_offl_1_isosurface': thresholds_1['present_plus_meandelta'],
                'depth_of_omega_arag_offl_1.5_isosurface': thresholds_1_5['present_plus_meandelta']
            }
        },
        'ssp585': {
            'present': {
                'depth_of_omega_arag_offl_1_isosurface': thresholds_1['present'],
                'depth_of_omega_arag_offl_1.5_isosurface': thresholds_1_5['present']
            },
            'present_plus_meandelta': {
                'depth_of_omega_arag_offl_1_isosurface': thresholds_1['present_plus_meandelta'],
                'depth_of_omega_arag_offl_1.5_isosurface': thresholds_1_5['present_plus_meandelta']
            }
        }
    }
}

# Now calculate the extreme masks

print("Calculating the extreme masks...")

def calc_extreme_masks(isosurface_dict, thresh_dict):
    extreme_mask_dict = {}

    for config in configs:
        extreme_mask_dict[config] = {}
        for scenario in scenarios:
            extreme_mask_dict[config][scenario] = {}
            for threshold_type in ['present', 'present_plus_meandelta']:
                print(f'Calculating extreme mask for {config}, {scenario}, {threshold_type}')

                # Initialize dictionary for each threshold type
                extreme_mask_dict[config][scenario][threshold_type] = {}

                for varia in isosurface_varias:
                    try:
                        # Extract actual isosurface data
                        inst_data = isosurface_dict[config][scenario][varia]

                        # Extract corresponding threshold data
                        thresh_data = thresh_dict[config][scenario][threshold_type][varia]

                        # Compute threshold exceedance (boolean mask)
                        extreme_mask_dict[config][scenario][threshold_type][varia] = inst_data > thresh_data
                        print(f"Processed exceedance mask for {varia}")

                    except KeyError:
                        print(f"Warning: {varia} not found in either inst_dict or thresh_dict for {config}, {scenario}, {threshold_type}.")

    return extreme_mask_dict

# Compute threshold exceedances
extreme_mask_dict = calc_extreme_masks(variables, thresholds_mult)


#%%
###############################
#         PLOTTING MAPS       #
###############################

# %%
# Plots the extreme mask for both isosurfaces and all scenarios (extreme vs. non-extreme)

# Load landmask
landmask_etopo = PlotFuncs.get_etopo_data()

def plot_extreme_mask_map(ocean_ds, extreme_mask_dict, landmask_etopo, bounds, varia, filename=None, config="romsoc_fully_coupled", scenario="ssp585"):
    """
    Plots extreme event occurrences for isosurface variables (extreme mask).
    """
    dataset = ocean_ds[config][scenario]

    # 🔍 **Detect correct latitude and longitude names**
    lat_name = None
    lon_name = None
    for coord in dataset.coords:
        if "lat" in coord.lower():
            lat_name = coord
        if "lon" in coord.lower():
            lon_name = coord

    if lat_name is None or lon_name is None:
        raise KeyError("⚠️ Could not find latitude and longitude in dataset!")

    print(f"✅ Found latitude: {lat_name}, longitude: {lon_name}")

    # Extract latitude & longitude dynamically
    lat_rho = dataset[lat_name].values
    lon_rho = dataset[lon_name].values

    # Extract extreme masks
    mask_fixed = extreme_mask_dict[config][scenario]["present"][varia].squeeze()
    mask_moving = extreme_mask_dict[config][scenario]["present_plus_meandelta"][varia].squeeze()

    if "time" in mask_fixed.dims:
        mask_fixed = mask_fixed.isel(time=0)
        mask_moving = mask_moving.isel(time=0)

    mask_fixed = mask_fixed.squeeze().values
    mask_moving = mask_moving.squeeze().values

    # Define spatial bounds
    lat_bounds, lon_bounds = bounds

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    for idx, (mask_data, ax, title) in enumerate(zip([mask_fixed, mask_moving], axs, ["Fixed Baseline", "Moving Baseline"])):
        ax.set_facecolor("white")

        # **Ensure landmask is binary (1=land, 0=ocean)**
        landmask_binary = np.where(landmask_etopo > 0.5, 1, np.nan)  # Land = 1, Ocean = NaN
        ax.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_binary, levels=[0.5, 1.5], colors="black", zorder=3)

        # **Plot extreme mask using `contourf`**
        contour = ax.contourf(lon_rho, lat_rho, mask_data, levels=[0, 0.5, 1], colors=["white", "red"], zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Longitude")
        if idx == 0:
            ax.set_ylabel("Latitude")

        # Add degree symbols and N/W indicators
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticklabels([f"{xtick}°E" if xtick <= 180 else f"{360 - xtick}°W" for xtick in xticks])

    # Add colorbar with labels
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label="Extreme Event Occurrence", ticks=[0, 1])
    cbar.ax.set_yticklabels(['Non-Extreme', 'Extreme'])

    plt.suptitle(f"Occurrence of Extreme Events ({scenario}, {varia})", fontsize=20, y=1.08)

    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    plt.show()


lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)
base_path = "/nfs/sea/work/fpfaeffli/plots/extreme_masks/"

# Loop for plotting
for varia in isosurface_varias:
    for scenario in scenarios:
        filename = f"{base_path}extreme_mask_map_{config}_{scenario}_{varia}.png"
        plot_extreme_mask_map(ocean_ds, extreme_mask_dict, landmask_etopo, bounds, varia, filename, config=config, scenario=scenario)


# %%

# Plots the mean depth of the saturation horizon over the CaCS during extreme events
# Compares the present scenario to the future scenarios (fixed and moving baselines) for omega = 1 and omega = 1.5 each


# Load landmask
landmask_etopo = PlotFuncs.get_etopo_data()

# Define bounds for the California Current System (CaCS)
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

# Function to plot extreme depth maps following the driver analysis format
def plot_extreme_depths_present_future(data_present, data_future_fixed, data_future_moving, lon_rho, lat_rho, landmask_etopo, bounds, vmin, vmax, title, filename):
    lat_bounds, lon_bounds = bounds

    fig, axs = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
    levels = np.linspace(vmin, vmax, 40)

    for idx, (data, ax, period) in enumerate(zip(
        [data_present, data_future_fixed, data_future_moving], 
        axs, 
        ["Present", "Future (Fixed Baseline)", "Future (Moving Baseline)"]
    )):
        

        # Add landmask
        ax.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, levels=1, colors='black', zorder=1)

        # Plot data
        contour = ax.contourf(lon_rho, lat_rho, data, levels=levels, cmap=cmo.deep_r, vmin=vmin, vmax=vmax, extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_title(period, fontsize=14)
        ax.set_xlabel("Longitude")
        if idx == 0:
            ax.set_ylabel("Latitude")

        # **Add degree symbols and N/W indicators**
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

    # **Add colorbar**
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label="Mean depth during extremes (m)", ticks=np.linspace(vmin, vmax, 11))

    # **Add title**
    plt.suptitle(title, fontsize=20, y=1.08)
    # **Add a small suptitle to indicate scenario**
    if varia == "depth_of_omega_arag_offl_1_isosurface":
        fig.text(1.0, 1.08, f'Ω = 1', ha='right', fontsize=14)
    elif varia == "depth_of_omega_arag_offl_1.5_isosurface":
        fig.text(1.0, 1.08, f'Ω = 1.5', ha='right', fontsize=14)

    # **Save and Show**
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()

# %%

# Plot the extreme depths for each variable and scenario
for varia in isosurface_varias:
    print(f"Plotting extreme depths for {varia}...")

    # Get latitude & longitude from dataset
    dataset = ocean_ds["romsoc_fully_coupled"]["present"]  # Use present scenario to get lat/lon
    lat_name = next(coord for coord in dataset.coords if "lat" in coord.lower())
    lon_name = next(coord for coord in dataset.coords if "lon" in coord.lower())

    lat_rho = dataset[lat_name].values
    lon_rho = dataset[lon_name].values

    # Extract present, future (fixed baseline), and future (moving baseline) data
    data_present_xr = ocean_ds["romsoc_fully_coupled"]["present"][varia].squeeze()
    data_future_fixed_xr_ssp245 = ocean_ds["romsoc_fully_coupled"]["ssp245"][varia].squeeze()
    data_future_moving_xr_ssp245 = extreme_mask_dict["romsoc_fully_coupled"]["ssp245"]["present_plus_meandelta"][varia].squeeze()

    data_future_fixed_xr_ssp585 = ocean_ds["romsoc_fully_coupled"]["ssp585"][varia].squeeze()
    data_future_moving_xr_ssp585 = extreme_mask_dict["romsoc_fully_coupled"]["ssp585"]["present_plus_meandelta"][varia].squeeze()

    # Compute the mean over time, only where extreme events occur
    def compute_mean_if_exists(mask, data):
        if np.any(mask):  # Only compute mean if mask contains any True values
            return np.nanmean(np.where(mask, data, np.nan), axis=0)

    data_present_mean = compute_mean_if_exists(extreme_mask_dict["romsoc_fully_coupled"]["present"]["present"][varia], data_present_xr)
    data_future_fixed_mean_ssp245 = compute_mean_if_exists(extreme_mask_dict["romsoc_fully_coupled"]["ssp245"]["present"][varia], data_future_fixed_xr_ssp245)
    data_future_moving_mean_ssp245 = compute_mean_if_exists(extreme_mask_dict["romsoc_fully_coupled"]["ssp245"]["present_plus_meandelta"][varia], data_future_moving_xr_ssp245)

    data_future_fixed_mean_ssp585 = compute_mean_if_exists(extreme_mask_dict["romsoc_fully_coupled"]["ssp585"]["present"][varia], data_future_fixed_xr_ssp585)
    data_future_moving_mean_ssp585 = compute_mean_if_exists(extreme_mask_dict["romsoc_fully_coupled"]["ssp585"]["present_plus_meandelta"][varia], data_future_moving_xr_ssp585)

    # Fixed depth color scale limits
    vmin_depth, vmax_depth = -300, 0  # Depth range fixed from -500m to 0m

    # **Define file names**
    filename_ssp245 = f"/nfs/sea/work/fpfaeffli/plots/extreme_depths/extreme_depths_{varia}_present_vs_future_fixed_vs_moving_ssp245_only300m.png"
    filename_ssp585 = f"/nfs/sea/work/fpfaeffli/plots/extreme_depths/extreme_depths_{varia}_present_vs_future_fixed_vs_moving_ssp585_only300m.png"

    # Call the plotting function for SSP245
    plot_extreme_depths_present_future(
        data_present_mean,
        data_future_fixed_mean_ssp245,
        data_future_moving_mean_ssp245,
        lon_rho, lat_rho, landmask_etopo, bounds, vmin_depth, vmax_depth,
        f"Comparison of depth of saturation horizon during extremes (present vs. future (ssp245))",
        filename_ssp245
    )

    # Call the plotting function for SSP585
    plot_extreme_depths_present_future(
        data_present_mean,
        data_future_fixed_mean_ssp585,
        data_future_moving_mean_ssp585,
        lon_rho, lat_rho, landmask_etopo, bounds, vmin_depth, vmax_depth,
        f"Comparison of depth of saturation horizon during extremes (present vs. future (ssp585))",
        filename_ssp585
    )

# %%
