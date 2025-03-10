"""
author: Fiona Pfäffli
description: This file plots present day vs future extremes for the different thresholds
"""

#%% enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')


#%%
# load the package
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter

from importlib import reload  # Python 3.4+

import get_obs_datasets 
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter

from func_for_clim_thresh import ThreshClimFuncs
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs

from plotting_functions_general import PlotFuncs as PlotFuncs

from matplotlib.lines import Line2D

#%% 
# Define the threshold 
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.Hplus_instance() #95th percentile threshold

#%%
# Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present','ssp585', 'ssp245'] # ,'ssp245'
configs = ['romsoc_fully_coupled'] # [ms_only'] 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '001'
vert_struct = 'zavg'    # 'avg'
depth = 0

#%% 
# Get the model datasets for the oceanic and atmospheric variables

ocean_ds = dict()
atmosphere_ds = dict()
pressure_ds = dict()
cloud_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     atmosphere_ds[config] = dict()
     pressure_ds[config] = dict()
     cloud_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)

#%% 
# load the data at the respective location

variables = dict()
for config in configs:
     variables[config] = dict()
     for scenario in scenarios:
          print(f'Getting the variables for config {config} and scenario {scenario}.')
          #
          # oceanic variables
          variables[config][scenario] = dict()
          print('Hplus')
          # Convert pH to [H+] concentration
          pH = ocean_ds[config][scenario].pH_offl.isel(depth=0).load()
          variables[config][scenario]['Hplus'] = np.power(10, -pH)
          

#%%
varias = ['Hplus']

#%% 
# Get the climatology for each variable
print('Get the climatology')
clims = dict()
for config in configs:
     clims[config] = dict()
     for scenario in scenarios:
          clims[config][scenario] = dict()
          for varia in varias:
               print(f'Calculate the climatology for: {config}, {scenario}, {varia}.')
               dummy_clim = ThreshClimFuncs.calc_clim(params,variables[config][scenario][varia])
               smoothed_clim = ThreshClimFuncs.smooth_array(params,dummy_clim)
               smoothed_clim = smoothed_clim.rename({'day_of_year_adjusted': 'time'})
               smoothed_clim = smoothed_clim.assign_coords({'time': pd.date_range('2015-01-01','2015-12-31')})
               clims[config][scenario][varia] = ThreshClimFuncs.repeat_array_multiple_years(smoothed_clim)


#%% 
# Get the threshold for each variable
print('Get the present day threshold')
varia = 'Hplus'
thresholds = dict()
for config in configs:
    thresholds[config] = dict()
    for scenario in ['present']:  # ,'ssp245','ssp585'
        thresholds[config][scenario] = dict()
        print(f'{config}, {scenario}')
        threshold, threshold_366 = ModelGetter.get_threshold('Hplus', 0, 'relative', 95, config, scenario)  # variable, depth_level, threshold_type, threshold_value, config, scenario
        concatenated_threshold = ModelGetter.concatenate_yearly_arrays(threshold, threshold_366, start_year=2011, end_year=2021)
        #rename 'day_of_year_adjusted' to 'time' in the concatenated threshold
        if 'day_of_year_adjusted' in concatenated_threshold.dims:
            concatenated_threshold = concatenated_threshold.rename({'day_of_year_adjusted': 'time'})
        elif 'day_of_year_adjusted' in concatenated_threshold.coords:
            concatenated_threshold = concatenated_threshold.rename_coords({'day_of_year_adjusted': 'time'})
        #assign the renamed threshold to the dictionary
        thresholds[config][scenario][varia] = concatenated_threshold

#%%
# Adjust the thresholds
print('Adjust the thresholds')
thresholds_mult = dict()
for config in configs:
     thresholds_mult[config] = dict()
     for scenario in ['present','ssp585']:
          thresholds_mult[config][scenario] = dict()
          print(f'{config}, {scenario}')
          thresholds_mult[config][scenario]['present'] = thresholds[config]['present'][varia]
          thresholds_mult[config][scenario]['present_plus_meandelta'] = thresholds[config]['present'][varia] + (clims[config][scenario][varia].mean(dim='time') - clims[config]['present'][varia].mean(dim='time'))


# %%
#% Get the distance to coast file from ROMS and the regions over which to calculate the statistics
model_d2coasts = dict()
model_d2coasts['roms_only'] = ModelGetter.get_distance_to_coast(vtype='oceanic')

# Get the model regions
model_regions = dict()
model_regions['roms_only'] = GetRegions.define_CalCS_regions(model_d2coasts['roms_only'].lon, model_d2coasts['roms_only'].lat, model_d2coasts['roms_only'])

# Get the model area
model_area = ModelGetter.get_model_area()



#%%

# Variables 
scenario = 'ssp585'
depthlevel = 0
eta_rho_cho = 500
xi_rho_cho = 200

# Load Hplus variables and both thresholds (moving an fixed)
present = variables['romsoc_fully_coupled']['present']['Hplus']
future = variables['romsoc_fully_coupled'][scenario]['Hplus']
thresholds_present = thresholds_mult['romsoc_fully_coupled'][scenario]['present']
thresholds_present_meandelta = thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_meandelta']

# Compute extremes with moving and fixed baselines
future_fixed_baseline = future - thresholds_present
future_moving_baseline = future - thresholds_present_meandelta
print("Future baselines computed")

#%% Cleaning dataarrays

# Clean data array function
def clean_dataarray(dataarray):
    """
    Cleans the dataarray by replacing 2D lat/lon with 1D indices for eta_rho and xi_rho.
    """
    eta_rho_1d = np.arange(dataarray.shape[1])  # Replace eta_rho with 1D indices
    xi_rho_1d = np.arange(dataarray.shape[2])  # Replace xi_rho with 1D indices

    return xr.DataArray(
        data=dataarray.values,  # Keep the data
        dims=["time", "eta_rho", "xi_rho"],  # Replace with 1D dimensions
        coords={
            "time": dataarray["time"].values,  # Copy time coordinate
            "eta_rho": eta_rho_1d,  # Replace eta_rho with 1D indices
            "xi_rho": xi_rho_1d,  # Replace xi_rho with 1D indices
        },
    )

# Clean the data arrays
future_fixed_baseline_cleaned = clean_dataarray(future_fixed_baseline)
future_moving_baseline_cleaned = clean_dataarray(future_moving_baseline)


# %%  Plotting

def plot_extremes_with_bounds_and_labels(data_fixed, data_moving, landmask_etopo, bounds, title, colorbar_label, filename):
    """
    Plots fixed and moving baseline extremes with proper lat/lon bounds, axis labels, and degree formatting.
    """
    lat_bounds, lon_bounds = bounds

    # Step 1: Extract dimensions for regridding landmask
    lat_2d = data_fixed["lat"].values  # Assuming "lat" exists in your dataset
    lon_2d = data_fixed["lon"].values  # Assuming "lon" exists in your dataset

    # Step 2: Regrid the landmask
    regridded_landmask = landmask_etopo.interp(
        lon=(("eta_rho", "xi_rho"), lon_2d),
        lat=(("eta_rho", "xi_rho"), lat_2d),
        method="nearest"
    )

    # Step 3: Initialize plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    # Step 4: Plot both fixed and moving baselines
    for idx, (data, ax, baseline) in enumerate(zip([data_fixed, data_moving], axs, ["Fixed baseline", "Moving baseline"])):
        # Plot data with masked values <= 0
        filtered_data = data.where(data > 0)
        contour = ax.contourf(
            filtered_data["lon"],  # Longitude coordinate
            filtered_data["lat"],  # Latitude coordinate
            filtered_data,         # Data to plot
            levels=40,             # Number of color levels
            cmap="cmo.balance",    # Colormap
            extend="both"          # Extend color range
        )

        # Overlay the landmask
        ax.contour(
            regridded_landmask["lon"],
            regridded_landmask["lat"],
            regridded_landmask,
            levels=1,
            colors="black",
            zorder=2
        )

        # Set plot title and axis limits
        ax.set_title(baseline, fontsize=14)
        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)

        # Add degree labels to axes
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])  # Convert to west longitude

        ax.set_xlabel("Longitude")
        if idx == 0:
            ax.set_ylabel("Latitude")

        # Add gridlines
        ax.grid(visible=True, linestyle="--", linewidth=0.5)

    # Step 5: Add colorbar
    cbar = fig.colorbar(
        contour,
        ax=axs,
        orientation="vertical",
        label=colorbar_label
    )

    # Step 6: Save and show plot
    plt.suptitle(title, fontsize=16)
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# %%
# Define bounds for the California Current System (CaCS)
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

# Load the landmask
landmask_etopo = PlotFuncs.get_etopo_data()

# Plot the first day's extremes
plot_extremes_with_bounds_and_labels(
    data_fixed=future_fixed_baseline.isel(time=0),
    data_moving=future_moving_baseline.isel(time=0),
    landmask_etopo=landmask_etopo,
    bounds=bounds,
    title="Comparison of extremes (Fixed vs Moving Baseline) - Day 1",
    colorbar_label="Contribution",
    filename="/nfs/sea/work/fpfaeffli/plots/extremes_fixed_vs_moving_baseline_day1.png"
)

# %%
