"""
author: Fiona Pfäffli
description: This file plots present day vs future extremes for the different thresholds.
"""

# %% enable the visibility of the modules for the import functions
import sys
import os

sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

# %%
# Load the necessary packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
from statsmodels.tsa.stattools import acf
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter
from get_model_datasets import ModelCarbonateChemistryGetter as ModelCarbonateChemistryGetter
from get_obs_datasets import ObsGetter as ObsGetter

import multiprocessing
from tqdm import tqdm

from importlib import reload  # Python 3.4+

import matplotlib.patheffects as pe

import get_obs_datasets 
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter

from func_for_clim_thresh import ThreshClimFuncs
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs

import xesmf as xe

from plotting_functions_general import PlotFuncs as PlotFuncs

# %%
# Define the threshold 
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.omega_arag_instance()  # 5th percentile threshold

# %%
# Defining variables
model_temp_resolution = 'daily'  # 'monthly'
scenarios = ['present', 'ssp245', 'ssp585']  # Different climate scenarios
configs = ['romsoc_fully_coupled']  # Model configuration
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'zavg'  # Vertical structure: 'avg'

# %%
# Variables
varias = [
    'depth_of_omega_arag_offl_1_isosurface',
    'depth_of_omega_arag_offl_1.3_isosurface',
    'depth_of_omega_arag_offl_1.5_isosurface',
    'depth_of_omega_arag_offl_1.7_isosurface'
]

# %%
# Get the model datasets for the oceanic variables
ocean_ds = dict()
for config in configs:
    ocean_ds[config] = dict()
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
            # Handle the case where no file is available
            print(f"Failed to load dataset for config '{config}' and scenario '{scenario}': {e}")

# %%
# Load the data at the respective location

variables = dict()
for config in configs:
    variables[config] = dict()
    for scenario in scenarios:
        print(f'Getting the variables for config {config} and scenario {scenario}.')
        variables[config][scenario] = dict()
        for var in varias:
            try:
                print(f'Loading variable: {var}')
                variables[config][scenario][var] = ocean_ds[config][scenario][var].load()
                print(f"Variable '{var}' loaded successfully.")
            except KeyError:
                # If the variable is not found in the dataset
                print(f"Variable '{var}' not found in the dataset for config '{config}' and scenario '{scenario}'.")
            


#%%
# Load the climatologies and thresholds
print("Loading climatologies...")
climatology = xr.open_dataset(f'/nfs/sea/work/fpfaeffli/future_sim/thresholds_and_climatologies/romsoc_fully_coupled/clim_thresh_all_omega_arag_saturation_horizon_depth_romsoc_fully_coupled_present_day_threshold.nc')
print(climatology)
print("Dataset loaded")


#%%
# Load the thresholds from netCDF 
print("Loading thresholds...")
threshold = xr.open_dataset(f'/nfs/sea/work/fpfaeffli/future_sim/thresholds_and_climatologies/romsoc_fully_coupled/thresholds_all_omega_arag_saturation_horizon_depth_all_configs_present_scenario.nc')
print(threshold)
print("Dataset loaded")

#%%
# Adjust the thresholds
print('Adjust the thresholds')
thresholds_mult = dict()
for config in configs:
     thresholds_mult[config] = dict()
     for scenario in ['present','ssp245','ssp585']:
          thresholds_mult[config][scenario] = dict()
          print(f'{config}, {scenario}')
          thresholds_mult[config][scenario]['present'] = threshold[config]['present'][varia]
          thresholds_mult[config][scenario]['present_plus_meandelta'] = threshold[config]['present'][varia] + (climatology[config][scenario][varia].mean(dim='time') - climatology[config]['present'][varia].mean(dim='time'))


#%%
# Calculate the masks for the extremes detection
print('Calculate masks for extremes detection')
masks = dict()
for config in configs:
    masks[config] = dict()
    for scenario in scenarios:
        masks[config][scenario] = dict()
        for varia in varias:
            print(f"Creating mask for {config}, {scenario}, {varia}.")
            
            variable_data = variables[config][scenario][varia]
            
            # Get the adjusted threshold
            threshold = thresholds_mult[config][scenario]['present_plus_meandelta']  # Or 'present' depending on need
            
            # Create the mask where values are below the threshold
            masks[config][scenario][varia] = variable_data < threshold

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
################################################ PLOTTING ##########################################################################

#%%
# Plotting maps of all three scenarios to compare the mean values for each isosurface

# Scenarios to be plotted
scenarios = ['present', 'ssp245', 'ssp585']

# Plotting the mean values for each variable across different scenarios
for varia in varias:
    print(f"Plotting scenarios for variable: {varia}")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    vmin, vmax = -500, 0

    # Plot each scenario
    for i, scenario in enumerate(scenarios):
        mean_values = variables['romsoc_fully_coupled'][scenario][varia].mean(dim='time')

        pcm = ax[i].pcolormesh(
            mean_values.lon,
            mean_values.lat,
            mean_values,
            vmin=vmin, vmax=vmax, cmap=plt.get_cmap('cmo.amp_r', 10)
        )
        ax[i].set_title(f'{scenario}')
        ax[i].set_xlim([230, 245])  
        ax[i].set_ylim([30, 50])  

    # Set tick labels and adjust subplot spacing
    yticks = np.arange(30, 55, 5)
    xticks = np.arange(230, 245, 5)
    for axi in ax:
        axi.set_yticks(yticks)
        axi.set_xticks(xticks)
        axi.set_yticklabels([f'{val}°N' for val in yticks])
        axi.set_xticklabels([f'{360 - val}°W' for val in xticks])

    plt.subplots_adjust(top=0.85, right=0.85)

    # Colorbar settings
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax, extend='both')
    cbar.set_label('Depth [m]', fontsize=12)

    # Add continent (landmask)
    landmask_etopo = PlotFuncs.get_etopo_data()
    for axi in ax.flatten():
        axi.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

    # Set figure title
    if varia == 'depth_of_omega_arag_offl_1_isosurface':
        fig.suptitle(f"Depth of saturation horizon for Ω = 1 ", fontsize=20, y=1.02)
    elif varia == 'depth_of_omega_arag_offl_1.3_isosurface':
        fig.suptitle(f"Depth of saturation horizon for Ω = 1.3 ", fontsize=20, y=1.02)
    elif varia == 'depth_of_omega_arag_offl_1.5_isosurface':
        fig.suptitle(f"Depth of saturation horizon for Ω = 1.5 ", fontsize=20, y=1.02)
    elif varia == 'depth_of_omega_arag_offl_1.7_isosurface':
        fig.suptitle(f"Depth of saturation horizon for Ω = 1.7 ", fontsize=20, y=1.02)
    

    # Save and show the figure
    savedir = '/nfs/sea/work/fpfaeffli/plots/future_vs_present/saturation_horizon_depth/'
    filename = f'{varia}_present_vs_future.png'
    plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')

    plt.show()



#%%
# Plotting anomalies (differences between present and future scenarios)

# Define the future scenarios to compare with the present
future_scenarios = ['ssp245', 'ssp585']

for varia in varias:
    print(f"Plotting anomalies for variable: {varia}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)  
    anomaly_vmin, anomaly_vmax = -200, 200  

    # Loop through the future scenarios
    for i, scenario in enumerate(future_scenarios):
        # Calculate anomaly (difference between future and present)
        present_mean = variables['romsoc_fully_coupled']['present'][varia].mean(dim='time')
        future_mean = variables['romsoc_fully_coupled'][scenario][varia].mean(dim='time')
        anomaly = future_mean - present_mean

        # Plot the anomaly
        pcm = ax[i].pcolormesh(
            anomaly.lon,
            anomaly.lat,
            anomaly,
            vmin=anomaly_vmin, vmax=anomaly_vmax, cmap = plt.get_cmap('cmo.balance', 10)
        )
        ax[i].set_title(f'{scenario} - present')
        ax[i].set_xlim([230, 245])  
        ax[i].set_ylim([30, 50])    

    # Set tick labels and adjust layout
    yticks = np.arange(30, 55, 5)
    xticks = np.arange(230, 245, 5)
    for axi in ax:
        axi.set_yticks(yticks)
        axi.set_xticks(xticks)
        axi.set_yticklabels([f'{val}°N' for val in yticks])
        axi.set_xticklabels([f'{360 - val}°W' for val in xticks])

    plt.subplots_adjust(top=0.85, right=0.85)

    # Add a colorbar for the anomalies
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax, extend='both')
    cbar.set_label('Anomaly [m]', fontsize=12)

    # Add continent (landmask)
    landmask_etopo = PlotFuncs.get_etopo_data()
    for axi in ax.flatten():
        axi.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

    # Set figure title
    if varia == 'depth_of_omega_arag_offl_1_isosurface':
        fig.suptitle(f"Anomalies in saturation horizon depth for Ω = 1 ", fontsize=20, y=1.02)
    elif varia == 'depth_of_omega_arag_offl_1.3_isosurface':
        fig.suptitle(f"Anomalies in saturation horizon depth for Ω = 1.3 ", fontsize=20, y=1.02)
    elif varia == 'depth_of_omega_arag_offl_1.5_isosurface':
        fig.suptitle(f"Anomalies in saturation horizon depth for Ω = 1.5 ", fontsize=20, y=1.02)
    elif varia == 'depth_of_omega_arag_offl_1.7_isosurface':
        fig.suptitle(f"Anomalies in saturation horizon depth for Ω = 1.7 ", fontsize=20, y=1.02)


    # Save the anomalies figure
    savedir = '/nfs/sea/work/fpfaeffli/plots/future_vs_present/saturation_horizon_depth/'
    filename = f'{varia}_present_vs_future_anomalies.png'
    plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()


#%%
# Define present and scenario for plotting extreme days per year
# Create a dictionary to store the results for each isosurface
present = {}
for varia in varias:
    present[varia] = (
        variables['romsoc_fully_coupled']['present'][varia] -
        threshold['romsoc_fully_coupled']['present'][varia]
    )

scenario = 'ssp585'

#%%
### THRESHOLD TYPE: 'present'
threshold_type = 'present'
future = {}  
masks = {}   

# Calculate present and future for each isosurface
for varia in varias:  
    print(f"Calculating present and future for {varia}")
 
    present = (
        variables['romsoc_fully_coupled']['present'][varia] -
        threshold['romsoc_fully_coupled']['present'][varia]
    )

    future[varia] = (
        variables['romsoc_fully_coupled'][scenario][varia] -
        threshold['romsoc_fully_coupled']['present'][varia] 
    )

    # Define masks for the different extreme types
    masks[varia] = {
        "non_extremes": (future[varia] > 0) & (present > 0),
        "new_extremes": (future[varia] <= 0) & (present > 0),
        "disappearing_extremes": (future[varia] > 0) & (present <= 0),
        "intensifying_extremes": (future[varia] <= 0) & (present <= 0) & (future[varia] <= present),
        "weakening_extremes": (future[varia] <= 0) & (present <= 0) & (future[varia] >= present),
    }

    print(f"Finished calculations for {varia}")

#%% 
# Plotting extreme days per year for each isosurface for threshold type 'present'

for varia in varias:
   
    # Retrieve masks for the current variable
    non_extremes_mask = masks[varia]["non_extremes"]
    new_extremes_mask = masks[varia]["new_extremes"]
    disappearing_extremes_mask = masks[varia]["disappearing_extremes"]
    intensifying_extremes_mask = masks[varia]["intensifying_extremes"]
    weakening_extremes_mask = masks[varia]["weakening_extremes"]

    # Create subplots for the current variable
    fig, ax = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

    # Define parameters for plotting
    vmax = 365
    vmin = 0
    cmap = plt.get_cmap('cmo.amp', 20)

    # Plot each mask 
    pcm0 = ax[0].pcolormesh(
        non_extremes_mask.lon,
        non_extremes_mask.lat,
        non_extremes_mask.sum(dim='time') / (2021 - 2011 + 1),
        vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax[0].contour(
        non_extremes_mask.lon,
        non_extremes_mask.lat,
        non_extremes_mask.sum(dim='time') / (2021 - 2011 + 1),
        levels=10, colors='k', linewidths=0.2
    )
    ax[0].set_title('non-extremes')

    pcm1 = ax[1].pcolormesh(
        new_extremes_mask.lon,
        new_extremes_mask.lat,
        new_extremes_mask.sum(dim='time') / (2021 - 2011 + 1),
        vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax[1].contour(
        new_extremes_mask.lon,
        new_extremes_mask.lat,
        new_extremes_mask.sum(dim='time') / (2021 - 2011 + 1),
        levels=10, colors='k', linewidths=0.2
    )
    ax[1].set_title('new extremes')

    pcm2 = ax[2].pcolormesh(
        disappearing_extremes_mask.lon,
        disappearing_extremes_mask.lat,
        disappearing_extremes_mask.sum(dim='time') / (2021 - 2011 + 1),
        vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax[2].contour(
        disappearing_extremes_mask.lon,
        disappearing_extremes_mask.lat,
        disappearing_extremes_mask.sum(dim='time') / (2021 - 2011 + 1),
        levels=5, colors='k', linewidths=0.2
    )
    ax[2].set_title('disappeared extremes')

    pcm3 = ax[3].pcolormesh(
        intensifying_extremes_mask.lon,
        intensifying_extremes_mask.lat,
        intensifying_extremes_mask.sum(dim='time') / (2021 - 2011 + 1),
        vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax[3].contour(
        intensifying_extremes_mask.lon,
        intensifying_extremes_mask.lat,
        intensifying_extremes_mask.sum(dim='time') / (2021 - 2011 + 1),
        levels=5, colors='k', linewidths=0.2
    )
    ax[3].set_title('intensified extremes')

    pcm4 = ax[4].pcolormesh(
        weakening_extremes_mask.lon,
        weakening_extremes_mask.lat,
        weakening_extremes_mask.sum(dim='time') / (2021 - 2011 + 1),
        vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax[4].contour(
        weakening_extremes_mask.lon,
        weakening_extremes_mask.lat,
        weakening_extremes_mask.sum(dim='time') / (2021 - 2011 + 1),
        levels=2, colors='k', linewidths=0.2
    )
    ax[4].set_title('weakening extremes')

    # Add continent (landmask)
    landmask_etopo = PlotFuncs.get_etopo_data()
    for axi in ax.flatten():
        axi.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

    # Set figure title
    fig.suptitle(f"Days per year of extremes for {varia}", fontsize=16, y=1.02)

    # Add the region limits to focus on the same area as in `Hplus`
    for axi in ax:
        axi.set_xlim([230, 245])  
        axi.set_ylim([30, 50])  

    # Set the tick labels
    yticks = np.arange(30, 55, 5)
    xticks = np.arange(230, 245, 5)
    for axi in ax:
        axi.set_yticks(yticks)
        axi.set_xticks(xticks)
        axi.set_yticklabels([f'{val}°N' for val in yticks])
        axi.set_xticklabels([f'{360 - val}°W' for val in xticks])

    # Adjust layout and add a single shared colorbar for all plots
    plt.subplots_adjust(top=0.85, right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # Position for the shared colorbar
    cbar = fig.colorbar(pcm4, cax=cbar_ax, extend='max')
    cbar.set_label('days per year', fontsize=12)
    cbar.set_ticks([0, 50, 100, 150, 200, 250, 300, 350])

    # Save and show the figure
    savedir = '/nfs/sea/work/fpfaeffli/plots/future_vs_present/days_per_year_extremes/omega_arag_sat/'
    filename = f'future_vs_present_days_per_year_extremes_omega_arag_sat_{threshold_type}_{varia}_threshold.png'
    #plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')

    plt.show()


# %%
