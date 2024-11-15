"""
author: Fiona Pfäffli
description: This script extracts the regions and times where omega aragonite saturation (Ω < 1) for all three scenarios: present, ssp245, and ssp585.
"""

#%% Enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

#%% Load required packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from get_model_datasets import ModelGetter as ModelGetter

#%% Load other scripts and functions
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.fiona_instance()

#%% Load datasets
model_temp_resolution = 'daily'
scenarios = ['present', 'ssp245', 'ssp585']
configs = ['romsoc_fully_coupled']
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'zavg'
depth = 0

#%% Get the model datasets for the oceanic variables
ocean_ds = dict()
for config in configs:
    ocean_ds[config] = dict()
    for scenario in scenarios:
        print(f'--{config}, {scenario}--')
        print('ocean...')
        ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, model_temp_resolution, vert_struct, vtype='oceanic', parent_model=parent_model)

#%% Load omega_arag_offl data for each scenario
variables = dict()
for config in configs:
    variables[config] = dict()
    for scenario in scenarios:
        print(f'Getting the variables for config {config} and scenario {scenario}.')
        variables[config][scenario] = dict()
        print('omega_arag_offl')
        variables[config][scenario]['omega_arag_offl'] = ocean_ds[config][scenario].omega_arag_offl.isel(depth=0).load()

#%% Extract regions where Ω < 1 for each scenario
omega_below_one_masks = dict()  

for config in configs:
    omega_below_one_masks[config] = dict()
    for scenario in scenarios:
        print(f'Extracting regions where Ω < 1 for scenario: {scenario}')

        # Create a mask where omega aragonite is less than 1
        omega_data = variables[config][scenario]['omega_arag_offl']
        omega_below_one_mask = omega_data < 1
        omega_below_one_masks[config][scenario] = omega_below_one_mask

#%% 
# Save the extracted masks as netCDf files
#for config in configs:
    #for scenario in scenarios:
        #mask = omega_below_one_masks[config][scenario]
        
        # Save the mask dataset to a NetCDF file
        #save_path = f'/nfs/sea/work/fpfaeffli/omega_below_one_masks/omega_below_one_mask_{config}_{scenario}_depth_{depth}m.nc'
        #print(f'Saving mask for scenario {scenario} to {save_path}')
        #mask.to_netcdf(save_path)

#%% Plotting the areas where Ω < 1 for each scenario
for config in configs:
    for scenario in scenarios:
        omega_below_one_mask = omega_below_one_masks[config][scenario]

        # Calculate the number of days per year where Ω < 1
        days_per_year_below_one = omega_below_one_mask.sum(dim='time') / (2021 - 2011 + 1)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        vmax = 365
        cmap = plt.get_cmap('cmo.matter', 10)

        pcm = ax.pcolormesh(days_per_year_below_one.lon, days_per_year_below_one.lat, days_per_year_below_one, 
                            vmin=0, vmax=vmax, cmap=cmap)
        cbar = plt.colorbar(pcm, ax=ax, extend='max')
        cbar.set_label('Days per year with Ω < 1', fontsize=12)

        # Set title and labels
        ax.set_title(f'Regions with Ω < 1 for {scenario} Scenario', fontsize=16)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)

        plt.show()


