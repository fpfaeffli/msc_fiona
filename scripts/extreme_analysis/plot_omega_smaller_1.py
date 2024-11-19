
"""
author: Fiona Pfäffli
description: This script extracts and plots where omega aragonite saturation (Ω < 1) for the scenarios present, ssp245, and ssp585.
"""

#%% enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

#%% Load required packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
from get_model_datasets import ModelGetter as ModelGetter
from plotting_functions_general import PlotFuncs as PlotFuncs


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
omega_below_one_masks = dict()  # This will store the mask arrays for each scenario

for config in configs:
    omega_below_one_masks[config] = dict()
    for scenario in scenarios:
        print(f'Extracting regions where Ω < 1 for scenario: {scenario}')

       
        omega_data = variables[config][scenario]['omega_arag_offl']
        # Create a mask where omega aragonite is less than 1
        omega_below_one_mask = omega_data < 1
        omega_below_one_masks[config][scenario] = omega_below_one_mask

#%% Extract regions where Ω < 1.5 for each scenario
omega_below_one_point_five_masks = dict() 

for config in configs:
    omega_below_one_point_five_masks[config] = dict()
    for scenario in scenarios:
        print(f'Extracting regions where Ω < 1.5 for scenario: {scenario}')

        omega_data = variables[config][scenario]['omega_arag_offl']
        # Create a mask where omega aragonite is less than 1.5
        omega_below_one_point_five_mask = omega_data < 1.5
        omega_below_one_point_five_masks[config][scenario] = omega_below_one_point_five_mask

#%% 
#################################### 1 ######################################
# Plotting avrage days per year where Ω < 1 and Ω < 1.5 for each scenario 
fig, ax = plt.subplots(2, 3, figsize=(10, 10), sharey=True)
vmax_large = 365
cmap = plt.get_cmap('cmo.matter', 10)

for idx, scenario in enumerate(scenarios):
    # Extract the masks for Ω < 1 and Ω < 1.5
    omega_below_one_mask = omega_below_one_masks['romsoc_fully_coupled'][scenario]
    omega_below_one_point_five_mask = omega_below_one_point_five_masks['romsoc_fully_coupled'][scenario]

    # Calculate the number of days per year where Ω < 1
    days_per_year_below_one = omega_below_one_mask.sum(dim='time') / (2021 - 2011 + 1)
    
    # Plotting the number of days per year with undersaturation (Ω < 1)
    pcm = ax[0, idx].pcolormesh(days_per_year_below_one.lon, days_per_year_below_one.lat, 
                                 days_per_year_below_one, vmin=0, vmax=vmax_large, cmap=cmap)
    ax[0, idx].contour(days_per_year_below_one.lon, days_per_year_below_one.lat, 
                       days_per_year_below_one, levels=10, colors='k', linewidths=0.2)

    # Add continent (landmask)
    landmask_etopo = PlotFuncs.get_etopo_data()
    ax[0, idx].contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

    # Set the title and labels for the top row (Ω < 1)
    ax[0, idx].set_title(f'{scenario} (Ω < 1)', fontsize=16)
    ax[0, idx].set_xlim([230, 245])
    ax[0, idx].set_ylim([30, 50])
    ax[0, idx].set_xlabel('Longitude', fontsize=12)

    # Calculate the number of days per year where Ω < 1.5
    days_per_year_below_one_point_five = omega_below_one_point_five_mask.sum(dim='time') / (2021 - 2011 + 1)
    
    # Plotting the number of days per year with undersaturation (Ω < 1.5)
    pcm = ax[1, idx].pcolormesh(days_per_year_below_one_point_five.lon, days_per_year_below_one_point_five.lat, 
                                 days_per_year_below_one_point_five, vmin=0, vmax=vmax_large, cmap=cmap)
    ax[1, idx].contour(days_per_year_below_one_point_five.lon, days_per_year_below_one_point_five.lat, 
                       days_per_year_below_one_point_five, levels=10, colors='k', linewidths=0.2)

    # Add continent (landmask)
    ax[1, idx].contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

    # Set the title and labels for the bottom row (Ω < 1.5)
    ax[1, idx].set_title(f'{scenario} (Ω < 1.5)', fontsize=16)
    ax[1, idx].set_xlim([230, 245])
    ax[1, idx].set_ylim([30, 50])
    ax[1, idx].set_xlabel('Longitude', fontsize=12)

# Set common y-axis label
ax[0, 0].set_ylabel('Latitude', fontsize=12)
ax[1, 0].set_ylabel('Latitude', fontsize=12)

# Add colorbars for both Ω < 1 and Ω < 1.5
cbax1 = fig.add_axes([ 1 , 0.3, 0.015, 0.35])  
cbar1 = plt.colorbar(ax[0, 0].collections[0], cax=cbax1, extend='max')
cbar1.set_label('Days per year with Ω subceeding threshold', fontsize=12)

# Set axis ticks for all subplots
yticks = np.arange(30, 55, 5)
xticks = np.arange(230, 245, 5)
for i in range(2):
    for axi in ax[i, :]:
        axi.set_yticks(yticks)
        axi.set_xticks(xticks)
        axi.set_yticklabels([f'{val}°N' for val in yticks])
        axi.set_xticklabels([f'{360 - val}°W' for val in xticks])

# Set the figure title
fig.suptitle('Averaged days per year with omega aragonite saturation (Ω < 1 and Ω < 1.5) for different scenarios at the surface', fontsize=18, y=1.02)

# Explicitly adjust the top margin to prevent clipping of the title
plt.subplots_adjust(top=0.92, right=0.9)


plt.tight_layout()

# Save and show the figure
savedir = '/nfs/sea/work/fpfaeffli/plots/omega_below/'
filename = f'days_per_year_omega_below_one_and_one_point_five_averaged_depth{depth}m.png'
plt.savefig(savedir + filename, dpi=200, transparent=True)

plt.show()

# %%
#################################### 2 ######################################
# Plotting days per year for each year separately for Ω < 1 and Ω < 1.5

vmax_large = 365
cmap = plt.get_cmap('cmo.matter', 10)

for scenario in scenarios:  # Loop through each scenario
    fig, ax = plt.subplots(2, 11, figsize=(30, 10), sharey=True)

    # Extract the masks for Ω < 1 and Ω < 1.5
    omega_below_one_mask = omega_below_one_masks['romsoc_fully_coupled'][scenario]
    omega_below_one_point_five_mask = omega_below_one_point_five_masks['romsoc_fully_coupled'][scenario]

    # Group by year and calculate the number of days with Ω < 1 and Ω < 1.5 for each year
    years = omega_below_one_mask['time.year']
    days_per_year_below_one_by_year = omega_below_one_mask.groupby(years).sum(dim='time')
    days_per_year_below_one_point_five_by_year = omega_below_one_point_five_mask.groupby(years).sum(dim='time')

    # Add continent (landmask)
    landmask_etopo = PlotFuncs.get_etopo_data()

    for year_idx, year in enumerate(range(2011, 2022)):  # Loop through each year
        # Plotting Ω < 1 for the current year
        pcm = ax[0, year_idx].pcolormesh(days_per_year_below_one_by_year.sel(year=year).lon, 
                                         days_per_year_below_one_by_year.sel(year=year).lat, 
                                         days_per_year_below_one_by_year.sel(year=year), 
                                         vmin=0, vmax=vmax_large, cmap=cmap)
        # Add continent (landmask) for Ω < 1
        ax[0, year_idx].contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

        ax[0, year_idx].set_title(f'{scenario} - {year}', fontsize=12)
        ax[0, year_idx].set_xlim([230, 245])
        ax[0, year_idx].set_ylim([30, 50])

        # Plotting Ω < 1.5 for the current year
        pcm = ax[1, year_idx].pcolormesh(days_per_year_below_one_point_five_by_year.sel(year=year).lon, 
                                         days_per_year_below_one_point_five_by_year.sel(year=year).lat, 
                                         days_per_year_below_one_point_five_by_year.sel(year=year), 
                                         vmin=0, vmax=vmax_large, cmap=cmap)
        # Add continent (landmask) for Ω < 1.5
        ax[1, year_idx].contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

        ax[1, year_idx].set_title(f'{year}', fontsize=12)
        ax[1, year_idx].set_xlim([230, 245])
        ax[1, year_idx].set_ylim([30, 50])

    # Set common y-axis label
    ax[0, 0].set_ylabel('Latitude', fontsize=12)
    ax[1, 0].set_ylabel('Latitude', fontsize=12)

    # Set axis ticks for all subplots
    yticks = np.arange(30, 55, 5)
    xticks = np.arange(230, 245, 5)
    for i in range(2):
        for axi in ax[i, :]:
            axi.set_yticks(yticks)
            axi.set_xticks(xticks)
            axi.set_yticklabels([f'{val}°N' for val in yticks])
            axi.set_xticklabels([f'{360 - val}°W' for val in xticks])

    # Add a single colorbar for all plots
    cbax = fig.add_axes([0.92, 0.3, 0.015, 0.4])  
    cbar = plt.colorbar(pcm, cax=cbax, extend='max')
    cbar.set_label('Days per year with Ω subceeding threshold', fontsize=12)

    # Adjust layout
    plt.subplots_adjust(top=0.9, right=0.88, hspace=0.3, wspace=0.2)

    # Set the figure title
    fig.suptitle(f'Yearly days per year with Ω < 1 for scenario {scenario}', fontsize=20, y=0.95)
    fig.text(0.5, 0.5, f'Yearly days per year with Ω < 1.5 for scenario {scenario}', ha='center', fontsize=20)

    # Save and show the figure
    savedir = '/nfs/sea/work/fpfaeffli/plots/omega_below/'
    filename = f'days_per_year_omega_below_one_and_one_point_five_all_years_scenario_{scenario}_depth{depth}m.png'
    plt.savefig(savedir + filename, dpi=200, transparent=True)
    plt.show()




# %%
#################################### 3 ######################################
