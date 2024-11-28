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
scenarios = ['present','ssp585'] # ,'ssp245'
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
################################################ PLOTTING ##########################################################################
# Plot the present day vs future extremes for the different thresholds
# Maps for new, disappearing, intensifying, and weakening extreme days per year (romsoc_fully_coupled: present vs ssp585)


present = variables['romsoc_fully_coupled']['present']['Hplus'] - thresholds['romsoc_fully_coupled']['present']['Hplus']
scenario = 'ssp585'

#%%
### THRESHOLD TYPE: 'present'
threshold_type = 'present'
future = variables['romsoc_fully_coupled'][scenario]['Hplus'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present']

# Define masks for the different extreme types
non_extremes_mask = (future <= 0) * (present <= 0)
new_extremes_mask = (future > 0) * (present <= 0)
disappearing_extremes_mask = (future <= 0) * (present > 0)
intensifying_extremes_mask = (future > 0) * (present > 0) * (future >= present)
weakening_extremes_mask = (future > 0) * (present > 0) * (future < present)

# Create subplots
fig, ax = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

# Parameters 
vmax_large = 300
vmax_small = 25
cmap_small_range = plt.get_cmap('cmo.tempo', 15)
cmap_big_range = plt.get_cmap('cmo.amp', 15)

# Plot each subplot
ax[0].pcolormesh(non_extremes_mask.lon, non_extremes_mask.lat, 
                 non_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
                 vmin=0, vmax=vmax_large, cmap=cmap_big_range)
ax[0].contour(non_extremes_mask.lon, non_extremes_mask.lat, 
              non_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
              levels=10, colors='k', linewidths=0.2)
ax[0].set_title('non-extremes')

ax[1].pcolormesh(new_extremes_mask.lon, new_extremes_mask.lat, 
                 new_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
                 vmin=0, vmax=vmax_large, cmap=cmap_big_range)
ax[1].contour(new_extremes_mask.lon, new_extremes_mask.lat, 
              new_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
              levels=10, colors='k', linewidths=0.2)
ax[1].set_title('new extremes')

ax[2].pcolormesh(disappearing_extremes_mask.lon, disappearing_extremes_mask.lat, 
                 disappearing_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
                 vmin=0, vmax=vmax_small, cmap=cmap_small_range)
ax[2].contour(disappearing_extremes_mask.lon, disappearing_extremes_mask.lat, 
              disappearing_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
              levels=5, colors='k', linewidths=0.2)
ax[2].set_title('disappeared extremes')

ax[3].pcolormesh(intensifying_extremes_mask.lon, intensifying_extremes_mask.lat, 
                 intensifying_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
                 vmin=0, vmax=vmax_small, cmap=cmap_small_range)
ax[3].contour(intensifying_extremes_mask.lon, intensifying_extremes_mask.lat, 
              intensifying_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
              levels=5, colors='k', linewidths=0.2)
ax[3].set_title('intensified extremes')

ax[4].pcolormesh(weakening_extremes_mask.lon, weakening_extremes_mask.lat, 
                 weakening_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
                 vmin=0, vmax=vmax_small, cmap=cmap_small_range)
ax[4].contour(weakening_extremes_mask.lon, weakening_extremes_mask.lat, 
              weakening_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
              levels=2, colors='k', linewidths=0.2)
ax[4].set_title('weakening extremes')

# Add continent (landmask)
landmask_etopo = PlotFuncs.get_etopo_data()
for axi in ax.flatten():
    axi.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

# Set the figure title 
fig.suptitle(f'Days per year of extreme surface Hplus concentrations based on present-day threshold (fixed baseline) (ensemble {ensemble_run})', fontsize=16, y=1.02)

# Explicitly adjust the top margin to prevent clipping of the title
plt.subplots_adjust(top=0.85, right=0.9)

# Colorbar settings 
cbax1 = fig.add_axes([0.05, 0.2, 0.025, 0.6])
plt.colorbar(ax[0].collections[0], cax=cbax1, extend='max')
cbax1.set_title('days\n    per year', pad=15)
yticks1 = np.array([0, 50, 100, 150, 200, 250, 300])
cbax1.set_yticks(yticks1)

cbax2 = fig.add_axes([0.91, 0.2, 0.025, 0.6])
plt.colorbar(ax[2].collections[0], cax=cbax2, extend='max')
cbax2.set_title('days\n    per year', pad=15)
yticks2 = np.array([0, 5, 10, 15, 20, 25])
cbax2.set_yticks(yticks2)

# Axis limits and ticks 
for axi in ax:
    axi.set_xlim([230, 245])
    axi.set_ylim([30, 50])

yticks = np.arange(30, 55, 5)
xticks = np.arange(230, 245, 5)
for axi in ax:
    axi.set_yticks(yticks)
    axi.set_xticks(xticks)
    axi.set_yticklabels([str(val) + '°N' for val in yticks])
    axi.set_xticklabels([str(360 - val) + '°W' for val in xticks])

# Save and show the figure
savedir = '/nfs/sea/work/fpfaeffli/plots/future_vs_present/days_per_year_extremes/Hplus/'
filename = f'future_vs_present_days_per_year_extremes_Hplus_{threshold_type}_threshold_ensemble{ensemble_run}.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')

plt.show()

#%%
### THRESHOLD TYPE: 'present_plus_meandelta'
threshold_type = 'present_plus_meandelta'
future = variables['romsoc_fully_coupled'][scenario]['Hplus'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_meandelta']

# Define masks for the different extreme types
non_extremes_mask = (future <= 0) * (present <= 0)
new_extremes_mask = (future > 0) * (present <= 0)
disappearing_extremes_mask = (future <= 0) * (present > 0)
intensifying_extremes_mask = (future > 0) * (present > 0) * (future >= present)
weakening_extremes_mask = (future > 0) * (present > 0) * (future < present)

# Create subplots
fig, ax = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

# Parameters 
vmax_large = 300
vmax_small = 20
cmap_small_range = plt.get_cmap('cmo.tempo', 10)
cmap_big_range = plt.get_cmap('cmo.amp', 15)

# Plot each subplot
ax[0].pcolormesh(non_extremes_mask.lon, non_extremes_mask.lat, 
                 non_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
                 vmin=0, vmax=vmax_large, cmap=cmap_big_range)
ax[0].contour(non_extremes_mask.lon, non_extremes_mask.lat, 
              non_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
              levels=10, colors='k', linewidths=0.2)
ax[0].set_title('non-extremes')

ax[1].pcolormesh(new_extremes_mask.lon, new_extremes_mask.lat, 
                 new_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
                 vmin=0, vmax=vmax_large, cmap=cmap_big_range)
ax[1].contour(new_extremes_mask.lon, new_extremes_mask.lat, 
              new_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
              levels=5, colors='k', linewidths=0.2)
ax[1].set_title('new extremes')

ax[2].pcolormesh(disappearing_extremes_mask.lon, disappearing_extremes_mask.lat, 
                 disappearing_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
                 vmin=0, vmax=vmax_small, cmap=cmap_small_range)
ax[2].contour(disappearing_extremes_mask.lon, disappearing_extremes_mask.lat, 
              disappearing_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
              levels=5, colors='k', linewidths=0.2)
ax[2].set_title('disappeared extremes')

ax[3].pcolormesh(intensifying_extremes_mask.lon, intensifying_extremes_mask.lat, 
                 intensifying_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
                 vmin=0, vmax=vmax_small, cmap=cmap_small_range)
ax[3].contour(intensifying_extremes_mask.lon, intensifying_extremes_mask.lat, 
              intensifying_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
              levels=5, colors='k', linewidths=0.2)
ax[3].set_title('intensified extremes')

ax[4].pcolormesh(weakening_extremes_mask.lon, weakening_extremes_mask.lat, 
                 weakening_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
                 vmin=0, vmax=vmax_small, cmap=cmap_small_range)
ax[4].contour(weakening_extremes_mask.lon, weakening_extremes_mask.lat, 
              weakening_extremes_mask.sum(dim='time') / (2021 - 2011 + 1), 
              levels=2, colors='k', linewidths=0.2)
ax[4].set_title('weakening extremes')

# Add continent (landmask)
landmask_etopo = PlotFuncs.get_etopo_data()
for axi in ax.flatten():
    axi.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

# Set the figure title
fig.suptitle(f'Days per year of extreme surface Hplus concentrations based on present_plus_meandelta threshold (moving baseline) (emsemble {ensemble_run})', fontsize=16, y=1.02)

plt.subplots_adjust(top=0.85, right=0.9)

# Colorbar settings 
cbax1 = fig.add_axes([0.05, 0.2, 0.025, 0.6])
plt.colorbar(ax[0].collections[0], cax=cbax1, extend='max')
cbax1.set_title('days\n    per year', pad=15)
yticks1 = np.array([0, 50, 100, 150, 200, 250, 300])
cbax1.set_yticks(yticks1)

cbax2 = fig.add_axes([0.91, 0.2, 0.025, 0.6])
plt.colorbar(ax[2].collections[0], cax=cbax2, extend='max')
cbax2.set_title('days\n    per year', pad=15)
yticks2 = np.array([0, 10, 20])
cbax2.set_yticks(yticks2)

# Axis limits and ticks 
for axi in ax:
    axi.set_xlim([230, 245])
    axi.set_ylim([30, 50])

yticks = np.arange(30, 55, 5)
xticks = np.arange(230, 245, 5)
for axi in ax:
    axi.set_yticks(yticks)
    axi.set_xticks(xticks)
    axi.set_yticklabels([str(val) + '°N' for val in yticks])
    axi.set_xticklabels([str(360 - val) + '°W' for val in xticks])

# Save and show the figure
savedir = '/nfs/sea/work/fpfaeffli/plots/future_vs_present/days_per_year_extremes/Hplus/'
filename = f'future_vs_present_days_per_year_extremes_Hplus_{threshold_type}_threshold_ensemble{ensemble_run}.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')


# %% # Plots extreme days per year relative to the distance to coast

#### ONLY WORKS IF SSP245 DATA IS LOADED (ONLY ENSEMBLE 000)

present = variables['romsoc_fully_coupled']['present']['Hplus'] - thresholds['romsoc_fully_coupled']['present']['Hplus']

for region_cho in ['all_dists_all_lats']:
    for threshold_type in ['present', 'present_plus_meandelta']:  # ,'present_plus_climdelta']:
        
        if threshold_type == 'present':
            sharey = False
        else:
            sharey = False
            
        fig, ax = plt.subplots(1, 4, figsize=(11, 4), sharey=sharey)
        
        for scenario in ['ssp245', 'ssp585']:

            if threshold_type == 'present_plus_meandelta':
                future = variables['romsoc_fully_coupled'][scenario]['Hplus'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_meandelta']
            elif threshold_type == 'present':
                future = variables['romsoc_fully_coupled'][scenario]['Hplus'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present']
            elif threshold_type == 'present_plus_climdelta':
                # Check if 'present_plus_climdelta' exists in the dataset
                if 'present_plus_climdelta' in variables['romsoc_fully_coupled']:
                    future = variables['romsoc_fully_coupled'][scenario]['Hplus'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_climdelta']
                else:
                    print("Warning: 'present_plus_climdelta' not found in the dataset.")
                continue  

            non_extremes_mask = (future <= 0) * (present <= 0) * model_regions['roms_only'][region_cho]['mask']
            new_extremes_mask = (future > 0) * (present <= 0) * model_regions['roms_only'][region_cho]['mask']
            disappearing_extremes_mask = (future <= 0) * (present > 0) * model_regions['roms_only'][region_cho]['mask']
            intensifying_extremes_mask = (future > 0) * (present > 0) * (future >= present) * model_regions['roms_only'][region_cho]['mask']
            weakening_extremes_mask = (future > 0) * (present > 0) * (future < present) * model_regions['roms_only'][region_cho]['mask']
            
            # Sum the arrays
            new_extremes_mask_sum = new_extremes_mask.sum(dim='time', skipna=True, min_count=1) / (2021 - 2011 + 1)
            disappearing_extremes_mask_sum = disappearing_extremes_mask.sum(dim='time', skipna=True, min_count=1) / (2021 - 2011 + 1)
            intensifying_extremes_mask_sum = intensifying_extremes_mask.sum(dim='time', skipna=True, min_count=1) / (2021 - 2011 + 1)
            weakening_extremes_mask_sum = weakening_extremes_mask.sum(dim='time', skipna=True, min_count=1) / (2021 - 2011 + 1)

            # Flatten the arrays
            new_flat = new_extremes_mask_sum.values.flatten()
            disappear_flat = disappearing_extremes_mask_sum.values.flatten()
            intensify_flat = intensifying_extremes_mask_sum.values.flatten()
            weakening_flat = weakening_extremes_mask_sum.values.flatten()

            # Flatten the distance to coast array
            d2coast_flat = model_d2coasts['roms_only'].values.flatten()

            # Drop the NaN entries
            nanmask = np.isnan(d2coast_flat) + np.isnan(new_flat) + np.isnan(disappear_flat) + np.isnan(intensify_flat) + np.isnan(weakening_flat)
            new_nonan = new_flat[~nanmask]
            disappear_nonan = disappear_flat[~nanmask]
            intensify_nonan = intensify_flat[~nanmask]
            weakening_nonan = weakening_flat[~nanmask]
            d2coast_flat_nonan = d2coast_flat[~nanmask]

            # Compute the mean and the IQR for bins as a function of distance to coast
            new_vs_d2coast_mean = []
            new_vs_d2coast_p25 = []
            new_vs_d2coast_p75 = []

            disappear_vs_d2coast_mean = []
            disappear_vs_d2coast_p25 = []
            disappear_vs_d2coast_p75 = []

            intensify_vs_d2coast_mean = []
            intensify_vs_d2coast_p25 = []
            intensify_vs_d2coast_p75 = []

            weakening_vs_d2coast_mean = []
            weakening_vs_d2coast_p25 = []
            weakening_vs_d2coast_p75 = []

            bins_d2coast = np.arange(0, 380, 10)
            for bdx, binn in enumerate(bins_d2coast[:-1]):
                dist_cond = (d2coast_flat_nonan >= bins_d2coast[bdx]) * (d2coast_flat_nonan < bins_d2coast[bdx + 1]) 
                new_vs_d2coast_mean.append(np.mean(new_nonan[dist_cond]))
                new_vs_d2coast_p25.append(np.percentile(new_nonan[dist_cond], 25))
                new_vs_d2coast_p75.append(np.percentile(new_nonan[dist_cond], 75))

                disappear_vs_d2coast_mean.append(np.mean(disappear_nonan[dist_cond]))
                disappear_vs_d2coast_p25.append(np.percentile(disappear_nonan[dist_cond], 25))
                disappear_vs_d2coast_p75.append(np.percentile(disappear_nonan[dist_cond], 75))
                
                intensify_vs_d2coast_mean.append(np.mean(intensify_nonan[dist_cond]))
                intensify_vs_d2coast_p25.append(np.percentile(intensify_nonan[dist_cond], 25))
                intensify_vs_d2coast_p75.append(np.percentile(intensify_nonan[dist_cond], 75))

                weakening_vs_d2coast_mean.append(np.mean(weakening_nonan[dist_cond]))
                weakening_vs_d2coast_p25.append(np.percentile(weakening_nonan[dist_cond], 25))
                weakening_vs_d2coast_p75.append(np.percentile(weakening_nonan[dist_cond], 75))

            if scenario == 'ssp245':
                color = 'green'
                label_name = 'SSP245'  
            elif scenario == 'ssp585':
                color = 'purple'
                label_name = 'SSP585'  
            
            # Plotting
            ax[0].plot(bins_d2coast[:-1], new_vs_d2coast_mean, color=color, alpha=1, linewidth=2, label=label_name)
            ax[0].fill_between(bins_d2coast[:-1], new_vs_d2coast_p75, new_vs_d2coast_p25, color=color, alpha=0.35)

            ax[1].plot(bins_d2coast[:-1], disappear_vs_d2coast_mean, color=color, alpha=1, linewidth=2, label=label_name)
            ax[1].fill_between(bins_d2coast[:-1], disappear_vs_d2coast_p75, disappear_vs_d2coast_p25, color=color, alpha=0.35)

            ax[2].plot(bins_d2coast[:-1], intensify_vs_d2coast_mean, color=color, alpha=1, linewidth=2, label=label_name)
            ax[2].fill_between(bins_d2coast[:-1], intensify_vs_d2coast_p75, intensify_vs_d2coast_p25, color=color, alpha=0.35)

            ax[3].plot(bins_d2coast[:-1], weakening_vs_d2coast_mean, color=color, alpha=1, linewidth=2, label=label_name)
            ax[3].fill_between(bins_d2coast[:-1], weakening_vs_d2coast_p75, weakening_vs_d2coast_p25, color=color, alpha=0.35)

            ax[0].set_title('new extremes', loc='left')
            ax[1].set_title('disappeared extremes', loc='left')
            ax[2].set_title('intensified extremes', loc='left')
            ax[3].set_title('weakened extremes', loc='left')

        for axi in ax:
            axi.set_xlim([0, 371])
            axi.grid(linestyle='--', alpha=0.25)
            axi.spines['top'].set_visible(False)
            axi.spines['right'].set_visible(False)
            axi.axvline(100, color='black', linestyle='--', linewidth=2)
            axi.set_xlabel('Dist. to coast in km')
        
        ax[0].set_ylabel('Extreme days per year\nfor respective extreme type')

        # Set y-limits 
        if threshold_type == 'present_plus_meandelta':
            ax[0].set_ylim(0, 250) # New extremes
            ax[1].set_ylim(0, 20)  # Disappeared extremes
            ax[2].set_ylim(0, 15)  # Intensified extremes
            ax[3].set_ylim(0, 5)  # Weakened extremes¨
        elif threshold_type == 'present':
            ax[0].set_ylim(150, 350) # New extremes
            ax[1].set_ylim(0, 2.5)  # Disappeared extremes
            ax[2].set_ylim(15, 23)  # Intensified extremes
            ax[3].set_ylim(0, 2)  # Weakened extremes

        # Set the figure title based on the threshold type
        if threshold_type == 'present_plus_meandelta':
            fig.suptitle(f'Extreme Hplus days per year based on present_plus_meandelta ({scenario}) threshold (moving baseline) (emsemble {ensemble_run})', fontsize=16, y=1.02)
        else:
            fig.suptitle(f'Extreme Hplus days per year based on present-day threshold (fixed threshold) (emsemble {ensemble_run})', fontsize=16, y=1.02)
        
        # Add the legend 
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='ssp245'),
            Line2D([0], [0], color='purple', lw=2, label='ssp585')
        ]
        
        fig.legend(
            handles = legend_elements,
            loc='center right',  
            bbox_to_anchor=(0.95, 0.5),  
            title='scenario',  
            fontsize='medium'  
        )

        plt.tight_layout(rect=[0, 0, 0.85, 1])  


        # Save and show the figure
        savedir = '/nfs/sea/work/fpfaeffli/plots/future_vs_present/days_per_year_extremes/Hplus/'
        filename = f'future_vs_present_dist2coast_days_per_year_Hplus_{threshold_type}_threshold__ensemble{ensemble_run}.png'
        plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')

        plt.show()

# %%

