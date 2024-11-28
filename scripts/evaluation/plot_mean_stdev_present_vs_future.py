#%% Enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

#%% Load the necessary packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
from statsmodels.tsa.stattools import acf
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter
from get_obs_datasets import ObsGetter as ObsGetter
import multiprocessing
from tqdm import tqdm
from importlib import reload
import matplotlib.patheffects as pe
import get_obs_datasets
reload(get_obs_datasets)
from func_for_clim_thresh import ThreshClimFuncs
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs
import xesmf as xe
from plotting_functions_general import PlotFuncs as PlotFuncs
from matplotlib.lines import Line2D


#%% Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present','ssp245','ssp585']
configs = ['romsoc_fully_coupled']
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'zavg'    # 'avg'

#%% Get the model datasets for the oceanic variables
ocean_ds = dict()
for config in configs:
    ocean_ds[config] = dict()
    for scenario in scenarios:
        print(f'--{config}, {scenario}--')
        print('Loading ocean data...')
        ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, model_temp_resolution, vert_struct, vtype='oceanic', parent_model=parent_model)

#%% Define variable
varias = ['Hplus'] # 'pH_offl' , 'Hplus' , 'omega_arag_offl'


#%% Define params based on the variable

if varias[0] == 'pH_offl' or varias[0] == 'Hplus':
    params = ThresholdParameters.Hplus_instance() #95th percentile threshold
elif varias[0] == 'omega_arag_offl':
    params = ThresholdParameters.omega_arag_instance() #5th percentile threshold

#%% Load the data at the respective location
variables = dict()
for config in configs:
    variables[config] = dict()
    for scenario in scenarios:
        print(f'Getting the variables for config {config} and scenario {scenario}.')
        # Oceanic variables
        variables[config][scenario] = dict()
        for var in varias:
            if var == 'Hplus':
                print('Hplus')
                # Convert pH to [H+] concentration
                pH = ocean_ds[config][scenario].pH_offl.isel(depth=0).load()
                variables[config][scenario]['Hplus'] = 10 ** (-pH)
            elif var == 'pH_offl':
                print('pH_offl')
                # Directly take pH_offl without conversion
                variables[config][scenario]['pH_offl'] = ocean_ds[config][scenario].pH_offl.isel(depth=0).load()
            elif var == 'omega_arag_offl':
                print('omega_arag_offl')
                variables[config][scenario]['omega_arag_offl'] = ocean_ds[config][scenario].omega_arag_offl.isel(depth=0).load()


#%%
# Calculate the climatology for each scenario
print('Get the climatology')
clims = dict()
for config in configs:
    clims[config] = dict()
    for scenario in scenarios:
        clims[config][scenario] = dict()
        for varia in varias:
            print(f'Calculate the climatology for: {config}, {scenario}, {varia}.')
            dummy_clim = ThreshClimFuncs.calc_clim(params, variables[config][scenario][varia])
            smoothed_clim = ThreshClimFuncs.smooth_array(params, dummy_clim)
            smoothed_clim = smoothed_clim.rename({'day_of_year_adjusted': 'time'})
            smoothed_clim = smoothed_clim.assign_coords({'time': pd.date_range('2015-01-01', '2015-12-31')})
            clims[config][scenario][varia] = ThreshClimFuncs.repeat_array_multiple_years(smoothed_clim)

#%% Calculate and plot the mean values for each variable and scenario
# Define the scenarios and configs as in the original script
scenarios = ['present', 'ssp245', 'ssp585']
configs = ['romsoc_fully_coupled']


#%%
# Calculate mean values for each variable and scenario
mean_values = dict()
for config in configs:
    mean_values[config] = dict()
    for scenario in scenarios:
        mean_values[config][scenario] = dict()
        for var in varias:
            print(f'Calculating mean {var} for {config}, {scenario}')
            mean_values[config][scenario][var] = variables[config][scenario][var].mean(dim='time')


#%%
# Calculate standard deviation for each variable and scenario
std_values = dict()
for config in configs:
    std_values[config] = dict()
    for scenario in scenarios:
        std_values[config][scenario] = dict()
        for var in varias:
            print(f'Calculating standard deviation of {var} for {config}, {scenario}')
            std_values[config][scenario][var] = variables[config][scenario][var].std(dim='time')

#%%
# Calculate anomalies relative to the present scenario for each variable
anomalies = dict()
for config in configs:
    anomalies[config] = dict()
    for scenario in ['ssp245', 'ssp585']:
        anomalies[config][scenario] = dict()
        for var in varias:
            print(f'Calculating anomaly for {var}, {config}, {scenario}')
            anomalies[config][scenario][var] = mean_values[config][scenario][var] - mean_values[config]['present'][var]


#%%
# Calculate deseasonalized mean by removing the climatology
print('Calculating deseasonalized variables by removing the climatology')
deseasonalized_mean = dict()
scenarios = ['present','ssp245','ssp585']
for config in configs:
    deseasonalized_mean[config] = dict()
    for scenario in scenarios:
        deseasonalized_mean[config][scenario] = dict()
        for var in varias:
            print(f'Removing seasonality for {var}, {config}, {scenario}')
            # Subtract the climatology from the original data to remove seasonality
            deseasonalized_mean[config][scenario][var] = mean_values[config][scenario][var] - clims[config][scenario][var]


#%% ###################### PLOTTING ######################
# Plotting the mean values for each variable and scenario
fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey='row')

# Automatically determine vmin and vmax for the entire dataset for mean values
all_means = [mean_values['romsoc_fully_coupled'][scenario][varias[0]] for scenario in scenarios]
combined_min = min([data.min().item() for data in all_means])
combined_max = max([data.max().item() for data in all_means])
# Filter out values above 10 for 'pH_offl'
if varias[0] == 'pH_offl' or varias[0] == 'omega_arag_offl':
    for config in configs:
        for scenario in scenarios:
            mean_values[config][scenario][varias[0]] = mean_values[config][scenario][varias[0]].where(mean_values[config][scenario][varias[0]] <= 10)

# Parameters for plotting means
vmin = combined_min
vmax = combined_max
cmap = plt.get_cmap('cmo.matter', 10)

# Plot each scenario's mean values
for idx, scenario in enumerate(scenarios):
    mean_var = mean_values['romsoc_fully_coupled'][scenario][varias[0]]
    mean_var_scaled, unit_label = scale_data(varias[0], mean_var)  # Scale data and get unit label
    plot = ax[idx].pcolormesh(mean_var.lon, mean_var.lat, mean_var_scaled, vmin=vmin, vmax=vmax, cmap=cmap)
    ax[idx].contour(mean_var.lon, mean_var.lat, mean_var, levels=5, colors='k', linewidths=0.2)
    ax[idx].set_title(f'{scenario}')
    

# Add continent (landmask) to all subplots
landmask_etopo = PlotFuncs.get_etopo_data()
for axi in ax.flatten():
    axi.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

# Set the figure title 
title_map = {
    'Hplus': 'Mean surface Hplus concentrations for different scenarios',
    'pH_offl': 'Mean surface pH for different scenarios',
    'omega_arag_offl': 'Mean surface omega aragonite for different scenarios'
}
fig.suptitle(title_map.get(varias[0], f'Mean surface {varias[0]} for different scenarios'), fontsize=16, y=1.02)
plt.subplots_adjust(top=0.85, right=0.9)

# Colorbar settings for mean values
cbax = fig.add_axes([0.92, 0.2, 0.025, 0.6])
cbar = plt.colorbar(ax[0].collections[0], cax=cbax, extend='both')
if varias[0] == 'Hplus':
    cbar.ax.set_title(f'[H+] \n(mol/L)', pad=25)
elif varias[0] == 'pH_offl':
    cbar.ax.set_title(f'pH', pad=25)
elif varias[0] == 'omega_arag_offl':
    cbar.ax.set_title(r'$\Omega_{arag}$', pad=25)


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

# Make sure that every subplot has y-axis labels visible
for axi in ax:
    axi.tick_params(axis='y', which='both', labelleft=True)

# Save and show the figure
savedir = '/nfs/sea/work/fpfaeffli/plots/future_vs_present/means_and_stdev/'
filename = f'maps_mean_{varias[0]}_concentrations_surface.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')
plt.show()


#%% ###################### PLOTTING STANDARD DEVIATION ######################
# Plotting the standard deviation for each variable and scenario
fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey='row')

# Automatically determine vmin and vmax for the entire dataset for standard deviation values
all_stds = [std_values['romsoc_fully_coupled'][scenario][varias[0]] for scenario in scenarios]
combined_min_std = min([data.min().values for data in all_stds])
combined_max_std = max([data.max().values for data in all_stds])
# Filter out values above 10 for 'pH_offl'
if varias[0] == 'pH_offl' or varias[0] == 'omega_arag_offl':
    for config in configs:
        for scenario in scenarios:
            std_values[config][scenario][varias[0]] = std_values[config][scenario][varias[0]].where(std_values[config][scenario][varias[0]] <= 10)

# Parameters for plotting standard deviations
vmin_std = combined_min_std  # Set a dynamic minimum value
vmax_std = combined_max_std  # Set a dynamic maximum value
cmap_std = plt.get_cmap('cmo.matter', 10)

# Plot each scenario's standard deviation values
for idx, scenario in enumerate(scenarios):
    std_var = std_values['romsoc_fully_coupled'][scenario][varias[0]]
    plot = ax[idx].pcolormesh(std_var.lon, std_var.lat, std_var, vmin=vmin_std, vmax=vmax_std, cmap=cmap_std)
    ax[idx].contour(std_var.lon, std_var.lat, std_var, levels=5, colors='k', linewidths=0.2)
    ax[idx].set_title(f'{scenario} standard deviation')

# Add continent (landmask) to all subplots
for axi in ax.flatten():
    axi.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

# Set the figure title 
title_map = {
    'Hplus': 'Standard deviation of surface Hplus concentrations for different scenarios',
    'pH_offl': 'Standard deviation of surface pH for different scenarios',
    'omega_arag_offl': 'Standard deviation of surface omega aragonite for different scenarios'
}
fig.suptitle(title_map.get(varias[0], f'Standard deviation of surface {varias[0]} for different scenarios'), fontsize=16, y=1.02)
plt.subplots_adjust(top=0.85, right=0.9)

# Colorbar settings for standard deviation values
cbax_std = fig.add_axes([0.92, 0.2, 0.025, 0.6])
cbar_std = plt.colorbar(ax[0].collections[0], cax=cbax_std, extend='both')
if varias[0] == 'Hplus':
    cbar_std.ax.set_title(f'Std Dev [H+] \n(mol/L)', pad=25)
elif varias[0] == 'pH_offl':
    cbar_std.ax.set_title(f'Std Dev pH', pad=25)
elif varias[0] == 'omega_arag_offl':
    cbar_std.ax.set_title(r'Std Dev $\Omega_{arag}$', pad=25)

# Axis limits and ticks 
for axi in ax:
    axi.set_xlim([230, 245])
    axi.set_ylim([30, 50])

for axi in ax:
    axi.set_yticks(yticks)
    axi.set_xticks(xticks)
    axi.set_yticklabels([str(val) + '°N' for val in yticks])
    axi.set_xticklabels([str(360 - val) + '°W' for val in xticks])

# Make sure that every subplot has y-axis labels visible
for axi in ax:
    axi.tick_params(axis='y', which='both', labelleft=True)

# Save and show the figure
filename = f'maps_std_{varias[0]}_concentrations_surface.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')
plt.show()

#%% ###################### PLOTTING MEAN ANOMALIES ######################
# Plotting the anomalies for each scenario (future - present)
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey='row')

# Automatically determine vmin and vmax for anomalies
all_anomalies = [anomalies['romsoc_fully_coupled'][scenario][varias[0]] for scenario in ['ssp245', 'ssp585']]
anomaly_min = min([data.min().values for data in all_anomalies])
anomaly_max = max([data.max().values for data in all_anomalies])

# Parameters for plotting anomalies
vmin_anomaly = anomaly_min  # Set a dynamic minimum value for anomalies
vmax_anomaly = anomaly_max  # Set a dynamic maximum value for anomalies
cmap_anomaly = plt.get_cmap('cmo.tempo_r', 10)

# Plot anomalies for ssp245 and ssp585
for idx, scenario in enumerate(['ssp245', 'ssp585']):
    var_anomaly = anomalies['romsoc_fully_coupled'][scenario][varias[0]]
    plot = ax[idx].pcolormesh(var_anomaly.lon, var_anomaly.lat, var_anomaly, vmin=vmin_anomaly, vmax=vmax_anomaly, cmap=cmap_anomaly)
    ax[idx].contour(var_anomaly.lon, var_anomaly.lat, var_anomaly, levels=10, colors='k', linewidths=0.2)
    ax[idx].set_title(f'{scenario} anomaly compared to present scenario')

# Add continent (landmask) to all subplots
for axi in ax.flatten():
    axi.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')


# Set the figure title 
title_map = {
    'Hplus': 'Anomalies of surface Hplus concentrations for future scenarios compared to present',
    'pH_offl': 'Anomalies of surface pH for future scenarios compared to present',
    'omega_arag_offl': 'Anomalies of surface omega aragonite for future scenarios compared to present'
}
fig.suptitle(title_map.get(varias[0], f'Anomalies of surface {varias[0]} concentrations for future scenarios compared to present'), fontsize=16, y=1.02)
plt.subplots_adjust(top=0.85, right=0.9)

# Colorbar settings for anomalies
cbax_anomaly = fig.add_axes([0.96, 0.2, 0.025, 0.6])
cbar_anomaly = plt.colorbar(ax[0].collections[0], cax=cbax_anomaly, extend='both')
if varias[0] == 'Hplus':
    cbar_anomaly.ax.set_title(f'[H+] anomaly \n(mol/L)', pad=25)
elif varias[0] == 'pH_offl':
    cbar_anomaly.ax.set_title(f'pH anomaly', pad=25)
elif varias[0] == 'omega_arag_offl':
    cbar_anomaly.ax.set_title(r'$\Omega_{arag}$ anomaly', pad=25)


# Axis limits and ticks 
for axi in ax:
    axi.set_xlim([230, 245])
    axi.set_ylim([30, 50])

for axi in ax:
    axi.set_yticks(yticks)
    axi.set_xticks(xticks)
    axi.set_yticklabels([str(val) + '°N' for val in yticks])
    axi.set_xticklabels([str(360 - val) + '°W' for val in xticks])

# Make sure that every subplot has y-axis labels visible
for axi in ax:
    axi.tick_params(axis='y', which='both', labelleft=True)

# Save and show the figure
filename = f'maps_{varias[0]}_anomalies_surface.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')
plt.show()



#%% ###################### PLOTTING DESEASONALIZED ######################
# Plotting the deseasonalized values for each variable and scenario

### !!!! this doesn't work for pH_offl, because i have no climatology for pH_offl, only for Hplus

fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey='row')

# Automatically determine vmin and vmax for deseasonalized dataset
all_deseasonalized = [deseasonalized_mean['romsoc_fully_coupled'][scenario][varias[0]].mean(dim='time') for scenario in scenarios]
combined_min_deseasonalized = min([data.min().values for data in all_deseasonalized])
combined_max_deseasonalized = max([data.max().values for data in all_deseasonalized])

# Parameters for plotting deseasonalized data
vmin_deseasonalized = -4e-12
vmax_deseasonalized = 4e-12
cmap_deseasonalized = plt.get_cmap('cmo.curl', 10)

# Plot deseasonalized data
for idx, scenario in enumerate(scenarios):
    # Take the mean over time to plot the spatial variability of deseasonalized data
    var_ds_mean = deseasonalized_mean['romsoc_fully_coupled'][scenario][varias[0]].mean(dim='time')
    plot = ax[idx].pcolormesh(var_ds_mean.lon, var_ds_mean.lat, var_ds_mean, 
                              vmin=vmin_deseasonalized, vmax=vmax_deseasonalized, cmap=cmap_deseasonalized)
    ax[idx].contour(var_ds_mean.lon, var_ds_mean.lat, var_ds_mean, levels=10, colors='k', linewidths=0.2)
    ax[idx].set_title(f'{scenario} deseasonalized')

# Add continent (landmask) to all subplots
for axi in ax.flatten():
    axi.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

# Set the figure title 
title_map = {
    'Hplus': 'Deseasonalized surface Hplus concentrations for different scenarios',
    'pH_offl': 'Deseasonalized surface pH for different scenarios'
}
fig.suptitle(title_map.get(varias[0], f'Deseasonalized surface {varias[0]} concentrations for different scenarios'), fontsize=16, y=1.02)
plt.subplots_adjust(top=0.85, right=0.9)

# Colorbar settings for deseasonalized data
cbax_deseasonalized = fig.add_axes([0.92, 0.2, 0.025, 0.6])
cbar_deseasonalized = plt.colorbar(ax[0].collections[0], cax=cbax_deseasonalized, extend='both')
cbar_deseasonalized.ax.set_title(f'Deseasonalized [{varias[0]}] \n(mol/L)', pad=25)

# Axis limits and ticks 
for axi in ax:
    axi.set_xlim([230, 245])
    axi.set_ylim([30, 50])

for axi in ax:
    axi.set_yticks(yticks)
    axi.set_xticks(xticks)
    axi.set_yticklabels([str(val) + '°N' for val in yticks])
    axi.set_xticklabels([str(360 - val) + '°W' for val in xticks])

# Make sure that every subplot has y-axis labels visible
for axi in ax:
    axi.tick_params(axis='y', which='both', labelleft=True)

# Save and show the figure
filename = f'maps_deseasonalized_{varias[0]}_concentrations.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')
plt.show()

# %%
