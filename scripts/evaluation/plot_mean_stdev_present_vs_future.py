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
varias = ['omega_arag_offl'] # 'pH_offl' , 'Hplus' , 'omega_arag_offl'

#%% Define params based on the variable

if varias[0] == 'pH_offl' or varias[0] == 'Hplus':
    params = ThresholdParameters.Hplus_instance() #95th percentile threshold
elif varias[0] == 'omega_arag_offl':
    params = ThresholdParameters.omega_arag_instance() #5th percentile threshold

#%%

#% Get the distance to coast file from ROMS and the regions over which to calculate the statistics
model_d2coasts = dict()
model_d2coasts['roms_only'] = ModelGetter.get_distance_to_coast(vtype='oceanic')

# Get the model regions
model_regions = dict()
model_regions['roms_only'] = GetRegions.define_CalCS_regions(model_d2coasts['roms_only'].lon, model_d2coasts['roms_only'].lat, model_d2coasts['roms_only'])

# Get the model area
model_area = ModelGetter.get_model_area()

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
regions = {
    "northern": (40.5, 50),
    "central": (34.7, 40.5),
    "southern": (30, 34.7)
}

for config in configs:
    for scenario in scenarios:
        for var in varias:
            print(f"\nRegional means for {var} in {config}, {scenario}:")

            data = variables[config][scenario][var]

            # Filter out all values over 15 to avoid the land points getting included in the mean
            data = data.where(data <= 15)

            for region_name, (lat_min, lat_max) in regions.items():
                # Create a mask for the latitude range
                lat = data.coords["lat"]
                mask = (lat >= lat_min) & (lat <= lat_max)

                region_data = data.where(mask, drop=True)

                # Calculate the mean 
                regional_mean = region_data.mean(dim=["eta_rho", "xi_rho", "time"], skipna=True).values.item()
                print(f"  {region_name.capitalize()} Region ({lat_min}° to {lat_max}°): {regional_mean}")


""" 
Results for Hplus:

Regional means for Hplus in romsoc_fully_coupled, present:
  Northern Region (40.5° to 50°): 7.208587094222702e-09
  Central Region (34.7° to 40.5°): 7.648157135052504e-09
  Southern Region (30° to 34.7°): 4.724110824803284e-09

Regional means for Hplus in romsoc_fully_coupled, ssp245:
  Northern Region (40.5° to 50°): 1.0347018446462063e-08
  Central Region (34.7° to 40.5°): 1.095028972410268e-08
  Southern Region (30° to 34.7°): 6.812238284520926e-09

Regional means for Hplus in romsoc_fully_coupled, ssp585:
  Northern Region (40.5° to 50°): 1.5046162445283874e-08
  Central Region (34.7° to 40.5°): 1.5803866854165455e-08
  Southern Region (30° to 34.7°): 9.90893191035909e-09

Results for pH_offl:

Regional means for pH_offl in romsoc_fully_coupled, present:
  Northern Region (40.5° to 50°): 8.046490784683161
  Central Region (34.7° to 40.5°): 8.045760011797617
  Southern Region (30° to 34.7°): 8.058400060750342

Regional means for pH_offl in romsoc_fully_coupled, ssp245:
  Northern Region (40.5° to 50°): 7.8892634837749505
  Central Region (34.7° to 40.5°): 7.889451641291051
  Southern Region (30° to 34.7°): 7.899176804943013

Regional means for pH_offl in romsoc_fully_coupled, ssp585:
  Northern Region (40.5° to 50°): 7.726728488531865
  Central Region (34.7° to 40.5°): 7.7303656935108345
  Southern Region (30° to 34.7°): 7.736400510504098

  
Results for omega_arag_offl:

Regional means for omega_arag_offl in romsoc_fully_coupled, present:
  Northern Region (40.5° to 50°): 1.9984736527400169
  Central Region (34.7° to 40.5°): 2.32724776285143
  Southern Region (30° to 34.7°): 2.7258183933369384

  Mean overall: 2.33

Regional means for omega_arag_offl in romsoc_fully_coupled, ssp245:
  Northern Region (40.5° to 50°): 1.5340139579761358
  Central Region (34.7° to 40.5°): 1.789018823479026
  Southern Region (30° to 34.7°): 2.0861851946906866

  Mean overall: 1.83

Regional means for omega_arag_offl in romsoc_fully_coupled, ssp585:
  Northern Region (40.5° to 50°): 1.1381265371558955
  Central Region (34.7° to 40.5°): 1.3444900265592823
  Southern Region (30° to 34.7°): 1.5593920796799703

  Mean overall: 1.35
"""

#%% 
# Define bins for distance to coast
bins_d2coast = [0, 100, np.max(model_d2coasts['roms_only'].values)]  # coastal: 0–100 km, offshore: >100 km
bin_labels = ['coastal', 'offshore']  

# %% Calculating the mean pH values for the different scenarios for coastal region and offshore region

# Initialize dictionary to store results
coastal_offshore_means = {}

for config in configs:
    coastal_offshore_means[config] = {}
    for scenario in scenarios:
        coastal_offshore_means[config][scenario] = {}
        for var in varias:
            print(f"\nProcessing {var} in {config}, {scenario}:")

            # Get Hplus data and distance to coast
            data = variables[config][scenario][var]
            d2coast = model_d2coasts['roms_only']

            # Flatten data and distance to coast for binning
            d2coast_flat = d2coast.values.flatten()
            data_flat = data.mean(dim="time", skipna=True).values.flatten()  # Mean over time

            # Mask NaNs
            nanmask = ~np.isnan(d2coast_flat) & ~np.isnan(data_flat)
            d2coast_flat = d2coast_flat[nanmask]
            data_flat = data_flat[nanmask]

            # Bin data based on distance to coast
            bin_indices = np.digitize(d2coast_flat, bins_d2coast)

            # Compute means for each bin (Coastal and Offshore)
            for bdx, label in enumerate(bin_labels):
                bin_data = data_flat[bin_indices == bdx + 1]
                if len(bin_data) > 0:  # Avoid errors with empty bins
                    bin_mean = np.mean(bin_data)  # Mean for current bin
                else:
                    bin_mean = np.nan  # No data for this bin

                # Print results
                print(f"  {label} Mean ({bins_d2coast[bdx]}–{bins_d2coast[bdx + 1]} km): {bin_mean}")

                # Store the results
                if label not in coastal_offshore_means[config][scenario]:
                    coastal_offshore_means[config][scenario][label] = []
                coastal_offshore_means[config][scenario][label].append(bin_mean)

print("\nSummary of coastal and offshore means:")
for config, scenarios_data in coastal_offshore_means.items():
    for scenario, regions_data in scenarios_data.items():
        for region, means in regions_data.items():
            print(f"{config} - {scenario} - {region}: {means}")

"""
Summary of Coastal and Offshore Means of Hplus:
romsoc_fully_coupled - present - coastal: [9.289525443908393e-09]
romsoc_fully_coupled - present - offshore: [9.031097905028417e-09]
romsoc_fully_coupled - ssp245 - coastal: [1.2868395852681426e-08]
romsoc_fully_coupled - ssp245 - offshore: [1.2789945992652557e-08]
romsoc_fully_coupled - ssp585 - coastal: [1.8078316372919678e-08]
romsoc_fully_coupled - ssp585 - offshore: [1.8332000763629608e-08]

Summary of coastal and offshore means of pH_offl:
romsoc_fully_coupled - present - coastal: [8.03868412049604]
romsoc_fully_coupled - present - offshore: [8.046310659641787]
romsoc_fully_coupled - ssp245 - coastal: [7.89671770352317]
romsoc_fully_coupled - ssp245 - offshore: [7.894714766597022]
romsoc_fully_coupled - ssp585 - coastal: [7.749082332476205]
romsoc_fully_coupled - ssp585 - offshore: [7.738296055063372]

Summary of coastal and offshore means of omega_arag_offl:
romsoc_fully_coupled - present - coastal: [2.4971079766514714]
romsoc_fully_coupled - present - offshore: [2.8220408175701928]
romsoc_fully_coupled - ssp245 - coastal: [1.9877593234378141]
romsoc_fully_coupled - ssp245 - offshore: [2.213561886821444]
romsoc_fully_coupled - ssp585 - coastal: [1.5308377876342993]
romsoc_fully_coupled - ssp585 - offshore: [1.6857885103446386]
"""

# %% Calculating the mean pH values for the different scenarios for coastal region and offshore region split up into southerm, central and northern regions
# Define latitude regions
regions = {
    "southern": (30, 34.7),
    "central": (34.7, 40.5),
    "northern": (40.5, 50)
}

# Initialize dictionary to store results
regional_coastal_offshore_means = {}

for config in configs:
    regional_coastal_offshore_means[config] = {}
    for scenario in scenarios:
        regional_coastal_offshore_means[config][scenario] = {}
        for var in varias:
            print(f"\nProcessing {var} in {config}, {scenario}:")
            
            # Get Hplus data and distance to coast
            data = variables[config][scenario][var]
            d2coast = model_d2coasts['roms_only']
            lat = data.coords["lat"]

            for region_name, (lat_min, lat_max) in regions.items():
                print(f"  Region: {region_name.capitalize()} ({lat_min}° to {lat_max}°)")

                # Create latitude mask for the region
                lat_mask = (lat >= lat_min) & (lat <= lat_max)

                # Apply the latitude mask
                region_data = data.where(lat_mask)

                # Flatten data and distance to coast for binning
                d2coast_flat = d2coast.values.flatten()
                region_data_flat = region_data.mean(dim="time", skipna=True).values.flatten()

                # Mask NaNs
                nanmask = ~np.isnan(d2coast_flat) & ~np.isnan(region_data_flat)
                d2coast_flat = d2coast_flat[nanmask]
                region_data_flat = region_data_flat[nanmask]

                # Bin data based on distance to coast
                bin_indices = np.digitize(d2coast_flat, bins_d2coast)

                # Compute means for each bin (Coastal and Offshore)
                for bdx, label in enumerate(bin_labels):
                    bin_data = region_data_flat[bin_indices == bdx + 1]
                    if len(bin_data) > 0:  # Avoid errors with empty bins
                        bin_mean = np.mean(bin_data)  # Mean for current bin
                    else:
                        bin_mean = np.nan  # No data for this bin

                    # Print results
                    print(f"    {label} Mean ({bins_d2coast[bdx]}–{bins_d2coast[bdx + 1]} km): {bin_mean}")

                    # Store the results
                    if region_name not in regional_coastal_offshore_means[config][scenario]:
                        regional_coastal_offshore_means[config][scenario][region_name] = {}
                    if label not in regional_coastal_offshore_means[config][scenario][region_name]:
                        regional_coastal_offshore_means[config][scenario][region_name][label] = []
                    regional_coastal_offshore_means[config][scenario][region_name][label].append(bin_mean)

print("\nSummary of regional coastal and offshore means:")
for config, scenarios_data in regional_coastal_offshore_means.items():
    for scenario, regions_data in scenarios_data.items():
        for region, bins_data in regions_data.items():
            for bin_label, means in bins_data.items():
                print(f"{config} - {scenario} - {region.capitalize()} - {bin_label}: {means}")

"""
Summary of regional coastal and offshore means of Hplus:
romsoc_fully_coupled - present - Southern - coastal: [9.069333008755343e-09]
romsoc_fully_coupled - present - Southern - offshore: [8.73930227024583e-09]
romsoc_fully_coupled - present - Central - coastal: [9.915313468078459e-09]
romsoc_fully_coupled - present - Central - offshore: [8.933067974990449e-09]
romsoc_fully_coupled - present - Northern - coastal: [9.411043119547298e-09]
romsoc_fully_coupled - present - Northern - offshore: [9.00421895544354e-09]
romsoc_fully_coupled - ssp245 - Southern - coastal: [1.2741764247195705e-08]
romsoc_fully_coupled - ssp245 - Southern - offshore: [1.2663965090890772e-08]
romsoc_fully_coupled - ssp245 - Central - coastal: [1.3443138748638023e-08]
romsoc_fully_coupled - ssp245 - Central - offshore: [1.2912128245195544e-08]
romsoc_fully_coupled - ssp245 - Northern - coastal: [1.269915532645469e-08]
romsoc_fully_coupled - ssp245 - Northern - offshore: [1.3061922499444426e-08]
romsoc_fully_coupled - ssp585 - Southern - coastal: [1.808456542802955e-08]
romsoc_fully_coupled - ssp585 - Southern - offshore: [1.850323044531201e-08]
romsoc_fully_coupled - ssp585 - Central - coastal: [1.8340881092799878e-08]
romsoc_fully_coupled - ssp585 - Central - offshore: [1.8807316745401582e-08]
romsoc_fully_coupled - ssp585 - Northern - coastal: [1.755435201804466e-08]
romsoc_fully_coupled - ssp585 - Northern - offshore: [1.9149049420301958e-08]

Summary of regional coastal and offshore means of pH_offl:
romsoc_fully_coupled - present - Southern - coastal: [8.047465246379714]
romsoc_fully_coupled - present - Southern - offshore: [8.060407924157756]
romsoc_fully_coupled - present - Central - coastal: [8.014109220669617]
romsoc_fully_coupled - present - Central - offshore: [8.050893659095633]
romsoc_fully_coupled - present - Northern - coastal: [8.038212525808195]
romsoc_fully_coupled - present - Northern - offshore: [8.047897422404793]
romsoc_fully_coupled - ssp245 - Southern - coastal: [7.899501142623039]
romsoc_fully_coupled - ssp245 - Southern - offshore: [7.899117249678885]
romsoc_fully_coupled - ssp245 - Central - coastal: [7.880185826714971]
romsoc_fully_coupled - ssp245 - Central - offshore: [7.8909545239130905]
romsoc_fully_coupled - ssp245 - Northern - coastal: [7.907510059008177]
romsoc_fully_coupled - ssp245 - Northern - offshore: [7.886163034585961]
romsoc_fully_coupled - ssp585 - Southern - coastal: [7.747342695595833]
romsoc_fully_coupled - ssp585 - Southern - offshore: [7.734391293675982]
romsoc_fully_coupled - ssp585 - Central - coastal: [7.745952988773907]
romsoc_fully_coupled - ssp585 - Central - offshore: [7.7278374888953945]
romsoc_fully_coupled - ssp585 - Northern - coastal: [7.7669554559813445]
romsoc_fully_coupled - ssp585 - Northern - offshore: [7.7198931417881225]

Summary of regional coastal and offshore means for omega_arag_offl:
romsoc_fully_coupled - present - Southern - coastal: [2.756667500577833]
romsoc_fully_coupled - present - Southern - offshore: [2.720153844174468]
romsoc_fully_coupled - present - Central - coastal: [2.1322224155675333]
romsoc_fully_coupled - present - Central - offshore: [2.358880190081989]
romsoc_fully_coupled - present - Northern - coastal: [1.9456395280314587]
romsoc_fully_coupled - present - Northern - offshore: [2.0074512015134243]
romsoc_fully_coupled - ssp245 - Southern - coastal: [2.1533917894742105]
romsoc_fully_coupled - ssp245 - Southern - offshore: [2.0738446411303992]
romsoc_fully_coupled - ssp245 - Central - coastal: [1.7136125257618788]
romsoc_fully_coupled - ssp245 - Central - offshore: [1.8012494604627816]
romsoc_fully_coupled - ssp245 - Northern - coastal: [1.5968663510504877]
romsoc_fully_coupled - ssp245 - Northern - offshore: [1.5233341099146145]
romsoc_fully_coupled - ssp585 - Southern - coastal: [1.6340630504626712]
romsoc_fully_coupled - ssp585 - Southern - offshore: [1.545680908666053]
romsoc_fully_coupled - ssp585 - Central - coastal: [1.3514246105106777]
romsoc_fully_coupled - ssp585 - Central - offshore: [1.3433652613671225]
romsoc_fully_coupled - ssp585 - Northern - coastal: [1.237084179502637]
romsoc_fully_coupled - ssp585 - Northern - offshore: [1.1213117026986121]
"""

#%%
# Calculate mean values for each variable and scenario over the whole research area
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
landmask_etopo = PlotFuncs.get_etopo_data()
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
yticks = np.arange(30, 55, 5)
xticks = np.arange(230, 245, 5)
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
savedir = '/nfs/sea/work/fpfaeffli/plots/future_vs_present/means_and_stdev/'
filename = f'maps_std_{varias[0]}_concentrations_surface.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')
plt.show()

#%% ###################### PLOTTING MEAN ANOMALIES ######################
# Plotting the anomalies for each scenario (future - present)
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey='row')

# Automatically determine vmin and vmax for anomalies
all_anomalies = [anomalies['romsoc_fully_coupled'][scenario][varias[0]] for scenario in ['ssp245', 'ssp585']]
anomaly_min = min([data.min().values for data in all_anomalies])
anomaly_max = 0

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
landmask_etopo = PlotFuncs.get_etopo_data()
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
yticks = np.arange(30, 55, 5)
xticks = np.arange(230, 245, 5)
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
savedir = '/nfs/sea/work/fpfaeffli/plots/future_vs_present/means_and_stdev/'
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
landmask_etopo = PlotFuncs.get_etopo_data()
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
savedir = '/nfs/sea/work/fpfaeffli/plots/future_vs_present/means_and_stdev/'
filename = f'maps_deseasonalized_{varias[0]}_concentrations.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')
plt.show()

# %%
