"""
author: Fiona Pfäffli
description: Driver analysis of ocean acidification extremes for the SSP585 scenario.
"""

#%% enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')
# sys.path.append('/home/koehne/Documents/supervision/fiona/msc_fiona/scripts/modules/')
# sys.path.append('/home/koehne/Documents/supervision/fiona/msc_fiona/scripts/evaluation/')
# sys.path.append('/home/koehne/Documents/supervision/fiona/msc_fiona/scripts/climatologies_and_thresholds/')

# load the packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
from get_study_regions import GetRegions as GetRegions
from get_model_datasets_omega_surf_only import ModelGetter as ModelGetter

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
import glob

#Get the distance to coast file from ROMS and the regions over which to calculate the statistics
model_d2coasts = dict()
model_d2coasts['roms_only'] = ModelGetter.get_distance_to_coast(vtype='oceanic')

# Get the model regions
model_regions = dict()
model_regions['roms_only'] = GetRegions.define_CalCS_regions(model_d2coasts['roms_only'].lon, model_d2coasts['roms_only'].lat, model_d2coasts['roms_only'])

# Get the model area
model_area = ModelGetter.get_model_area()

#%% 

# Define the threshold 
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.omega_arag_offl_instance() #95th percentile threshold

# Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present', 'ssp245', 'ssp585'] # ,'ssp585'
configs = ['romsoc_fully_coupled'] # [ms_only'] 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'zavg'    # 'avg'
depth = 0

#%% 

################ Loading all the data ##################

# Get the model datasets for the oceanic variables

ocean_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)

#%%
# load the data at the respective location
def load_data_in(varia):
    vdict = dict()
    for config in configs:
        vdict[config] = dict()
        for scenario in scenarios:
            print(f'Getting the variable for config {config} and scenario {scenario}.')
            print(varia)
            if varia == 'omega_arag_offl':
                vdict[config][scenario] = ocean_ds[config][scenario].omega_arag_offl.load()
            elif varia == 'temp':
                vdict[config][scenario] = ocean_ds[config][scenario].temp.load()
            elif varia == 'salt':
                vdict[config][scenario] = ocean_ds[config][scenario].salt.load()
            elif varia == 'DIC':
                vdict[config][scenario] = ocean_ds[config][scenario].DIC.load()
            elif varia == 'Alk':
                vdict[config][scenario] = ocean_ds[config][scenario].Alk.load()
    return vdict

omega = load_data_in('omega_arag_offl')
tem = load_data_in('temp')
sal = load_data_in('salt')
dic = load_data_in('DIC')
alk = load_data_in('Alk')

#%% LOAD in all the climatologies

def load_clim_in(varia):
    cdict = dict()
    for config in configs:
        cdict[config] = dict()
        for scenario in scenarios:
            print(f'Getting the variable for config {config} and scenario {scenario}.')
            clim_dir = f'/nfs/sea/work/fpfaeffli/future_sim/thresholds_and_climatologies/{config}/{scenario}/'
            clim_file = f'hobday2016_threshold_and_climatology_{varia}_5.0perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing_0depthlevelindex_ensemble000.nc'
            ds_dummy = xr.open_dataset(clim_dir+clim_file)
            clim_365 = ds_dummy.clim_smoothed
            clim_366 = ModelGetter.include_feb29(clim_365)
            concatenated_clim = ModelGetter.concatenate_yearly_arrays(clim_365, clim_366, start_year=2011, end_year=2021)
            if 'day_of_year_adjusted' in concatenated_clim.dims:
                concatenated_clim = concatenated_clim.rename({'day_of_year_adjusted': 'time'})
            elif 'day_of_year_adjusted' in concatenated_clim.coords:
                concatenated_clim = concatenated_clim.rename_coords({'day_of_year_adjusted': 'time'})
            cdict[config][scenario] = concatenated_clim
    return cdict

omega_clim = load_clim_in('omega_arag_offl')
tem_clim = load_clim_in('temp')
sal_clim = load_clim_in('salt')
dic_clim = load_clim_in('DIC')
alk_clim = load_clim_in('Alk')

#%% LOAD in the threshold for omega

print("Loading the threshold for omega...")
def load_thresh_in(varia,thresh_value):
    print('Get the present day threshold')
    config = configs[0]
    threshold, threshold_366 = ModelGetter.get_threshold(varia, 0, 'relative', thresh_value, config, 'present')
    concatenated_threshold = ModelGetter.concatenate_yearly_arrays(threshold, threshold_366, start_year=2011, end_year=2021)
    if 'day_of_year_adjusted' in concatenated_threshold.dims:
        concatenated_threshold = concatenated_threshold.rename({'day_of_year_adjusted': 'time'})
    elif 'day_of_year_adjusted' in concatenated_threshold.coords:
        concatenated_threshold = concatenated_threshold.rename_coords({'day_of_year_adjusted': 'time'})

    thresholds_mult = dict()
    thresholds_mult[config] = dict()
    for scenario in ['present', 'ssp245', 'ssp585']:
        thresholds_mult[config][scenario] = dict()
        print(f'{config}, {scenario}')
        thresholds_mult[config][scenario]['present'] = concatenated_threshold
        thresholds_mult[config][scenario]['present_plus_meandelta'] = concatenated_threshold + (omega_clim[config][scenario].mean(dim='time') - omega_clim[config]['present'].mean(dim='time'))
    
    return thresholds_mult

thresholds_mult = load_thresh_in('omega_arag_offl',5)

# %% COMPUTE the anomalies (instantaneous minus climatology)

def calc_anomalies_rel_to_clim(inst_dict,clim_dict):
    anom_dict = dict()
    for config in configs:
        anom_dict[config] = dict()
        for scenario in scenarios:
            print(f'Calculating anomalies for {scenario} and {config}')
            anom_dict[config][scenario] = dict()
            for scenario_that_anoms_are_relative_to in scenarios:
                anom_dict[config][scenario][scenario_that_anoms_are_relative_to] = inst_dict[config][scenario] - clim_dict[config][scenario_that_anoms_are_relative_to]
    return anom_dict

omega_anom = calc_anomalies_rel_to_clim(omega,omega_clim)
tem_anom   = calc_anomalies_rel_to_clim(tem,tem_clim)
sal_anom   = calc_anomalies_rel_to_clim(sal,sal_clim)
dic_anom   = calc_anomalies_rel_to_clim(dic,dic_clim)
alk_anom   = calc_anomalies_rel_to_clim(alk,alk_clim)


# %% COMPUTE THE THRESHOLD EXCEEDANCES (EXTREMES MASK)

print("Calculating the extreme masks...")
def calc_extreme_masks(inst_dict,thresh_dict):
    extreme_mask_dict = dict()
    for config in configs:
        extreme_mask_dict[config] = dict()
        for scenario in scenarios:
            extreme_mask_dict[config][scenario] = dict()
            for threshold_type in ['present','present_plus_meandelta']:
                extreme_mask_dict[config][scenario][threshold_type] = inst_dict[config][scenario] > thresh_dict[config][scenario][threshold_type]
    return extreme_mask_dict

extreme_mask_dict = calc_extreme_masks(omega,thresholds_mult)

#%% LOAD IN THE SENSITIVITIES

def load_in_sensitivities(configs,scenarios):
    sens_dict = dict()
    for config in configs:
        sens_dict[config] = dict()
        for scenario in scenarios:
            print(f'Loading sensitivities for {config} and {scenario}')
            root_dir = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/{config}/{scenario}/daily/z_avg/mocsy_co2_chemistry/'
            file_dum = f'pactcs30_{config}_{scenario}_daily_*_mocsy_sensitivities_depthlevel_0.nc'
            years = np.arange(2011,2022)
            all_files = []
            for year in years:
                search_here = f'{root_dir}{year}/{file_dum}'
                list_dum = sorted(glob.glob(search_here))
                for list_dummy in list_dum:
                    all_files.append(list_dummy)
            print(len(all_files))
            ds_dummy =  xr.open_mfdataset(all_files)
            sens_dict[config][scenario] = ds_dummy.compute()
    return sens_dict

sens_dict = load_in_sensitivities(configs,scenarios)

#%% multiply anomalies with sensitivities

def multiply_anom_with_sens(varia,sens_wrt_var,anoms,sens_dict):
    mult_dict = dict()
    for config in configs:
        mult_dict[config] = dict()
        for scenario in scenarios:
            print(config,scenario)
            mult_dict[config][scenario] = dict()
            if varia == 'temp' and sens_wrt_var == 'h':
                sensitivity = sens_dict[config][scenario].dh_dtem
            elif varia == 'salt' and sens_wrt_var == 'h':
                sensitivity = sens_dict[config][scenario].dh_dsal
            elif varia == 'DIC' and sens_wrt_var == 'h':
                sensitivity = sens_dict[config][scenario].dh_ddic
            elif varia == 'Alk' and sens_wrt_var == 'h':
                sensitivity = sens_dict[config][scenario].dh_dalk
            elif varia == 'temp' and sens_wrt_var == 'omegaa':
                sensitivity = sens_dict[config][scenario].domegaa_dtem
            elif varia == 'salt' and sens_wrt_var == 'omegaa':
                sensitivity = sens_dict[config][scenario].domegaa_dsal
            elif varia == 'DIC' and sens_wrt_var == 'omegaa':
                sensitivity = sens_dict[config][scenario].domegaa_ddic
            elif varia == 'Alk' and sens_wrt_var == 'omegaa':
                sensitivity = sens_dict[config][scenario].domegaa_dalk

            for scenario_that_anoms_are_relative_to in scenarios:
                print(scenario_that_anoms_are_relative_to)
                mult_dict[config][scenario][scenario_that_anoms_are_relative_to] = anoms[config][scenario][scenario_that_anoms_are_relative_to] * sensitivity

    return mult_dict

dtem_times_dhdtem = multiply_anom_with_sens('temp','h',tem_anom,sens_dict)
dsal_times_dhdsal = multiply_anom_with_sens('salt','h',sal_anom,sens_dict)
ddic_times_dhddic = multiply_anom_with_sens('DIC','h',dic_anom,sens_dict)
dalk_times_dhdalk = multiply_anom_with_sens('Alk','h',alk_anom,sens_dict)



#%% now mask all the non-extreme values, using the present-day threshold

def apply_extreme_masks(mult_dict,mask_dict):
    applied_mask_dict = dict()
    for config in configs:
        applied_mask_dict[config] = dict()
        for scenario in scenarios:
            applied_mask_dict[config][scenario] = dict()
            for scenario_that_anoms_are_relative_to in scenarios:
                applied_mask_dict[config][scenario][scenario_that_anoms_are_relative_to] = dict()
                for threshold_type in ['present','present_plus_meandelta']:
                    applied_mask_dict[config][scenario][scenario_that_anoms_are_relative_to][threshold_type] = mult_dict[config][scenario][scenario_that_anoms_are_relative_to].where(mask_dict[config][scenario][threshold_type]==True,np.NaN) 
    return applied_mask_dict

dtem_times_dhdtem_masked = apply_extreme_masks(dtem_times_dhdtem,extreme_mask_dict)
dsal_times_dhdsal_masked = apply_extreme_masks(dsal_times_dhdsal,extreme_mask_dict)
ddic_times_dhddic_masked = apply_extreme_masks(ddic_times_dhddic,extreme_mask_dict)
dalk_times_dhdalk_masked = apply_extreme_masks(dalk_times_dhdalk,extreme_mask_dict)


#%%
###############################
#         PLOTTING MAPS       #
###############################


#%% Functions to plot contributions for fixed and moving baselines of each driver 

# Load landmask
landmask_etopo = PlotFuncs.get_etopo_data()

# Define bounds for the California Current System (CaCS)
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

# Define the drivers and their respective datasets
drivers = {
    "temperature": dtem_times_dhdtem_masked,
    "salinity": dsal_times_dhdsal_masked,
    "DIC": ddic_times_dhddic_masked,
    "alkalinity": dalk_times_dhdalk_masked
}

# Function to compute vmin and vmax dynamically
def compute_vmin_vmax(data):
    max_val = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    return -max_val, max_val

# Function to plot contributions
def plot_contributions(data_fixed, data_moving, landmask_etopo, bounds, vmin, vmax, title, filename, cmap="cmo.balance"):
    lat_bounds, lon_bounds = bounds


    fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    levels = np.linspace(vmin, vmax, 40)

    for idx, (data, ax, baseline) in enumerate(zip([data_fixed, data_moving], axs, ["Fixed baseline", "Moving baseline"])):
        # Add landmask
        ax.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, levels=1, colors='black', zorder=1)

        # Plot data
        contour = ax.contourf(data["lon"], data["lat"], data, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_title(baseline, fontsize=14)
        ax.set_xlabel("Longitude")
        if idx == 0:
            ax.set_ylabel("Latitude")

        # Add degree symbols and N/W indicators
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

    # Add colorbar
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label="Change in [H+]", ticks=np.linspace(vmin, vmax, 11))

    # Add title
    plt.suptitle(title, fontsize=20, y=1.08)

    # Save and Show
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()


# %%
# Plotting for mean over time 

# Loop through each driver and plot
for driver_name, driver_data in drivers.items():
    print(f"Plotting {driver_name} contributions...")

    # Extract fixed and moving baseline data
    data_fixed_xr = driver_data["romsoc_fully_coupled"]["ssp585"]["present"]["present"].squeeze()
    data_moving_xr = driver_data["romsoc_fully_coupled"]["ssp585"]["ssp585"]["present_plus_meandelta"].squeeze()

    # Compute the mean over time for the entire period
    data_fixed_mean = data_fixed_xr.mean(dim="time")
    data_moving_mean = data_moving_xr.mean(dim="time")

    # Define fixed color scale limits
    vmin_driver, vmax_driver = -3e-9, 3e-9  # Example fixed limits, adjust as needed

    # Define file name
    filename = f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/contributions_maps/contributions_{driver_name.lower()}_fixed_vs_moving_baseline_mean_fixed_color_scale_3e-9.png"

    # Call the plotting function
    plot_contributions(
        data_fixed_mean,
        data_moving_mean,
        landmask_etopo,
        bounds,
        vmin_driver,
        vmax_driver,
        f"Comparison of mean {driver_name} contribution to H+ changes (fixed vs. moving baseline)",
        filename
    )

# PRESENT VS FIXED VS MOVING BASELINE 

# %%

# Load landmask
landmask_etopo = PlotFuncs.get_etopo_data()

# Define bounds for the California Current System (CaCS)
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

# Define the drivers and their respective datasets
drivers = {
    "temperature": dtem_times_dhdtem_masked,
    "salinity": dsal_times_dhdsal_masked,
    "DIC": ddic_times_dhddic_masked,
    "alkalinity": dalk_times_dhdalk_masked
}

# Function to plot contributions
def plot_contributions_present_future(data_present, data_future_fixed, data_future_moving, landmask_etopo, bounds, vmin, vmax, title, filename, cmap="cmo.balance"):
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
        contour = ax.contourf(data["lon"], data["lat"], data, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_title(period, fontsize=14)
        ax.set_xlabel("Longitude")
        if idx == 0:
            ax.set_ylabel("Latitude")

        # Add degree symbols and N/W indicators
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

    # Add colorbar
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label="Change in [H+]", ticks=np.linspace(vmin, vmax, 11))

    # Add title
    plt.suptitle(title, fontsize=20, y=1.08)

    # Save and Show
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()


# %% 
# Plotting for mean over time 

# Loop through each driver and plot for **Present** and **Future** (Fixed & Moving Baseline)
for driver_name, driver_data in drivers.items():
    print(f"Plotting {driver_name} contributions...")

    # Extract present, future fixed baseline, and future moving baseline data
    data_present_xr = driver_data["romsoc_fully_coupled"]["present"]["present"]["present"].squeeze()
    data_future_fixed_xr = driver_data["romsoc_fully_coupled"]["ssp585"]["present"]["present"].squeeze()
    data_future_moving_xr = driver_data["romsoc_fully_coupled"]["ssp585"]["ssp585"]["present_plus_meandelta"].squeeze()

    # Compute the mean over time for the entire period
    data_present_mean = data_present_xr.mean(dim="time")
    data_future_fixed_mean = data_future_fixed_xr.mean(dim="time")
    data_future_moving_mean = data_future_moving_xr.mean(dim="time")

    # Define fixed color scale limits dynamically
    vmin_driver, vmax_driver = compute_vmin_vmax(data_present_mean, data_future_fixed_mean, data_future_moving_mean)

    # Define file name
    filename = f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/contributions_maps/contributions_{driver_name.lower()}_present_vs_future_fixed_vs_moving_dyn_colorbar.png"

    # Call the plotting function
    plot_contributions_present_future(
        data_present_mean,
        data_future_fixed_mean,
        data_future_moving_mean,
        landmask_etopo,
        bounds,
        vmin_driver,
        vmax_driver,
        f"Comparison of mean {driver_name} contribution to H+ changes (present vs. future under fixed & moving baseline)",
        filename
    )


# %%

# Plotting for mean over time (without skipping nan values)

# Compute the mean over time for the entire period
data_fixed_mean = data_fixed_xr.mean(dim="time")
data_moving_mean = data_moving_xr.mean(dim="time")

# Call the plotting function
plot_contributions(
    data_fixed_mean,  # Use the mean over time instead of a single day
    data_moving_mean,  # Use the mean over time instead of a single day
    landmask_etopo,
    bounds,
    vmin_driver,
    vmax_driver,
    f"Comparison of mean {driver_name} contribution to H+ changes (fixed vs. moving baseline)",
    filename
)


#%%
###############################
#     PLOTTING TIMESERIES     #
###############################

# %%
# Function to find the closest grid point to a given latitude and longitude

def find_closest_grid_point(lat_rho, lon_rho, target_lat, target_lon):
    """
    Finds the closest (eta_rho, xi_rho) indices to a given latitude and longitude.
    """
    # Compute distance (great-circle distance approximation)
    dist = np.sqrt((lat_rho - target_lat) ** 2 + (lon_rho - target_lon) ** 2)
    
    print(f"Distance array shape: {dist.shape}")  # Debugging

    # Get the index of the minimum distance
    min_idx = np.argmin(dist)  # Flattened index
    indices = np.unravel_index(min_idx, dist.shape)  # Convert to 2D indices

    print(f"Closest grid indices: {indices}")  # Debugging

    # Ensure only two values are returned
    if len(indices) == 2:
        return int(indices[0]), int(indices[1])
    else:
        raise ValueError(f"Unexpected number of indices returned: {indices}")

#%% Plotting the time series at a random grid point

# this doesnt include present

def plot_driver_timeseries_at_point(ocean_ds, drivers, drivers_unmasked, target_lat, target_lon, config="romsoc_fully_coupled", scenario="ssp585"):
    """
    Plots the time series of multiple drivers at a single selected point.
    Uses latitude & longitude instead of grid indices.
    Adds black background lines for unmasked contributions.
    """
    dataset = ocean_ds[config][scenario]

    # Extract lat_rho and lon_rho (first time step only)
    lat_rho = dataset.lat_rho.values[0, :, :]
    lon_rho = dataset.lon_rho.values[0, :, :]

    # Convert West longitudes (e.g., -125°W → 235°E)
    if target_lon < 0:
        target_lon = 360 + target_lon

    # Find closest grid point
    eta_rho, xi_rho = find_closest_grid_point(lat_rho, lon_rho, target_lat, target_lon)

    # Define colors for each driver
    driver_colors = {
        "Temperature": "red",
        "Salinity": "blue",
        "DIC": "green",
        "Alkalinity": "purple"
    }

    # Create figure with two subplots: One for fixed, one for moving baseline
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Loop through each driver and plot both baselines
    for driver_name, driver_data in drivers.items():
        # Extract the unmasked version of this driver correctly
        driver_data_unmasked = drivers_unmasked[driver_name][config][scenario]  # ✅ Now correctly structured!

        # Extract time series at the identified closest grid point
        time_series_fixed = driver_data[config][scenario]["present"].isel(xi_rho=xi_rho, eta_rho=eta_rho).squeeze()
        time_series_moving = driver_data[config][scenario]["present_plus_meandelta"].isel(xi_rho=xi_rho, eta_rho=eta_rho).squeeze()

        # Extract the unmasked time series at the same point correctly
        time_series_fixed_unmasked = driver_data_unmasked.isel(xi_rho=xi_rho, eta_rho=eta_rho).squeeze()
        time_series_moving_unmasked = driver_data_unmasked.isel(xi_rho=xi_rho, eta_rho=eta_rho).squeeze()

        # Plot **unmasked** contributions in black (as background)
        axs[0].plot(time_series_fixed.time, time_series_fixed_unmasked, color="grey", linewidth=0.8, alpha=0.6, zorder=1)
        axs[1].plot(time_series_moving.time, time_series_moving_unmasked, color="grey", linewidth=0.8, alpha=0.6, zorder=1)

        # Plot **masked** contributions on top
        axs[0].plot(time_series_fixed.time, time_series_fixed, label=driver_name, color=driver_colors[driver_name], zorder=2)
        axs[1].plot(time_series_moving.time, time_series_moving, label=driver_name, color=driver_colors[driver_name], zorder=2)

        # Debugging: Print selected point information
        print(f"Target Lat/Lon: ({target_lat}°N, {target_lon}°E) -> Closest Grid Point: ({lat_rho[eta_rho, xi_rho]}, {lon_rho[eta_rho, xi_rho]})")

    # Formatting
    axs[0].set_title("Contribution of different drivers to change in [H+] (fixed baseline)", fontsize=14)
    axs[1].set_title("Contribution of different drivers to change in [H+] (moving baseline)", fontsize=14)

    # Common x and y labels
    axs[1].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("Change in [H+]", fontsize=12)
    axs[1].set_ylabel("Change in [H+]", fontsize=12)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Add legends
    axs[0].legend()
    axs[1].legend()

    # Add grid
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[1].grid(True, linestyle="--", alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    filename = f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/contributions_timeseries/contributions_all_drivers_timeseries_at_lat_{target_lat}_lon{target_lon}.png'
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')

    # Show the plot
    plt.show()

# %%
# Define a target location (Latitude, Longitude)
target_lat = 40   # Example: 40°N
target_lon = -125  # Example: 125°W (Converted to 235°E)

# Dictionary of masked driver data (contributions with missing values)
drivers_masked = {
    "Temperature": dtem_times_dhdtem_masked,
    "Salinity": dsal_times_dhdsal_masked,
    "DIC": ddic_times_dhddic_masked,
    "Alkalinity": dalk_times_dhdalk_masked
}

# Dictionary of unmasked driver data (original full contributions) 
drivers_unmasked = {
    "Temperature": dtem_times_dhdtem, 
    "Salinity": dsal_times_dhdsal,
    "DIC": ddic_times_dhddic,
    "Alkalinity": dalk_times_dhdalk
}

# Call the function
plot_driver_timeseries_at_point(ocean_ds, drivers_masked, drivers_unmasked, target_lat, target_lon, config="romsoc_fully_coupled", scenario="ssp585")


#%%¨
# this does include present





#%%
# Plotting driver timeseries at two selected points


def plot_driver_timeseries_at_two_points(ocean_ds, drivers, drivers_unmasked, driver_name, points_latlon, filename=None, config="romsoc_fully_coupled", scenario="ssp585"):
    """
    Plots the time series of a specific driver at two selected points.
    Uses latitude & longitude instead of grid indices.
    Adds black background lines for unmasked contributions.
    Saves the plot if a filename is provided.
    """
    dataset = ocean_ds[config][scenario]

    # Extract lat_rho and lon_rho (first time step only)
    lat_rho = dataset.lat_rho.values[0, :, :]
    lon_rho = dataset.lon_rho.values[0, :, :]

    # Define colors and labels for the two points
    point_colors = ["red", "blue"]
    point_labels = ["Point 1", "Point 2"]

    # Get the driver dataset
    driver_data = drivers[driver_name]
    driver_data_unmasked = drivers_unmasked[driver_name][config][scenario]

    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Loop through each of the two points
    for i, (target_lat, target_lon) in enumerate(points_latlon):
        # Convert West longitudes (e.g., -125°W → 235°E)
        if target_lon < 0:
            target_lon = 360 + target_lon

        # Find closest grid point
        eta_rho, xi_rho = find_closest_grid_point(lat_rho, lon_rho, target_lat, target_lon)

        # Extract time series at this grid point
        time_series_fixed = driver_data[config][scenario]["present"].isel(xi_rho=xi_rho, eta_rho=eta_rho).squeeze()
        time_series_moving = driver_data[config][scenario]["present_plus_meandelta"].isel(xi_rho=xi_rho, eta_rho=eta_rho).squeeze()

        # Extract the unmasked time series at the same point
        time_series_fixed_unmasked = driver_data_unmasked.isel(xi_rho=xi_rho, eta_rho=eta_rho).squeeze()
        time_series_moving_unmasked = driver_data_unmasked.isel(xi_rho=xi_rho, eta_rho=eta_rho).squeeze()

        # Plot **unmasked** contributions in black (as background)
        axs[0].plot(time_series_fixed.time, time_series_fixed_unmasked, color="grey", linewidth=0.8, alpha=0.6, zorder=1)
        axs[1].plot(time_series_moving.time, time_series_moving_unmasked, color="grey", linewidth=0.8, alpha=0.6, zorder=1)

        # Plot **masked** contributions on top
        axs[0].plot(time_series_fixed.time, time_series_fixed, label=f"{point_labels[i]} ({target_lat}°N, {target_lon}°E)", color=point_colors[i], zorder=2)
        axs[1].plot(time_series_moving.time, time_series_moving, label=f"{point_labels[i]} ({target_lat}°N, {target_lon}°E)", color=point_colors[i], zorder=2)

        # Debugging
        print(f"{point_labels[i]} -> Target: ({target_lat}, {target_lon}) | Closest Grid: ({lat_rho[eta_rho, xi_rho]}, {lon_rho[eta_rho, xi_rho]})")

    # Formatting
    axs[0].set_title(f"Contribution of {driver_name} to H+ changes (fixed baseline)", fontsize=14)
    axs[1].set_title(f"Contribution of {driver_name} to H+ changes (moving baseline)", fontsize=14)

    # Common labels
    axs[1].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("Change in [H+]", fontsize=12)
    axs[1].set_ylabel("Change in [H+]", fontsize=12)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Add legends
    axs[0].legend()
    axs[1].legend()

    # Add grid
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[1].grid(True, linestyle="--", alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    # Show the plot
    plt.show()



# %%

# %%
# Define the 2 selected points (lat, lon)
points_latlon = [
    (40, -125),  # Point 1: 40°N, 125°W  (Converted to 235°E)
    (40, -132)   # Point 2: 40°N, 132°W  (Converted to 228°E)
]

# Dictionary of driver names and their corresponding datasets
drivers = {
    "Temperature": dtem_times_dhdtem_masked,
    "Salinity": dsal_times_dhdsal_masked,
    "DIC": ddic_times_dhddic_masked,
    "Alkalinity": dalk_times_dhdalk_masked
}

# Dictionary of unmasked driver data (original full contributions)
drivers_unmasked = {
    "Temperature": dtem_times_dhdtem,
    "Salinity": dsal_times_dhdsal,
    "DIC": ddic_times_dhddic,
    "Alkalinity": dalk_times_dhdalk
}

# Define file paths
base_path = "/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/contributions_timeseries/"
temp_filename = f"{base_path}temperature_timeseries.png"
salinity_filename = f"{base_path}salinity_timeseries.png"
dic_filename = f"{base_path}dic_timeseries.png"
alkalinity_filename = f"{base_path}alkalinity_timeseries.png"

# Call the function for each driver individually with custom filenames
plot_driver_timeseries_at_two_points(ocean_ds, drivers, drivers_unmasked, "Temperature", points_latlon, filename=temp_filename)
plot_driver_timeseries_at_two_points(ocean_ds, drivers, drivers_unmasked, "Salinity", points_latlon, filename=salinity_filename)
plot_driver_timeseries_at_two_points(ocean_ds, drivers, drivers_unmasked, "DIC", points_latlon, filename=dic_filename)
plot_driver_timeseries_at_two_points(ocean_ds, drivers, drivers_unmasked, "Alkalinity", points_latlon, filename=alkalinity_filename)

# %%

# %%

def plot_extreme_mask_fraction(ocean_ds, extreme_mask_dict, target_variable, filename=None, config="romsoc_fully_coupled", scenario="ssp585"):
    """
    Plots the fraction of the domain experiencing extreme events over time.
    Uses the extreme mask for the entire spatial domain.
    """
    dataset = ocean_ds[config][scenario]

    # Extract time steps
    time_steps = dataset.time.values

    # Extract extreme mask and **squeeze** to remove extra dimensions
    mask_fixed = extreme_mask_dict[config][scenario]["present"]
    mask_moving = extreme_mask_dict[config][scenario]["present_plus_meandelta"]

    if "depth" in mask_fixed.dims:
        mask_fixed = mask_fixed.squeeze("depth")
        mask_moving = mask_moving.squeeze("depth")

    # Compute the fraction of extreme events over the entire domain at each time step
    fraction_fixed = mask_fixed.mean(dim=("eta_rho", "xi_rho"))
    fraction_moving = mask_moving.mean(dim=("eta_rho", "xi_rho"))

    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot fraction of extreme events
    axs[0].plot(time_steps, fraction_fixed, color="red", label="Fixed Baseline")
    axs[1].plot(time_steps, fraction_moving, color="blue", label="Moving Baseline")

    # Formatting
    axs[0].set_title(f"Fraction of grid cells with {target_variable} extremes (Fixed Baseline ({scenario}))", fontsize=14)
    axs[1].set_title(f"Fraction of grid cells with {target_variable} extremes (Moving Baseline ({scenario}))", fontsize=14)

    # Common labels
    axs[1].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("Fraction of extreme grid cells", fontsize=12)
    axs[1].set_ylabel("Fraction of extreme grid cells", fontsize=12)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Add legends
    axs[0].legend()
    axs[1].legend()

    # Add grid
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[1].grid(True, linestyle="--", alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    # Show the plot
    plt.show()

# %%
# Define file paths
base_path = "/nfs/sea/work/fpfaeffli/plots/extreme_masks/"
omega_filename = f"{base_path}omega_extreme_mask_fraction_{config}_{scenario}.png"

# Call the function to plot the extreme mask fraction for H+
plot_extreme_mask_fraction(ocean_ds, extreme_mask_dict, "H+", filename=omega_filename)


# %%
# Load landmask
landmask_etopo = PlotFuncs.get_etopo_data()



def plot_extreme_mask_map(ocean_ds, extreme_mask_dict, landmask_etopo, bounds, target_variable, filename=None, config="romsoc_fully_coupled", scenario="ssp585"):
    """
    Plots a spatial map of extreme event occurrences with:
    - Black landmask (fully masked)
    - White non-extreme areas
    - Red extreme areas
    - Labels for extreme and non-extreme regions
    """
    dataset = ocean_ds[config][scenario]

    # Extract latitude & longitude from ocean dataset (first time step only)
    lat_rho = dataset.lat_rho.values[0, :, :]
    lon_rho = dataset.lon_rho.values[0, :, :]

    # Extract extreme mask and **squeeze** to remove extra dimensions (e.g., depth, time)
    mask_fixed = extreme_mask_dict[config][scenario]["present"].squeeze()
    mask_moving = extreme_mask_dict[config][scenario]["present_plus_meandelta"].squeeze()

    # If there is a time dimension, select the first time step
    if "time" in mask_fixed.dims:
        mask_fixed = mask_fixed.isel(time=0)
        mask_moving = mask_moving.isel(time=0)

    # Ensure it's 2D (eta_rho, xi_rho)
    mask_fixed = mask_fixed.squeeze().values
    mask_moving = mask_moving.squeeze().values

    # Define spatial bounds
    lat_bounds, lon_bounds = bounds

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    for idx, (mask_data, ax, title) in enumerate(zip([mask_fixed, mask_moving], axs, ["Fixed Baseline", "Moving Baseline"])):
        # Set background to white (ocean)
        ax.set_facecolor("white")

        # **Ensure landmask is binary (1=land, 0=ocean)**
        landmask_binary = np.where(landmask_etopo > 0.5, 1, np.nan)  # Set land=1, ocean=NaN

        # **Plot landmask in black**
        ax.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_binary, levels=[0.5, 1.5], colors="black", zorder=3)

        # **Plot extreme mask using `contourf` for smooth visualization**
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

    # Add colorbar with labels for "Extreme" and "Non-Extreme"
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label="H+ Extreme Event Occurrence", ticks=[0, 1])
    cbar.ax.set_yticklabels(['Non-Extreme', 'Extreme'])

    # Add title
    plt.suptitle(f"Occurence of H+ extreme events ({scenario})", fontsize=20, y=1.08)

    # Save and Show
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    plt.show()


# %%
# Define spatial bounds (adjust based on your data)
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

# Define file paths
base_path = "/nfs/sea/work/fpfaeffli/plots/extreme_masks/"
omega_map_filename = f"{base_path}omega_extreme_mask_map_{config}_{scenario}.png"

# Call the function to plot the spatial distribution of extreme events for H+
plot_extreme_mask_map(ocean_ds, extreme_mask_dict, landmask_etopo, bounds, "H+", filename=omega_map_filename)

# %%


# %%
import matplotlib.pyplot as plt

def plot_driver_timeseries_at_point(
    ocean_ds, drivers, drivers_unmasked, target_lat, target_lon, config="romsoc_fully_coupled"
):
    """
    Plots time series of different drivers contributing to H+ changes at a given latitude and longitude.
    
    Includes present-day data and ensures fixed & moving baselines follow the correct climatologies.
    """
    dataset = ocean_ds[config]["ssp585"]  # Using SSP585 for future data

    # Extract lat/lon grid (assuming first time step is representative)
    lat_rho = dataset.lat_rho.values[0, :, :]
    lon_rho = dataset.lon_rho.values[0, :, :]

    print("Drivers keys inside time series:", drivers.keys())

    # Print inside one driver (temperature)
    print("Keys inside one driver (timeseries):", drivers["temperature"].keys())


    # Convert negative longitudes (e.g., -125°W to 235°E)
    if target_lon < 0:
        target_lon = 360 + target_lon

    # Find the closest grid point
    eta_rho, xi_rho = find_closest_grid_point(lat_rho, lon_rho, target_lat, target_lon)

    # Define colors for each driver
    driver_colors = {
        "temperature": "red",
        "salinity": "blue",
        "DIC": "green",
        "alkalinity": "purple",
    }

    # Create figure with three subplots: Present, Fixed, Moving Baseline
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for driver_name, driver_data in drivers.items():
        driver_data_unmasked = drivers_unmasked[driver_name][config]["ssp585"]

        # Extract present, future fixed baseline, and future moving baseline data
        # Extract present, future fixed baseline, and future moving baseline data
        time_series_present = driver_data["romsoc_fully_coupled"]["present"]["present"]["present"].squeeze()
        time_series_fixed = driver_data["romsoc_fully_coupled"]["ssp585"]["present"]["present"].squeeze()
        time_series_moving = driver_data["romsoc_fully_coupled"]["ssp585"]["ssp585"]["present_plus_meandelta"].squeeze()

        # Extract the unmasked time series
        time_series_fixed_unmasked = driver_data_unmasked.isel(xi_rho=xi_rho, eta_rho=eta_rho).squeeze()
        time_series_moving_unmasked = driver_data_unmasked.isel(xi_rho=xi_rho, eta_rho=eta_rho).squeeze()

        # Plot unmasked data as background (grey lines)
        axs[1].plot(time_series_fixed.time, time_series_fixed_unmasked, color="grey", linewidth=0.8, alpha=0.6)
        axs[2].plot(time_series_moving.time, time_series_moving_unmasked, color="grey", linewidth=0.8, alpha=0.6)

        # Plot masked contributions
        axs[0].plot(time_series_present.time, time_series_present, label=driver_name, color=driver_colors[driver_name])
        axs[1].plot(time_series_fixed.time, time_series_fixed, label=driver_name, color=driver_colors[driver_name])
        axs[2].plot(time_series_moving.time, time_series_moving, label=driver_name, color=driver_colors[driver_name])

    # Titles and labels
    axs[0].set_title("Present-Day Contributions", fontsize=14)
    axs[1].set_title("Future Contributions (Fixed Baseline)", fontsize=14)
    axs[2].set_title("Future Contributions (Moving Baseline)", fontsize=14)
    
    axs[2].set_xlabel("Time", fontsize=12)
    for ax in axs:
        ax.set_ylabel("Change in [H+]", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    
    # Adjust layout and save
    plt.tight_layout()
    filename = f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/contributions_timeseries/contributions_all_drivers_timeseries_at_lat_{target_lat}_lon{target_lon}.png'
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()

# Execution example
target_lat = 35.0  # Example latitude
target_lon = -125.0  # Example longitude (negative for west longitude)
plot_driver_timeseries_at_point(ocean_ds, drivers, drivers_unmasked, target_lat, target_lon)

# %%
