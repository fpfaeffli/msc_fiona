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
from get_model_datasets_Hplus_surf_only import ModelGetter as ModelGetter

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
params = ThresholdParameters.Hplus_instance() #95th percentile threshold

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
#% Get the distance to coast file from ROMS and the regions over which to calculate the statistics
model_d2coasts = dict()
model_d2coasts['roms_only'] = ModelGetter.get_distance_to_coast(vtype='oceanic')

# Get the model regions
model_regions = dict()
model_regions['roms_only'] = GetRegions.define_CalCS_regions(model_d2coasts['roms_only'].lon, model_d2coasts['roms_only'].lat, model_d2coasts['roms_only'])

# Get the model area
model_area = ModelGetter.get_model_area()

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
            if varia == 'Hplus':
                pH = ocean_ds[config][scenario].pH_offl.load()
                vdict[config][scenario] = np.power(10, -pH)
            elif varia == 'temp':
                vdict[config][scenario] = ocean_ds[config][scenario].temp.load()
            elif varia == 'salt':
                vdict[config][scenario] = ocean_ds[config][scenario].salt.load()
            elif varia == 'DIC':
                vdict[config][scenario] = ocean_ds[config][scenario].DIC.load()
            elif varia == 'Alk':
                vdict[config][scenario] = ocean_ds[config][scenario].Alk.load()
    return vdict

hplus = load_data_in('Hplus')
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
            clim_file = f'hobday2016_threshold_and_climatology_{varia}_95.0perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing_0depthlevelindex_ensemble000.nc'
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

hplus_clim = load_clim_in('Hplus')
tem_clim = load_clim_in('temp')
sal_clim = load_clim_in('salt')
dic_clim = load_clim_in('DIC')
alk_clim = load_clim_in('Alk')

#%% LOAD in the threshold for Hplus

print("Loading the threshold for Hplus...")
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
        thresholds_mult[config][scenario]['present_plus_meandelta'] = concatenated_threshold + (hplus_clim[config][scenario].mean(dim='time') - hplus_clim[config]['present'].mean(dim='time'))
    
    return thresholds_mult

thresholds_mult = load_thresh_in('Hplus',95)

# %% COMPUTE THE THRESHOLD EXCEEDANCES (EXTREMES MASK)

print("Calculating the extreme masks...")
def calc_extreme_masks(inst_dict,thresh_dict):
    extreme_mask_dict = dict()
    for config in configs:
        extreme_mask_dict[config] = dict()
        for scenario in scenarios:
            extreme_mask_dict[config][scenario] = dict()
            for threshold_type in ['present','present_plus_meandelta']:
                print(f'Calculating extreme mask for {inst_dict}: {config} and {scenario} and {threshold_type}')
                extreme_mask_dict[config][scenario][threshold_type] = inst_dict[config][scenario] > thresh_dict[config][scenario][threshold_type]
    return extreme_mask_dict

extreme_mask_dict = calc_extreme_masks(hplus,thresholds_mult)

# %% COMPUTE the anomalies (instantaneous minus climatology)

def calc_anomalies_rel_to_clim(inst_dict,clim_dict):
    anom_dict = dict()
    for config in configs:
        anom_dict[config] = dict()
        for scenario in scenarios:
            print(f'Calculating anomalies for {inst_dict} and {scenario} and {config}')
            anom_dict[config][scenario] = dict()
            for scenario_that_anoms_are_relative_to in scenarios:
                anom_dict[config][scenario][scenario_that_anoms_are_relative_to] = inst_dict[config][scenario] - clim_dict[config][scenario_that_anoms_are_relative_to]
    return anom_dict

hplus_anom = calc_anomalies_rel_to_clim(hplus,hplus_clim)
tem_anom   = calc_anomalies_rel_to_clim(tem,tem_clim)
sal_anom   = calc_anomalies_rel_to_clim(sal,sal_clim)
dic_anom   = calc_anomalies_rel_to_clim(dic,dic_clim)
alk_anom   = calc_anomalies_rel_to_clim(alk,alk_clim)


# %%

# Load sensitivities based on climatologies

def add_feb29_if_leap_year(ds, year):
    """Adds 29.2. for leap years by interpolating values from 28.2. and 1.3."""
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):  # Leap year condition
        # Extract February 28 and March 1
        feb28 = ds.sel(time="2001-02-28")
        mar01 = ds.sel(time="2001-03-01")

        # Compute the mean for February 29
        feb29 = (feb28 + mar01) / 2
        feb29 = feb29.assign_coords(time=pd.to_datetime(f"{year}-02-29"))

        # Split dataset before and after Feb 29
        before_feb29 = ds.sel(time=slice(None, "2001-02-28"))
        after_feb29 = ds.sel(time=slice("2001-03-01", None))

        # Concatenate everything with Feb 29 inserted
        ds = xr.concat([before_feb29, feb29, after_feb29], dim="time")

    return ds

def load_and_aggregate_sensitivities(scenarios):
    sens_dict = {}

    for scenario in scenarios:
        print(f'Aggregating sensitivities for {scenario}')

        # Define file path for the single-year dataset
        root_dir = f'/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled/{scenario}/daily/z_avg/mocsy_co2_chemistry/based_on_t_s_dic_alk_po4_sio3_clims/'
        file_name = f'pactcs30_romsoc_fully_coupled_{scenario}_daily_climatology_mocsy_sensitivities_depthlevel_0.nc'
        file_path = os.path.join(root_dir, file_name)

        # Load the single-year dataset
        ds_single_year = xr.open_dataset(file_path)

        # Ensure the dataset has a generic time dimension (2001 as a placeholder)
        ds_single_year = ds_single_year.assign_coords(time=pd.date_range("2001-01-01", periods=365, freq='D'))

        # Expand dataset to 11 years (2011-2021)
        years = np.arange(2011, 2022)
        extended_datasets = []

        for year in years:
            ds_copy = ds_single_year.copy()

            # First, add Feb 29 if it's a leap year
            ds_copy = add_feb29_if_leap_year(ds_copy, year)

            # Now, generate the correct time array for the dataset length
            days_in_year = ds_copy.sizes["time"]  # This is now 365 or 366
            new_time = pd.date_range(f'{year}-01-01', periods=days_in_year, freq='D')

            # Assign the correct year-specific time
            ds_copy = ds_copy.assign_coords(time=new_time)

            extended_datasets.append(ds_copy)

        # Concatenate all years into a single dataset
        aggregated_ds = xr.concat(extended_datasets, dim='time')

        # Store in dictionary
        sens_dict[scenario] = aggregated_ds.compute()

        # Save aggregated dataset to NetCDF file (if needed)
        # output_dir = "/nfs/sea/work/fpfaeffli/aggregated_sensitivities/"
        # os.makedirs(output_dir, exist_ok=True)
        # output_file = os.path.join(output_dir, f'aggregated_sensitivities_{scenario}_2011_2021.nc')
        # aggregated_ds.to_netcdf(output_file)
        # print(f"Saved aggregated dataset to {output_file}")

    return sens_dict

# Define scenarios
scenarios = ['present', 'ssp245', 'ssp585']

# Run the function
sens_dict = load_and_aggregate_sensitivities(scenarios)



#%%

#### For the sensitivities based on climatologies

def multiply_anom_with_sens(varia, sens_wrt_var, anoms, sens_dict):
    mult_dict = {}

    # Extract the actual scenarios from anoms (since config level is extra)
    anoms = anoms["romsoc_fully_coupled"]

    for scenario in scenarios:
        print(f'Processing {scenario}')

        mult_dict[scenario] = {}

        sensitivity = None
        if varia == 'temp' and sens_wrt_var == 'h':
            sensitivity = sens_dict[scenario].dh_dtem
        elif varia == 'salt' and sens_wrt_var == 'h':
            sensitivity = sens_dict[scenario].dh_dsal
        elif varia == 'DIC' and sens_wrt_var == 'h':
            sensitivity = sens_dict[scenario].dh_ddic
        elif varia == 'Alk' and sens_wrt_var == 'h':
            sensitivity = sens_dict[scenario].dh_dalk
        elif varia == 'temp' and sens_wrt_var == 'omegaa':
            sensitivity = sens_dict[scenario].domegaa_dtem
        elif varia == 'salt' and sens_wrt_var == 'omegaa':
            sensitivity = sens_dict[scenario].domegaa_dsal
        elif varia == 'DIC' and sens_wrt_var == 'omegaa':
            sensitivity = sens_dict[scenario].domegaa_ddic
        elif varia == 'Alk' and sens_wrt_var == 'omegaa':
            sensitivity = sens_dict[scenario].domegaa_dalk

        for scenario_that_anoms_are_relative_to in scenarios:
            print(f'  Multiplying anomalies for {scenario} relative to {scenario_that_anoms_are_relative_to}')
            mult_dict[scenario][scenario_that_anoms_are_relative_to] = (
                anoms[scenario][scenario_that_anoms_are_relative_to] * sensitivity
            )

    return mult_dict


dtem_times_dhdtem = multiply_anom_with_sens('temp', 'h', tem_anom, sens_dict)
dsal_times_dhdsal = multiply_anom_with_sens('salt', 'h', sal_anom, sens_dict)
ddic_times_dhddic = multiply_anom_with_sens('DIC', 'h', dic_anom, sens_dict)
dalk_times_dhdalk = multiply_anom_with_sens('Alk', 'h', alk_anom, sens_dict)

#%% now mask all the non-extreme values, using the present-day threshold

## For sensitivities based on climatologies

# %%
def apply_extreme_masks(mult_dict, mask_dict):
    applied_mask_dict = {}

    # Extract the actual scenarios from mask_dict (since config level is extra)
    mask_dict = mask_dict["romsoc_fully_coupled"]

    for scenario in scenarios:
        applied_mask_dict[scenario] = {}
        for scenario_that_anoms_are_relative_to in scenarios:
            applied_mask_dict[scenario][scenario_that_anoms_are_relative_to] = {}
            for threshold_type in ['present', 'present_plus_meandelta']:
                print(f"Applying mask for {scenario} relative to {scenario_that_anoms_are_relative_to} using {threshold_type}")
                applied_mask_dict[scenario][scenario_that_anoms_are_relative_to][threshold_type] = (
                    mult_dict[scenario][scenario_that_anoms_are_relative_to]
                    .where(mask_dict[scenario][threshold_type] == True, np.NaN)
                )

    return applied_mask_dict

dtem_times_dhdtem_masked = apply_extreme_masks(dtem_times_dhdtem, extreme_mask_dict)
dsal_times_dhdsal_masked = apply_extreme_masks(dsal_times_dhdsal, extreme_mask_dict)
ddic_times_dhddic_masked = apply_extreme_masks(ddic_times_dhddic, extreme_mask_dict)
dalk_times_dhdalk_masked = apply_extreme_masks(dalk_times_dhdalk, extreme_mask_dict)

#%%
###############################
#         PLOTTING MAPS       #
###############################


# Plots maps of the contributions of different drivers to [H+] changes for present and future scenarios.

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

# Function to plot contributions for Present vs. Future (Fixed & Moving Baseline)
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
    # Add a second smaller suptitle to the right upper corner
    fig.text(1.0, 1.08, f'ssp585', ha='right', fontsize=14)

    # Save and Show
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()


# Plotting for mean over time 

# Loop through each driver and plot for both present and future
for driver_name, driver_data in drivers.items():
    print(f"Plotting {driver_name} contributions...")

    # Extract present, future fixed baseline, and future moving baseline data (Using `ssp585` instead of `ssp245`)
    data_present_xr = driver_data["present"]["present"]["present"].squeeze()
    data_future_fixed_xr = driver_data["ssp585"]["present"]["present"].squeeze()
    data_future_moving_xr = driver_data["ssp585"]["ssp585"]["present_plus_meandelta"].squeeze()

    # Compute the mean over time for the entire period
    data_present_mean = data_present_xr.mean(dim="time")
    data_future_fixed_mean = data_future_fixed_xr.mean(dim="time")
    data_future_moving_mean = data_future_moving_xr.mean(dim="time")

    # **Fixed color scale limits**
    vmin_driver, vmax_driver = -3e-9, 3e-9  # **Use fixed limits instead of dynamic computation**

    # Define file name
    filename = f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/contributions_maps/contributions_clim_based_{driver_name.lower()}_present_vs_future_fixed_vs_moving_ssp585_fixed_colorbar_ssp585.png"

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
# Plots the fraction of the domain experiencing extreme events over time.

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


# Define file paths
base_path = "/nfs/sea/work/fpfaeffli/plots/extreme_masks/"
hplus_filename = f"{base_path}hplus_extreme_mask_fraction_{config}_{scenario}.png"

# Call the function to plot the extreme mask fraction for H+
plot_extreme_mask_fraction(ocean_ds, extreme_mask_dict, "H+", filename=hplus_filename)


# %%
# Plots the extreme map (extreme vs non-extreme)


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


# Define spatial bounds (adjust based on your data)
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

# Define file paths
base_path = "/nfs/sea/work/fpfaeffli/plots/extreme_masks/"
hplus_map_filename = f"{base_path}hplus_extreme_mask_map_{config}_{scenario}.png"

# Call the function to plot the spatial distribution of extreme events for H+
plot_extreme_mask_map(ocean_ds, extreme_mask_dict, landmask_etopo, bounds, "H+", filename=hplus_map_filename)



#%%
###############################
#     PLOTTING TIMESERIES     #
###############################
# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

def plot_weighted_driver_timeseries(drivers, model_area):
    """
    Plots time series of the contributions of multiple drivers to [H+] changes using weighted averaging.
    Includes present-day scenario and loops over both future scenarios (ssp245, ssp585).
    
    Parameters:
        drivers (dict): Dictionary containing driver datasets.
        model_area (xarray.DataArray): Area weight data for weighted averaging.
    """

    # Define colors for each driver
    driver_colors = {
        "Temperature": "red",
        "Salinity": "blue",
        "DIC": "green",
        "Alkalinity": "purple"
    }

    # Create figure with five subplots (Present + Fixed & Moving for `ssp245` and `ssp585`)
    fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

    # Loop through drivers (Temperature, Salinity, DIC, Alkalinity)
    for driver_name, driver_data in drivers.items():

        time_series_present_mean = []
        time_series_fixed_ssp245_mean = []
        time_series_fixed_ssp585_mean = []
        time_series_moving_ssp245_mean = []
        time_series_moving_ssp585_mean = []

        # Extract **present-day scenario**
        present_data = driver_data["present"]["present"]["present"]
        if present_data is not None:
            time_series_present = present_data.weighted(model_area.fillna(0)).mean(dim=["eta_rho", "xi_rho"])
            time_series_present_mean.append(time_series_present)

        # Extract **future scenarios (ssp245 & ssp585)**
        for scenario in ["ssp245", "ssp585"]:
            if scenario not in driver_data:
                continue

            fixed_data = driver_data[scenario]["present"]["present"]
            moving_data = driver_data[scenario]["present"]["present_plus_meandelta"]

            if fixed_data is None or moving_data is None:
                continue

            # Compute weighted mean over model grid dimensions
            time_series_fixed = fixed_data.weighted(model_area.fillna(0)).mean(dim=["eta_rho", "xi_rho"])
            time_series_moving = moving_data.weighted(model_area.fillna(0)).mean(dim=["eta_rho", "xi_rho"])

            if scenario == "ssp245":
                time_series_fixed_ssp245_mean.append(time_series_fixed)
                time_series_moving_ssp245_mean.append(time_series_moving)
            elif scenario == "ssp585":
                time_series_fixed_ssp585_mean.append(time_series_fixed)
                time_series_moving_ssp585_mean.append(time_series_moving)

        if not time_series_present_mean or not time_series_fixed_ssp245_mean or not time_series_fixed_ssp585_mean:
            continue

        # Convert lists to xarray and average over scenarios
        time_series_present_mean = sum(time_series_present_mean) / len(time_series_present_mean)
        time_series_fixed_ssp245_mean = sum(time_series_fixed_ssp245_mean) / len(time_series_fixed_ssp245_mean)
        time_series_fixed_ssp585_mean = sum(time_series_fixed_ssp585_mean) / len(time_series_fixed_ssp585_mean)
        time_series_moving_ssp245_mean = sum(time_series_moving_ssp245_mean) / len(time_series_moving_ssp245_mean)
        time_series_moving_ssp585_mean = sum(time_series_moving_ssp585_mean) / len(time_series_moving_ssp585_mean)

        # Plot present scenario in the first panel
        axs[0].plot(time_series_present_mean.time, time_series_present_mean, label=driver_name, color=driver_colors[driver_name], zorder=2)

        # Plot fixed baseline contributions
        axs[1].plot(time_series_fixed_ssp245_mean.time, time_series_fixed_ssp245_mean, label=driver_name, color=driver_colors[driver_name], zorder=2)
        axs[2].plot(time_series_fixed_ssp585_mean.time, time_series_fixed_ssp585_mean, label=driver_name, color=driver_colors[driver_name], zorder=2)

        # Plot moving baseline contributions
        axs[3].plot(time_series_moving_ssp245_mean.time, time_series_moving_ssp245_mean, label=driver_name, color=driver_colors[driver_name], zorder=2)
        axs[4].plot(time_series_moving_ssp585_mean.time, time_series_moving_ssp585_mean, label=driver_name, color=driver_colors[driver_name], zorder=2)

    # Formatting
    axs[0].set_title("Present-day mean contribution of different drivers to change in [H+]", fontsize=14)
    axs[1].set_title("Mean contribution of different drivers to change in [H+] during extremes (fixed baseline, ssp245)", fontsize=14)
    axs[2].set_title("Mean contribution of different drivers to change in [H+] during extremes (fixed baseline, ssp585)", fontsize=14)
    axs[3].set_title("Mean contribution of different drivers to change in [H+] during extremes (moving baseline, ssp245)", fontsize=14)
    axs[4].set_title("Mean contribution of different drivers to change in [H+] during extremes (moving baseline, ssp585)", fontsize=14)

    axs[4].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("Change in [H+]", fontsize=12)
    axs[1].set_ylabel("Change in [H+]", fontsize=12)
    axs[2].set_ylabel("Change in [H+]", fontsize=12)
    axs[3].set_ylabel("Change in [H+]", fontsize=12)
    axs[4].set_ylabel("Change in [H+]", fontsize=12)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Add legends
    for ax in axs:
        ax.legend()

    # Add grid for better readability
    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save the plot (optional)
    filename = "/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/contributions_all_drivers_timeseries_weighted.png"
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')

    # Show plot
    plt.show()


# Example Usage:
# Load your dataset and area weighting data first
# model_area = xr.open_dataarray("your_area_weights.nc")

# Call function with drivers and area weighting
plot_weighted_driver_timeseries(
    drivers={
        "Temperature": dtem_times_dhdtem_masked,
        "Salinity": dsal_times_dhdsal_masked,
        "DIC": ddic_times_dhddic_masked,
        "Alkalinity": dalk_times_dhdalk_masked
    },
    model_area=model_area
)


















#%%
###############################
#        OLD TIMESERIES       #
###############################


#%%

# Plots time series of the contributions of multiple drivers to [H+] changes for present and future scenarios.

def plot_driver_timeseries(drivers):
    """
    Plots time series of the contributions of multiple drivers to [H+] changes.
    Includes present-day scenario and loops over both future scenarios (ssp245, ssp585).
    """

    # Define colors for each driver
    driver_colors = {
        "Temperature": "red",
        "Salinity": "blue",
        "DIC": "green",
        "Alkalinity": "purple"
    }

    # Create figure with five subplots (Present + Fixed & Moving for `ssp245` and `ssp585`)
    fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

    # Loop through drivers (Temperature, Salinity, DIC, Alkalinity)
    for driver_name, driver_data in drivers.items():

        time_series_present_mean = []
        time_series_fixed_ssp245_mean = []
        time_series_fixed_ssp585_mean = []
        time_series_moving_ssp245_mean = []
        time_series_moving_ssp585_mean = []

        # Extract **present-day scenario**
        present_data = driver_data["present"]["present"]["present"]
        if present_data is not None:
            time_series_present = present_data.mean(dim=["eta_rho", "xi_rho"], skipna=True).squeeze()
            time_series_present_mean.append(time_series_present)

        # Extract **future scenarios (ssp245 & ssp585)**
        for scenario in ["ssp245", "ssp585"]:
            if scenario not in driver_data:
                continue

            fixed_data = driver_data[scenario]["present"]["present"]
            moving_data = driver_data[scenario]["present"]["present_plus_meandelta"]

            if fixed_data is None or moving_data is None:
                continue

            # Compute mean over model grid dimensions
            time_series_fixed = fixed_data.mean(dim=["eta_rho", "xi_rho"], skipna=True).squeeze()
            time_series_moving = moving_data.mean(dim=["eta_rho", "xi_rho"], skipna=True).squeeze()

            if scenario == "ssp245":
                time_series_fixed_ssp245_mean.append(time_series_fixed)
                time_series_moving_ssp245_mean.append(time_series_moving)
            elif scenario == "ssp585":
                time_series_fixed_ssp585_mean.append(time_series_fixed)
                time_series_moving_ssp585_mean.append(time_series_moving)

        if not time_series_present_mean or not time_series_fixed_ssp245_mean or not time_series_fixed_ssp585_mean:
            continue

        # Convert lists to xarray and average over scenarios
        time_series_present_mean = sum(time_series_present_mean) / len(time_series_present_mean)
        time_series_fixed_ssp245_mean = sum(time_series_fixed_ssp245_mean) / len(time_series_fixed_ssp245_mean)
        time_series_fixed_ssp585_mean = sum(time_series_fixed_ssp585_mean) / len(time_series_fixed_ssp585_mean)
        time_series_moving_ssp245_mean = sum(time_series_moving_ssp245_mean) / len(time_series_moving_ssp245_mean)
        time_series_moving_ssp585_mean = sum(time_series_moving_ssp585_mean) / len(time_series_moving_ssp585_mean)

        # Plot present scenario in the first panel
        axs[0].plot(time_series_present_mean.time, time_series_present_mean, label=driver_name, color=driver_colors[driver_name], zorder=2)

        # Plot fixed baseline contributions
        axs[1].plot(time_series_fixed_ssp245_mean.time, time_series_fixed_ssp245_mean, label=driver_name, color=driver_colors[driver_name], zorder=2)
        axs[2].plot(time_series_fixed_ssp585_mean.time, time_series_fixed_ssp585_mean, label=driver_name, color=driver_colors[driver_name], zorder=2)

        # Plot moving baseline contributions
        axs[3].plot(time_series_moving_ssp245_mean.time, time_series_moving_ssp245_mean, label=driver_name, color=driver_colors[driver_name], zorder=2)
        axs[4].plot(time_series_moving_ssp585_mean.time, time_series_moving_ssp585_mean, label=driver_name, color=driver_colors[driver_name], zorder=2)

    # Formatting
    axs[0].set_title("Present-day mean contribution of different drivers to change in [H+]", fontsize=14)
    axs[1].set_title("Mean contribution of different drivers to change in [H+] during extremes (fixed baseline, ssp245)", fontsize=14)
    axs[2].set_title("Mean contribution of different drivers to change in [H+] during extremes (fixed baseline, ssp585)", fontsize=14)
    axs[3].set_title("Mean contribution of different drivers to change in [H+] during extremes (moving baseline, ssp245)", fontsize=14)
    axs[4].set_title("Mean contribution of different drivers to change in [H+] during extremes (moving baseline, ssp585)", fontsize=14)

    axs[4].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("Change in [H+]", fontsize=12)
    axs[1].set_ylabel("Change in [H+]", fontsize=12)
    axs[2].set_ylabel("Change in [H+]", fontsize=12)
    axs[3].set_ylabel("Change in [H+]", fontsize=12)
    axs[4].set_ylabel("Change in [H+]", fontsize=12)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Add legends
    for ax in axs:
        ax.legend()

    # Add grid for better readability
    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save the plot (optional)
    filename = "/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/contributions_all_drivers_timeseries_full.png"
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')

    # Show plot
    plt.show()


# Call the function with corrected structure
plot_driver_timeseries(
    drivers={
        "Temperature": dtem_times_dhdtem_masked,
        "Salinity": dsal_times_dhdsal_masked,
        "DIC": ddic_times_dhddic_masked,
        "Alkalinity": dalk_times_dhdalk_masked
    }
)

#%%
# Plots timeseries of each driver seperately with present vs future (both scenarios, and moving vs fixed baseline)


def plot_driver_comparison(drivers):
    """
    Creates a separate plot for each driver comparing present-day, 
    fixed baseline (ssp245 & ssp585), and moving baseline (ssp245 & ssp585).
    """

    # Define colors for scenarios
    scenario_colors = {
        "present": "black",
        "ssp245_fixed": "blue",
        "ssp585_fixed": "green",
        "ssp245_moving": "red",
        "ssp585_moving": "purple"
    }

    # Loop through drivers (Temperature, Salinity, DIC, Alkalinity) and create separate figures
    for driver_name, driver_data in drivers.items():

        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        time_series_present_mean = []
        time_series_fixed_ssp245_mean = []
        time_series_fixed_ssp585_mean = []
        time_series_moving_ssp245_mean = []
        time_series_moving_ssp585_mean = []

        # Extract **present-day scenario**
        present_data = driver_data["present"]["present"]["present"]
        if present_data is not None:
            time_series_present = present_data.mean(dim=["eta_rho", "xi_rho"], skipna=True).squeeze()
            time_series_present_mean.append(time_series_present)

        # Extract **future scenarios (ssp245 & ssp585)**
        for scenario in ["ssp245", "ssp585"]:
            if scenario not in driver_data:
                continue

            fixed_data = driver_data[scenario]["present"]["present"]
            moving_data = driver_data[scenario]["present"]["present_plus_meandelta"]

            if fixed_data is None or moving_data is None:
                continue

            # Compute mean over model grid dimensions
            time_series_fixed = fixed_data.mean(dim=["eta_rho", "xi_rho"], skipna=True).squeeze()
            time_series_moving = moving_data.mean(dim=["eta_rho", "xi_rho"], skipna=True).squeeze()

            if scenario == "ssp245":
                time_series_fixed_ssp245_mean.append(time_series_fixed)
                time_series_moving_ssp245_mean.append(time_series_moving)
            elif scenario == "ssp585":
                time_series_fixed_ssp585_mean.append(time_series_fixed)
                time_series_moving_ssp585_mean.append(time_series_moving)

        if not time_series_present_mean or not time_series_fixed_ssp245_mean or not time_series_fixed_ssp585_mean:
            continue

        # Convert lists to xarray and average over scenarios
        time_series_present_mean = sum(time_series_present_mean) / len(time_series_present_mean)
        time_series_fixed_ssp245_mean = sum(time_series_fixed_ssp245_mean) / len(time_series_fixed_ssp245_mean)
        time_series_fixed_ssp585_mean = sum(time_series_fixed_ssp585_mean) / len(time_series_fixed_ssp585_mean)
        time_series_moving_ssp245_mean = sum(time_series_moving_ssp245_mean) / len(time_series_moving_ssp245_mean)
        time_series_moving_ssp585_mean = sum(time_series_moving_ssp585_mean) / len(time_series_moving_ssp585_mean)

        # Panel 1: Present-day scenario
        axs[0].plot(time_series_present_mean.time, time_series_present_mean, label="Present", color=scenario_colors["present"], zorder=2)

        # Panel 2: Fixed Baseline for SSP245 and SSP585
        axs[1].plot(time_series_fixed_ssp245_mean.time, time_series_fixed_ssp245_mean, label="Fixed Baseline SSP245", color=scenario_colors["ssp245_fixed"], zorder=2)
        axs[1].plot(time_series_fixed_ssp585_mean.time, time_series_fixed_ssp585_mean, label="Fixed Baseline SSP585", color=scenario_colors["ssp585_fixed"], zorder=2)

        # Panel 3: Moving Baseline for SSP245 and SSP585
        axs[2].plot(time_series_moving_ssp245_mean.time, time_series_moving_ssp245_mean, label="Moving Baseline SSP245", color=scenario_colors["ssp245_moving"], zorder=2)
        axs[2].plot(time_series_moving_ssp585_mean.time, time_series_moving_ssp585_mean, label="Moving Baseline SSP585", color=scenario_colors["ssp585_moving"], zorder=2)

        # Formatting
        axs[0].set_title(f"{driver_name}: Present-day Mean Contribution to [H+]", fontsize=14)
        axs[1].set_title(f"{driver_name}: Fixed Baseline Contribution to [H+]", fontsize=14)
        axs[2].set_title(f"{driver_name}: Moving Baseline Contribution to [H+]", fontsize=14)

        axs[2].set_xlabel("Time", fontsize=12)
        axs[0].set_ylabel("Change in [H+]", fontsize=12)
        axs[1].set_ylabel("Change in [H+]", fontsize=12)
        axs[2].set_ylabel("Change in [H+]", fontsize=12)

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        # Add legends
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        # Add grid for better readability
        for ax in axs:
            ax.grid(True, linestyle="--", alpha=0.5)

        # Adjust layout
        plt.tight_layout()

        # Save the plot (optional)
        filename = f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/contributions_{driver_name}_timeseries.png"
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')

        # Show plot
        plt.show()


# Call the function with corrected structure
plot_driver_comparison(
    drivers={
        "Temperature": dtem_times_dhdtem_masked,
        "Salinity": dsal_times_dhdsal_masked,
        "DIC": ddic_times_dhddic_masked,
        "Alkalinity": dalk_times_dhdalk_masked
    }
)

