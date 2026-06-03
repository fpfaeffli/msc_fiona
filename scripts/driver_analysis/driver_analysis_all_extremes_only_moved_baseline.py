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
            elif varia == 'DIC':
                vdict[config][scenario] = ocean_ds[config][scenario].DIC.load()
            elif varia == 'Alk':
                vdict[config][scenario] = ocean_ds[config][scenario].Alk.load()
    return vdict

hplus = load_data_in('Hplus')
#tem = load_data_in('temp')
#sal = load_data_in('salt')
dic = load_data_in('DIC')
#alk = load_data_in('Alk')

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
#tem_clim = load_clim_in('temp')
dic_clim = load_clim_in('DIC')
#alk_clim = load_clim_in('Alk')

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
            for threshold_type in ['present_plus_meandelta']:
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
#tem_anom   = calc_anomalies_rel_to_clim(tem,tem_clim)
dic_anom   = calc_anomalies_rel_to_clim(dic,dic_clim)
#alk_anom   = calc_anomalies_rel_to_clim(alk,alk_clim)


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


#dtem_times_dhdtem = multiply_anom_with_sens('temp', 'h', tem_anom, sens_dict)
ddic_times_dhddic = multiply_anom_with_sens('DIC', 'h', dic_anom, sens_dict)
#dalk_times_dhdalk = multiply_anom_with_sens('Alk', 'h', alk_anom, sens_dict)

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

#dtem_times_dhdtem_masked = apply_extreme_masks(dtem_times_dhdtem, extreme_mask_dict)
#dsal_times_dhdsal_masked = apply_extreme_masks(dsal_times_dhdsal, extreme_mask_dict)
ddic_times_dhddic_masked = apply_extreme_masks(ddic_times_dhddic, extreme_mask_dict)
#dalk_times_dhdalk_masked = apply_extreme_masks(dalk_times_dhdalk, extreme_mask_dict)

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


