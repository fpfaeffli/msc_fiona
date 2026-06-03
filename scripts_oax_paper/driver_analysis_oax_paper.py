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

import gc

from importlib import reload  # Python 3.4+

import get_obs_datasets 
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter

from func_for_clim_thresh import ThreshClimFuncs
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs

from get_auxiliary_data import AuxGetter as AuxGetter

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
model_regions_dict = GetRegions.define_CalCS_regions(model_area.lon, model_area.lat, model_d2coasts['roms_only'])


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
#sal = load_data_in('salt')
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
#sal_clim = load_clim_in('salt')
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
#sal_anom   = calc_anomalies_rel_to_clim(sal,sal_clim)
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
#dsal_times_dhdsal = multiply_anom_with_sens('salt', 'h', sal_anom, sens_dict)
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
#dsal_times_dhdsal_masked = apply_extreme_masks(dsal_times_dhdsal, extreme_mask_dict)
ddic_times_dhddic_masked = apply_extreme_masks(ddic_times_dhddic, extreme_mask_dict)
dalk_times_dhdalk_masked = apply_extreme_masks(dalk_times_dhdalk, extreme_mask_dict)

#%%
# Define driver input dictionaries
driver_data = {
    "DIC": ddic_times_dhddic_masked,
    "temp": dtem_times_dhdtem_masked,
    "Alk": dalk_times_dhdalk_masked
}

# Extract reference lat/lon for all maps
lat = dtem_times_dhdtem["present"]["present"].lat
lon = dtem_times_dhdtem["present"]["present"].lon


# Initialize output container
driver_contrib_anoms = {
    "DIC": {},
    "temp": {},
    "Alk": {}
}

# Loop through all drivers and both scenarios
for driver_name, masked_dict in driver_data.items():
    driver_contrib_anoms[driver_name] = {}

    for scenario in ["ssp245", "ssp585"]:
        # Extract raw arrays
        present_raw = masked_dict["present"]["present"]["present"]
        future_fixed_raw = masked_dict[scenario]["present"]["present"]
        future_shifted_raw = masked_dict[scenario][scenario]["present_plus_meandelta"]

        # Subtract values to get anomalies (ignore coordinates)
        dims = present_raw.dims
        anom_fixed = xr.DataArray(future_fixed_raw.values - present_raw.values, dims=dims)
        anom_shifted = xr.DataArray(future_shifted_raw.values - present_raw.values, dims=dims)

        # Time mean and squeeze to 2D
        fixed_mean = anom_fixed.mean(dim="time", skipna=True).squeeze()
        shifted_mean = anom_shifted.mean(dim="time", skipna=True).squeeze()

        # Attach lat/lon coordinates properly
        fixed_mean.coords['lat'] = (('eta_rho', 'xi_rho'), lat.data)
        fixed_mean.coords['lon'] = (('eta_rho', 'xi_rho'), lon.data)
        shifted_mean.coords['lat'] = (('eta_rho', 'xi_rho'), lat.data)
        shifted_mean.coords['lon'] = (('eta_rho', 'xi_rho'), lon.data)

        # Store results
        driver_contrib_anoms[driver_name][scenario] = {
            "fixed": fixed_mean,
            "shifted": shifted_mean
        }



#%%
# Delete sensitivities dictionary 

del sens_dict  # remove reference
gc.collect()   # force garbage collection to free memory



#%%
###############################
#         PLOTTING MAPS       #
###############################

#%%
# Plots maps of the contributions of different drivers to [H+] changes for present and future scenarios.

# Load landmask
landmask_etopo = AuxGetter.get_etopo_data()

# Define bounds for the California Current System (CaCS)
lat_bounds = (29, 48)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

# Define the drivers and their respective datasets
drivers = {
    "temp": dtem_times_dhdtem_masked,
    "Alk": dalk_times_dhdalk_masked,
    "DIC": ddic_times_dhddic_masked
    
}

# Function to plot contributions for Present vs. Future (Fixed & Moving Baseline)
def plot_contributions_present_future(data_present, data_future_fixed, data_future_moving, landmask_etopo, bounds, vmin, vmax, title, filename, cmap="cmo.balance"):
    lat_bounds, lon_bounds = bounds

    fig, axs = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
    levels = np.linspace(vmin, vmax, 40)

    for idx, (data, ax, period) in enumerate(zip(
        [data_present, data_future_fixed, data_future_moving], 
        axs, 
        ["present", "future (fixed baseline)", "future (shifted baseline)"]
    )):
        # Add landmask
        ax.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, levels=1, colors='black', zorder=1)

        # Plot data
        contour = ax.contourf(data["lon"], data["lat"], data, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_title(period, fontsize=20)
        ax.set_xlabel("Longitude", fontsize=18)
        if idx == 0:
            ax.set_ylabel("Latitude", fontsize=18)

        # Add offshore vs. coastal boundaries as contour lines
        downsample_factor = 1  # Optional for cleaner contour lines
        region_contour_colors = {
            'offshore_central': 'saddlebrown',
            'coastal_central': 'coral'
        }

        # Add offshore vs. coastal boundaries as contour lines
        downsample_factor = 1  # Optional for cleaner contour lines
        region_contour_colors = {
            'offshore_central': 'saddlebrown',
            'coastal_central': 'coral'
        }

        for region in region_contour_colors:
            region_mask_da = model_regions_dict[region]['mask']
            region_mask_dummy = xr.where(np.isnan(region_mask_da), 0, 1)

            ax.contour(
                region_mask_da.lon[::downsample_factor, ::downsample_factor],
                region_mask_da.lat[::downsample_factor, ::downsample_factor],
                region_mask_dummy[::downsample_factor, ::downsample_factor],
                levels=[0.5],
                colors=region_contour_colors[region],
                linewidths=1,
                zorder=3
            )



        # Add degree symbols and N/W indicators, and make ticks bigger
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks], fontsize=16)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks], fontsize=16)

    # Add colorbar
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label="Change in [H+]", ticks=np.linspace(vmin, vmax, 11))

    # Add title
    plt.suptitle(title, fontsize=20, y=1.08)
    # Add a second smaller suptitle to the right upper corner
    fig.text(1.0, 1.08, f'ssp245', ha='right', fontsize=14)

    # Save and Show
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()


# Plotting for mean over time 

# Loop through each driver and plot for both present and future
for driver_name, driver_data in drivers.items():
    print(f"Plotting {driver_name} contributions...")

    # Extract present, future fixed baseline, and future moving baseline data (Using `ssp245` instead of `ssp585`)
    data_present_xr = driver_data["present"]["present"]["present"].squeeze()
    data_future_fixed_xr = driver_data["ssp245"]["present"]["present"].squeeze()
    data_future_moving_xr = driver_data["ssp245"]["ssp245"]["present_plus_meandelta"].squeeze()

    # Compute the mean over time for the entire period
    data_present_mean = data_present_xr.mean(dim="time")
    data_future_fixed_mean = data_future_fixed_xr.mean(dim="time")
    data_future_moving_mean = data_future_moving_xr.mean(dim="time")

    # **Fixed color scale limits**
    vmin_driver, vmax_driver = -3e-9, 3e-9 # **Use fixed limits instead of dynamic computation**

    # Define file name
    filename = f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/paper_plots/aa_finished_{driver_name.lower()}_contributions_present_vs_future_fixed_vs_shifted_ssp245.png"

    # Call the plotting function
    plot_contributions_present_future(
        data_present_mean,
        data_future_fixed_mean,
        data_future_moving_mean,
        landmask_etopo,
        bounds,
        vmin_driver,
        vmax_driver,
        f"Comparison of mean {driver_name} contribution to H+ changes (present vs. future under fixed & shifted baseline)",
        filename
    )

#%%
# Plots all drivers for each baseline in a 3x3 figure format for the paper


def plot_drivers_3x3(drivers_dict, landmask_etopo, model_regions_dict,
                     baseline="fixed", vmin=-3e-9, vmax=3e-9):
    """
    baseline: "fixed"  → fixed baseline
              "moving" → shifted baseline
    """
    drivers_order = ["DIC", "temp", "Alk"]
    drivers_label = {"DIC":"DIC", "temp":"Temperature", "Alk":"Alkalinity"}
    scenarios     = ["present", "ssp245", "ssp585"]
    title_colors  = {"present":"#f08c3a","ssp245":"#65a342","ssp585":"#9553e0"}
    lat_bounds    = [32, 46]
    lon_bounds    = [225, 245]

    def get_mean(driver, scn, moving):
        if scn == "present":
            arr = drivers_dict[driver]["present"]["present"]["present"].squeeze()
        else:
            if not moving:
                arr = drivers_dict[driver][scn]["present"]["present"].squeeze()
            else:
                arr = drivers_dict[driver][scn][scn]["present_plus_meandelta"].squeeze()
        return arr.mean(dim="time")

    moving_flag = (baseline == "moving")
    data_grid = [
        [get_mean(drv, scn, moving_flag) for scn in scenarios]
        for drv in drivers_order
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    levels = np.linspace(vmin, vmax, 40)

    for i, drv in enumerate(drivers_order):
        for j, scn in enumerate(scenarios):
            ax = axes[i, j]
            da = data_grid[i][j]

            cf = ax.contourf(
                da.lon, da.lat, da,
                levels=levels, cmap="cmo.balance",
                vmin=vmin, vmax=vmax, extend="both", zorder=1
            )
            ax.contourf(
                landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
                levels=[0,1], colors=["black"], zorder=3
            )
            for region, colr in {
                "offshore_central":"black",
                "coastal_central":"black"
            }.items():
                mask2d = model_regions_dict[region]["mask"]
                mp = xr.where(np.isnan(mask2d), 0, 1)
                ax.contour(
                    mask2d.lon, mask2d.lat, mp,
                    levels=[0.5], colors=[colr], linewidths=1, zorder=4
                )

            ax.set_title(scn, fontsize=16, color=title_colors[scn])
            ax.set_xlim(lon_bounds); ax.set_ylim(lat_bounds)
            ax.set_xlabel("Longitude", fontsize=13)
            # remove per‐subplot ylabel
            if j == 0:
                ax.set_ylabel("Latitude", fontsize=13)  

            yt = np.arange(lat_bounds[0], lat_bounds[1]+1, 3)
            xt = np.arange(lon_bounds[0], lon_bounds[1]+1, 5)
            ax.set_yticks(yt); ax.set_xticks(xt)
            ax.set_yticklabels([f"{y}°N" for y in yt])
            ax.set_xticklabels([f"{360 - x}°W" for x in xt])


    # thinner horizontal colorbar
    cax = fig.add_axes([0.25, -0.02, 0.5, 0.01])
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label("Change in [H⁺] (nmol L$^{-1}$)", fontsize=14)
    cbar.set_ticks(np.linspace(vmin, vmax, 7))
    cbar.ax.tick_params(labelsize=12)

    blabel = "shifted baseline" if baseline=="fixed" else "shifted baseline"

    out = f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/3x3_{baseline}_baseline.png"
    plt.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# Call for both panels:
plot_drivers_3x3(drivers, landmask_etopo, model_regions_dict, baseline="fixed")
plot_drivers_3x3(drivers, landmask_etopo, model_regions_dict, baseline="moving")


#%%
# Plotting for anomalies (difference between present and future contributions)

# Load landmask
landmask_etopo = AuxGetter.get_etopo_data()

# Define bounds for the California Current System (CaCS)
bounds = ((29, 48), (225, 245))  # lat_bounds, lon_bounds


def plot_contribution_anomalies(future_fixed, future_shifted, landmask_etopo, bounds, vmin, vmax, title, filename, cmap="cmo.balance"):
    lat_bounds, lon_bounds = bounds
    bounds = ((29, 48), (225, 245))  # lat_bounds, lon_bounds
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    levels = np.linspace(vmin, vmax, 40)

    for idx, (data, ax, label) in enumerate(zip(
        [future_fixed, future_shifted],
        axs,
        ["future – present (fixed baseline)", "future – present (shifted baseline)"]
    )):
        # Plot landmask
        ax.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, levels=1, colors='black', zorder=1)

        # Plot anomaly
        contour = ax.contourf(
            data.lon if 'lon' in data.coords else data.coords['xi_rho'],
            data.lat if 'lat' in data.coords else data.coords['eta_rho'],
            data,
            levels=levels,
            cmap=cmap,
            extend='both',
            zorder=2
        )

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_title(label, fontsize=18)
        ax.set_xlabel("Longitude", fontsize=16)
        if idx == 0:
            ax.set_ylabel("Latitude", fontsize=16)

        # Add region boundaries
        for region in ['coastal_all_lats', 'offshore_all_lats']:
            region_mask_da = model_regions_dict[region]['mask']
            region_mask_dummy = xr.where(np.isnan(region_mask_da), 0, 1)

            ax.contour(
                region_mask_da.lon,
                region_mask_da.lat,
                region_mask_dummy,
                levels=[0.5],
                colors='coral' if region == 'coastal_all_lats' else 'saddlebrown',
                linewidths=2,
                zorder=3
            )

        # Axis ticks
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks], fontsize=14)
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks], fontsize=14)

    # Add colorbar
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label="Δ DIC contribution to [H⁺]")

    plt.suptitle(title, fontsize=20, y=1.05)
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()

# plot_contribution_anomalies(
#     driver_contrib_anoms["DIC"]["ssp585"]["fixed"],
#     driver_contrib_anoms["DIC"]["ssp585"]["shifted"],
#     landmask_etopo,
#     bounds=bounds,
#     vmin=-2e-8,
#     vmax=2e-8,
#     title="ΔDIC contribution to [H⁺] during extreme events (ssp585 – present)",
#     filename="/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/paper_plots/ddic_anomaly_ssp585_2e-8.png"
# )

plot_contribution_anomalies(
    driver_contrib_anoms["temp"]["ssp585"]["fixed"],
    driver_contrib_anoms["temp"]["ssp585"]["shifted"],
    landmask_etopo,
    bounds=bounds,
    vmin=-5e-9,
    vmax=5e-9,
    title="ΔTemperature contribution to [H⁺] during extreme events (ssp585 – present)",
    filename="/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/paper_plots/dtemp_anomaly_ssp585_5e-9.png"
)

plot_contribution_anomalies(
    driver_contrib_anoms["Alk"]["ssp585"]["fixed"],
    driver_contrib_anoms["Alk"]["ssp585"]["shifted"],
    landmask_etopo,
    bounds=bounds,
    vmin=-5e-9,
    vmax=5e-9,
    title="ΔAlkalinity contribution to [H⁺] during extreme events (ssp585 – present)",
    filename="/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/paper_plots/dalk_anomaly_ssp585_5e-9.png"
)






# %%
################################
#         TIMESERIES           #
################################

#%%


# 1) Gather your drivers and the masked‐by‐extremes data:
driver_masks = {
    "Temperature":    dtem_times_dhdtem_masked,
    "DIC":            ddic_times_dhddic_masked,
    "Alkalinity":     dalk_times_dhdalk_masked
}

# 2) Define the regions you want:
regions = ["offshore_central", "coastal_central"]

# 3) Loop over drivers, scenarios, baselines, and regions to compute mean time series:
timeseries = {}  # timeseries[driver][scenario][baseline][region] = pandas.Series

for drv_name, drv_dict in driver_masks.items():
    timeseries[drv_name] = {}
    for scenario in ["present", "ssp245", "ssp585"]:
        timeseries[drv_name][scenario] = {}
        for baseline in ["present", "present_plus_meandelta"]:
            # pick the DataArray for this driver/scenario/baseline
            da = drv_dict[scenario][scenario][baseline] if scenario != "present" else drv_dict["present"]["present"][baseline]
            # da dims should be (time, eta_rho, xi_rho)
            for region in regions:
                mask2d = model_regions_dict[region]["mask"]  # 2D land/ocean mask
                # apply region mask (True over region, NaN elsewhere)
                region_da = da.where(mask2d == 1)
                # spatial average over grid
                ts = region_da.mean(dim=["eta_rho","xi_rho"], skipna=True)
                # store as pandas.Series
                timeseries[drv_name][scenario].setdefault(baseline, {})[region] = ts.to_pandas()

# 4) Plotting
for region in regions:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, drv_name in zip(axes, driver_masks):
        for scenario, color in zip(["present","ssp245","ssp585"], ["tab:gray","tab:green","tab:purple"]):
            for baseline, ls in zip(["present","present_plus_meandelta"], ["-", "--"]):
                series = timeseries[drv_name][scenario][baseline][region]
                label = f"{scenario} [{ 'fixed' if baseline=='present' else 'shifted' }]"
                ax.plot(series.index, series.values, label=label, linestyle=ls, color=color)
        ax.set_title(f"{drv_name} contribution in {region.replace('_',' ')}")
        ax.set_ylabel("Δ[H⁺] (nmol L⁻¹)")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    plt.show()

# %%

def plot_daily_mean_drivers(drivers, model_regions_dict):
    """
    For each scenario (present, ssp245, ssp585) and each region (offshore_central,
    coastal_central), compute the daily-mean regional average of each driver and plot them.
    """
    scenarios = ["present", "ssp245", "ssp585"]
    regions   = ["offshore_central", "coastal_central"]
    driver_colors = {
        "Temperature": "firebrick",
        "DIC":         "royalblue",
        "Alkalinity":  "darkgreen"
    }

    for scenario in scenarios:
        for region in regions:
            # 1) build the 2D mask as a NumPy boolean array
            mask2d = (model_regions_dict[region]["mask"].values == 1)

            fig, ax = plt.subplots(figsize=(12,4))

            for drv_name, drv_dict in drivers.items():
                # pick the correct DataArray
                if scenario == "present":
                    da = drv_dict["present"]["present"]["present"]
                else:
                    da = drv_dict[scenario][scenario]["present"]

                # broadcast 2D→3D if needed
                if "time" not in da.dims:
                    ref = drivers["Temperature"]["present"]["present"]["present"]
                    da = da.expand_dims(time=ref["time"])

                # ***** NEW: squeeze out any extra singleton dims (depth, z, etc.)
                da = da.squeeze(drop=True)

                # now da.dims should be exactly ("time","eta_rho","xi_rho")
                # pull out the raw values and time index
                data_np = da.values             # shape (T, η, ξ)
                times   = da["time"].to_index() # DatetimeIndex length T

                # spatial mask & mean
                regional_ts = np.nanmean(
                    np.where(mask2d, data_np, np.nan),
                    axis=(1,2)
                )  # now guaranteed 1D of length T

                # turn into pandas Series and daily‐mean
                ser = pd.Series(regional_ts, index=times)
                daily = ser.resample("1D").mean()

                # plot
                ax.plot(
                    daily.index, daily.values,
                    label=drv_name,
                    color=driver_colors[drv_name],
                    linewidth=1
                )

            ax.set_title(f"{region.replace('_',' ')} — {scenario}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Δ[H⁺] (nmol L⁻¹)")
            ax.grid(True, linestyle=":", alpha=0.5)
            # Move legend outside the box on the right
            ax.legend(title="Driver", loc="upper left", bbox_to_anchor=(1.02, 1))
            plt.tight_layout()
            plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/daily_mean_drivers_{region}_{scenario}.png", dpi=200, transparent=True)
            plt.show()


# Call it with your dicts:
plot_daily_mean_drivers(
    drivers={
        "Temperature": dtem_times_dhdtem_masked,
        "DIC":         ddic_times_dhddic_masked,
        "Alkalinity":  dalk_times_dhdalk_masked
    },
    model_regions_dict=model_regions_dict
)

# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_timeseries_at_points(drivers, driver_points):
    scenarios = ["present","ssp245","ssp585"]
    colors    = {"Temperature":"tab:orange","DIC":"tab:blue","Alkalinity":"tab:green"}
    styles    = {"offshore_central":"-","coastal_central":"--"}

    sample = drivers["Temperature"]["present"]["present"]["present"]
    lat2d, lon2d = sample.lat.values, sample.lon.values
    time_idx = sample["time"].to_index()

    # find i,j for each point
    point_idx = {}
    for name,(plat,plon) in driver_points.items():
        plon360 = plon if plon>=0 else plon+360
        dist2 = (lat2d-plat)**2 + (lon2d-plon360)**2
        point_idx[name] = np.unravel_index(np.nanargmin(dist2), dist2.shape)

    for scen in scenarios:
        fig, ax = plt.subplots(figsize=(12,4))
        for drv_name, drv_dict in drivers.items():
            for pt_name,(i,j) in point_idx.items():
                # pick baseline
                da = (drv_dict["present"]["present"]["present"]
                      if scen=="present"
                      else drv_dict[scen][scen]["present"])

                if "time" not in da.dims:
                    da = da.expand_dims(time=time_idx)
                da = da.squeeze(drop=True)

                # select point and drop non-time coords
                ts = da.isel(eta_rho=i, xi_rho=j).reset_coords(drop=True)

                # daily mean series
                ser   = pd.Series(ts.values, index=ts["time"].to_index())
                daily = ser.resample("1D").mean()

                ax.plot(
                    daily.index, daily.values,
                    color=colors[drv_name],
                    linestyle=styles[pt_name],
                    label=f"{drv_name} ({pt_name.replace('_',' ')})"
                )

        ax.set_title(f"Drivers at points — {scen}")
        ax.set_xlabel("Date"); ax.set_ylabel("Δ[H⁺]")
        ax.grid(True,linestyle=":",alpha=0.5)
        h,l = ax.get_legend_handles_labels()
        uniq = dict(zip(l,h))
        ax.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()



# Define your two points:
driver_points = {
    "offshore_central": (38.0, 233.0),
    "coastal_central":  (36.5, 236.5)
}

# Call the function:
plot_timeseries_at_points(
    drivers={
        "Temperature": dtem_times_dhdtem_masked,
        "DIC":         ddic_times_dhddic_masked,
        "Alkalinity":  dalk_times_dhdalk_masked
    },
    driver_points=driver_points
)


# %%

def plot_daily_mean_by_region_pair(drivers, model_regions_dict):
    """
    For each scenario, plot offshore and coastal side‐by‐side,
    showing daily‐mean regional contribution time series for each driver.
    All font sizes set to 14.
    """
    scenarios = ["present", "ssp245", "ssp585"]
    regions   = ["offshore_central", "coastal_central"]
    driver_colors = {
        "Temperature": "firebrick",
        "DIC":         "royalblue",
        "Alkalinity":  "darkgreen"
    }

    for scenario in scenarios:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)
        fig.suptitle(f"{scenario}", fontsize=14)

        for ax, region in zip(axes, regions):
            mask2d = model_regions_dict[region]["mask"].values == 1

            for drv_name, drv_dict in drivers.items():
                # pick the correct DataArray
                if scenario == "present":
                    da = drv_dict["present"]["present"]["present"]
                else:
                    da = drv_dict[scenario][scenario]["present"]

                if "time" not in da.dims:
                    ref = drivers["Temperature"]["present"]["present"]["present"]
                    da = da.expand_dims(time=ref["time"])
                da = da.squeeze(drop=True)

                data_np = da.values
                times   = da["time"].to_index()

                regional_ts = np.nanmean(
                    np.where(mask2d, data_np, np.nan),
                    axis=(1,2)
                )
                daily = pd.Series(regional_ts, index=times).resample("1D").mean()

                ax.plot(
                    daily.index, daily.values,
                    label=drv_name,
                    color=driver_colors[drv_name],
                    lw=1
                )

            # use just "Offshore" or "Coastal" as subplot titles
            short_label = "Offshore region" if region.startswith("offshore") else "Coastal region"
            ax.set_title(short_label, fontsize=14)

            ax.set_xlabel("Year", fontsize=14)
            if region == "offshore_central":
                if scenario == "present":
                    ax.set_ylabel("Δ[H⁺] (mol L⁻¹)", fontsize=14)
                else:
                    ax.set_ylabel("Δ[H⁺] (nmol L⁻¹)", fontsize=14)

            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True, linestyle=":", alpha=0.5)

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            title="Driver",
            title_fontsize=14,
            fontsize=14,
            loc="upper center", ncol=len(drivers),
            bbox_to_anchor=(0.5, 0.02)
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/timeseries_daily_mean_by_region_pair_{scenario}.png", dpi=200, transparent=True, bbox_inches='tight')
        plt.show()


# Usage:
plot_daily_mean_by_region_pair(
    drivers={
        "Temperature": dtem_times_dhdtem_masked,
        "DIC":         ddic_times_dhddic_masked,
        "Alkalinity":  dalk_times_dhdalk_masked
    },
    model_regions_dict=model_regions_dict
)

# %%


def plot_point_timeseries_by_region_pair(drivers, driver_points):
    """
    For each scenario, plot offshore and coastal side‐by‐side,
    showing daily‐mean point time series for each driver.
    All fonts and colors match the regional‐mean plot.
    """
    scenarios = ["present", "ssp245", "ssp585"]
    regions   = ["offshore_central", "coastal_central"]
    driver_colors = {
        "Temperature": "firebrick",
        "DIC":         "royalblue",
        "Alkalinity":  "darkgreen"
    }
    linestyles = {
        "offshore_central": "-",
        "coastal_central":  "--"
    }

    # grab a sample to get time axis
    sample = drivers["Temperature"]["present"]["present"]["present"]
    time_idx = sample["time"].to_index()

    # find nearest grid cell for each point
    lat2d, lon2d = sample.lat.values, sample.lon.values
    point_idx = {}
    for name,(plat,plon) in driver_points.items():
        plon360 = plon if plon >= 0 else plon + 360
        dist2 = (lat2d - plat)**2 + (lon2d - plon360)**2
        point_idx[name] = np.unravel_index(np.nanargmin(dist2), lat2d.shape)

    for scen in scenarios:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)
        fig.suptitle(f"{scen}", fontsize=14)

        for ax, region in zip(axes, regions):
            i, j = point_idx[region]

            for drv_name, drv_dict in drivers.items():
                # pick the correct DataArray
                if scen == "present":
                    da = drv_dict["present"]["present"]["present"]
                else:
                    da = drv_dict[scen][scen]["present"]

                # ensure time dim
                if "time" not in da.dims:
                    da = da.expand_dims(time=time_idx)
                da = da.squeeze(drop=True)

                # extract point timeseries
                arr = da.values
                ts  = arr[:, i, j] if arr.ndim == 3 else arr
                ser = pd.Series(ts, index=da["time"].to_index())
                daily = ser.resample("1D").mean()

                ax.plot(
                    daily.index, daily.values,
                    color=driver_colors[drv_name],
                    linestyle=linestyles[region],
                    linewidth=1,
                    label=drv_name
                )

            # subplot title and labels
            short_label = "Offshore (38.5°N, 233.0°E)" if region.startswith("offshore") else "Coastal (38.5°N, 236.5°E)"
            ax.set_title(short_label, fontsize=14)
            ax.set_xlabel("Year", fontsize=14)
            # Set y-label units: mol L⁻¹ for present, nmol L⁻¹ for ssp245/ssp585
            if region == "offshore_central":
                if scen == "present":
                    ax.set_ylabel("Δ[H⁺] (mol L⁻¹)", fontsize=14)
                else:
                    ax.set_ylabel("Δ[H⁺] (mol L⁻¹)", fontsize=14)

            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # shared legend at bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            title="Driver",
            title_fontsize=14,
            fontsize=14,
            loc="lower center",
            ncol=len(drivers),
            bbox_to_anchor=(0.5, -0.1)
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/timeseries_point_by_region_pair_{scen}.png", dpi=200, transparent=True, bbox_inches='tight')
        plt.show()


# Usage:
driver_points = {
    "offshore_central": (38.5, 233.0),
    "coastal_central":  (38.5, 236.5)
}

plot_point_timeseries_by_region_pair(
    drivers={
        "Temperature": dtem_times_dhdtem_masked,
        "DIC":         ddic_times_dhddic_masked,
        "Alkalinity":  dalk_times_dhdalk_masked
    },
    driver_points=driver_points
)

# %%
