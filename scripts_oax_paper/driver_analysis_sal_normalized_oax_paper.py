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
# Salinity-normalized contributions 

def multiply_anom_with_sens_salnorm(tem_anom, sal_anom, dic_anom, alk_anom,
                                    dic, alk, sal, sens_dict,
                                    configs, scenarios):
    S0 = 35.0
    mult_dict = {}

    for scen in scenarios:
        print(f"Processing {scen}")
        sens = sens_dict[scen]
        dHdT  = sens.dh_dtem
        dHdS  = sens.dh_dsal
        dHdDIC = sens.dh_ddic
        dHdAlk = sens.dh_dalk

        # instantaneous fields
        S   = sal[configs[0]][scen]
        DIC = dic[configs[0]][scen]
        Alk = alk[configs[0]][scen]

        mult_dict[scen] = {}
        for ref in scenarios:
            print(f"  Relative to {ref}")

            dT   = tem_anom[configs[0]][scen][ref]
            dS   = sal_anom[configs[0]][scen][ref]
            dDIC = dic_anom[configs[0]][scen][ref]
            dAlk = alk_anom[configs[0]][scen][ref]

            # contributions
            temp_contrib = dHdT * dT

            # salinity direct + dilution
            S_direct = dHdS * dS
            dic_dil  = (DIC / S) * dHdDIC * dS
            alk_dil  = (Alk / S) * dHdAlk * dS
            sal_contrib = S_direct + dic_dil + alk_dil

            # salinity-normalized anomalies
            sDIC_anom = (dDIC / S) * S0
            sAlk_anom = (dAlk / S) * S0
            dic_contrib = (S / S0) * dHdDIC * sDIC_anom
            alk_contrib = (S / S0) * dHdAlk * sAlk_anom

            mult_dict[scen][ref] = {
                "temp_sal_normalized": temp_contrib,
                "DIC_sal_normalized":  dic_contrib,
                "Alk_sal_normalized":  alk_contrib,
                "salinity_sal_normalized": sal_contrib,
            }
    return mult_dict



salnorm_contribs = multiply_anom_with_sens_salnorm(
    tem_anom, sal_anom, dic_anom, alk_anom,
    dic, alk, sal, sens_dict,
    configs=configs, scenarios=scenarios
)



#%% now mask all the non-extreme values, using the present-day threshold


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


dtem_times_dhdtem_masked = apply_extreme_masks({scen: {ref: salnorm_contribs[scen][ref]["temp_sal_normalized"] for ref in scenarios} for scen in scenarios}, extreme_mask_dict)
ddic_times_dhddic_masked = apply_extreme_masks({scen: {ref: salnorm_contribs[scen][ref]["DIC_sal_normalized"] for ref in scenarios} for scen in scenarios}, extreme_mask_dict)
dalk_times_dhdalk_masked = apply_extreme_masks({scen: {ref: salnorm_contribs[scen][ref]["Alk_sal_normalized"] for ref in scenarios} for scen in scenarios}, extreme_mask_dict)
dsal_times_dhdsal_masked = apply_extreme_masks({scen: {ref: salnorm_contribs[scen][ref]["salinity_sal_normalized"] for ref in scenarios} for scen in scenarios}, extreme_mask_dict)



#%%
# Define driver input dictionaries
driver_data = {
    "DIC": ddic_times_dhddic_masked,
    "temp": dtem_times_dhdtem_masked,
    "Alk": dalk_times_dhdalk_masked
}

# Extract reference lat/lon for all maps
lat = salnorm_contribs["present"]["present"]["temp_sal_normalized"].lat
lon = salnorm_contribs["present"]["present"]["temp_sal_normalized"].lon


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

    out = f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/sal_norm_3x3_{baseline}_baseline.png"
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
        plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/sal_norm_timeseries_point_by_region_pair_{scen}.png", dpi=200, transparent=True, bbox_inches='tight')
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
        "Alkalinity":  dalk_times_dhdalk_masked,
        "Salinity" : dsal_times_dhdsal_masked  # if you want to include salinity as well
    },
    driver_points=driver_points
)

#%%

# %%

# Sanity check: summed up contributions vs. actual H+ concentration at a point


def _strip_to_index_coords(da: xr.DataArray) -> xr.DataArray:
    """
    Keep only coords that are 1-D and exactly match a dimension (e.g. 'time').
    Drop scalars and any 2-D coords that share a dim name.
    """
    # ensure we don't mutate the original
    da0 = da.squeeze(drop=True)

    keep = {}
    for dim in da0.dims:
        if dim in da0.coords:
            coord = da0.coords[dim]
            if coord.ndim == 1 and coord.sizes.get(dim, None) == da0.sizes[dim]:
                keep[dim] = coord

    # rebuild DA with only index-like coords
    return xr.DataArray(
        da0.data,
        dims=da0.dims,
        coords=keep,
        name=getattr(da0, "name", None),
        attrs=getattr(da0, "attrs", {})
    )

def sum_contributions_safe(dtem_masked, ddic_masked, dalk_masked, dsal_masked):
    """
    Sum Temperature + DIC + Alkalinity + Salinity contributions
    with robust coord handling to avoid xarray index errors.
    """
    out = {}
    for scen in ["present", "ssp245", "ssp585"]:
        out[scen] = {}
        for ref in ["present", "ssp245", "ssp585"]:
            out[scen][ref] = {}
            for thresh in ["present", "present_plus_meandelta"]:
                a = _strip_to_index_coords(dtem_masked[scen][ref][thresh])
                b = _strip_to_index_coords(ddic_masked[scen][ref][thresh])
                c = _strip_to_index_coords(dalk_masked[scen][ref][thresh])
                d = _strip_to_index_coords(dsal_masked[scen][ref][thresh])

                # align on common dims/indexes to avoid any mis-match
                a, b, c, d = xr.align(a, b, c, d, join="inner")

                out[scen][ref][thresh] = ((a + b) + (c + d))
    return out

# build it
sum_contribs_masked = sum_contributions_safe(
    dtem_times_dhdtem_masked,
    ddic_times_dhddic_masked,
    dalk_times_dhdalk_masked,
    dsal_times_dhdsal_masked
)

def build_hplus_concentration_dict(hplus, config="romsoc_fully_coupled"):
    """
    Returns a dict with the structure drivers[name][scenario][scenario]['present'] -> DataArray
    holding the *instantaneous* [H+] concentration (NOT anomalies).

    Keys filled (others unnecessary for your plotter but added for completeness):
      - present → ["present"]["present"]["present"]
      - future  → [scen][scen]["present"]  (scen in {"ssp245","ssp585"})
    """
    out = {"present": {}, "ssp245": {}, "ssp585": {}}

    # present
    out["present"]["present"] = {
        "present": hplus[config]["present"]
    }

    # futures (store under [scen][scen]["present"])
    for scen in ["ssp245", "ssp585"]:
        out[scen][scen] = {
            "present": hplus[config][scen]
        }

    # (optional) put placeholders so access patterns never KeyError
    for scen in ["present","ssp245","ssp585"]:
        for ref in ["present","ssp245","ssp585"]:
            out[scen].setdefault(ref, {})
            out[scen][ref].setdefault("present_plus_meandelta", out[scen][ref].get("present", None))

    return out

# Build it
hplus_conc = build_hplus_concentration_dict(hplus, config="romsoc_fully_coupled")

#%%
# Plotting function for  the sanity check

def plot_sanity_check(drivers, driver_points):
    """
    For each scenario, plot offshore and coastal side‐by‐side,
    showing daily‐mean point time series for each driver.
    All fonts and colors match the regional‐mean plot.
    """
    scenarios = ["present", "ssp245", "ssp585"]
    regions   = ["offshore_central", "coastal_central"]
    driver_colors = {
        "Summed up drivers": "firebrick",
        "Actual [H+]":         "royalblue",
    }
    linestyles = {
        "offshore_central": "-",
        "coastal_central":  "--"
    }

    # grab a sample to get time axis
    sample = dtem_times_dhdtem_masked["present"]["present"]["present"]
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
        plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/driver_analysis/sum_cont_masked{scen}.png", dpi=200, transparent=True, bbox_inches='tight')
        plt.show()


# Usage:
driver_points = {
    "offshore_central": (38.5, 233.0),
    "coastal_central":  (38.5, 236.5)
}

plot_sanity_check(
    drivers={
        "Summed up drivers": sum_contribs_masked,
        "Actual [H+]":       hplus_conc
    },
    driver_points=driver_points
)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_contrib_pies_all_scenarios(
    dtem_masked, ddic_masked, dalk_masked, dsal_masked,
    points,                         # {"offshore_central": (lat,lon), "coastal_central": (lat,lon)}
    use_abs=True,                   # percent of |contribution| (recommended)
    savepath="/nfs/sea/work/fpfaeffli/plots/driver_analysis/sal_norm_contrib_pies_all_scens.png"
):
    """
    Make pie charts of % contributions (T, DIC, Alk, Sal (+dil))
    at two points for all scenarios. Uses shifted ref (ref=scenario).
    Threshold key: 'present' for present; 'present_plus_meandelta' for futures.
    Percentages appear outside with leader lines and color-matched text.
    """

    scenarios  = ["present", "ssp245", "ssp585"]
    ref_map    = {"present":"present", "ssp245":"ssp245", "ssp585":"ssp585"}
    thresh_map = {"present":"present", "ssp245":"present_plus_meandelta", "ssp585":"present_plus_meandelta"}

    # sample grid/time
    sample = dtem_masked["present"]["present"]["present"]
    lat2d, lon2d = sample.lat.values, sample.lon.values
    time_idx = sample["time"].to_index()

    def nearest_ij(lat, lon):
        plon = lon if lon >= 0 else lon + 360
        d2 = (lat2d - lat)**2 + (lon2d - plon)**2
        return np.unravel_index(np.nanargmin(d2), lat2d.shape)

    def point_mean(da, ij):
        """Time-mean at grid cell (daily mean first)."""
        if "time" not in da.dims:
            da = da.expand_dims(time=time_idx)
        i, j = ij
        arr = da.values
        ts = arr[:, i, j] if arr.ndim == 3 else arr
        ser = pd.Series(ts, index=da["time"].to_index()).resample("1D").mean()
        return float(np.nanmean(ser.values))

    # Labels + Okabe–Ito colorblind-friendly palette
    labels = ["Temperature", "DIC", "Alkalinity", "Salinity (+ dil.)"]
    colors = ["#E69F00", "#56B4E9", "#009E73", "#D55E00"]  # orange, sky blue, bluish green, vermillion

    # Figure: 3 scenarios × 2 points
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(12, 12))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]  # safety

    # Precompute ij for points
    point_items = list(points.items())
    point_ij = {name: nearest_ij(lat, lon) for name, (lat, lon) in point_items}

    for r, scen in enumerate(scenarios):
        ref = ref_map[scen]
        thresh_key = thresh_map[scen]

        # select DAs for this (scenario, ref, thresh)
        T_da   = dtem_masked[scen][ref][thresh_key].squeeze(drop=True)
        DIC_da = ddic_masked[scen][ref][thresh_key].squeeze(drop=True)
        Alk_da = dalk_masked[scen][ref][thresh_key].squeeze(drop=True)
        Sal_da = dsal_masked[scen][ref][thresh_key].squeeze(drop=True)

        for c, (pname, (plat, plon)) in enumerate(point_items):
            ax = axes[r, c]
            ij = point_ij[pname]

            vals = np.array([
                point_mean(T_da, ij),
                point_mean(DIC_da, ij),
                point_mean(Alk_da, ij),
                point_mean(Sal_da, ij),
            ])

            # percentages
            weights = np.abs(vals) if use_abs else vals
            total = np.nansum(np.abs(vals)) if use_abs else np.nansum(vals)
            fracs = np.zeros_like(vals) if (not np.isfinite(total) or total == 0) else 100.0 * weights / total
            fracs = np.clip(fracs, 0, None)

            # Make the pie without labels/autopct (we’ll draw our own outside)
            wedges, _ = ax.pie(
                fracs, labels=None, colors=colors,
                startangle=90, counterclock=False, radius=1.0
            )

            # External labels with leader lines, one decimal percentage
            # Compute angles for each wedge mid, then place annotation outside
            kw_line = dict(arrowprops=dict(arrowstyle="-", lw=1.0, color=None),
                           va="center", ha="center")

            # Track used y-positions to avoid overlap (simple nudging)
            used_y = []

            for w, pct, col, lab in zip(wedges, fracs, colors, labels):
                if pct <= 0:
                    continue

                # angle in degrees where label should go
                ang = (w.theta2 + w.theta1) / 2.0
                ang_rad = np.deg2rad(ang)

                # anchor point just outside the pie
                r_label = 1.15
                x = r_label * np.cos(ang_rad)
                y = r_label * np.sin(ang_rad)

                # nudging to reduce overlap
                # push slightly up/down if too close to a previous label
                for _ in range(10):
                    if any(abs(y - yy) < 0.06 for yy in used_y):
                        y += 0.06 * np.sign(y if y != 0 else 1.0)
                    else:
                        break
                used_y.append(y)

                # alignment based on side
                ha = "left" if x >= 0 else "right"
                # leader line from wedge center to label
                # set arrow color to the wedge color
                arrowprops = dict(arrowstyle="-", lw=1.0, color=col, shrinkA=0, shrinkB=0)

                ax.annotate(f"{pct:.1f}%", xy=(np.cos(ang_rad)*0.8, np.sin(ang_rad)*0.8),
                            xytext=(x, y), color=col, ha=ha, va="center",
                            arrowprops=arrowprops, fontsize=11)

            # Titles
            if r == 0:  # only top row gets titles
                short = "offshore" if pname.startswith("offshore") else "coastal"
                ax.set_title(short, fontsize=14)

            # Row label on left side
            if c == 0:
                ax.text(-1.5, 0.5, f"{scen}", fontsize=14)

    # Legend (shared)
    legend_handles = [Patch(facecolor=col, edgecolor="none", label=lab) for lab, col in zip(labels, colors)]
    fig.legend(
        handles=legend_handles, labels=labels,
        loc="lower center", ncol=4, frameon=True, fontsize=12, borderaxespad=0.8
    )

    plt.tight_layout(rect=[0.02, 0.06, 1, 0.94])
    plt.savefig(savepath, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()


driver_points = {
    "offshore_central": (38.5, 233.0),
    "coastal_central":  (38.5, 236.5)
}

plot_contrib_pies_all_scenarios(
    dtem_times_dhdtem_masked,
    ddic_times_dhddic_masked,
    dalk_times_dhdalk_masked,
    dsal_times_dhdsal_masked,
    points=driver_points,
    use_abs=True,
    savepath="/nfs/sea/work/fpfaeffli/plots/driver_analysis/sal_norm_contrib_pies_all_scens_abs.png"
)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_contrib_pies_area_all_scenarios(
    dtem_masked, ddic_masked, dalk_masked, dsal_masked,
    use_abs=True,
    savepath="/nfs/sea/work/fpfaeffli/plots/driver_analysis/sal_norm_contrib_pies_AREA_all_scens.png"
):
    """
    Three pies (present, ssp245, ssp585), each showing the percent share of
    Temperature, DIC, Alkalinity, Salinity(+dil.) contributions averaged over
    the entire masked domain (time & space mean of masked fields).

    Future scenarios use the 'present_plus_meandelta' threshold; present uses 'present'.
    """

    scenarios  = ["present", "ssp245", "ssp585"]
    ref_map    = {"present":"present", "ssp245":"ssp245", "ssp585":"ssp585"}  # shifted-style ref
    thresh_map = {"present":"present", "ssp245":"present_plus_meandelta", "ssp585":"present_plus_meandelta"}

    # color-blind friendly (Okabe–Ito) + labels
    labels = ["Temperature contribution", "DIC contribution", "Alkalinity contribution", "Salinity contribution"]
    colors = ["#D55E00", "#56B4E9", "#009E73", "#E69F00"]

    # fallback time index in case a DA is missing 'time'
    sample = dtem_masked["present"]["present"]["present"]
    time_idx = sample["time"].to_index() if "time" in sample.dims else None

    def ensure_time(da):
        if "time" not in da.dims:
            if time_idx is None:
                raise ValueError("No reference 'time' axis found to expand arrays.")
            da = da.expand_dims(time=time_idx)
        return da

    def area_time_mean(da):
        da = ensure_time(da).squeeze(drop=True)
        # average over all available dims with skipna=True (masked areas are NaN)
        dims = [d for d in da.dims if d in ("time", "eta_rho", "xi_rho")]
        return float(da.mean(dim=dims, skipna=True).values)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, scen in zip(axes, scenarios):
        ref        = ref_map[scen]
        thresh_key = thresh_map[scen]

        # pick arrays
        T_da   = dtem_masked[scen][ref][thresh_key].squeeze(drop=True)
        DIC_da = ddic_masked[scen][ref][thresh_key].squeeze(drop=True)
        Alk_da = dalk_masked[scen][ref][thresh_key].squeeze(drop=True)
        Sal_da = dsal_masked[scen][ref][thresh_key].squeeze(drop=True)

        # time–space means
        vals = np.array([
            area_time_mean(T_da),
            area_time_mean(DIC_da),
            area_time_mean(Alk_da),
            area_time_mean(Sal_da),
        ])

        # convert to % shares
        weights = np.abs(vals) if use_abs else vals
        total   = np.nansum(np.abs(vals)) if use_abs else np.nansum(vals)
        fracs   = np.zeros_like(vals) if (not np.isfinite(total) or total == 0) else 100.0 * weights / total
        fracs   = np.clip(fracs, 0, None)

        # draw pie (no internal labels)
        wedges, _ = ax.pie(fracs, labels=None, colors=colors,
                           startangle=90, counterclock=False, radius=1.0)

        # outside labels with leader lines, one decimal, color-matched text
        used_y = []
        for w, pct, col, lab in zip(wedges, fracs, colors, labels):
            if pct <= 0:
                continue
            ang = (w.theta2 + w.theta1) / 2.0
            ang_rad = np.deg2rad(ang)
            # anchor positions
            r_text = 1.15
            x = r_text * np.cos(ang_rad)
            y = r_text * np.sin(ang_rad)
            # simple anti-overlap nudging
            for _ in range(12):
                if any(abs(y - yy) < 0.06 for yy in used_y):
                    y += 0.06 * (1 if y >= 0 else -1)
                else:
                    break
            used_y.append(y)
            ha = "left" if x >= 0 else "right"
            arrowprops = dict(arrowstyle="-", lw=1.0, color=col, shrinkA=0, shrinkB=0)
            ax.annotate(f"{pct:.1f}%", xy=(np.cos(ang_rad)*0.8, np.sin(ang_rad)*0.8),
                        xytext=(x, y), color=col, ha=ha, va="center",
                        arrowprops=arrowprops, fontsize=11)

        # Move the scenario title higher above the pie
        ax.set_title(scen, fontsize=14, pad=24)

    # shared legend
    legend_handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in zip(labels, colors)]
    fig.legend(legend_handles, labels, loc="lower center", ncol=4, frameon=True, fontsize=12)

    plt.tight_layout(rect=[0.02, 0.08, 1, 1])
    plt.savefig(savepath, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()


plot_contrib_pies_area_all_scenarios(
    dtem_times_dhdtem_masked,
    ddic_times_dhddic_masked,
    dalk_times_dhdalk_masked,
    dsal_times_dhdsal_masked,
    use_abs=True,
    savepath="/nfs/sea/work/fpfaeffli/plots/driver_analysis/sal_norm_contrib_pies_all_scens_whole_area.png"
)
