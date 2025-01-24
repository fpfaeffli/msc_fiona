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

# load the packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
from get_study_regions import GetRegions as GetRegions
from get_model_datasets_Hplus import ModelGetter as ModelGetter

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
scenarios = ['present','ssp585'] # ,'ssp585'
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

print("Load Hplus data")
hplus = dict()
for config in configs:
    hplus[config] = dict()
    for scenario in scenarios:
        print(f'Getting the variable for config {config} and scenario {scenario}.')
        #
        # oceanic variables
        hplus[config][scenario] = dict()
        print('Hplus')
        # Convert pH to [H+] concentration
        pH = ocean_ds[config][scenario].pH_offl.isel(depth=0).load()
        hplus[config][scenario]['Hplus'] = np.power(10, -pH)
print("Hplus data loaded.")

print("Load temperature data")
tem = dict()
for config in configs:
     tem[config] = dict()
     for scenario in scenarios:
          print(f'Getting the variable for config {config} and scenario {scenario} and ensemble run {ensemble_run}.')
          #
          # oceanic variables
          tem[config][scenario] = dict()
          print('temp')
          tem[config][scenario]['temp'] = ocean_ds[config][scenario].temp.isel(depth=0).load()
print("Temperature data loaded.")

#%%
print("Load salinity data")
sal = dict()
for config in configs:
     sal[config] = dict()
     for scenario in scenarios:
          print(f'Getting the variable for config {config} and scenario {scenario} and ensemble run {ensemble_run}.')
          #
          # oceanic variables
          sal[config][scenario] = dict()
          print('sal')
          sal[config][scenario]['salt'] = ocean_ds[config][scenario].salt.isel(depth=0).load()
print("Salinity data loaded.")

print("Load DIC data")
dic = dict()
for config in configs:
     dic[config] = dict()
     for scenario in scenarios:
          print(f'Getting the variable for config {config} and scenario {scenario} and ensemble run {ensemble_run}.')
          #
          # oceanic variables
          dic[config][scenario] = dict()
          print('DIC')
          dic[config][scenario]['DIC'] = ocean_ds[config][scenario].DIC.isel(depth=0).load()
print("DIC data loaded.")

print("Load Alk data")
alk = dict()
for config in configs:
     alk[config] = dict()
     for scenario in scenarios:
          print(f'Getting the variable for config {config} and scenario {scenario} and ensemble run {ensemble_run}.')
          #
          # oceanic variables
          alk[config][scenario] = dict()
          print('DIC')
          alk[config][scenario]['Alk'] = ocean_ds[config][scenario].Alk.isel(depth=0).load()
print("Alk data loaded.")

#%%
# Save the loaded data into new netCDF files with just depth=0 for faster loading

def save_to_netcdf(data_dict, variable_name, output_dir):
    """
    Save the data dictionary to netCDF files with just depth=0.

    """
    os.makedirs(output_dir, exist_ok=True)
    for config in data_dict:
        for scenario in data_dict[config]:
            file_path = os.path.join(output_dir, f"{variable_name}_{config}_{scenario}_only_surface.nc")
            data_dict[config][scenario][variable_name].to_netcdf(file_path)
            print(f"Saved {variable_name} data for {config} {scenario} to {file_path}")

# Define the output directory
output_dir = "/nfs/sea/work/fpfaeffli/sliced_ocean_ds/"

# Save Hplus data
save_to_netcdf(hplus, "Hplus", output_dir)

# Save temperature data
save_to_netcdf(tem, "temp", output_dir)


#%%

# Loading the aggregated sensitivity fields from netCDF files

# Define the directory where the files are stored
aggregated_dir = "/nfs/sea/work/fpfaeffli/aggregated_sensitivities/"

# Define the sensitivity variables
sensitivity_variables = ['dh_dtem', 'dh_dsal', 'dh_ddic', 'dh_dalk']

# Load each variable as its own 
sensitivities_dh_dtem = xr.open_dataarray(os.path.join(aggregated_dir, "aggregated_dh_dtem_all_years.nc"))
sensitivities_dh_dsal = xr.open_dataarray(os.path.join(aggregated_dir, "aggregated_dh_dsal_all_years.nc"))
sensitivities_dh_ddic = xr.open_dataarray(os.path.join(aggregated_dir, "aggregated_dh_ddic_all_years.nc"))
sensitivities_dh_dalk = xr.open_dataarray(os.path.join(aggregated_dir, "aggregated_dh_dalk_all_years.nc"))

# Verify each variable
print(f"sensitivities_dh_dtem: dims = {sensitivities_dh_dtem.dims}, shape = {sensitivities_dh_dtem.shape}")
print(f"sensitivities_dh_dsal: dims = {sensitivities_dh_dsal.dims}, shape = {sensitivities_dh_dsal.shape}")
print(f"sensitivities_dh_ddic: dims = {sensitivities_dh_ddic.dims}, shape = {sensitivities_dh_ddic.shape}")
print(f"sensitivities_dh_dalk: dims = {sensitivities_dh_dalk.dims}, shape = {sensitivities_dh_dalk.shape}")


# %% 
# Get the present day climatology for Hplus, T, DIC, Alk and S

clim_dir = '/nfs/sea/work/fpfaeffli/future_sim/thresholds_and_climatologies/romsoc_fully_coupled/present/'

# Get climatology for Hplus 
hplus_clim = xr.open_dataset(clim_dir+'hobday2016_threshold_and_climatology_Hplus_95.0perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing_0depthlevelindex_ensemble000.nc').clim_smoothed
print("Hplus climatology loaded.")

# Get the climatology for tem 
tem_clim = xr.open_dataset(clim_dir+'hobday2016_threshold_and_climatology_temp_95.0perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing_0depthlevelindex_ensemble000.nc').clim_smoothed
print("Temperature climatology loaded.")

# Get the climatology for sal
sal_clim = xr.open_dataset(clim_dir+'hobday2016_threshold_and_climatology_salt_95.0perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing_0depthlevelindex_ensemble000.nc').clim_smoothed
print("Salinity climatology loaded.")

# Get the climatology for dic
dic_clim = xr.open_dataset(clim_dir+'hobday2016_threshold_and_climatology_DIC_95.0perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing_0depthlevelindex_ensemble000.nc').clim_smoothed
print("DIC climatology loaded.")

# Get the climatology for alk
alk_clim = xr.open_dataset(clim_dir+'hobday2016_threshold_and_climatology_Alk_95.0perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing_0depthlevelindex_ensemble000.nc').clim_smoothed
print("Alkalinity climatology loaded.")


#%%

##################### Extreme detection #####################§


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
            dummy_clim = ThreshClimFuncs.calc_clim(params,hplus[config][scenario][varia])
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
    for scenario in ['present']:  # ,'ssp585','ssp585'
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



#%%

# Variables 
scenario = 'ssp585'
depthlevel = 0
eta_rho_cho = 500
xi_rho_cho = 200

# Load Hplus variables and both thresholds (moving an fixed)
present = hplus['romsoc_fully_coupled']['present']['Hplus']
future = hplus['romsoc_fully_coupled'][scenario]['Hplus']
thresholds_present = thresholds_mult['romsoc_fully_coupled'][scenario]['present']
thresholds_present_meandelta = thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_meandelta']

# Compute extremes with moving and fixed baselines
future_fixed_baseline = future - thresholds_present
future_moving_baseline = future - thresholds_present_meandelta
print("Future baselines computed")

# Define a mask for all extremes (fixed baseline)
future_extremes_fixed_mask = future_fixed_baseline > 0

# Define a mask for all extremes (moving baseline)
future_extremes_moving_mask = future_moving_baseline > 0



#%%

############# Applying masks to sensitivities #############

# Function to apply extreme masks to sensitivity fields
def apply_extreme_mask(sensitivity_field, extreme_mask):
    """
    Apply extreme masks to a sensitivity field using a Boolean mask.

    """
    return sensitivity_field.where(extreme_mask, other=np.nan)

# Apply extreme masks to sensitivities for fixed baseline
print("Applying fixed baseline masks...")
masked_sensitivities_fixed_dtem = apply_extreme_mask(sensitivities_dh_dtem, future_extremes_fixed_mask)
masked_sensitivities_fixed_dsal = apply_extreme_mask(sensitivities_dh_dsal, future_extremes_fixed_mask)
masked_sensitivities_fixed_ddic = apply_extreme_mask(sensitivities_dh_ddic, future_extremes_fixed_mask)
masked_sensitivities_fixed_dalk = apply_extreme_mask(sensitivities_dh_dalk, future_extremes_fixed_mask)
print("Fixed baseline masks applied.")

# Apply extreme masks to sensitivities for moving baseline
print("Applying moving baseline masks...")
masked_sensitivities_moving_dtem = apply_extreme_mask(sensitivities_dh_dtem, future_extremes_moving_mask)
masked_sensitivities_moving_dsal = apply_extreme_mask(sensitivities_dh_dsal, future_extremes_moving_mask)
masked_sensitivities_moving_ddic = apply_extreme_mask(sensitivities_dh_ddic, future_extremes_moving_mask)
masked_sensitivities_moving_dalk = apply_extreme_mask(sensitivities_dh_dalk, future_extremes_moving_mask)
print("Moving baseline masks applied.")



#%% ############### Anomaly calculations #################


print("Loading hindcast temperature data...")
hindcast_temp = tem['romsoc_fully_coupled']['present']['temp']

# Step 1: Add 'day_of_year_adjusted' to the climatology if not present
if 'day_of_year_adjusted' not in tem_clim.dims:
    print("Adding 'day_of_year_adjusted' dimension to climatology...")
    tem_clim = tem_clim.assign_coords(day_of_year_adjusted=("time", np.arange(1, 366)))


# Handle leap years: Add February 29 to the climatology
tem_clim_with_leap = ModelGetter.include_feb29(tem_clim)

# Step 3: Expand climatology for the full hindcast period (2011–2021)
print("Expanding climatology for the full hindcast period...")
expanded_climatology = ModelGetter.concatenate_yearly_arrays(
    tem_clim,
    tem_clim_with_leap,
    start_year=2011,
    end_year=2021
)

# Rename 'day_of_year_adjusted' to 'time' in the expanded climatology
expanded_climatology = expanded_climatology.rename({'day_of_year_adjusted': 'time'})

# Check dimensions after renaming
print("Renamed climatology dimensions:", expanded_climatology.dims)

# Step 4: Align hindcast and climatology
print("Aligning hindcast data with expanded climatology...")
hindcast_temp, expanded_climatology = xr.align(hindcast_temp, expanded_climatology)

print("Hindcast temperature data dimensions:", hindcast_temp.dims)
print("Hindcast temperature data shape:", hindcast_temp.shape)
print("Expanded climatology dimensions:", expanded_climatology.dims)
print("Expanded climatology shape:", expanded_climatology.shape)

# Step 4: Calculate anomalies
print("Calculating temperature anomalies...")
tem_anomalies = hindcast_temp - expanded_climatology

print("Temperature anomalies calculated.")

# Now for DIC

print("Loading hindcast DIC data...")
hindcast_dic = dic['romsoc_fully_coupled']['present']['DIC']

# Step 1: Add 'day_of_year_adjusted' to the climatology if not present
if 'day_of_year_adjusted' not in dic_clim.dims:
    print("Adding 'day_of_year_adjusted' dimension to climatology...")
    dic_clim = dic_clim.assign_coords(day_of_year_adjusted=("time", np.arange(1, 366)))


# Handle leap years: Add February 29 to the climatology
dic_clim_with_leap = ModelGetter.include_feb29(dic_clim)

# Step 3: Expand climatology for the full hindcast period (2011–2021)
print("Expanding climatology for the full hindcast period...")
expanded_climatology_dic = ModelGetter.concatenate_yearly_arrays(
    dic_clim,
    dic_clim_with_leap,
    start_year=2011,
    end_year=2021
)

# Rename 'day_of_year_adjusted' to 'time' in the expanded climatology
expanded_climatology_dic = expanded_climatology_dic.rename({'day_of_year_adjusted': 'time'})

# Check dimensions after renaming
print("Renamed climatology dimensions:", expanded_climatology_dic.dims)

# Step 4: Align hindcast and climatology
print("Aligning hindcast data with expanded climatology...")
hindcast_dic, expanded_climatology_dic = xr.align(hindcast_dic, expanded_climatology_dic)

print("Hindcast DIC data dimensions:", hindcast_dic.dims)
print("Hindcast DIC data shape:", hindcast_dic.shape)
print("Expanded climatology dimensions:", expanded_climatology_dic.dims)
print("Expanded climatology shape:", expanded_climatology_dic.shape)

# Step 4: Calculate anomalies
print("Calculating DIC anomalies...")
dic_anomalies = hindcast_dic - expanded_climatology_dic

print("DIC anomalies calculated.")

# Now for Alk

print("Loading hindcast Alk data...")
hindcast_alk = alk['romsoc_fully_coupled']['present']['Alk']

# Step 1: Add 'day_of_year_adjusted' to the climatology if not present
if 'day_of_year_adjusted' not in alk_clim.dims:
    print("Adding 'day_of_year_adjusted' dimension to climatology...")
    alk_clim = alk_clim.assign_coords(day_of_year_adjusted=("time", np.arange(1, 366)))


# Handle leap years: Add February 29 to the climatology
alk_clim_with_leap = ModelGetter.include_feb29(alk_clim)

# Step 3: Expand climatology for the full hindcast period (2011–2021)
print("Expanding climatology for the full hindcast period...")
expanded_climatology_alk = ModelGetter.concatenate_yearly_arrays(
    alk_clim,
    alk_clim_with_leap,
    start_year=2011,
    end_year=2021
)

# Rename 'day_of_year_adjusted' to 'time' in the expanded climatology
expanded_climatology_alk = expanded_climatology_alk.rename({'day_of_year_adjusted': 'time'})

# Check dimensions after renaming
print("Renamed climatology dimensions:", expanded_climatology_alk.dims)

# Step 4: Align hindcast and climatology
print("Aligning hindcast data with expanded climatology...")
hindcast_alk, expanded_climatology_alk = xr.align(hindcast_alk, expanded_climatology_alk)

print("Hindcast Alk data dimensions:", hindcast_alk.dims)
print("Hindcast Alk data shape:", hindcast_alk.shape)
print("Expanded climatology dimensions:", expanded_climatology_alk.dims)
print("Expanded climatology shape:", expanded_climatology_alk.shape)

# Step 4: Calculate anomalies
print("Calculating Alk anomalies...")
alk_anomalies = hindcast_alk - expanded_climatology_alk

print("Alk anomalies calculated.")

# Now for Sal

print("Loading hindcast Sal data...")
hindcast_sal = sal['romsoc_fully_coupled']['present']['salt']

# Step 1: Add 'day_of_year_adjusted' to the climatology if not present
if 'day_of_year_adjusted' not in sal_clim.dims:
    print("Adding 'day_of_year_adjusted' dimension to climatology...")
    sal_clim = sal_clim.assign_coords(day_of_year_adjusted=("time", np.arange(1, 366)))


# Handle leap years: Add February 29 to the climatology
sal_clim_with_leap = ModelGetter.include_feb29(sal_clim)

# Step 3: Expand climatology for the full hindcast period (2011–2021)
print("Expanding climatology for the full hindcast period...")
expanded_climatology_sal = ModelGetter.concatenate_yearly_arrays(
    sal_clim,
    sal_clim_with_leap,
    start_year=2011,
    end_year=2021
)

# Rename 'day_of_year_adjusted' to 'time' in the expanded climatology
expanded_climatology_sal = expanded_climatology_sal.rename({'day_of_year_adjusted': 'time'})

# Check dimensions after renaming
print("Renamed climatology dimensions:", expanded_climatology_sal.dims)

# Step 4: Align hindcast and climatology
print("Aligning hindcast data with expanded climatology...")
hindcast_sal, expanded_climatology_sal = xr.align(hindcast_sal, expanded_climatology_sal)

print("Hindcast Sal data dimensions:", hindcast_sal.dims)
print("Hindcast Sal data shape:", hindcast_sal.shape)
print("Expanded climatology dimensions:", expanded_climatology_sal.dims)
print("Expanded climatology shape:", expanded_climatology_sal.shape)

# Step 4: Calculate anomalies
print("Calculating Sal anomalies...")
sal_anomalies = hindcast_sal - expanded_climatology_sal

print("Sal anomalies calculated.")



#%%
# Calculate the contributions of each driver to Hplus changes by multiplying sensitivities with anomalies

# Contribution to H+ changes

# Extract 1D indices for eta_rho and xi_rho
eta_rho_1d = np.arange(masked_sensitivities_fixed_dtem.shape[1])  # Size of eta_rho dimension
xi_rho_1d = np.arange(masked_sensitivities_fixed_dtem.shape[2])   # Size of xi_rho dimension

# Create new DataArrays with corrected dimensions
def clean_dataarray(dataarray):
    return xr.DataArray(
        dataarray.values,  # Keep the data
        dims=["time", "eta_rho", "xi_rho"],  # Define new dimensions
        coords={
            "time": dataarray["time"].values,  # Copy time coordinate
            "eta_rho": eta_rho_1d,  # Replace with 1D indices
            "xi_rho": xi_rho_1d,  # Replace with 1D indices
        },
    )

# Clean the data arrays for fixed baseline
print("Cleaning data arrays for fixed baseline...")
masked_sensitivities_fixed_cleaned_dtem = clean_dataarray(masked_sensitivities_fixed_dtem)
masked_sensitivities_fixed_cleaned_dsal = clean_dataarray(masked_sensitivities_fixed_dsal)
masked_sensitivities_fixed_cleaned_ddic = clean_dataarray(masked_sensitivities_fixed_ddic)
masked_sensitivities_fixed_cleaned_dalk = clean_dataarray(masked_sensitivities_fixed_dalk)

print("Cleaning data arrays for moving baseline...")
masked_sensitivities_moving_cleaned_dtem = clean_dataarray(masked_sensitivities_moving_dtem)
masked_sensitivities_moving_cleaned_dsal = clean_dataarray(masked_sensitivities_moving_dsal)
masked_sensitivities_moving_cleaned_ddic = clean_dataarray(masked_sensitivities_moving_ddic)
masked_sensitivities_moving_cleaned_dalk = clean_dataarray(masked_sensitivities_moving_dalk)

print("Cleaning arrays for anomalies")
tem_anomalies_cleaned = clean_dataarray(tem_anomalies)
sal_anomalies_cleaned = clean_dataarray(sal_anomalies)
dic_anomalies_cleaned = clean_dataarray(dic_anomalies)
alk_anomalies_cleaned = clean_dataarray(alk_anomalies)

# Align the arrays
def align_arrays(anomalies_cleaned, sensitivities_cleaned):
    return xr.align(anomalies_cleaned, sensitivities_cleaned, join="inner")

# Align the arrays for fixed baseline
print("Aligning arrays for fixed baseline...")
tem_anomalies_aligned, sensitivities_aligned_fixed_dtem = align_arrays(tem_anomalies_cleaned, masked_sensitivities_fixed_cleaned_dtem)
sal_anomalies_aligned, sensitivities_aligned_fixed_dsal = align_arrays(sal_anomalies_cleaned, masked_sensitivities_fixed_cleaned_dsal)
dic_anomalies_aligned, sensitivities_aligned_fixed_ddic = align_arrays(dic_anomalies_cleaned, masked_sensitivities_fixed_cleaned_ddic)
alk_anomalies_aligned, sensitivities_aligned_fixed_dalk = align_arrays(alk_anomalies_cleaned, masked_sensitivities_fixed_cleaned_dalk)

# Align the arrays for moving baseline
print("Aligning arrays for moving baseline...")
tem_anomalies_aligned, sensitivities_aligned_moving_dtem = align_arrays(tem_anomalies_cleaned, masked_sensitivities_moving_cleaned_dtem)
sal_anomalies_aligned, sensitivities_aligned_moving_dsal = align_arrays(sal_anomalies_cleaned, masked_sensitivities_moving_cleaned_dsal)
dic_anomalies_aligned, sensitivities_aligned_moving_ddic = align_arrays(dic_anomalies_cleaned, masked_sensitivities_moving_cleaned_ddic)
alk_anomalies_aligned, sensitivities_aligned_moving_dalk = align_arrays(alk_anomalies_cleaned, masked_sensitivities_moving_cleaned_dalk)

# Step 4: Multiply the aligned arrays for both fixed and baseline
print("Multiplying aligned arrays for fixed baseline...")
tem_contribution_fixed = tem_anomalies_aligned * sensitivities_aligned_fixed_dtem
sal_contribution_fixed = sal_anomalies_aligned * sensitivities_aligned_fixed_dsal
dic_contribution_fixed = dic_anomalies_aligned * sensitivities_aligned_fixed_ddic
alk_contribution_fixed = alk_anomalies_aligned * sensitivities_aligned_fixed_dalk

print("Multiplying aligned arrays for moving baseline...")
tem_contribution_moving = tem_anomalies_aligned * sensitivities_aligned_moving_dtem
sal_contribution_moving = sal_anomalies_aligned * sensitivities_aligned_moving_dsal
dic_contribution_moving = dic_anomalies_aligned * sensitivities_aligned_moving_ddic
alk_contribution_moving = alk_anomalies_aligned * sensitivities_aligned_moving_dalk


print("Multiplication successful. Shapes:")
print("Temperature contribution (fixed):", tem_contribution_fixed.shape)
print("Salinity contribution (fixed):", sal_contribution_fixed.shape)
print("DIC contribution (fixed):", dic_contribution_fixed.shape)
print("Alkalinity contribution (fixed):", alk_contribution_fixed.shape)

print("Temperature contribution (moving):", tem_contribution_moving.shape)
print("Salinity contribution (moving):", sal_contribution_moving.shape)
print("DIC contribution (moving):", dic_contribution_moving.shape)
print("Alkalinity contribution (moving):", alk_contribution_moving.shape)

#%% Regrid the landmask

# Load the landmask 
landmask_etopo = PlotFuncs.get_etopo_data() 

lat_2d = tem_anomalies["lat"].values
lon_2d = tem_anomalies["lon"].values

# Assign these coordinates back to tem_contribution_fixed
tem_contribution_fixed = tem_contribution_fixed.assign_coords(
    lat=(("eta_rho", "xi_rho"), lat_2d),
    lon=(("eta_rho", "xi_rho"), lon_2d),
)

# Regrid the landmask to the same grid as tem_contribution_fixed
regridded_landmask = landmask_etopo.interp(
    lon=tem_contribution_fixed["lon"],
    lat=tem_contribution_fixed["lat"],
    method="nearest"
)

# Assign these coordinates back to dic_contribution_fixed
dic_contribution_fixed = dic_contribution_fixed.assign_coords(
    lat=(("eta_rho", "xi_rho"), lat_2d),
    lon=(("eta_rho", "xi_rho"), lon_2d),
)

# Regrid the landmask to the same grid as dic_contribution_fixed
regridded_landmask_dic = landmask_etopo.interp(
    lon=dic_contribution_fixed["lon"],
    lat=dic_contribution_fixed["lat"],
    method="nearest"
)

# Assign these coordinates back to alk_contribution_fixed
alk_contribution_fixed = alk_contribution_fixed.assign_coords(
    lat=(("eta_rho", "xi_rho"), lat_2d),
    lon=(("eta_rho", "xi_rho"), lon_2d),
)

# Regrid the landmask to the same grid as alk_contribution_fixed
regridded_landmask_alk = landmask_etopo.interp(
    lon=alk_contribution_fixed["lon"],
    lat=alk_contribution_fixed["lat"],
    method="nearest"
)

# Assign these coordinates back to sal_contribution_fixed
sal_contribution_fixed = sal_contribution_fixed.assign_coords(
    lat=(("eta_rho", "xi_rho"), lat_2d),
    lon=(("eta_rho", "xi_rho"), lon_2d),
)

# Regrid the landmask to the same grid as sal_contribution_fixed
regridded_landmask_sal = landmask_etopo.interp(
    lon=sal_contribution_fixed["lon"],
    lat=sal_contribution_fixed["lat"],
    method="nearest"
)

# Assign these coordinates back to tem_contribution_moving
tem_contribution_moving = tem_contribution_moving.assign_coords(
    lat=(("eta_rho", "xi_rho"), lat_2d),
    lon=(("eta_rho", "xi_rho"), lon_2d),
)

# Regrid the landmask to the same grid as tem_contribution_moving
regridded_landmask_moving = landmask_etopo.interp(
    lon=tem_contribution_moving["lon"],
    lat=tem_contribution_moving["lat"],
    method="nearest"
)

# Assign these coordinates back to dic_contribution_moving
dic_contribution_moving = dic_contribution_moving.assign_coords(
    lat=(("eta_rho", "xi_rho"), lat_2d),
    lon=(("eta_rho", "xi_rho"), lon_2d),
)

# Regrid the landmask to the same grid as dic_contribution_moving
regridded_landmask_dic_moving = landmask_etopo.interp(
    lon=dic_contribution_moving["lon"],
    lat=dic_contribution_moving["lat"],
    method="nearest"
)

# Assign these coordinates back to alk_contribution_moving
alk_contribution_moving = alk_contribution_moving.assign_coords(
    lat=(("eta_rho", "xi_rho"), lat_2d),
    lon=(("eta_rho", "xi_rho"), lon_2d),
)

# Regrid the landmask to the same grid as alk_contribution_moving
regridded_landmask_alk_moving = landmask_etopo.interp(
    lon=alk_contribution_moving["lon"],
    lat=alk_contribution_moving["lat"],
    method="nearest"
)

# Assign these coordinates back to sal_contribution_moving
sal_contribution_moving = sal_contribution_moving.assign_coords(
    lat=(("eta_rho", "xi_rho"), lat_2d),
    lon=(("eta_rho", "xi_rho"), lon_2d),
)

# Regrid the landmask to the same grid as sal_contribution_moving
regridded_landmask_sal_moving = landmask_etopo.interp(
    lon=sal_contribution_moving["lon"],
    lat=sal_contribution_moving["lat"],
    method="nearest"
)


#%%
###############################
#           PLOTTING          #
###############################

#%%

# Plotting function for masked sensitivities
def plot_masked_sensitivity(data, title, lat_bounds, lon_bounds, save_dir="/nfs/sea/work/fpfaeffli/plots/driver_analysis/masked_sensitivities"):
    """
    Plots the masked sensitivities of T, Alk, S and DIC for fixed and moving baselines
    with degree symbols and N/W indicators.
    """
    if data is None:
        print(f"No data available for {title}")
        return

    # Extract coordinates
    lon = data.lon
    lat = data.lat
    values = data.mean(dim='time')  # Taking the mean over time

    # Plot
    plt.figure(figsize=(12, 6))
    contour = plt.contourf(lon, lat, values, levels=20, cmap='viridis')

    # Add landmask
    landmask_etopo = PlotFuncs.get_etopo_data()
    plt.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

    # Add colorbar
    plt.colorbar(contour, label='Sensitivity')

    # Set axis limits
    plt.xlim(lon_bounds)
    plt.ylim(lat_bounds)

    # Define ticks with degree symbols and N/W indicators
    yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
    xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)

    # Set tick labels
    plt.yticks(yticks, [f"{lat}°N" for lat in yticks])
    plt.xticks(xticks, [f"{360 - lon}°W" for lon in xticks])

    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)

    # Add gridlines for better orientation
    plt.grid(visible=True, linestyle='--', linewidth=0.5)

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    #plt.savefig(save_path, dpi=300)
    #print(f"Plot saved to {save_path}")

    # Show the plot
    plt.show()


# Define bounds
lat_bounds = (30, 47)
lon_bounds = (225, 245)

# Plot fixed baseline sensitivities
print("Plotting fixed baseline sensitivities...")
plot_masked_sensitivity(masked_sensitivities_fixed_dtem, "Fixed baseline: dh/dTem (ssp585)", lat_bounds, lon_bounds)
plot_masked_sensitivity(masked_sensitivities_fixed_dsal, "Fixed baseline: dh/dSal (ssp585)", lat_bounds, lon_bounds)
plot_masked_sensitivity(masked_sensitivities_fixed_ddic, "Fixed baseline: dh/dDIC (ssp585)", lat_bounds, lon_bounds)
plot_masked_sensitivity(masked_sensitivities_fixed_dalk, "Fixed baseline: dh/dAlk (ssp585)", lat_bounds, lon_bounds)

# Plot moving baseline sensitivities
print("Plotting moving baseline sensitivities...")
plot_masked_sensitivity(masked_sensitivities_moving_dtem, "Moving baseline: dh/dTem (ssp585)", lat_bounds, lon_bounds)
plot_masked_sensitivity(masked_sensitivities_moving_dsal, "Moving baseline: dh/dSal (ssp585)", lat_bounds, lon_bounds)
plot_masked_sensitivity(masked_sensitivities_moving_ddic, "Moving baseline: dh/dDIC (ssp585)", lat_bounds, lon_bounds)
plot_masked_sensitivity(masked_sensitivities_moving_dalk, "Moving baseline: dh/dAlk (ssp585)", lat_bounds, lon_bounds)

# %%

# Plotting function for sensitivity comparison

def plot_sensitivity_comparison(fixed_data, moving_data, variable_name, lat_bounds, lon_bounds):
    """
    Plots sensitivity fields for fixed and moving baselines side by side, with degree symbols and N/W indicators.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Add landmask to both subplots
    landmask_etopo = PlotFuncs.get_etopo_data()
    for ax in axs:
        ax.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

    # Plot Fixed Baseline
    fixed_mean = fixed_data.mean(dim='time')
    contour_fixed = axs[0].contourf(
        fixed_mean.lon, fixed_mean.lat, fixed_mean, levels=20, cmap='viridis'
    )
    axs[0].set_title(f"Fixed Baseline: {variable_name}")
    axs[0].set_xlim(lon_bounds)
    axs[0].set_ylim(lat_bounds)
    axs[0].set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")
    fig.colorbar(contour_fixed, ax=axs[0], orientation='vertical', label="Sensitivity")

    # Plot Moving Baseline
    moving_mean = moving_data.mean(dim='time')
    contour_moving = axs[1].contourf(
        moving_mean.lon, moving_mean.lat, moving_mean, levels=20, cmap='viridis'
    )
    axs[1].set_title(f"Moving Baseline: {variable_name}")
    axs[1].set_xlim(lon_bounds)
    axs[1].set_ylim(lat_bounds)
    axs[1].set_xlabel("Longitude")
    fig.colorbar(contour_moving, ax=axs[1], orientation='vertical', label="Sensitivity")

    # Add degree symbols and N/W indicators to axes
    yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
    xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
    
    for ax in axs:
        # Set ticks and labels for latitude
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])

        # Set ticks and labels for longitude
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

        # Add gridlines for better orientation
        ax.grid(visible=True, linestyle='--', linewidth=0.5)

    # Set a shared title
    plt.suptitle(f"Comparison of {variable_name} sensitivities (ssp585)", fontsize=16)
    plt.show()


# Define bounds
lat_bounds = (30, 47)
lon_bounds = (225, 245)

# Plot comparison for dh/dAlk
print("Plotting sensitivity comparison for dh/dAlk...")
plot_sensitivity_comparison(masked_sensitivities_fixed_dalk, masked_sensitivities_moving_dalk, "dh/dAlk", lat_bounds, lon_bounds)

# Plot comparison for dh/dTem
print("Plotting sensitivity comparison for dh/dTem...")
plot_sensitivity_comparison(masked_sensitivities_fixed_dtem, masked_sensitivities_moving_dtem, "dh/dTem", lat_bounds, lon_bounds)

# Plot comparison for dh/dAlk
print("Plotting sensitivity comparison for dh/dSal...")
plot_sensitivity_comparison(masked_sensitivities_fixed_dsal, masked_sensitivities_moving_dsal, "dh/dSal", lat_bounds, lon_bounds)

# Plot comparison for dh/dAlk
print("Plotting sensitivity comparison for dh/dDIC...")
plot_sensitivity_comparison(masked_sensitivities_fixed_ddic, masked_sensitivities_moving_ddic, "dh/dDIC", lat_bounds, lon_bounds)


# %%

#################### Plotting driver contributions ####################

#%%

# Plotting contribution of temperature to H+ changes

# Define the bounds for the California Current System (CaCS)
lat_bounds = (30, 47)
lon_bounds = (225, 245)

# Define the data to plot
data_to_plot = tem_contribution_fixed.mean(dim="time")  # Averaging over time

# Load the landmask 
landmask_etopo = PlotFuncs.get_etopo_data()  

# Dynamically set the colorbar limits based on the data, ensuring zero is centered
vmin_tem = -max(abs(np.nanmin(data_to_plot.values)), abs(np.nanmax(data_to_plot.values)))
vmax_tem = max(abs(np.nanmin(data_to_plot.values)), abs(np.nanmax(data_to_plot.values)))

vmin_tem = -1e-10
vmax_tem = 1e-10

# Define specific ticks for the colorbar
colorbar_ticks = np.arange(-1e-10, 1.1e-10, 0.1e-10)  # Custom tick values every 0.1e-10


# Plotting
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

# Add the landmask to the map
ax.contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

# Plot the data
contour = ax.contourf(
    data_to_plot["lon"],  # 2D longitude array
    data_to_plot["lat"],  # 2D latitude array
    data_to_plot,         # Data to plot
    levels=np.linspace(vmin_tem, vmax_tem, 40),  # Ensure the levels span symmetrically around zero
    cmap="cmo.balance",   # Colormap
    vmin=vmin_tem,            # Minimum value for the colorbar
    vmax=vmax_tem,            # Maximum value for the colorbar
    zorder=2
)

# Add colorbar
cbar = fig.colorbar(contour, ax=ax, orientation="vertical", label="Contribution", ticks=colorbar_ticks)


# Set lat/lon bounds
ax.set_xlim(lon_bounds)
ax.set_ylim(lat_bounds)

# Add labels and title
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Temperature Contribution (Fixed Baseline)", fontsize=14)

# Add degree symbols and N/W indicators to axes
yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)

ax.set_yticks(yticks)
ax.set_yticklabels([f"{lat}°N" for lat in yticks])

ax.set_xticks(xticks)
ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])  # Convert to west longitude

# Save the plot
savedir = '/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/'
filename = f'contributions_temperature_fixed_baseline_instant_sensitivities.png'
plt.savefig(savedir + filename, dpi=200, transparent=True)

# Show the plot
plt.show()

# %%

# Plotting contribution of salinity to H+ changes
# Define the bounds for the California Current System (CaCS)
lat_bounds = (30, 47)
lon_bounds = (225, 245)

# Define the data to plot
data_to_plot = sal_contribution_fixed.mean(dim="time")  # Averaging over time

# Dynamically set the colorbar limits based on the data, ensuring zero is centered
vmin_sal = -max(abs(np.nanmin(data_to_plot.values)), abs(np.nanmax(data_to_plot.values)))
vmax_sal = max(abs(np.nanmin(data_to_plot.values)), abs(np.nanmax(data_to_plot.values)))

vmin_sal = -1e-11
vmax_sal = 1e-11

# Define specific ticks for the colorbar
colorbar_ticks = np.arange(-1e-11, 1.1e-11, 0.1e-11)  # Custom tick values every 0.1e-10

# Load the landmask 
landmask_etopo = PlotFuncs.get_etopo_data()  

# Plotting
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

# Add the landmask to the map
ax.contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

# Plot the data
contour = ax.contourf(
    data_to_plot["lon"],  # 2D longitude array
    data_to_plot["lat"],  # 2D latitude array
    data_to_plot,         # Data to plot
    levels=np.linspace(vmin_sal, vmax_sal, 40),            # Number of levels in the colorbar
    cmap="cmo.balance",   # Colormap
    vmin=vmin_sal,            # Minimum value for the colorbar
    vmax=vmax_sal,            # Maximum value for the colorbar
    zorder=2
)

# Add colorbar
cbar = fig.colorbar(contour, ax=ax, orientation="vertical", label="Contribution", ticks=colorbar_ticks)

# Set lat/lon bounds
ax.set_xlim(lon_bounds)
ax.set_ylim(lat_bounds)

# Add labels and title
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Salinity Contribution (Fixed Baseline)", fontsize=14)

# Add degree symbols and N/W indicators to axes
yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)

ax.set_yticks(yticks)
ax.set_yticklabels([f"{lat}°N" for lat in yticks])

ax.set_xticks(xticks)
ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])  # Convert to west longitude

# Add gridlines for better orientation
ax.grid(visible=True, linestyle="--", linewidth=0.5)

savedir = '/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/'
filename = f'contributions_salinity_fixed_baseline_instant_sensitivities.png'
plt.savefig(savedir+filename,dpi=200,transparent=True)

# Show the plot
plt.show()


# %%
# Plotting contribution of alkalinity to H+ changes
# Define the bounds for the California Current System (CaCS)
lat_bounds = (30, 47)
lon_bounds = (225, 245)

# Define the data to plot
data_to_plot = alk_contribution_fixed.mean(dim="time")  # Averaging over time

# Dynamically set the colorbar limits based on the data, ensuring zero is centered
vmin_alk = -max(abs(np.nanmin(data_to_plot.values)), abs(np.nanmax(data_to_plot.values)))
vmax_alk = max(abs(np.nanmin(data_to_plot.values)), abs(np.nanmax(data_to_plot.values)))

vmin_alk = -1e-09
vmax_alk = 1e-09

# Define specific ticks for the colorbar
colorbar_ticks = np.arange(-1e-09, 1.1e-09, 0.1e-09)  # Custom tick values every 0.1e-10


# Load the landmask 
landmask_etopo = PlotFuncs.get_etopo_data()  

# Plotting
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

# Add the landmask to the map
ax.contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

# Plot the data
contour = ax.contourf(
    data_to_plot["lon"],  # 2D longitude array
    data_to_plot["lat"],  # 2D latitude array
    data_to_plot,         # Data to plot
    levels=np.linspace(vmin_alk, vmax_alk, 40),            # Number of levels in the colorbar
    cmap="cmo.balance",   # Colormap
    vmin=vmin_alk,            # Minimum value for the colorbar
    vmax=vmax_alk,            # Maximum value for the colorbar
    zorder=2
)

# Add colorbar
cbar = fig.colorbar(contour, ax=ax, orientation="vertical", label="Contribution", ticks=colorbar_ticks)

# Set lat/lon bounds
ax.set_xlim(lon_bounds)
ax.set_ylim(lat_bounds)

# Add labels and title
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Alkalinity Contribution (Fixed Baseline)", fontsize=14)

# Add degree symbols and N/W indicators to axes
yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)

ax.set_yticks(yticks)
ax.set_yticklabels([f"{lat}°N" for lat in yticks])

ax.set_xticks(xticks)
ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])  # Convert to west longitude

# Add gridlines for better orientation
ax.grid(visible=True, linestyle="--", linewidth=0.5)

savedir = '/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/'
filename = f'contributions_alkalinity_fixed_baseline_instant_sensitivities.png'
plt.savefig(savedir+filename,dpi=200,transparent=True)

# Show the plot
plt.show()

# %%

# Plotting contribution of DIC to H+ changes

# Define the bounds for the California Current System (CaCS)
lat_bounds = (30, 47)
lon_bounds = (225, 245)

# Define the data to plot
data_to_plot = dic_contribution_fixed.mean(dim="time")  # Averaging over time

# Dynamically set the colorbar limits based on the data, ensuring zero is centered
vmin_dic = -max(abs(np.nanmin(data_to_plot.values)), abs(np.nanmax(data_to_plot.values)))
vmax_dic = max(abs(np.nanmin(data_to_plot.values)), abs(np.nanmax(data_to_plot.values)))

vmin_dic = -1e-09
vmax_dic = 1e-09

# Define specific ticks for the colorbar
colorbar_ticks = np.arange(-1e-09, 1.1e-09, 0.1e-09)  # Custom tick values every 0.1e-10


# Load the landmask 
landmask_etopo = PlotFuncs.get_etopo_data()  

# Plotting
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

# Add the landmask to the map
ax.contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

# Plot the data
contour = ax.contourf(
    data_to_plot["lon"],  # 2D longitude array
    data_to_plot["lat"],  # 2D latitude array
    data_to_plot,         # Data to plot
    levels=np.linspace(vmin_dic, vmax_dic, 40),  # Number of levels in the colorbar
    cmap="cmo.balance",   # Colormap
    vmin=vmin_dic,            # Minimum value for the colorbar
    vmax=vmax_dic,            # Maximum value for the colorbar
    zorder=2
)

# Add colorbar
cbar = fig.colorbar(contour, ax=ax, orientation="vertical", label="Contribution", ticks=colorbar_ticks)

# Set lat/lon bounds
ax.set_xlim(lon_bounds)
ax.set_ylim(lat_bounds)

# Add labels and title
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("DIC Contribution (Fixed Baseline)", fontsize=14)

# Add degree symbols and N/W indicators to axes
yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)

ax.set_yticks(yticks)
ax.set_yticklabels([f"{lat}°N" for lat in yticks])

ax.set_xticks(xticks)
ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])  # Convert to west longitude

# Add gridlines for better orientation
ax.grid(visible=True, linestyle="--", linewidth=0.5)

savedir = '/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/'
filename = f'contributions_DIC_fixed_baseline_instant_sensitivities.png'
plt.savefig(savedir+filename,dpi=200,transparent=True)

# Show the plot
plt.show()

# %%

### Plotting fixed vs moving baselines

# Plotting contributions for fixed and moving baselines of contributions of temperature to H+ changes

# Define the bounds for the California Current System (CaCS)
lat_bounds = (30, 47)
lon_bounds = (225, 245)

# Define the data to plot
data_fixed = tem_contribution_fixed.mean(dim="time")  # Averaging over time for fixed baseline
data_moving = tem_contribution_moving.mean(dim="time")  # Averaging over time for moving baseline

# Load the landmask
landmask_etopo = PlotFuncs.get_etopo_data()

# Dynamically set the colorbar limits based on the data, ensuring zero is centered
vmin_tem = -1e-10
vmax_tem = 1e-10

# Define specific ticks for the colorbar
colorbar_ticks = np.linspace(vmin_tem, vmax_tem, 11)  # Custom tick values

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

# Add landmask and plot data for fixed baseline
axs[0].contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

contour_fixed = axs[0].contourf(
    data_fixed["lon"],  # 2D longitude array
    data_fixed["lat"],  # 2D latitude array
    data_fixed,         # Data to plot
    levels=np.linspace(vmin_tem, vmax_tem, 40),  # Symmetrically span around zero
    cmap="cmo.balance",  # Colormap
    vmin=vmin_tem,       # Minimum value for the colorbar
    vmax=vmax_tem,       # Maximum value for the colorbar
    zorder=2
)
axs[0].set_xlim(lon_bounds)
axs[0].set_ylim(lat_bounds)
axs[0].set_title("Fixed baseline", fontsize=14)
axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Latitude")

# Add degree symbols and N/W indicators to axes for fixed baseline
yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
axs[0].set_yticks(yticks)
axs[0].set_yticklabels([f"{lat}°N" for lat in yticks])
axs[0].set_xticks(xticks)
axs[0].set_xticklabels([f"{360 - lon}°W" for lon in xticks])

# Add landmask and plot data for moving baseline
axs[1].contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

contour_moving = axs[1].contourf(
    data_moving["lon"],  # 2D longitude array
    data_moving["lat"],  # 2D latitude array
    data_moving,         # Data to plot
    levels=np.linspace(vmin_tem, vmax_tem, 40),  # Symmetrically span around zero
    cmap="cmo.balance",  # Colormap
    vmin=vmin_tem,       # Minimum value for the colorbar
    vmax=vmax_tem,       # Maximum value for the colorbar
    zorder=2
)
axs[1].set_xlim(lon_bounds)
axs[1].set_ylim(lat_bounds)
axs[1].set_title("Moving baseline", fontsize=14)
axs[1].set_xlabel("Longitude")

# Add degree symbols and N/W indicators to axes for moving baseline
axs[1].set_yticks(yticks)
axs[1].set_yticklabels([f"{lat}°N" for lat in yticks])
axs[1].set_xticks(xticks)
axs[1].set_xticklabels([f"{360 - lon}°W" for lon in xticks])

# Add colorbar for both plots
cbar = fig.colorbar(contour_fixed, ax=axs, orientation="vertical", label="Contribution of temperature", ticks=colorbar_ticks)

# Add an overall title for the figure
plt.suptitle("Comparison of temperature contributions to H+ changes during extremes (fixed vs. moving baseline)", fontsize=20, y=1.1)

# Save the plot
savedir = '/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/'
filename = f'contributions_temperature_fixed_vs_moving_baseline_instant_sens_ssp585.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')

# Show the plot
plt.show()

# Plotting contributions for fixed and moving baselines of contributions of DIC to H+ changes

# Define the data to plot
data_fixed = dic_contribution_fixed.mean(dim="time")  # Averaging over time for fixed baseline
data_moving = dic_contribution_moving.mean(dim="time")  # Averaging over time for moving baseline

# Dynamically set the colorbar limits based on the data, ensuring zero is centered
vmin_dic = -1e-09
vmax_dic = 1e-09

# Define specific ticks for the colorbar
colorbar_ticks = np.linspace(vmin_dic, vmax_dic, 11)  # Custom tick values

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

# Add landmask and plot data for fixed baseline
axs[0].contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

contour_fixed = axs[0].contourf(
    data_fixed["lon"],  # 2D longitude array
    data_fixed["lat"],  # 2D latitude array
    data_fixed,         # Data to plot
    levels=np.linspace(vmin_dic, vmax_dic, 40),  # Symmetrically span around zero
    cmap="cmo.balance",  # Colormap
    vmin=vmin_dic,       # Minimum value for the colorbar
    vmax=vmax_dic,       # Maximum value for the colorbar
    zorder=2
)
axs[0].set_xlim(lon_bounds)
axs[0].set_ylim(lat_bounds)
axs[0].set_title("Fixed baseline", fontsize=14)
axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Latitude")

# Add degree symbols and N/W indicators to axes for fixed baseline
yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
axs[0].set_yticks(yticks)
axs[0].set_yticklabels([f"{lat}°N" for lat in yticks])
axs[0].set_xticks(xticks)
axs[0].set_xticklabels([f"{360 - lon}°W" for lon in xticks])

# Add landmask and plot data for moving baseline
axs[1].contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

contour_moving = axs[1].contourf(
    data_moving["lon"],  # 2D longitude array
    data_moving["lat"],  # 2D latitude array
    data_moving,         # Data to plot
    levels=np.linspace(vmin_dic, vmax_dic, 40),  # Symmetrically span around zero
    cmap="cmo.balance",  # Colormap
    vmin=vmin_dic,       # Minimum value for the colorbar
    vmax=vmax_dic,       # Maximum value for the colorbar
    zorder=2
)
axs[1].set_xlim(lon_bounds)
axs[1].set_ylim(lat_bounds)
axs[1].set_title("Moving baseline", fontsize=14)
axs[1].set_xlabel("Longitude")

# Add degree symbols and N/W indicators to axes for moving baseline
axs[1].set_yticks(yticks)
axs[1].set_yticklabels([f"{lat}°N" for lat in yticks])
axs[1].set_xticks(xticks)
axs[1].set_xticklabels([f"{360 - lon}°W" for lon in xticks])

# Add colorbar for both plots
cbar = fig.colorbar(contour_fixed, ax=axs, orientation="vertical", label="Contribution of DIC", ticks=colorbar_ticks)

# Add an overall title for the figure
plt.suptitle("Comparison of DIC contributions to H+ changes during extremes (fixed vs. moving baseline)", fontsize=20, y=1.1)

# Save the plot
savedir = '/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/'
filename = f'contributions_DIC_fixed_vs_moving_baseline_instant_sens_ssp585.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')

# Show the plot
plt.show()

# Plotting contributions for fixed and moving baselines of contributions of Alkalinity to H+ changes

# Define the data to plot
data_fixed = alk_contribution_fixed.mean(dim="time")  # Averaging over time for fixed baseline
data_moving = alk_contribution_moving.mean(dim="time")  # Averaging over time for moving baseline

# Dynamically set the colorbar limits based on the data, ensuring zero is centered
vmin_alk = -1e-09
vmax_alk = 1e-09

# Define specific ticks for the colorbar
colorbar_ticks = np.linspace(vmin_alk, vmax_alk, 11)  # Custom tick values

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

# Add landmask and plot data for fixed baseline
axs[0].contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

contour_fixed = axs[0].contourf(
    data_fixed["lon"],  # 2D longitude array
    data_fixed["lat"],  # 2D latitude array
    data_fixed,         # Data to plot
    levels=np.linspace(vmin_alk, vmax_alk, 40),  # Symmetrically span around zero
    cmap="cmo.balance",  # Colormap
    vmin=vmin_alk,       # Minimum value for the colorbar
    vmax=vmax_alk,       # Maximum value for the colorbar
    zorder=2
)
axs[0].set_xlim(lon_bounds)
axs[0].set_ylim(lat_bounds)
axs[0].set_title("Fixed baseline", fontsize=14)
axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Latitude")

# Add degree symbols and N/W indicators to axes for fixed baseline
yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
axs[0].set_yticks(yticks)
axs[0].set_yticklabels([f"{lat}°N" for lat in yticks])
axs[0].set_xticks(xticks)
axs[0].set_xticklabels([f"{360 - lon}°W" for lon in xticks])

# Add landmask and plot data for moving baseline
axs[1].contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

contour_moving = axs[1].contourf(
    data_moving["lon"],  # 2D longitude array
    data_moving["lat"],  # 2D latitude array
    data_moving,         # Data to plot
    levels=np.linspace(vmin_alk, vmax_alk, 40),  # Symmetrically span around zero
    cmap="cmo.balance",  # Colormap
    vmin=vmin_alk,       # Minimum value for the colorbar
    vmax=vmax_alk,       # Maximum value for the colorbar
    zorder=2
)
axs[1].set_xlim(lon_bounds)
axs[1].set_ylim(lat_bounds)
axs[1].set_title("Moving baseline", fontsize=14)
axs[1].set_xlabel("Longitude")

# Add degree symbols and N/W indicators to axes for moving baseline
axs[1].set_yticks(yticks)
axs[1].set_yticklabels([f"{lat}°N" for lat in yticks])
axs[1].set_xticks(xticks)
axs[1].set_xticklabels([f"{360 - lon}°W" for lon in xticks])

# Add colorbar for both plots
cbar = fig.colorbar(contour_fixed, ax=axs, orientation="vertical", label="Contribution of Alkalinity", ticks=colorbar_ticks)

# Add an overall title for the figure
plt.suptitle("Comparison of Alkalinity contributions to H+ changes during extremes (fixed vs. moving baseline)", fontsize=20, y=1.1)

# Save the plot
savedir = '/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/'
filename = f'contributions_Alkalinity_fixed_vs_moving_baseline_instant_sens_ssp585.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')

# Show the plot
plt.show()

# Plotting contributions for fixed and moving baselines of contributions of Salinity to H+ changes

# Define the data to plot
data_fixed = sal_contribution_fixed.mean(dim="time")  # Averaging over time for fixed baseline
data_moving = sal_contribution_moving.mean(dim="time")  # Averaging over time for moving baseline

# Dynamically set the colorbar limits based on the data, ensuring zero is centered
vmin_sal = -1e-11
vmax_sal = 1e-11

# Define specific ticks for the colorbar
colorbar_ticks = np.linspace(vmin_sal, vmax_sal, 11)  # Custom tick values

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

# Add landmask and plot data for fixed baseline
axs[0].contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

contour_fixed = axs[0].contourf(
    data_fixed["lon"],  # 2D longitude array
    data_fixed["lat"],  # 2D latitude array
    data_fixed,         # Data to plot
    levels=np.linspace(vmin_sal, vmax_sal, 40),  # Symmetrically span around zero
    cmap="cmo.balance",  # Colormap
    vmin=vmin_sal,       # Minimum value for the colorbar
    vmax=vmax_sal,       # Maximum value for the colorbar
    zorder=2
)
axs[0].set_xlim(lon_bounds)
axs[0].set_ylim(lat_bounds)
axs[0].set_title("Fixed baseline", fontsize=14)
axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Latitude")

# Add degree symbols and N/W indicators to axes for fixed baseline
yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
axs[0].set_yticks(yticks)
axs[0].set_yticklabels([f"{lat}°N" for lat in yticks])
axs[0].set_xticks(xticks)
axs[0].set_xticklabels([f"{360 - lon}°W" for lon in xticks])

# Add landmask and plot data for moving baseline
axs[1].contourf(
    landmask_etopo.lon,
    landmask_etopo.lat,
    landmask_etopo,
    levels=1,
    colors='black',
    zorder=1
)

contour_moving = axs[1].contourf(
    data_moving["lon"],  # 2D longitude array
    data_moving["lat"],  # 2D latitude array
    data_moving,         # Data to plot
    levels=np.linspace(vmin_sal, vmax_sal, 40),  # Symmetrically span around zero
    cmap="cmo.balance",  # Colormap
    vmin=vmin_sal,       # Minimum value for the colorbar
    vmax=vmax_sal,       # Maximum value for the colorbar
    zorder=2
)
axs[1].set_xlim(lon_bounds)
axs[1].set_ylim(lat_bounds)
axs[1].set_title("Moving baseline", fontsize=14)
axs[1].set_xlabel("Longitude")

# Add degree symbols and N/W indicators to axes for moving baseline
axs[1].set_yticks(yticks)
axs[1].set_yticklabels([f"{lat}°N" for lat in yticks])
axs[1].set_xticks(xticks)
axs[1].set_xticklabels([f"{360 - lon}°W" for lon in xticks])

# Add colorbar for both plots
cbar = fig.colorbar(contour_fixed, ax=axs, orientation="vertical", label="Contribution of Salinity", ticks=colorbar_ticks)

# Add an overall title for the figure
plt.suptitle("Comparison of Salinity contributions to H+ changes during extremes (fixed vs. moving baseline)", fontsize=20, y=1.1)

# Save the plot
savedir = '/nfs/sea/work/fpfaeffli/plots/driver_analysis/contributions/'
filename = f'contributions_Salinity_fixed_vs_moving_baseline_instant_sens_ssp585.png'
plt.savefig(savedir + filename, dpi=200, transparent=True, bbox_inches='tight')

# Show the plot
plt.show()


# %%
