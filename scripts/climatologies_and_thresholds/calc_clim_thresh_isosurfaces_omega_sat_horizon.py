"""
author: Fiona Pfäffli
description: This file calculates the climatologies and thresholds for the omega aragonite saturation horizon depth for the different climate scenarios and model configurations.
"""

# %% enable the visibility of the modules for the import functions
import sys
import os

sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

# %%
# Load the necessary packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
from statsmodels.tsa.stattools import acf
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter
from get_model_datasets import ModelCarbonateChemistryGetter as ModelCarbonateChemistryGetter
from get_obs_datasets import ObsGetter as ObsGetter

import multiprocessing
from tqdm import tqdm

from importlib import reload  # Python 3.4+

import matplotlib.patheffects as pe

import get_obs_datasets 
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter

from func_for_clim_thresh import ThreshClimFuncs
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs

import xesmf as xe
from datetime import date

from plotting_functions_general import PlotFuncs as PlotFuncs

# %%
# Define the threshold 
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.omega_arag_instance()  # 5th percentile threshold

# %%
# Defining variables
model_temp_resolution = 'daily'  # 'monthly'
scenarios = ['present', 'ssp245', 'ssp585']  # Different climate scenarios
configs = ['romsoc_fully_coupled']  # Model configuration
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'avg'  # Vertical structure: 'avg'

# %%
# Variables
varias = ['hbls']

# %%
# Get the model datasets for the oceanic variables

ocean_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)

# %%
# Load the data at the respective location

variables = dict()
for config in configs:
    variables[config] = dict()
    for scenario in scenarios:
        print(f'Getting the variables for config {config} and scenario {scenario}.')
        variables[config][scenario] = dict()
        for var in varias:
            try:
                print(f'Loading variable: {var} for scneario {scenario} and config {config}.')
                variables[config][scenario][var] = ocean_ds[config][scenario][var].load()
                print(f"Variable '{var}' loaded successfully.")
            except KeyError:
                # If the variable is not found in the dataset
                print(f"Variable '{var}' not found in the dataset for config '{config}' and scenario '{scenario}'.")
            

#%% 
############################ CALCULATING THE CLIMATOLOGIES AND THRESHOLDS ############################
# Calculating the climatology for each variable
print('Get the climatology')
climatology = dict()
for config in configs:
     climatology[config] = dict()
     for scenario in scenarios:
          climatology[config][scenario] = dict()
          for varia in varias:
               print(f'Calculate the climatology for: {config}, {scenario}, {varia}.')
               dummy_clim = ThreshClimFuncs.calc_clim(params,variables[config][scenario][varia])
               smoothed_clim = ThreshClimFuncs.smooth_array(params,dummy_clim)
               smoothed_clim = smoothed_clim.rename({'day_of_year_adjusted': 'time'})
               smoothed_clim = smoothed_clim.assign_coords({'time': pd.date_range('2015-01-01','2015-12-31')})
               climatology[config][scenario][varia] = ThreshClimFuncs.repeat_array_multiple_years(smoothed_clim)


#%% 
# Get the threshold for each variable
print('Get the present day threshold')
threshold = dict()
for config in configs:
    threshold[config] = dict()
    for scenario in ['present']:
        threshold[config][scenario] = dict()
        for varia in varias:
            print(f'Calculating the threshold for: {config}, {scenario}, {varia}')
            # Calculate the raw threshold using the calc_thresh function
            dummy_thresh = ThreshClimFuncs.calc_thresh(params, variables[config][scenario][varia])
            
            # Smooth the threshold
            smoothed_thresh = ThreshClimFuncs.smooth_array(params, dummy_thresh)
            
            # Rename and assign time coordinates to match the standard year structure
            smoothed_thresh = smoothed_thresh.rename({'day_of_year_adjusted': 'time'})
            smoothed_thresh = smoothed_thresh.assign_coords({'time': pd.date_range('2015-01-01', '2015-12-31')})
            
            # Repeat the smoothed threshold across the required year range
            threshold[config][scenario][varia] = ThreshClimFuncs.repeat_array_multiple_years(smoothed_thresh)
            
            print(f"Threshold for variable '{varia}' has been successfully calculated and stored.")


#%% Do the intensity normalizer calculations
print('Calc the intensity normalizer')

intensity_normalizer = {}
for config in configs:
    intensity_normalizer[config] = {}
    for scenario in ['present']:
        intensity_normalizer[config][scenario] = {}
        for varia in varias:
            print(f'Calculating intensity normalizer for: {config}, {scenario}, {varia}')
            intensity_normalizer[config][scenario][varia] = ThreshClimFuncs.calc_intensity_normalizer(
                threshold[config][scenario][varia], 
                climatology[config][scenario][varia]
            )

def calc_intensity_normalizer(threshold_array, climatology_array):
    return threshold_array - climatology_array

#%% smoothing
print('Smoothing')

print('Smoothing')

# Initialize dictionaries to store the smoothed results
climatology_smoothed = {}
threshold_smoothed = {}

for config in configs:
    climatology_smoothed[config] = {}
    threshold_smoothed[config] = {}

    for scenario in scenarios:
        climatology_smoothed[config][scenario] = {}

        if scenario == "present":  # Only initialize threshold for 'present'
            threshold_smoothed[config][scenario] = {}

        for varia in varias:
            print(f'Smoothing for {config}, {scenario}, {varia}')

            # Extract individual DataArray for climatology
            da_clim = climatology[config][scenario][varia]

            # Rename 'time' to 'day_of_year_adjusted' if necessary
            if 'time' in da_clim.dims and 'day_of_year_adjusted' not in da_clim.dims:
                da_clim = da_clim.rename({'time': 'day_of_year_adjusted'})

            # Apply smoothing to climatology
            climatology_smoothed[config][scenario][varia] = ThreshClimFuncs.smooth_array(params, da_clim)

            # Only process threshold for 'present' scenario
            if scenario == "present":
                da_thresh = threshold[config][scenario][varia]

                if 'time' in da_thresh.dims and 'day_of_year_adjusted' not in da_thresh.dims:
                    da_thresh = da_thresh.rename({'time': 'day_of_year_adjusted'})

                threshold_smoothed[config][scenario][varia] = ThreshClimFuncs.smooth_array(params, da_thresh)

# Now compute intensity normalizer using the smoothed data 
intensity_normalizer_smoothed = {}

for config in configs:
    intensity_normalizer_smoothed[config] = {}

    for scenario in ['present']:  
        intensity_normalizer_smoothed[config][scenario] = {}

        for varia in varias:
            intensity_normalizer_smoothed[config][scenario][varia] = ThreshClimFuncs.calc_intensity_normalizer(
                threshold_smoothed[config][scenario][varia],
                climatology_smoothed[config][scenario][varia]
            )


#%% 
####################### SAVE THE CLIMATOLOGIES AND THRESHOLDS TO NETCDF FILES #######################
# Define a function to put the smoothed fields into a dataset

def put_fields_into_dataset(params, climatology_smoothed, threshold_smoothed, intensity_normalizer_smoothed):
    """
    Convert nested dictionaries into an xarray Dataset and add attributes.
    """
    out_ds = xr.Dataset()

    # Convert nested dictionary to Dataset
    def dict_to_dataset(nested_dict, var_name):
        data_vars = {}
        for config, scenarios in nested_dict.items():
            for scenario, variables in scenarios.items():
                for varia, da in variables.items():
                    data_vars[f"{var_name}_{config}_{scenario}_{varia}"] = da
        return data_vars

    # Merge all fields into a single dataset
    out_ds = xr.Dataset(
        {
            **dict_to_dataset(climatology_smoothed, "clim_smoothed"),
            **dict_to_dataset(threshold_smoothed, "thresh_smoothed"),
            **dict_to_dataset(intensity_normalizer_smoothed, "intensity_normalizer_smoothed"),
        }
    )

    # Add metadata attributes
    out_ds.attrs['case_description'] = 'Given by the following attributes:'
    for pattribute in dir(params):
        if not pattribute.startswith('__') and not callable(getattr(params, pattribute)):
            if pattribute != 'param_names':
                print(pattribute)
                out_ds.attrs['params.' + pattribute] = getattr(params, pattribute)

    out_ds.attrs['leap_years'] = 'Feb 29th values were discarded for the computation of the threshold and climatologies. They are thus 365 days long. In order to get a 366th value for leap years, I propose to linearly interpolate the values between Feb 28 and Mar 1.'

    return out_ds

# Put smoothed fields into dataset
print('Put smoothed fields into dataset')
out_ds = put_fields_into_dataset(params, climatology_smoothed, threshold_smoothed, intensity_normalizer_smoothed)

# Add metadata attributes
out_ds.attrs['author'] = 'Fiona Pfäffli'
out_ds.attrs['date'] = str(date.today())

# Print dataset to confirm its structure
print("Dataset successfully created:")
print(out_ds)

#%% ####################### SAVE TO NETCDF #######################

# Define save path
savepath = '/nfs/sea/work/fpfaeffli/future_sim/thresholds_and_climatologies/romsoc_fully_coupled/hobday2016_threshold_and_climatology_5th_percentile_isosurface_1.nc'

# Ensure directory exists before saving
os.makedirs(os.path.dirname(savepath), exist_ok=True)

# Print dataset structure before saving
print("Saving dataset (WITHOUT compression)...")
print(out_ds)

# Save the dataset WITHOUT compression (Fastest method)
out_ds.to_netcdf(savepath)

print(f"✅ Dataset successfully saved to: {savepath}")


