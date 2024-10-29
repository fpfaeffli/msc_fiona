"""
Author: Fiona Pfäffli
Date: 23.10.2024
Description: Plotting of climatologies and thresholds of Hplus.
"""

#%% Load packages
import sys
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Enable visibility of custom modules
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')


# Paths to the datasets
base_path = '/nfs/sea/work/fpfaeffli/future_sim/thresholds_and_climatologies'
plot_path = '/nfs/sea/work/fpfaeffli/plots/climatologies_and_thresholds'

#%% Functions 
def load_dataset(scenario, model_type, period):
    """
    Parameters:
    scenario: present, ssp245, ssp585
    model_type (str): roms_only, romsoc_fully_coupled
    """
    nc_file = f'{base_path}/{model_type}/{scenario}/hobday2016_threshold_and_climatology_Hplus_95.0perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing_0depthlevelindex.nc'
    return xr.open_dataset(nc_file)


def plot_variable(ds, var_name, model_type, scenario, var_type):
    """
    Parameters:
    ds (xarray.Dataset): dataset containing the variable.
    var_name: Name of the variable to plot 
    var_type: climatology or threshold
    """
    ds[var_name].plot()
    title = f'smoothed {var_type} {model_type} {scenario}'
    plt.title(title)
    save_path = f'{plot_path}/{model_type}/{var_type}_{model_type}_{scenario}.png'
    plt.savefig(save_path)
    plt.show()
    plt.xlim(left=0.25)
    plt.close()


def process_scenario(scenario, model_type):
    """
    Load dataset for a given scenario and model type, then plot climatology and threshold.
    """
    ds = load_dataset(scenario, model_type, period='2011-2021')
    
    var_clim = 'clim_smoothed'
    var_thresh = 'thresh_smoothed'
    
    # Plot and save climatology
    plot_variable(ds, var_clim, model_type, scenario, var_type='climatology')
    
    # Plot and save threshold
    plot_variable(ds, var_thresh, model_type, scenario, var_type='threshold')

#%% Main

# List of scenarios and models to process
scenarios = ['present', 'ssp245', 'ssp585']
model_types = ['roms_only', 'romsoc_fully_coupled']

# Process each scenario for each model type
for model_type in model_types:
    for scenario in scenarios:
        process_scenario(scenario, model_type)

#%%
