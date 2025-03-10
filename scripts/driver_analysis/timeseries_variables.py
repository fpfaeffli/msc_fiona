"""
author: Fiona Pfäffli
description: This file plots present day vs future extremes for the different thresholds
"""

#%% enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')


#%%
# load the package
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

#%%
# Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present','ssp585', 'ssp245'] # ,'ssp245'
configs = ['romsoc_fully_coupled'] # [ms_only'] 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'avg'    # 'avg'
depth = 0

#%% 
# Get the model datasets for the oceanic variables

ocean_ds_raw = dict()
for config in configs:
     ocean_ds_raw[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds_raw[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)

#%% 
# load the data at the respective location

variables = dict()
for config in configs:
    variables[config] = dict()
    for scenario in scenarios:
        print(f'Getting the variables for config {config} and scenario {scenario}.')
        
        # Initialize dictionary structure
        variables[config][scenario] = dict()

        # Check if the dataset exists
        if config in ocean_ds_raw and scenario in ocean_ds_raw[config]:
            print("Loading total CHL")
            variables[config][scenario]["TOT_CHL"] = ocean_ds_raw[config][scenario]["TOT_CHL"].isel(s_rho=0).load()
            #print("Loading total production")
            #variables[config][scenario]["TOT_PROD"] = ocean_ds_raw[config][scenario]["TOT_PROD"].isel(s_rho=0).load()
          
#%%
# Plotting

def plot_weighted_timeseries(var_name, model_da, model_area, plot_resolution='monthly', savefig=False):
    """
    Plots a weighted time series for a given variable from the ROMS model dataset.

    """
    print(f"Processing variable: {var_name}")
    
    # Extract the variable and apply weighted averaging
    model_var = model_da[var_name].isel(s_rho=0)  # Extract surface level
    weighted_mean = model_var.weighted(model_area.fillna(0)).mean(dim=('eta_rho', 'xi_rho'))
    
    # Resample to desired resolution
    if plot_resolution == 'monthly' and pd.infer_freq(weighted_mean.time) == 'D':
        weighted_mean = weighted_mean.resample(time='1M').mean(dim='time')
    
    # Plot the time series
    plt.figure(figsize=(12, 5))
    plt.plot(weighted_mean.time, weighted_mean, label=f"{var_name} (Weighted Avg)", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel(f"{var_name} (units)")
    plt.title(f"Time Series of {var_name} (Weighted Average)")
    plt.legend()
    plt.grid()
    
    if savefig:
        plt.savefig(f"{var_name}_timeseries.png", dpi=200, transparent=True)
    
    plt.show()
    return weighted_mean

# Example usage (assuming you have loaded the datasets properly)
plot_weighted_timeseries("TOT_CHL", ocean_ds_raw['romsoc_fully_coupled']['present'], model_area)
plot_weighted_timeseries("TOT_PROD", ocean_ds_raw['romsoc_fully_coupled']['present'], model_area)

