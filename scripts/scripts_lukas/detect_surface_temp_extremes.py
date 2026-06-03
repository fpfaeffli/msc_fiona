
"""
author: Fiona Pfäffli
description: Calculates the extreme temperature days per year for the present scenario.
"""

#%% enable the visibility of the modules for the import functions
import sys
import os

# Add the paths to the modules
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')



# load the packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
import pprint
from get_study_regions import GetRegions as GetRegions
from get_model_datasets_eike import ModelGetter as ModelGetter 
from matplotlib import colors as mcolors
from importlib import reload  # Python 3.4+
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates


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


# Load landmask
landmask_etopo = AuxGetter.get_etopo_data()


#%% 

# Define the threshold (here 90th percentile) and climatology parameters
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.standard_instance() #90th percentile threshold, which is used for temperature extremes

# Defining variables
model_temp_resolution = 'daily' # 'monthly': you should use daily resolution 
scenarios = ['present'] # 'ssp245', 'ssp585': You only use the present scenario, so you can ignore the two future scenarios (ssp245, ssp585)
configs = ['romsoc_fully_coupled'] # ['roms_only']  # You only use the romsoc_fully_coupled configuration
simulation_type = 'hindcast' 
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000' #'001'
vert_struct = 'zavg'    # 'avg'
depth = 0 #you only detect surface temperature extremes, so depth = 0
varias = ['temp'] #variables to analyse. Here only temperature

#%% 

################ Loading all the data ##################

# Get the model datasets for the oceanic variables
# This part of the code loads in the model output into a dictionary, which is called ocean_ds
# The dictionary has the structure ocean_ds[config][scenario], where config is the model configuration (here only 'romsoc_fully_coupled') 
# and scenario is the scenario (here only 'present')

# You load the data in with using the get_model_dataset function from the ModelGetter class in the get_model_datasets module (path: /home/fpfaeffli/msc_fiona/scripts/modules/get_model_datasets.py)

ocean_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)

#%%
# load the data at the respective location

# This function loads in the variable you want to analyse (here: temperature) into a dictionary
# Don't be confused if it takes a few minutes to load the data.
# If you wanna check, use "htop" in the terminal to see how much memory is used on the server sea. If a lot of people are using the server, it can take a while to load in the data.


def load_data_in(varia):
    vdict = dict()
    for config in configs:
        vdict[config] = dict()
        for scenario in scenarios:
            print(f'Getting the variable for config {config} and scenario {scenario}.')
            print(varia)
            if varia == 'temp':
                vdict[config][scenario] = ocean_ds[config][scenario].temp.isel(depth=0).load() #This loads in the surface temperature (depth=0) data from the ocean_ds (=the model output) into a dictionary
            
    return vdict

temp = load_data_in('temp') #This is the execution of the function to load in the surface temperature data into a dictionary called temp


#%% LOAD in all the climatologies
# This loads in the (already calculated) climatologies for the coupled configuration and the present day. Its now Eike's path because he calculated the climatologies for temperature

def load_clim_in(varia):
    cdict = dict()

    for config in configs:
     cdict[config] = dict()
     for scenario in scenarios:
          cdict[config][scenario] = dict()
          for varia in varias:
               print(f'Calculate the climatology for: {config}, {scenario}, {varia}.')
               dummy_clim = ThreshClimFuncs.calc_clim(params,temp[config][scenario])
               smoothed_clim = ThreshClimFuncs.smooth_array(params,dummy_clim)
               smoothed_clim = smoothed_clim.rename({'day_of_year_adjusted': 'time'})
               smoothed_clim = smoothed_clim.assign_coords({'time': pd.date_range('2015-01-01','2015-12-31')})
               cdict[config][scenario]= ThreshClimFuncs.repeat_array_multiple_years(smoothed_clim)
    return cdict

temp_clim = load_clim_in('temp')



#%% LOAD in the threshold for Hplus
# This loads in the (already calculated) thresholds for the coupled configuration and the present day. 

print("Loading the threshold for temperature...")
def load_thresh_in(varia, thresh_value):
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
    for scenario in ['present']:
        thresholds_mult[config][scenario] = dict()
        print(f'{config}, {scenario}')
        thresholds_mult[config][scenario]['present'] = concatenated_threshold
        
    return thresholds_mult

thresholds_mult = load_thresh_in('temp', 90)    #90th percentile threshold for temperature


# %% COMPUTE THE THRESHOLD EXCEEDANCES (EXTREMES MASK) for temperature for whole area

# This function calculates the extreme masks for the different scenarios and configurations. It looks if the temperature at a certain day is above the threshold for that day.
# The output is a dictionary extreme_mask_dict[config][scenario]['present']. Only present threshold, as you only use the present scenario.
# This mask includes then all the days over the whole area which are above the threshold for that day (meaning, are extreme)!

print("Calculating the extreme masks for temperature...")
def calc_extreme_masks(inst_dict, thresh_dict):
    extreme_mask_dict = dict()
    for config in configs:
        extreme_mask_dict[config] = dict()
        for scenario in scenarios:
            extreme_mask_dict[config][scenario] = dict()
            for threshold_type in ['present']:
                extreme_mask_dict[config][scenario][threshold_type] = inst_dict[config][scenario] > thresh_dict[config][scenario][threshold_type]
    return extreme_mask_dict

extreme_mask_dict = calc_extreme_masks(temp, thresholds_mult)


#%%
# In the following there are a few basic plotting functions, which you can use to visualise the extremes on a map. 
# You can play around with the lat / lon, depending on which region you want to look at.

