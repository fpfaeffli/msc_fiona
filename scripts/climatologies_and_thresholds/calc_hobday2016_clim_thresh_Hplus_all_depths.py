"""
Content: calculate the threshold for H+ ions according to Hobday et al. 2016 based on a 30 year reference period for all depth levels
Author: Eike E Koehn
Changes: Fiona Pfäffli, Oct 11, 2024
Adaptation: <Your Name>, <Date>
"""

#%% DEFINE THE SCRIPTNAME
import os
import sys
# enable the visibility of the modules for the import functions
scriptdir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/'
scriptname = 'calc_hobday2016_clim_thresh_Hplus_all_depths.py'
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')

#%% load packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.ndimage
from datetime import date
import glob

#import modules
from get_model_datasets import ModelGetter as ModelGetter
from set_thresh_and_clim_params import ThresholdParameters
from func_for_clim_thresh import ThreshClimFuncs

from importlib import reload  # Python 3.4+
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs as ThreshClimFuncs



#%% DIFFERENT SCENARIOS AND CONFIGURATIONS
var = 'Hplus'  # Change to your desired variable
config =  'roms_only'
scenario =  'present'
simulation_type =   'hindcast'
ensemble_run = '000'  
temp_resolution = 'daily'
vert_struct = 'zavg'
vtype = 'oceanic'

params = ThresholdParameters.fiona_instance() #Fiona's Instance = 95

#%% Get the model data
print('Getting model data...')
model_ds = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, temp_resolution, vert_struct, parent_model='mpi-esm1-2-hr', vtype=vtype)

# Get all depth levels available
depth_levels = model_ds['depth'].values

# Prepare a list to store datasets for each depth level
datasets = []

for depth_level_index, depth_level in enumerate(depth_levels):
    print(f"Processing depth level index {depth_level_index}, depth: {depth_level}m")

    if var == 'Hplus':
        var2 = 'pH_offl'
        model_da = model_ds[var2].isel(depth=depth_level_index)
        #% Load the model data into memory
        print('Set the data type to float32.')
        model_da = model_da.astype('float32')
        print('Done setting the datatype. Start to load...')
        model_da = model_da.compute()
        model_da = 10**(-1*model_da) # convert to Hplus ion concentration
        print('Done')

    #Do the climatology calculations
    print('Calc the climatology')
    climatology = ThreshClimFuncs.calc_clim(params, model_da)
    print('Climatology calculated')

    # Do the threshold calculations
    print('Calc the threshold')
    threshold = ThreshClimFuncs.calc_thresh(params, model_da)
    print('Threshold calculated')

    # Do the intensity normalizer calculations
    print('Calc the intensity normalizer')
    intensity_normalizer = ThreshClimFuncs.calc_intensity_normalizer(threshold, climatology)

    # smoothing
    print('Smoothing')
    climatology_smoothed = ThreshClimFuncs.smooth_array(params, climatology)
    threshold_smoothed = ThreshClimFuncs.smooth_array(params, threshold)
    intensity_normalizer_smoothed = ThreshClimFuncs.calc_intensity_normalizer(threshold_smoothed, climatology_smoothed)

    #Put the smoothed climatology, threshold, and intensity normalizer into a dataset and add some attributes
    print('Put smoothed fields into dataset')
    out_ds = ThreshClimFuncs.put_fields_into_dataset(params, climatology_smoothed, threshold_smoothed, intensity_normalizer_smoothed, model_ds)
    out_ds.attrs['author'] = 'Fiona Pfäffli'
    out_ds.attrs['date'] = str(date.today())
    out_ds.attrs['scriptdir'] = scriptdir
    out_ds.attrs['scriptname'] = scriptname
    out_ds.attrs['depth_level'] = depth_level

    # Append the dataset for this depth level to the list
    datasets.append(out_ds)

#%% Combine all depth level datasets into one dataset
print("Combining all depth level datasets...")
combined_ds = xr.concat(datasets, dim='depth')
combined_ds['depth'] = depth_levels

#%% Save the combined dataset
print("Saving the combined dataset...")
savepath = params.rootdir + 'roms_only/' + 'present/'
save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing_all_depths.nc'
combined_ds.to_netcdf(savepath + save_filename)

print("Done saving.")
