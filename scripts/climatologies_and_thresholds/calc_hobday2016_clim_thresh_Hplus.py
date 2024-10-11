"""
Content: calculate the threshold for H+ ions according to Hobday et al. 2016 based on a 30 year reference period
Author: Eike E Koehn
Changes: Fiona Pfäffli, Oct 11, 2024
Date: Apr 26, 2022
"""

#%% DEFINE THE SCRIPTNAME
import os
import sys
scriptdir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/'
scriptname = 'calc_hobday2016_clim_thresh_Hplus.py'
# enable the visibility of the modules for the import functions
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

#%% set the variable and threshold parameters
var = 'temp'  # Change to your desired variable

depth_level = 0  # m, i.e., surface
config =  'roms_only' #'romsoc_fully_coupled'
scenario = 'present' # 'ssp245', 'ssp585'
simulation_type = 'spinup' # 'hindcast'
ensemble_run = '000'  
temp_resolution = 'daily'# 'monthly'
vert_struct = 'avg'# 'zavg' 
vtype = 'oceanic' #'atmospheric'


#%% Get the model data
print('Getting model data...')
model_ds = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, temp_resolution, vert_struct, parent_model='mpi-esm1-2-hr', vtype=vtype)
model_da = model_ds[var]

#%% Load the model data into memory
print('Loading the model data')
model_da = model_da.compute()

#%% Do the climatology calculations
print('Calc the climatology')
climatology = ThreshClimFuncs.calc_clim(params,model_da)

#%% Do the threshold calculations
print('Calc the threshold')
threshold = ThreshClimFuncs.calc_thresh(params,model_da)

#%% Do the intensity normalizer calculations
print('Calc the intensity normalizer')
intensity_normalizer = ThreshClimFuncs.calc_intensity_normalizer(threshold,climatology)

#%% smoothing
print('Smoothing')
climatology_smoothed = ThreshClimFuncs.smooth_array(params,climatology)
threshold_smoothed = ThreshClimFuncs.smooth_array(params,threshold)
intensity_normalizer_smoothed = ThreshClimFuncs.calc_intensity_normalizer(threshold_smoothed,climatology_smoothed)

#%% Put the smoothed climatology, thershold and intensity normalizer into a dataset and add some attributes
print('Put smoothed fields into dataset')
out_ds = ThreshClimFuncs.put_fields_into_dataset(params, climatology_smoothed, threshold_smoothed, intensity_normalizer_smoothed, model_ds)
out_ds.attrs['author'] = 'Fiona Pfäffli'
out_ds.attrs['date'] = str(date.today())
out_ds.attrs['scriptdir'] = scriptdir
out_ds.attrs['scriptname'] = scriptname

#%% Save the arrays
print("Saving the arrays...")
savepath = params.rootdir + 'model_output/'
save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing.nc'
out_ds.to_netcdf(savepath + save_filename)

#%%