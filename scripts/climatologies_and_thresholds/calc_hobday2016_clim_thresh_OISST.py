"""
Content: calculate the threshold for SST values according to Hobday et al. 2016 based on a 30 year reference period
Author: Eike E Koehn
Date: Apr 26, 2022
"""

#%% DEFINE THE SCRIPTNAME
import os
import sys
scriptdir = '/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/climatologies_and_thresholds/'#observations/'
scriptname = 'calc_hobday2016_clim_thresh_OISST.py'
# enable the visibility of the modules for the import functions
#sys.path.append('/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/')
#sys.path.append('/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/climatologies_and_thresholds/')

#%% load packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.ndimage
from datetime import date
import glob
from get_obs_datasets import ObsGetter as ObsGetter
from set_thresh_and_clim_params import ThresholdParameters
from funcs_for_clim_thresh import ThreshClimFuncs

from importlib import reload  # Python 3.4+
import funcs_for_clim_thresh
reload(funcs_for_clim_thresh)
from funcs_for_clim_thresh import ThreshClimFuncs as ThreshClimFuncs

#%% set the variable and threshold parameters
var = 'temp'
dep = 0 # m, i.e. surface
params = ThresholdParameters.standard_instance()
obs_temp_resolution = 'daily' # 'monthly'

#%% Get the observational data
print('Get the data...')
if var == 'temp' and dep == 0:
    obs_ds, obs_da = ObsGetter.get_sst_data(res=obs_temp_resolution)
print(obs_da)

#%% Load the observational data
print('Load the data...')
obs_da = obs_da.compute()

#%% Do the climatology calculations
print('Calc the climatology')
climatology = ThreshClimFuncs.calc_clim(params,obs_da)

#%% Do the threshold calculations
print('Calc the threshold')
threshold = ThreshClimFuncs.calc_thresh(params,obs_da)

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
out_ds = ThreshClimFuncs.put_fields_into_dataset(params,climatology_smoothed,threshold_smoothed,intensity_normalizer_smoothed,obs_ds)
out_ds.attrs['author'] = 'E. E. Koehn'
out_ds.attrs['date'] = str(date.today())
out_ds.attrs['scriptdir'] = scriptdir
out_ds.attrs['scriptname'] = scriptname

#%% Save the arrays
print("Save the arrays")
print("Saving")
# set the path and filename
savepath = params.rootdir + 'oisst/'
save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing.nc'
# save
out_ds.to_netcdf(savepath+save_filename)

#%%