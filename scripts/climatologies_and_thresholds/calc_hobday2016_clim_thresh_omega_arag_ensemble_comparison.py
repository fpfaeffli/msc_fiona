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
scriptname = 'calc_hobday2016_clim_thresh_omega-arag_offl.py'
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





############## DIFFERENT SCENARIOS AND CONFIGURATIONS ####################


#%% 
# ############################ ROMSOC COUPLED SSP585 0M ###############################

# set the variable and threshold parameters
var = 'omega-arag_offl'  # Change to your desired variable

depth_level_index = 0  # m, i.e., surface
config =  'romsoc_fully_coupled' # 'roms_only' 
scenario = 'ssp585' # 'present'  'ssp245'
simulation_type =   'hindcast' #'spinup'
ensemble_run = '001'  
temp_resolution = 'daily'# 'monthly'
vert_struct = 'zavg' # 'avg'  #(for pH zavg because offline carbonate chemistry only on z-levels for model output)
vtype = 'oceanic' #'atmospheric'

params = ThresholdParameters.fiona_instance() #Fiona's Instance = 95.

#%% Get the model data
print('Getting model data for romsoc ssp585...')
model_ds = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, temp_resolution, vert_struct, parent_model='mpi-esm1-2-hr', vtype=vtype)

if var == 'omega-arag_offl':
    var2 = 'pH_offl'
    model_da = model_ds[var2].isel(depth=depth_level_index)
    #% Load the model data into memory
    print('Set the data type to float32.')
    model_da = model_da.astype('float32')
    print('Done setting the datatype. Start to load...')
    model_da = model_da.compute()
    model_da = 10**(-1*model_da) # convert to omega-arag_offl ion concentration
    print('Done')

#%% Do the climatology calculations
print('Calc the climatology for romsoc ssp585 ensemble001...')
climatology = ThreshClimFuncs.calc_clim(params,model_da)
print('Climatology calculated')

#%% Do the threshold calculations
print('Calc the threshold for romsoc ssp585 ensemble001...')
threshold = ThreshClimFuncs.calc_thresh(params,model_da)
print('Threshold calculated')

#%% Do the intensity normalizer calculations
print('Calc the intensity normalizer for romsoc ssp585 ensemble001...')
intensity_normalizer = ThreshClimFuncs.calc_intensity_normalizer(threshold,climatology)

#%% smoothing
print('Smoothing for romsoc ssp585 ensemble001...')
climatology_smoothed = ThreshClimFuncs.smooth_array(params,climatology)
threshold_smoothed = ThreshClimFuncs.smooth_array(params,threshold)
intensity_normalizer_smoothed = ThreshClimFuncs.calc_intensity_normalizer(threshold_smoothed,climatology_smoothed)

#%% Put the smoothed climatology, thershold and intensity normalizer into a dataset and add some attributes
print('Put smoothed fields into dataset for romsoc ssp585 ensemble001...')
out_ds = ThreshClimFuncs.put_fields_into_dataset(params, climatology_smoothed, threshold_smoothed, intensity_normalizer_smoothed, model_ds)
out_ds.attrs['author'] = 'Fiona Pfäffli'
out_ds.attrs['date'] = str(date.today())
out_ds.attrs['scriptdir'] = scriptdir
out_ds.attrs['scriptname'] = scriptname

#%% Save the arrays
print("Saving the arrays for romsoc ssp585 ensemble001...")
savepath = params.rootdir + 'romsoc_fully_coupled/' + 'ssp585/'
save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing_{depth_level_index}depthlevelindex_ensemble001.nc'
out_ds.to_netcdf(savepath + save_filename)


#%%  ############################### ROMSOC COUPLED SSP245 0M ####################################
# 
# set the variable and threshold parameters
var = 'omega-arag_offl'  # Change to your desired variable

depth_level_index = 0  # m, i.e., surface
config =  'romsoc_fully_coupled' # 'roms_only' 
scenario = 'ssp245' # 'present'  
simulation_type =   'hindcast' #'spinup'
ensemble_run = '001'  
temp_resolution = 'daily'# 'monthly'
vert_struct = 'zavg' # 'avg'  #(for pH zavg because offline carbonate chemistry only on z-levels for model output)
vtype = 'oceanic' #'atmospheric'

params = ThresholdParameters.fiona_instance() #Fiona's Instance = 95.

#%% Get the model data
print('Getting model data for romsoc ssp245 for ensemble001...')
model_ds = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, temp_resolution, vert_struct, parent_model='mpi-esm1-2-hr', vtype=vtype)

if var == 'omega-arag_offl':
    var2 = 'pH_offl'
    model_da = model_ds[var2].isel(depth=depth_level_index)
    #% Load the model data into memory
    print('Set the data type to float32.')
    model_da = model_da.astype('float32')
    print('Done setting the datatype. Start to load...')
    model_da = model_da.compute()
    model_da = 10**(-1*model_da) # convert to omega-arag_offl ion concentration
    print('Done')

#%% Do the climatology calculations
print('Calc the climatology for romsoc ssp245 ensemble001...')
climatology = ThreshClimFuncs.calc_clim(params,model_da)
print('Climatology calculated')

#%% Do the threshold calculations
print('Calc the threshold  for romsoc ssp245 ensemble001...')
threshold = ThreshClimFuncs.calc_thresh(params,model_da)
print('Threshold calculated')

#%% Do the intensity normalizer calculations
print('Calc the intensity normalizer for romsoc ssp245 ensemble001...')
intensity_normalizer = ThreshClimFuncs.calc_intensity_normalizer(threshold,climatology)

#%% smoothing
print('Smoothing for romsoc ssp245 ensemble001...')
climatology_smoothed = ThreshClimFuncs.smooth_array(params,climatology)
threshold_smoothed = ThreshClimFuncs.smooth_array(params,threshold)
intensity_normalizer_smoothed = ThreshClimFuncs.calc_intensity_normalizer(threshold_smoothed,climatology_smoothed)

#%% Put the smoothed climatology, thershold and intensity normalizer into a dataset and add some attributes
print('Put smoothed fields into dataset for romsoc ssp245 ensemble001...')
out_ds = ThreshClimFuncs.put_fields_into_dataset(params, climatology_smoothed, threshold_smoothed, intensity_normalizer_smoothed, model_ds)
out_ds.attrs['author'] = 'Fiona Pfäffli'
out_ds.attrs['date'] = str(date.today())
out_ds.attrs['scriptdir'] = scriptdir
out_ds.attrs['scriptname'] = scriptname

#%% Save the arrays
print("Saving the arrays for romsoc ssp245 ensemble001...")
savepath = params.rootdir + 'romsoc_fully_coupled/' + 'ssp245/'
save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing_{depth_level_index}depthlevelindex_ensemble001.nc'
out_ds.to_netcdf(savepath + save_filename)



#%% ######################### ROMSOC COUPLED PRESENT 0M ##################################
# 
# set the variable and threshold parameters
var = 'omega-arag_offl'  # Change to your desired variable

depth_level_index = 0  # m, i.e., surface
config =  'romsoc_fully_coupled' # 'roms_only' 
scenario = 'present'  
simulation_type =   'hindcast' #'spinup'
ensemble_run = '001'  
temp_resolution = 'daily'# 'monthly'
vert_struct = 'zavg' # 'avg'  #(for pH zavg because offline carbonate chemistry only on z-levels for model output)
vtype = 'oceanic' #'atmospheric'

params = ThresholdParameters.fiona_instance() #Fiona's Instance = 95.

#%% Get the model data
print('Getting model data for romsoc present ensemble001...')
model_ds = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, temp_resolution, vert_struct, parent_model='mpi-esm1-2-hr', vtype=vtype)

if var == 'omega-arag_offl':
    var2 = 'pH_offl'
    model_da = model_ds[var2].isel(depth=depth_level_index)
    #% Load the model data into memory
    print('Set the data type to float32.')
    model_da = model_da.astype('float32')
    print('Done setting the datatype. Start to load...')
    model_da = model_da.compute()
    model_da = 10**(-1*model_da) # convert to omega-arag_offl ion concentration
    print('Done')

    
#%% Do the climatology calculations
print('Calc the climatology for romsoc present ensemble001...')
climatology = ThreshClimFuncs.calc_clim(params,model_da)
print('Climatology calculated')

#%% Do the threshold calculations
print('Calc the threshold for romsoc present ensemble001...')
threshold = ThreshClimFuncs.calc_thresh(params,model_da)
print('Threshold calculated')

#%% Do the intensity normalizer calculations
print('Calc the intensity normalizer for romsoc present ensemble001...')
intensity_normalizer = ThreshClimFuncs.calc_intensity_normalizer(threshold,climatology)


#%% smoothing
print('Smoothing for romsoc present ensemble001...')
climatology_smoothed = ThreshClimFuncs.smooth_array(params,climatology)
threshold_smoothed = ThreshClimFuncs.smooth_array(params,threshold)
intensity_normalizer_smoothed = ThreshClimFuncs.calc_intensity_normalizer(threshold_smoothed,climatology_smoothed)


#%% Put the smoothed climatology, thershold and intensity normalizer into a dataset and add some attributes
print('Put smoothed fields into dataset for romsoc present ensemble001...')
out_ds = ThreshClimFuncs.put_fields_into_dataset(params, climatology_smoothed, threshold_smoothed, intensity_normalizer_smoothed, model_ds)
out_ds.attrs['author'] = 'Fiona Pfäffli'
out_ds.attrs['date'] = str(date.today())
out_ds.attrs['scriptdir'] = scriptdir
out_ds.attrs['scriptname'] = scriptname

#%% Save the arrays
print("Saving the arrays for romsoc present ensemble001...")
savepath = params.rootdir + 'romsoc_fully_coupled/' + 'present/'
save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing_{depth_level_index}depthlevelindex_ensemble001.nc'
out_ds.to_netcdf(savepath + save_filename)

