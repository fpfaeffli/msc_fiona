"""
Content: calculate the threshold for S, T, Alk and DIC according to Hobday et al. 2016 based on a 30 year reference period
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
from get_model_datasets_Hplus import ModelGetter as ModelGetter
from set_thresh_and_clim_params import ThresholdParameters
from func_for_clim_thresh import ThreshClimFuncs

from importlib import reload  # Python 3.4+
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs as ThreshClimFuncs


#%% 
# Calculate climatology for salinity at the surface


# set the variable and threshold parameters
var = 'salt'  # Change to you desired variable

depth_level_index = 0  # m, i.e., surface
config =  'romsoc_fully_coupled' #'romsoc_fully_coupled'
scenario =  'ssp245' # 'ssp245' 'ssp245'
simulation_type =   'hindcast' #'spinup'
ensemble_run = '000'  
temp_resolution = 'daily'# 'monthly'
vert_struct = 'zavg' # 'avg'  #(for pH zavg because offline carbonate chemistry only on z-levels for model output)
vtype = 'oceanic' #'atmospheric'

params = ThresholdParameters.Hplus_instance() #95th

#%% Get the model data
print('Getting model data...')
model_ds = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, temp_resolution, vert_struct, parent_model='mpi-esm1-2-hr', vtype=vtype)

if var == 'salt':
    model_da = model_ds[var].isel(depth=depth_level_index)
    # Load the model data into memory
    print('Set the data type to float32.')
    model_da = model_da.astype('float32')
    print('Done setting the datatype. Start to load...')
    model_da = model_da.compute()
    print('Done')

#%% Do the climatology calculations
print('Calc the climatology')
climatology = ThreshClimFuncs.calc_clim(params,model_da)
print('Climatology calculated')

#%% Do the threshold calculations
print('Calc the threshold')
threshold = ThreshClimFuncs.calc_thresh(params,model_da)
print('Threshold calculated')

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
savepath = params.rootdir + 'romsoc_fully_coupled/' + 'ssp245/'
save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing_{depth_level_index}depthlevelindex.nc'
out_ds.to_netcdf(savepath + save_filename)

print('Done')
print("***********************************************")



#%%

# Calculate climatology for temperature at the surface

# set the variable and threshold parameters
var = 'temp'  # Change to your desired variable

depth_level_index = 0  # m, i.e., surface
config =  'romsoc_fully_coupled' #'romsoc_fully_coupled'
scenario =  'ssp245' # 'ssp245' 'ssp245'
simulation_type =   'hindcast' #'spinup'
ensemble_run = '000'  
temp_resolution = 'daily'# 'monthly'
vert_struct = 'zavg' # 'avg'  #(for pH zavg because offline carbonate chemistry only on z-levels for model output)
vtype = 'oceanic' #'atmospheric'

params = ThresholdParameters.Hplus_instance() #95th

#%% Get the model data
print('Getting model data...')
model_ds = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, temp_resolution, vert_struct, parent_model='mpi-esm1-2-hr', vtype=vtype)

if var == 'temp':
    model_da = model_ds[var].isel(depth=depth_level_index)
    # Load the model data into memory
    print('Set the data type to float32.')
    model_da = model_da.astype('float32')
    print('Done setting the datatype. Start to load...')
    model_da = model_da.compute()
    print('Done')

#%% Do the climatology calculations
print('Calc the climatology')
climatology = ThreshClimFuncs.calc_clim(params,model_da)
print('Climatology calculated')

#%% Do the threshold calculations
print('Calc the threshold')
threshold = ThreshClimFuncs.calc_thresh(params,model_da)
print('Threshold calculated')

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
savepath = params.rootdir + 'romsoc_fully_coupled/' + 'ssp245/'
save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing_{depth_level_index}depthlevelindex.nc'
out_ds.to_netcdf(savepath + save_filename)

print('Done')
print("***********************************************")

#%%

# Calculate climatology for DIC at the surface

# set the variable and threshold parameters
var = 'DIC'  # Change to your desired variable

depth_level_index = 0  # m, i.e., surface
config =  'romsoc_fully_coupled' #'romsoc_fully_coupled'
scenario =  'ssp245' # 'ssp245' 'ssp245'
simulation_type =   'hindcast' #'spinup'
ensemble_run = '000'  
temp_resolution = 'daily'# 'monthly'
vert_struct = 'zavg' # 'avg'  #(for pH zavg because offline carbonate chemistry only on z-levels for model output)
vtype = 'oceanic' #'atmospheric'

params = ThresholdParameters.Hplus_instance() #95th

#%% Get the model data
print('Getting model data...')
model_ds = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, temp_resolution, vert_struct, parent_model='mpi-esm1-2-hr', vtype=vtype)

if var == 'DIC':
    model_da = model_ds[var].isel(depth=depth_level_index)
    # Load the model data into memory
    print('Set the data type to float32.')
    model_da = model_da.astype('float32')
    print('Done setting the datatype. Start to load...')
    model_da = model_da.compute()
    print('Done')

#%% Do the climatology calculations
print('Calc the climatology') 
climatology = ThreshClimFuncs.calc_clim(params,model_da)
print('Climatology calculated')

#%% Do the threshold calculations
print('Calc the threshold')
threshold = ThreshClimFuncs.calc_thresh(params,model_da)
print('Threshold calculated')

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
savepath = params.rootdir + 'romsoc_fully_coupled/' + 'ssp245/'
save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing_{depth_level_index}depthlevelindex.nc'
out_ds.to_netcdf(savepath + save_filename)

print('Done')
print("***********************************************")

#%%

# Calculate climatology for Alk at the surface

# set the variable and threshold parameters
var = 'Alk'  # Change to your desired variable

depth_level_index = 0  # m, i.e., surface
config =  'romsoc_fully_coupled' #'romsoc_fully_coupled'
scenario =  'ssp245' # 'ssp245' 'ssp245'
simulation_type =   'hindcast' #'spinup'
ensemble_run = '000'  
temp_resolution = 'daily'# 'monthly'
vert_struct = 'zavg' # 'avg'  #(for pH zavg because offline carbonate chemistry only on z-levels for model output)
vtype = 'oceanic' #'atmospheric'

params = ThresholdParameters.Hplus_instance() #95th

#%% Get the model data
print('Getting model data...')
model_ds = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, temp_resolution, vert_struct, parent_model='mpi-esm1-2-hr', vtype=vtype)

if var == 'Alk':
    model_da = model_ds[var].isel(depth=depth_level_index)
    # Load the model data into memory
    print('Set the data type to float32.')
    model_da = model_da.astype('float32')
    print('Done setting the datatype. Start to load...')
    model_da = model_da.compute()
    print('Done')

#%% Do the climatology calculations
print('Calc the climatology')
climatology = ThreshClimFuncs.calc_clim(params,model_da)
print('Climatology calculated')

#%% Do the threshold calculations
print('Calc the threshold')
threshold = ThreshClimFuncs.calc_thresh(params,model_da)
print('Threshold calculated')

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
savepath = params.rootdir + 'romsoc_fully_coupled/' + 'ssp245/'
save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing_{depth_level_index}depthlevelindex.nc'
out_ds.to_netcdf(savepath + save_filename)


print('Done')
print("***********************************************")
