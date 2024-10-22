"""
Content: calculate the threshold for H+ ions according to Hobday et al. 2016 based on a 30-year reference period
Author: Eike E Koehn
Changes: Fiona Pfäffli, Oct 21, 2024
Date: Apr 26, 2022
"""

#%% DEFINE THE SCRIPTNAME
import os
import sys
import concurrent.futures
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

# import modules
from get_model_datasets import ModelGetter as ModelGetter
from set_thresh_and_clim_params import ThresholdParameters
from func_for_clim_thresh import ThreshClimFuncs

from importlib import reload  # Python 3.4+
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs as ThreshClimFuncs

#%% set the variable and threshold parameters
var = 'Hplus'  # Change to your desired variable

config = 'roms_only'  #'romsoc_fully_coupled'
scenario = 'present'  # 'ssp245', 'ssp585'
simulation_type = 'hindcast'  #'spinup'
ensemble_run = '000'
temp_resolution = 'daily'  # 'monthly'
vert_struct = 'zavg'  # 'avg' (for pH zavg because offline carbonate chemistry only on z-levels for model output)
vtype = 'oceanic'  #'atmospheric'

#%% Create the parameters instance 
params = ThresholdParameters.fiona_instance()  # Fiona's Instance = 95.

#%% Get the model data
print('Getting model data...')
model_ds = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, temp_resolution, vert_struct, parent_model='mpi-esm1-2-hr', vtype=vtype)

#%% Available depth levels
depth_levels = [-0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55,
                -60, -65, -70, -75, -80, -85, -90, -95, -100, -110, -120, -130,
                -140, -150, -165, -180, -200, -225, -250, -280, -310, -350, -400, -450, -500]

#%% Function to process a range of depth levels
def process_depth_range(start, end, params, model_ds, var, scriptdir, scriptname):
    for depth_level_index in range(start, end):
        print(f"Processing depth level index {depth_level_index}")

        if var == 'Hplus':
            var2 = 'pH_offl'
            model_da = model_ds[var2].isel(depth=depth_level_index)

            # Load and process model data
            model_da = model_da.astype('float32')
            model_da = model_da.compute()
            model_da = 10**(-1*model_da)  # convert to Hplus ion concentration

            # Do the climatology calculations
            climatology = ThreshClimFuncs.calc_clim(params, model_da)

            # Do the threshold calculations
            threshold = ThreshClimFuncs.calc_thresh(params, model_da)

            # Do the intensity normalizer calculations
            intensity_normalizer = ThreshClimFuncs.calc_intensity_normalizer(threshold, climatology)

            # Smoothing
            climatology_smoothed = ThreshClimFuncs.smooth_array(params, climatology)
            threshold_smoothed = ThreshClimFuncs.smooth_array(params, threshold)
            intensity_normalizer_smoothed = ThreshClimFuncs.calc_intensity_normalizer(threshold_smoothed, climatology_smoothed)

            # Put smoothed fields into dataset
            out_ds = ThreshClimFuncs.put_fields_into_dataset(params, climatology_smoothed, threshold_smoothed, intensity_normalizer_smoothed, model_ds)
            out_ds.attrs['author'] = 'Fiona Pfäffli'
            out_ds.attrs['date'] = str(date.today())
            out_ds.attrs['scriptdir'] = scriptdir
            out_ds.attrs['scriptname'] = scriptname

            # Save the arrays
            savepath = params.rootdir + 'model_output/'
            save_filename = f'hobday2016_threshold_and_climatology_{var}_{params.percentile}perc_{params.baseline_start_year}-{params.baseline_end_year}baseperiod_{params.baseline_type}baseline_{params.aggregation_window_size}aggregation_{params.smoothing_window_size}smoothing_{depth_level_index}depthlevelindex.nc'
            out_ds.to_netcdf(savepath + save_filename)
            print(f"Saved results for depth level index {depth_level_index}")

#%% Parallelize the depth level processing
def parallel_depth_processing(params, model_ds, var, scriptdir, scriptname):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #depth levels get split in 4 groups 
        num_processes = 4
        depth_range_size = len(depth_levels) // num_processes
        futures = []

        #dividing depth levels into smaller groups (0-8, 9-17, 18-26, 27-36)
        for i in range(num_processes):
            start = i * depth_range_size
            if i == num_processes - 1:
                end = len(depth_levels)  #remaining levels because last group is smaller
            else:
                end = (i + 1) * depth_range_size

            futures.append(executor.submit(process_depth_range, start, end, params, model_ds, var, scriptdir, scriptname))

        #completing processing
        concurrent.futures.wait(futures)
        print("All depth level processing complete.")

#%% Main script execution
if __name__ == "__main__": #so that the when script is imported, the parallel processing doesn't run automatically
    # run the parallel depth processing
    parallel_depth_processing(params, model_ds, var, scriptdir, scriptname)

# %%
