"""
author: Eike Köhn
date: June 10, 2024
description: This file serves to calculate the depth of the saturation horizon (omega_aragonite = 1, 1.3, 1.5., 1.7, ...)
"""

#%% load packages

# enable the visibility of the modules for the import functions
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/'))
sys.path.append('/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/')

# load the package
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from get_model_datasets import ModelGetter as ModelGetter

#%%

outpath = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/processed_model_data/isosurfaces/'

def calc_depth_of_omega_arag_surface(da,ds,isosurface_value,vert_struct):

    print(da, isosurface_value, vert_struct)
    if vert_struct == 'zavg':
        # da = xr.where(da>1e10,np.NaN,da)
        # find the first instances where the values are below the critical value
        cond = xr.where(da<isosurface_value,1,0)

        # get the max value along cond
        maxval = cond.max(dim='depth')
        minval = cond.min(dim='depth')

        # get the depth index of first value smaller than the critical value
        depthidx_of_first_value_below = (cond.argmax(dim='depth'))#.compute()

        # get the depth index of the last value greater than the critical value
        depthidx_of_last_value_above = depthidx_of_first_value_below - 1
        depthidx_of_last_value_above = xr.where(depthidx_of_last_value_above<0,0,depthidx_of_last_value_above)   # make sure that the index is 0 at least

        # get the depth value and the data value at the two different depths
        depth_of_first_value_below = da.depth.isel(depth=depthidx_of_first_value_below.compute())
        value_of_first_value_below = da.isel(depth=depthidx_of_first_value_below.compute())
        depth_of_last_value_above = da.depth.isel(depth=depthidx_of_last_value_above.compute())
        value_of_last_value_above = da.isel(depth=depthidx_of_last_value_above.compute())

        # compute the linearly interpolated depth
        slope = (value_of_last_value_above-value_of_first_value_below)/(depth_of_last_value_above-depth_of_first_value_below)
        d_v = isosurface_value-value_of_first_value_below
        d_z = d_v/slope
        interp_depth = depth_of_first_value_below + d_z
        interp_depth = xr.where(maxval==0,-1*ds.h,interp_depth)
        interp_depth = xr.where(minval==1,0,interp_depth)
        interp_depth = xr.where((depthidx_of_first_value_below==0)*(maxval==1),0,interp_depth)
        interp_depth = xr.where(ds.mask_rho==1,interp_depth,np.NaN)

    elif vert_struct == 'avg':
        raise Exception('Not yet implemented.')
    
    return interp_depth



#%% Setting the variable to compare and across which models to compare

var = 'omega_arag_offl'
temp_resolution = 'daily'
vert_struct = 'zavg'
scenarios = ['present']#,'ssp245','ssp585']
simulation_type = 'hindcast'
ensemble_run = '001'
vtype = 'oceanic'
configs = ['romsoc_fully_coupled']#,'romsoc_fully_coupled']
parent_model='mpi-esm1-2-hr'


#%% Getting model datasets
model_ds = dict()
model_da = dict()
for config in configs:
    model_ds[config] = dict()
    model_da[config] = dict()
    for scenario in scenarios:
        print('--------------------')
        print(f'{config},{scenario},{simulation_type},{ensemble_run},{temp_resolution},{vert_struct},{vtype}')
        model_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,temp_resolution,vert_struct,parent_model=parent_model,vtype=vtype)
        model_da[config][scenario] = model_ds[config][scenario][var]
        if vtype == 'oceanic':
            model_mask = ModelGetter.get_model_mask()
            model_area = ModelGetter.get_model_area()
            model_d2coast = ModelGetter.get_distance_to_coast()
        elif vtype == 'atmospheric':
            raise Exception('Not yet implemented.')
        
#%% Load the model data into memory
isosurface_values = [1,1.3,1.5,1.7]
for config in configs: #['romsoc_fully_coupled']:#
    for scenario in scenarios: #['ssp585']:#
        print('Loading the corresponding model data for config {}.'.format(config))
        model_ds_dummy = model_ds[config][scenario]#.sel(time=slice(pd.to_datetime('2011-01-01'),pd.to_datetime('2011-03-11')))
        model_da_dummy = model_da[config][scenario].load()#.sel(time=slice(pd.to_datetime('2011-01-01'),pd.to_datetime('2011-03-11'))).load()
        #model_da[config][scenario].sel(time=slice(pd.to_datetime('2011-01-01'),pd.to_datetime('2011-01-11'))).load()
        print('Done')
        #% Perform the calculation of the saturation horizon
        # for config in ['roms_only']:#configs:
        #    for scenario in ['present']:#scenarios:
        print('Perform the calculation of the depth of a certain property isosurface, e.g. Omega=1.')
        ds = xr.Dataset()
        for isosurface_value in isosurface_values:
            model_isosurface_da = calc_depth_of_omega_arag_surface(model_da_dummy,model_ds_dummy,isosurface_value,vert_struct)
            model_isosurface_da = model_isosurface_da.drop_vars('depth')
            model_isosurface_da = model_isosurface_da.assign_attrs(unit='meters')
            model_isosurface_da = model_isosurface_da.assign_attrs(variable=var)
            model_isosurface_da = model_isosurface_da.assign_attrs(isosurface_value=isosurface_value)
            model_isosurface_da = model_isosurface_da.assign_attrs(description='calculated from {}'.format(model_ds_dummy.encoding.get('source', 'Unknown source')))
            model_isosurface_da = model_isosurface_da.assign_attrs(note='When all values in water column (down to 500m or full water column depending on vertical structure of the underlying input data) are higher than the isosurface value, then the depth of the isosurface is set to the depth of the sea floor. If all values are lower than the isosurface value, then the depth of the isosurface is set to 0m.')
            ds[f'depth_of_{var}_{isosurface_value}_isosurface'] = model_isosurface_da
        ds = ds.assign_attrs(author='Eike E. Koehn')
        if ensemble_run == '000':
            outdir = f'{outpath}{config}/{scenario}/'
        elif ensemble_run != '000':
            outdir = f'{outpath}{config}/ensemble_members/{scenario}/ens{ensemble_run}/'
        outfilename = f'isosurface_{var}_{temp_resolution}'            
        ds.to_netcdf(f'{outdir}{outfilename}.nc')


# %% Save the depth of the saturation 
