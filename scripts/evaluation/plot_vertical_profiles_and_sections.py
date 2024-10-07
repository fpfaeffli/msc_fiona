"""
author: Eike Köhn
date: June 10, 2024
description: This file loads in the model data and compares it with the respective observational dataset. The comparison consists of:
1. map of annual mean
2. for subregions full timeseries of variable (calculating trends)
3. for subregions climatological timeseries
"""

#%% load packages

# enable the visibility of the modules for the import functions
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/'))
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')

# load the package
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from get_obs_datasets import ObsGetter as ObsGetter
from get_model_datasets import ModelGetter as ModelGetter
from get_study_regions import GetRegions as GetRegions
from plotting_functions_evaluation import Plotter as Plotter

from importlib import reload  # Python 3.4+
import plotting_functions_evaluation
reload(plotting_functions_evaluation)
from plotting_functions_evaluation import Plotter as Plotter

import regridding_tools
reload(regridding_tools)
from regridding_tools import Regridder as Regridder

import get_obs_datasets
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter
#%% Setting the variable to compare and across which models to compare

var = 'temp'
plot_resolution = 'monthly'
dep = 'profile'

obs_temp_resolution = 'monthly'#'monthly_clim' # 'monthly'
vert_struct = 'zavg'    # 'zavg'

model_temp_resolution = 'monthly'

scenario = 'present'
simulation_type = 'hindcast'
ensemble_run = '000'
vtype = 'oceanic'
configs = ['roms_only','romsoc_fully_coupled']

#%% Getting model datasets

model_ds = dict()
model_da = dict()
model_regs = dict()
for config in configs:
    print('--------------------')
    print(f'{config},{scenario},{simulation_type},{ensemble_run},{model_temp_resolution},{vert_struct},{vtype}')
    model_ds[config] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution.replace('_clim',''),vert_struct,vtype=vtype)
    if vert_struct == 'zavg':
        model_da[config] = model_ds[config][var]#.sel(depth=dep)
    elif vert_struct == 'avg' and np.size(np.shape(model_ds[config][var]))==3:
        model_da[config] = model_ds[config][var]
    else:
        raise Exception('Not yet implemented.')
    model_da[config] = xr.where(model_da[config]>1e19,np.NaN,model_da[config])
    if vtype == 'oceanic':
        model_mask = ModelGetter.get_model_mask()
        model_area = ModelGetter.get_model_area()
        model_d2coast = ModelGetter.get_distance_to_coast()
    elif vtype == 'atmospheric':
        raise Exception('Not yet implemented.')
    

#%% Getting observational dataset

if var == 'temp' and dep == 'profile':
    obs_ds, obs_da = ObsGetter.get_temp_data(res=obs_temp_resolution)#+'_clim')
if var == 'salt' and dep == 'profile':
    obs_ds, obs_da = ObsGetter.get_salt_data(res=obs_temp_resolution)#+'_clim')
elif var == 'omega_arag_offl' and dep == 'profile':
    obs_ds, obs_da = ObsGetter.get_omega_arag_data(res=obs_temp_resolution)#+'_clim')
elif var == 'O2' and dep == 'profile':
    obs_ds, obs_da = ObsGetter.get_o2_data(res=obs_temp_resolution+'_clim')
# ... add further datasets to be analyzed

# get the area and distance to coast fields
obs_area = ObsGetter.get_obs_area(obs_ds)
obs_d2coast = ObsGetter.get_distance_to_coast(model_d2coast,obs_da)   # note that there are some artifacts in the polar regions
obs_d2coast_lat = obs_d2coast.lat[:,0].values
obs_d2coast_lon = obs_d2coast.lon[0,:].values
obs_d2coast = xr.DataArray(data=obs_d2coast.values,dims=["lat","lon"],coords={"lat":("lat",obs_d2coast_lat),"lon":("lon",obs_d2coast_lon)})

#%% Get regions over which to calculate the statistics and to plot

model_regions_dict = GetRegions.define_CalCS_regions(model_area.lon,model_area.lat,model_d2coast)
obs_regions_dict = GetRegions.define_CalCS_regions(obs_area.lon,obs_area.lat,obs_d2coast)

#%% Load in the model and observational data

print('Loading the observational data profile for var {}.'.format(var))
obs_da.load()
print('Done')
for config in configs:
    print('Loading the corresponding model data for config {}.'.format(config))
    model_da[config].load()
    print('Done')


# %% 
print('Calculate the mean profiles for individual regions')
obs_mean_profiles = dict()
model_mean_profiles = dict()
model_mean_profiles['roms_only'] = dict()
model_mean_profiles['romsoc_fully_coupled'] = dict()

row_names = ['all_lats','northern','central','southern']
col_names = ['all_dists','offshore','coastal'] # 'transition'
for rdx, rn in enumerate(row_names):
    for cdx, cn in enumerate(col_names):
        region_name = f'{cn}_{rn}'
        print(region_name)
        # get the observations and compute the regional average
        obs_reg = obs_regions_dict[region_name]['mask']
        if 'time' in obs_da.dims:
            obs_regional_mean = obs_da.weighted((obs_area*obs_reg).fillna(0)).mean(dim=('time','lat','lon'))
        elif 'month' in obs_da.dims:
            obs_regional_mean = obs_da.weighted((obs_area*obs_reg).fillna(0)).mean(dim=('month','lat','lon'))
        else:
            obs_regional_mean = obs_da.weighted((obs_area*obs_reg).fillna(0)).mean(dim=('lat','lon'))
        obs_mean_profiles[region_name] = obs_regional_mean
        # get the model and compute the regional average
        model_reg = model_regions_dict[region_name]['mask']
        model_mean_profiles['roms_only'][region_name] = model_da['roms_only'].weighted((model_area*model_reg).fillna(0)).mean(dim=('time','eta_rho','xi_rho')) 
        model_mean_profiles['romsoc_fully_coupled'][region_name] = model_da['romsoc_fully_coupled'].weighted((model_area*model_reg).fillna(0)).mean(dim=('time','eta_rho','xi_rho'))


#%%##################
# START THE PLOTTING#
#####################

#%%
var = 'O2'
print('Generate plot: Vertical profiles')
plotted_values = Plotter.plot_vertical_profiles(var,obs_da,obs_mean_profiles,model_da,model_mean_profiles,model_regions_dict,plot_resolution,savefig=True)

# %% 
var = 'omega_arag_offl'
print('Generate plot: Time vs depth sections')
#if 'time' in obs_da.dims or 'month' in obs_da.dims or 'day' in obs_da.dims:
plotted_values = Plotter.plot_time_vs_depth_sections(var,obs_da,obs_area,model_da,model_area,obs_regions_dict,model_regions_dict,plot_resolution,savefig=True)

# %% 
var = 'O2'
print('Generate plot: Time vs depth sections climatology')
#if 'time' in obs_da.dims or 'month' in obs_da.dims or 'day' in obs_da.dims:
plotted_values = Plotter.plot_time_vs_depth_sections_climatology(var,obs_da,obs_area,model_da,model_area,obs_regions_dict,model_regions_dict,plot_resolution,savefig=True)

# %% UNDER DEVELOPMENT
var = 'O2'
print('Generate plot: Depth vs. distance to coast transects')

target_lats = np.array([36,43])
lonres = 0.5
target_lons = np.arange(225,240+lonres,lonres)

# compute the time mean of the dataset
if 'month' in obs_da.dims: 
    obs_da = obs_da.rename({"month": "time"})
elif 'day' in obs_da.dims: 
    obs_da = obs_da.rename({"day": "time"})

if 'time' in obs_da.dims:
    obs_da = obs_da.mean(dim='time')

model_da['roms_only'] = model_da['roms_only'].mean(dim='time')
model_da['romsoc_fully_coupled'] = model_da['romsoc_fully_coupled'].mean(dim='time')

# interpolate model data
model_da_interp = dict()
model_da_interp['roms_only'] = Regridder.regrid_xr_dataarray(model_da['roms_only'],target_lons,target_lats)
model_da_interp['romsoc_fully_coupled'] = Regridder.regrid_xr_dataarray(model_da['romsoc_fully_coupled'],target_lons,target_lats)
model_d2coast_interp = Regridder.regrid_xr_dataarray(model_d2coast,target_lons,target_lats)
# interpolate obs data
obs_da_interp = Regridder.regrid_xr_dataarray(obs_da,target_lons,target_lats)
obs_d2coast_interp = Regridder.regrid_xr_dataarray(obs_d2coast,target_lons,target_lats)

#%%
plotted_values = Plotter.plot_depth_vs_dist2coast_transect(var,target_lats,obs_da_interp,obs_d2coast_interp,model_da_interp,model_d2coast_interp,savefig=True)

# %%
### AUTOCORRELATION TIMESCALE MAPS ###