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
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')



# load the package
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from get_obs_datasets import ObsGetter as ObsGetter
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter
from plotting_functions_evaluation import Plotter as Plotter

from importlib import reload  # Python 3.4+
import plotting_functions_evaluation
reload(plotting_functions_evaluation)
from plotting_functions_evaluation import Plotter as Plotter

import get_model_datasets
reload(get_model_datasets)
from get_model_datasets import ModelGetter as ModelGetter

import get_obs_datasets
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter
#%% Setting the variable to compare and across which models to compare

var = 'mld_holte'

dep = 0 # m
vert_struct = 'zavg'    # 'zavg'

plot_resolution = 'monthly'
model_temp_resolution = 'daily' # 'monthly'
obs_temp_resolution = 'monthly' # 'monthly'

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
    model_ds[config] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype=vtype)
    if vert_struct == 'zavg':
        if np.size(np.shape(model_ds[config][var])) == 4:
            model_da[config] = model_ds[config][var].sel(depth=dep)
        else:
            model_da[config] = model_ds[config][var]
    elif vert_struct == 'avg' and np.size(np.shape(model_ds[config][var]))==3:
        model_da[config] = model_ds[config][var]
    else:
        raise Exception('Not yet implemented.')
    if vtype == 'oceanic':
        model_mask = ModelGetter.get_model_mask()
        model_area = ModelGetter.get_model_area()
        model_d2coast = ModelGetter.get_distance_to_coast()
    elif vtype == 'atmospheric':
        raise Exception('Not yet implemented.')

#%% Getting observational dataset

if var == 'temp' and dep == 0:
    obs_ds, obs_da = ObsGetter.get_sst_data(res=obs_temp_resolution)
elif var == 'salt' and dep == 0:
    obs_ds, obs_da = ObsGetter.get_sss_data(res=obs_temp_resolution)
elif var == 'NO3' and dep == 0:
    obs_ds, obs_da = ObsGetter.get_no3_data(res=obs_temp_resolution+'_clim')
    obs_da = obs_da.sel(depth=dep)
elif var == 'PO4' and dep == 0:
    obs_ds, obs_da = ObsGetter.get_po4_data(res=obs_temp_resolution+'_clim')
    obs_da = obs_da.sel(depth=dep)
elif var == 'DIC' and dep == 0:
    obs_ds, obs_da = ObsGetter.get_dic_data(res=obs_temp_resolution+'_clim')
    obs_da = obs_da.sel(depth=dep)
elif var == 'Alk' and dep == 0:
    obs_ds, obs_da = ObsGetter.get_dic_data(res=obs_temp_resolution+'_clim')
    obs_da = obs_da.sel(depth=dep)
elif var == 'pH_offl' and dep == 0:
    obs_ds, obs_da = ObsGetter.get_ph_data(res=obs_temp_resolution)
    obs_da = obs_da
elif var == 'omega_arag_offl' and dep == 0:
    obs_ds, obs_da = ObsGetter.get_omega_arag_data(res=obs_temp_resolution)
    obs_da = obs_da
elif var == 'zeta' and dep == 0:
    obs_ds, obs_da = ObsGetter.get_ssh_data(res=obs_temp_resolution)
    obs_da = obs_da
elif var == 'mld_holte':
    obs_ds, obs_da = ObsGetter.get_mld_data(res=obs_temp_resolution+'_clim')
    obs_da = obs_da*-1    
# ... add further datasets to be analyzed

# get the area and distance to coast fields
obs_area = ObsGetter.get_obs_area(obs_ds)
obs_d2coast = ObsGetter.get_distance_to_coast(model_d2coast,obs_da)   # note that there are some artifacts in the polar regions

#%% Get regions over which to calculate the statistics and to plot

model_regions_dict = GetRegions.define_CalCS_regions(model_area.lon,model_area.lat,model_d2coast)
obs_regions_dict = GetRegions.define_CalCS_regions(obs_area.lon,obs_area.lat,obs_d2coast)

#%% Load in the model and observational data

print('Loading the observational data for var {} at depth {}m.'.format(var,dep))
obs_da.load()
print('Done')
for config in configs:
    print('Loading the corresponding model data for config {}.'.format(config))
    model_da[config].load()
    print('Done')

#%% Calculate regional means
print('calculate the monthly means for the regions')
print('------Obs-------')
regional_data_obs = dict()
for region in obs_regions_dict.keys():
    if region != 'full_map':
        print(region)
        intermediate = (obs_da*obs_regions_dict[region]['mask']).weighted(obs_area.fillna(0)).mean(("lat","lon"))
        if "time" in obs_da.dims:
            regional_data_obs[region] = intermediate.groupby("time.month").mean("time")
        elif "month" in obs_da.dims and np.size(obs_da.month)==12:
            regional_data_obs[region] = intermediate

print('------Model------')
regional_data_model = dict()
for config in configs:
    regional_data_model[config] = dict()
    for region in model_regions_dict.keys():
        if region != 'full_map':
            print(config,region)
            regional_data_model[config][region] = (model_da[config]*model_regions_dict[region]['mask']).weighted(model_area.fillna(0)).mean(("eta_rho","xi_rho")).groupby("time.month").mean("time")

#%%##################
# START THE PLOTTING#
#####################

# %% 

print('Generate plot: Map for annual mean')
plotted_values = Plotter.plot_full_map_annual_mean(var,dep,obs_da,model_da,obs_regions_dict,model_regions_dict,regional_data=[regional_data_obs,regional_data_model],regional_data_plottype='lines',savefig=True)


# %% 

print('Generate plot: Averaged timeseries in regions')
plotted_values = Plotter.plot_area_averaged_timeseries(var,dep,obs_da,obs_area,model_da,model_area,obs_regions_dict,model_regions_dict,plot_resolution,savefig=True)

# %% 

print('Generate plot: Averaged timeseries climatology in regions')
plotted_values = Plotter.plot_area_averaged_climatology_timeseries(var,dep,obs_da,obs_area,model_da,model_area,obs_regions_dict,model_regions_dict,plot_resolution,savefig=True)

# %%


