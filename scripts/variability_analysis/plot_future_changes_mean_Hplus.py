"""
author: Eike Köhn
date: June 18, 2024
description: This file calculates and plots the mean of a model variable, as well as the future changes in the mean field. If requested, the plot also contains an inset showing changes in the monthly climatology.
"""

#%% enable the visibility of the modules for the import functions
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/'))
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/variability_analysis/')

#%%
# load the package
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import acf
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter
from plotting_functions_variability_analysis import VariaAnalysisPlotter as VariaAnalysisPlotter

from importlib import reload  # Python 3.4+
import plotting_functions_variability_analysis
reload(plotting_functions_variability_analysis)
from plotting_functions_variability_analysis import VariaAnalysisPlotter as VariaAnalysisPlotter

import multiprocessing
from tqdm import tqdm

from get_parent_anomalies import ParentGetter as ParentGetter

from importlib import reload  # Python 3.4+
import get_parent_anomalies
reload(get_parent_anomalies)
from get_parent_anomalies import ParentGetter as ParentGetter

from importlib import reload  # Python 3.4+
import get_study_regions
reload(get_study_regions)
from get_study_regions import GetRegions as GetRegions

#%%

var = 'pH_offl'#'temp'#'mld_holte'

dep = 0 # m
vert_struct = 'zavg'#'zavg'    # 'zavg'

model_temp_resolution = 'daily'#'monthly'#'daily' # 'monthly'

scenarios = ['present','ssp245','ssp585']
simulation_type = 'hindcast'
ensemble_run = '000'
vtype = 'oceanic'
configs = ['roms_only','romsoc_fully_coupled']

parent_model = 'mpi-esm1-2-hr'

# mld holte works with 'zavg' and 'daily'

#%% Getting model datasets

model_ds = dict()
model_da = dict()
model_regs = dict()
for config in configs:
     model_ds[config] = dict()
     model_da[config] = dict()
     for scenario in scenarios:
          print('--------------------')
          print(f'{config},{scenario},{simulation_type},{ensemble_run},{model_temp_resolution},{vert_struct},{vtype},{parent_model}')
          model_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype=vtype,parent_model=parent_model)
          if vert_struct == 'zavg' and 'depth' in model_ds[config][scenario][var].dims:
               model_da[config][scenario] = model_ds[config][scenario][var].sel(depth=dep)
          elif vert_struct == 'zavg' and 'depth' not in model_ds[config][scenario][var].dims:
               model_da[config][scenario] = model_ds[config][scenario][var]
          elif vert_struct == 'avg' and np.size(np.shape(model_ds[config][scenario][var]))==3:
               model_da[config][scenario] = model_ds[config][scenario][var]
          else:
               raise Exception('Not yet implemented.')
          if vtype == 'oceanic':
               model_mask = ModelGetter.get_model_mask()
               model_area = ModelGetter.get_model_area()
               model_d2coast = ModelGetter.get_distance_to_coast()
          elif vtype == 'atmospheric':
               raise Exception('Not yet implemented.')

#%% Get regions over which to calculate the statistics and to plot
model_regions_dict = GetRegions.define_CalCS_regions(model_area.lon,model_area.lat,model_d2coast)

#%% Load in the model data
for config in configs:
     for scenario in scenarios:
         print('Loading the corresponding model data for config {}, scenario {}.'.format(config,scenario))
         model_da[config][scenario].load()
         print('Done')

#%% Calculate the mean for the data
mean_results = dict()
for config in configs:
    mean_results[config] = dict()
    for scenario in scenarios:
        print(config,scenario)
        mean_results[config][scenario] = model_da[config][scenario].mean(dim='time')

#%% Calculate regional means
print('calculate the monthly means for the regions')
regional_data = dict()
for config in configs:
     regional_data[config] = dict()
     for scenario in scenarios:
          regional_data[config][scenario] = dict()
          for region in ['coastal_all_lats','offshore_all_lats']:#model_regions_dict.keys():
               print(config,scenario,region)
               regional_data[config][scenario][region] = (model_da[config][scenario]*model_regions_dict[region]['mask']).weighted(model_area.fillna(0)).mean(("eta_rho","xi_rho")).groupby("time.month").mean("time")

#%% Make the plot
plotted_values = VariaAnalysisPlotter.plot_future_changes_map(mean_results,var,dep,model_regions_dict,regional_data=regional_data,regional_data_plottype='lines',savefig=True)

#%%

# %% MAKE SEASONAL HOVMOELLER DIAGRAMS in 1° latitude bands in coastal region

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize

config = 'roms_only' #'romsoc_fully_coupled'

if var == 'temp' and dep == 0:

     binned_avgs = dict()
     for scenario in scenarios:
          binned_avgs[scenario] = dict()
          print(scenario)
          region_mask = model_regions_dict['coastal_all_lats']['mask']

          data_dum = model_da[config][scenario].groupby('time.month').mean('time')
          area_dum = ModelGetter.get_model_area()
          lat = model_da[config][scenario].lat

          masked_data = xr.where(region_mask==1.,data_dum,np.NaN)
          masked_area = xr.where(region_mask==1.,area_dum,np.NaN)
          weighted_data = masked_data * masked_area

          lat_bins = range(30,48)
          lat_bin_indices = np.digitize(data_dum['lat'], bins=lat_bins) - 1
          weighted_data.coords['lat_bin'] = (('eta_rho','xi_rho'), lat_bin_indices)
          masked_area.coords['lat_bin'] = (('eta_rho','xi_rho'), lat_bin_indices)

          binned_avgs[scenario] = (weighted_data.groupby('lat_bin').sum() / masked_area.groupby('lat_bin').sum())

     cmap1 = plt.cm.Greys(np.linspace(0,1,256))    # Normal Greys
     cmap2 = plt.cm.Greys_r(np.linspace(0,1,256))    # Reversed Greys
     # Create an array of evenly spaced values
     # Combine the colors
     combined_colors = np.vstack((cmap1, cmap2))
     # Create a new colormap from the combined colors
     combined_cmap = LinearSegmentedColormap.from_list('CombinedGreys', combined_colors)
     normalize = Normalize(vmin=-.015, vmax=0.015)

     #%
     fontsize=16
     plt.rcParams['font.size']=fontsize
     fig, ax = plt.subplots(1,3,figsize=(15,7),sharey=True,sharex=True,gridspec_kw={'width_ratios':[1.2,1,1.2]})
     MON,LAT = np.meshgrid(np.arange(1,13),lat_bins[1:])
     #if scenario == 'present':
     ax[0].set_title('Present day')
     present_vals = binned_avgs['present'][:,1:-1].T.values
     ssp245_vals = binned_avgs['ssp245'][:,1:-1].T.values
     ssp585_vals = binned_avgs['ssp585'][:,1:-1].T.values

     c00 = ax[0].pcolormesh(MON,LAT,present_vals,cmap=plt.get_cmap('RdYlBu_r',15),vmin=8,vmax=23)
     cbar0 = plt.colorbar(c00,ax=ax[0])#,pad=0.1)
     cbar0.ax.set_title(r'°C')

     ax[1].set_title('SSP245 - present')
     ax[1].pcolormesh(MON,LAT,(ssp245_vals-present_vals),cmap=plt.get_cmap('cmo.amp',16),vmin=0,vmax=4)
     ax[2].set_title('SSP585 - present')
     c20 = ax[2].pcolormesh(MON,LAT,(ssp585_vals-present_vals),cmap=plt.get_cmap('cmo.amp',16),vmin=0,vmax=4)
     cbar2 = plt.colorbar(c20,ax=ax[2],extend='both')
     cbar2.ax.set_title(r'°C',pad=20)
     for axi in ax:
          axi.set_xlabel('Month')
          axi.set_xticks(np.arange(1,13))
          axi.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
          axi.set_yticks(np.arange(30,48,2)+0.5)
          axi.set_yticklabels(np.arange(30,48,2))
     ax[0].set_ylabel('Latitude in °N')
     outdir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/variability_analysis/future_changes/regional_means/climatologies/'
     outfile = f'sst_climatology_binned_1degree_{config}.png'
     plt.savefig(outdir+outfile,dpi=200,transparent=True)
     plt.show()

elif var == 'mld_holte':

     binned_avgs = dict()
     for scenario in scenarios:
          binned_avgs[scenario] = dict()
          print(scenario)
          region_mask = model_regions_dict['coastal_all_lats']['mask']

          data_dum = model_da[config][scenario].groupby('time.month').mean('time')
          area_dum = ModelGetter.get_model_area()
          lat = model_da[config][scenario].lat

          masked_data = xr.where(region_mask==1.,data_dum,np.NaN)
          masked_area = xr.where(region_mask==1.,area_dum,np.NaN)
          weighted_data = masked_data * masked_area

          lat_bins = range(30,48)
          lat_bin_indices = np.digitize(data_dum['lat'], bins=lat_bins) - 1
          weighted_data.coords['lat_bin'] = (('eta_rho','xi_rho'), lat_bin_indices)
          masked_area.coords['lat_bin'] = (('eta_rho','xi_rho'), lat_bin_indices)

          binned_avgs[scenario] = (weighted_data.groupby('lat_bin').sum() / masked_area.groupby('lat_bin').sum())

     cmap1 = plt.cm.Greys(np.linspace(0,1,256))    # Normal Greys
     cmap2 = plt.cm.Greys_r(np.linspace(0,1,256))    # Reversed Greys
     # Create an array of evenly spaced values
     # Combine the colors
     combined_colors = np.vstack((cmap1, cmap2))
     # Create a new colormap from the combined colors
     combined_cmap = LinearSegmentedColormap.from_list('CombinedGreys', combined_colors)
     normalize = Normalize(vmin=-.015, vmax=0.015)

     #%
     fontsize=16
     plt.rcParams['font.size']=fontsize
     fig, ax = plt.subplots(1,3,figsize=(15,7),sharey=True,sharex=True,gridspec_kw={'width_ratios':[1.2,1,1.2]})
     MON,LAT = np.meshgrid(np.arange(1,13),lat_bins[1:])
     #if scenario == 'present':
     ax[0].set_title('Present day')
     present_vals = binned_avgs['present'][:,1:-1].T.values
     ssp245_vals = binned_avgs['ssp245'][:,1:-1].T.values
     ssp585_vals = binned_avgs['ssp585'][:,1:-1].T.values

     c00 = ax[0].pcolormesh(MON,LAT,present_vals,cmap=plt.get_cmap('cmo.tempo_r',10),vmin=-50,vmax=0)
     cbar0 = plt.colorbar(c00,ax=ax[0])#,pad=0.1)
     cbar0.ax.set_title(r'm')

     ax[1].set_title('SSP245 - present')
     ax[1].pcolormesh(MON,LAT,(ssp245_vals-present_vals),cmap=plt.get_cmap('cmo.balance',14),vmin=-7,vmax=7)
     ax[2].set_title('SSP585 - present')
     c20 = ax[2].pcolormesh(MON,LAT,(ssp585_vals-present_vals),cmap=plt.get_cmap('cmo.balance',14),vmin=-7,vmax=7)
     cbar2 = plt.colorbar(c20,ax=ax[2],extend='both')
     cbar2.ax.set_title(r'm',pad=20)
     for axi in ax:
          axi.set_xlabel('Month')
          axi.set_xticks(np.arange(1,13))
          axi.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
          axi.set_yticks(np.arange(30,48,2)+0.5)
          axi.set_yticklabels(np.arange(30,48,2))
     ax[0].set_ylabel('Latitude in °N')
     outdir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/variability_analysis/future_changes/regional_means/climatologies/'
     outfile = f'mld_climatology_binned_1degree_{config}.png'
     plt.savefig(outdir+outfile,dpi=200,transparent=True)
     plt.show()

#%%
















# %% Make the climatology timeseries plot
plotted_values = VariaAnalysisPlotter.plot_future_changes_climatology_timeseries(regional_data,var,dep,model_regions_dict,num_regions=3,savefig=False)
plotted_values = VariaAnalysisPlotter.plot_future_changes_climatology_timeseries(regional_data,var,dep,model_regions_dict,num_regions=9,savefig=False)


#%%
parent_anomalies_ds = dict()
parent_anomalies_da = dict()
for scenario in ['ssp245','ssp585']:
     parent_path = ParentGetter.get_parent_anomaly_paths(scenario,vtype,parent_model=parent_model)
     parent_ds,parent_da = ParentGetter.open_parent_anomaly_datasets(parent_path,var,dep)
     parent_anomalies_ds[scenario] = parent_ds
     parent_anomalies_da[scenario] = parent_da

#%% Calculate regional means for the parent model
print('calculate the monthly means for the regions')
parent_regional_data = dict()
for scenario in ['ssp245','ssp585']:
     parent_regional_data[scenario] = dict()
     for region in model_regions_dict.keys():
          print(scenario,region)
          parent_regional_data[scenario][region] = (parent_anomalies_da[scenario]*model_regions_dict[region]['mask']).weighted(model_area.fillna(0)).mean(("eta_rho","xi_rho"))

#%% Re calculate the regional means for the downscaled model, but mask the fields according to the global parent model field to make a fair comparison
print('calculate the monthly means for the regions')
recalc_regional_data = dict()
for config in configs:
     recalc_regional_data[config] = dict()
     for scenario in scenarios:
          recalc_regional_data[config][scenario] = dict()
          for region in model_regions_dict.keys():
               print(config,scenario,region)
               parent_mask = xr.where(np.isnan(parent_anomalies_da['ssp245'].isel(time=0).values),np.NaN,1)
               recalc_regional_data[config][scenario][region] = (model_da[config][scenario]*model_regions_dict[region]['mask']).weighted((parent_mask*model_area).fillna(0)).mean(("eta_rho","xi_rho")).groupby("time.month").mean("time")

# %%
plotted_values = VariaAnalysisPlotter.plot_delta_comparison_with_parent_model(var,dep,recalc_regional_data,parent_regional_data,savefig=False)



# %%
# # %% MAKE SEASONAL HOVMOELLER DIAGRAMS in 1° latitude bands in coastal region

# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.colors import Normalize

# binned_avgs = dict()
# for scenario in scenarios:
#      binned_avgs[scenario] = dict()
#      print(scenario)
#      config = 'romsoc_fully_coupled'
#      region_mask = model_regions_dict['coastal_all_lats']['mask']

#      data_dum = model_da[config][scenario]
#      #region_mask = region_mask.assign_coords({"rlon": data_dum.rlon, "rlat": data_dum.rlat})
#      area_dum = model_area#.rename({'y_cosp':'rlat','x_cosp':'rlon'})
#      #area_dum = area_dum.assign_coords({"rlon": data_dum.rlon, "rlat": data_dum.rlat})
#      lat = model_da[config][scenario].lat

#      masked_data = xr.where(region_mask==1.,data_dum.transpose("time", "eta_rho", "xi_rho"),np.NaN)
#      masked_area = xr.where(region_mask==1.,area_dum,np.NaN)
#      weighted_data = masked_data * masked_area

#      lat_bins = range(30,48)
#      lat_bin_indices = np.digitize(masked_data['lat'], bins=lat_bins) - 1
#      weighted_data.coords['lat_bin'] = (('eta_rho','xi_rho'), lat_bin_indices)
#      masked_area.coords['lat_bin'] = (('eta_rho','xi_rho'), lat_bin_indices)

#      binned_avgs[scenario] = (weighted_data.groupby('lat_bin').sum() / masked_area.groupby('lat_bin').sum())

# cmap1 = plt.cm.Greys(np.linspace(0,1,256))    # Normal Greys
# cmap2 = plt.cm.Greys_r(np.linspace(0,1,256))    # Reversed Greys
# # Create an array of evenly spaced values
# # Combine the colors
# combined_colors = np.vstack((cmap1, cmap2))
# # Create a new colormap from the combined colors
# combined_cmap = LinearSegmentedColormap.from_list('CombinedGreys', combined_colors)
# normalize = Normalize(vmin=-.015, vmax=0.015)
# %%
