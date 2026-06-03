"""
author: Eike Köhn
date: June 18, 2024
description: This file plots a snapshot of the present day and future blob for the different thresholds
"""

#%% enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

# load the package
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import acf
from get_study_regions import GetRegions as GetRegions
from get_model_datasets_Hplus_surf_only import ModelGetter as ModelGetter
from get_obs_datasets import ObsGetter as ObsGetter
#from plotting_functions_extremes import ExtAnalysisPlotter as ExtAnalysisPlotter

import multiprocessing
from tqdm import tqdm

from importlib import reload  # Python 3.4+
#import plotting_functions_extremes
#reload(plotting_functions_extremes)
#from plotting_functions_extremes import ExtAnalysisPlotter as ExtAnalysisPlotter

import matplotlib.patheffects as pe


import get_obs_datasets 
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter

# Define the threshold 
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.Hplus_instance() #95th percentile threshold

import xesmf as xe

from plotting_functions_general import PlotFuncs as PlotFuncs

from get_auxiliary_data import AuxGetter as AuxGetter

import pickle 

import define_netcdf_functions as ncfunc
from get_scriptdir_scriptname import *
#scriptdir, scriptname = get_scriptdir_scriptname(__file__)


# Load landmask
landmask_etopo = AuxGetter.get_etopo_data()


from func_for_clim_thresh import ThreshClimFuncs
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs

#%%

model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present','ssp245','ssp585'] # ,'ssp245'
configs = ['romsoc_fully_coupled'] # 'roms_only',
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'zavg'    # 'avg'
threshold_types = ['present','present_plus_meandelta']#,'present_plus_climdelta','present_plus_meandelta']
#%% Get the model datasets for the oceanic and atmospheric variables

# Get the model datasets for the oceanic and atmospheric variables

ocean_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)

#%% 
# load the data at the respective location

variables = dict()
for config in configs:
     variables[config] = dict()
     for scenario in scenarios:
          print(f'Getting the variables for config {config} and scenario {scenario}.')
          #
          # oceanic variables
          variables[config][scenario] = dict()
          print('Hplus')
          # Convert pH to [H+] concentration
          pH = ocean_ds[config][scenario].pH_offl.isel(depth=0).load()
          variables[config][scenario]['Hplus'] = np.power(10, -pH)
          

#%%
varias = ['Hplus']

#%% 
# Get the climatology for each variable
print('Get the climatology')
clims = dict()
for config in configs:
     clims[config] = dict()
     for scenario in scenarios:
          clims[config][scenario] = dict()
          for varia in varias:
               print(f'Calculate the climatology for: {config}, {scenario}, {varia}.')
               dummy_clim = ThreshClimFuncs.calc_clim(params,variables[config][scenario][varia])
               smoothed_clim = ThreshClimFuncs.smooth_array(params,dummy_clim)
               smoothed_clim = smoothed_clim.rename({'day_of_year_adjusted': 'time'})
               smoothed_clim = smoothed_clim.assign_coords({'time': pd.date_range('2015-01-01','2015-12-31')})
               clims[config][scenario][varia] = ThreshClimFuncs.repeat_array_multiple_years(smoothed_clim)


#%% 
# Get the threshold for each variable
print('Get the present day threshold')
varia = 'Hplus'
thresholds = dict()
for config in configs:
    thresholds[config] = dict()
    for scenario in ['present']:  # ,'ssp245','ssp585'
        thresholds[config][scenario] = dict()
        print(f'{config}, {scenario}')
        threshold, threshold_366 = ModelGetter.get_threshold('Hplus', 0, 'relative', 95, config, scenario)  # variable, depth_level, threshold_type, threshold_value, config, scenario
        concatenated_threshold = ModelGetter.concatenate_yearly_arrays(threshold, threshold_366, start_year=2011, end_year=2021)
        #rename 'day_of_year_adjusted' to 'time' in the concatenated threshold
        if 'day_of_year_adjusted' in concatenated_threshold.dims:
            concatenated_threshold = concatenated_threshold.rename({'day_of_year_adjusted': 'time'})
        elif 'day_of_year_adjusted' in concatenated_threshold.coords:
            concatenated_threshold = concatenated_threshold.rename_coords({'day_of_year_adjusted': 'time'})
        #assign the renamed threshold to the dictionary
        thresholds[config][scenario][varia] = concatenated_threshold


# %%
#% Get the distance to coast file from ROMS and the regions over which to calculate the statistics
model_d2coasts = dict()
model_d2coasts['roms_only'] = ModelGetter.get_distance_to_coast(vtype='oceanic')

# Get the model regions
model_regions = dict()
model_regions['roms_only'] = GetRegions.define_CalCS_regions(model_d2coasts['roms_only'].lon, model_d2coasts['roms_only'].lat, model_d2coasts['roms_only'])

# Get the model area
model_area = ModelGetter.get_model_area()


#%% Get the anomaly to thresholds for all different cases (from this i will deduce the boolean arrays)
anom_rel_to_thresh = dict()
anom_rel_to_thresh_normalized = dict()
for config in configs:
     anom_rel_to_thresh[config] = dict()
     anom_rel_to_thresh_normalized[config] = dict()
     for scenario in scenarios:
          anom_rel_to_thresh[config][scenario] = dict()
          anom_rel_to_thresh_normalized[config][scenario] = dict()
          for varia in varias:
              anom_rel_to_thresh[config][scenario][varia] = dict()
              anom_rel_to_thresh_normalized[config][scenario][varia] = dict()
              for threshold_type in threshold_types: 
                  print(config,scenario,varia,threshold_type)
                  if threshold_type == 'present_plus_climdelta':
                      shiftval   = clims[config][scenario][varia] - clims[config]['present'][varia]
                      thresh_val = thresholds[config]['present'][varia] + shiftval
                      clim_val   = clims[config]['present'][varia] + shiftval
                  elif threshold_type == 'present_plus_meandelta':
                      shiftval = variables[config][scenario][varia].mean(dim='time') - variables[config]['present'][varia].mean(dim='time')
                      thresh_val = thresholds[config]['present'][varia] + shiftval
                      clim_val   = clims[config]['present'][varia]      + shiftval
                  elif threshold_type == 'present':
                      thresh_val = thresholds[config]['present'][varia]
                      clim_val   = clims[config]['present'][varia]
                  anom_rel_to_thresh[config][scenario][varia][threshold_type]            = variables[config][scenario][varia] - thresh_val
                  anom_rel_to_thresh_normalized[config][scenario][varia][threshold_type] = (variables[config][scenario][varia] - clim_val)/(thresh_val-clim_val)


#%% Get the boolean arrays within the region of choice

reg_choice = 'all_dists_central'

mask_dum = xr.where(model_regions['roms_only'][reg_choice]['mask']==1,1,0)
booleans = dict()
for config in configs:
     booleans[config] = dict()
     for scenario in scenarios:
          booleans[config][scenario] = dict()
          for varia in varias:
              booleans[config][scenario][varia] = dict()
              for threshold_type in threshold_types: 
                  print(config,scenario,varia,threshold_type)
                  booleans[config][scenario][varia][threshold_type] = (anom_rel_to_thresh[config][scenario][varia][threshold_type]>0)*mask_dum


#%% Do the labeling
from scipy import ndimage as spndimage
structuring_element = np.zeros((3,3,3))
structuring_element[:,1,1]=True    
labeled_arrays = dict()
for config in configs:
     labeled_arrays[config] = dict()
     for scenario in scenarios:
          labeled_arrays[config][scenario] = dict()
          for varia in varias:
              labeled_arrays[config][scenario][varia] = dict()
              for threshold_type in threshold_types: 
                  print(config,scenario,varia,threshold_type)
                  labeled_dum = spndimage.label(booleans[config][scenario][varia][threshold_type], structuring_element)[0]
                  labeled_arrays[config][scenario][varia][threshold_type] = xr.DataArray(labeled_dum,dims=["time", "eta_rho", "xi_rho"],coords=dict(time=booleans[config][scenario][varia][threshold_type].time))

#%% Extract the label coordinates
import sparse as spa
from joblib import Parallel, delayed

# Function to process a single label
def process_label(lab, sparse_coords, sparse_labels):
    if np.mod(lab, 10000) == 0:
        print(lab)
    coords = sparse_coords[:, sparse_labels == lab]
    return lab, coords

print('go')
extremes_dict = dict()
for config in configs:
     extremes_dict[config] = dict()
     for scenario in scenarios:
          extremes_dict[config][scenario] = dict()
          for varia in varias:
              extremes_dict[config][scenario][varia] = dict()
              for threshold_type in threshold_types: 
                  print(config,scenario,varia,threshold_type)
                  lab_array_dum = labeled_arrays[config][scenario][varia][threshold_type].values
                  extremes_dict[config][scenario][varia][threshold_type] = dict()
                  sparse_labeled_array = spa.as_coo(labeled_arrays[config][scenario][varia][threshold_type].values)
                  sparse_coords = sparse_labeled_array.coords
                  sparse_labels = sparse_labeled_array.data

                  results = Parallel(n_jobs=96)(
                        delayed(process_label)(lab, sparse_coords, sparse_labels)
                        for lab in np.unique(sparse_labels)
                        )
                  
                  # Store results in the dictionary
                  for lab, coords in results:
                      extremes_dict[config][scenario][varia][threshold_type][lab] = dict()
                      extremes_dict[config][scenario][varia][threshold_type][lab]['coords'] = coords
                #   for lab in np.unique(sparse_labels):
                #       if np.mod(lab,100)==0:
                #           print(lab)
                #       extremes_dict[config][scenario][varia][threshold_type][lab] = dict()
                #       extremes_dict[config][scenario][varia][threshold_type][lab]['coords'] = sparse_coords[:,sparse_labels==lab]

#%% SAVE THE EXTREMES DICT
print('save the dictionary.')
outdir = '/nfs/sea/work/fpfaeffli/extremes_char_dict/'
outfile = 'extremes_dict_oax_not_extracted.pkl'
with open(f'{outdir}{outfile}', 'wb') as f:
    pickle.dump(extremes_dict, f)

#%% Extract associated values/anomalies/areas/d2coasts
model_area_dum = model_area
model_d2coasts_dum = model_d2coasts['roms_only']
for config in configs:
     for scenario in scenarios:
          for varia in varias:
              variable_dum = variables[config][scenario][varia]
              for threshold_type in threshold_types: 
                  print(config,scenario,varia,threshold_type)
                  anom_rel_to_thresh_dum            = anom_rel_to_thresh[config][scenario][varia][threshold_type]
                  anom_rel_to_thresh_normalized_dum = anom_rel_to_thresh_normalized[config][scenario][varia][threshold_type]
                  for lab in extremes_dict[config][scenario][varia][threshold_type].keys():
                      if np.mod(lab,10000)==0:
                          print(lab)
                      coords = extremes_dict[config][scenario][varia][threshold_type][lab]['coords']
                      timecoord = xr.DataArray(coords[0,:],dims="points")
                      etacoord  = xr.DataArray(coords[1,:],dims="points")
                      xicoord   = xr.DataArray(coords[2,:],dims="points")
                      Hplus = variable_dum.isel(time=timecoord,eta_rho=etacoord,xi_rho=xicoord)
                      Hplus_anom_rel_to_thresh = anom_rel_to_thresh_dum.isel(time=timecoord,eta_rho=etacoord,xi_rho=xicoord)
                      Hplus_anom_rel_to_thresh_normalized = anom_rel_to_thresh_normalized_dum.isel(time=timecoord,eta_rho=etacoord,xi_rho=xicoord)
                      areas = model_area_dum.isel(eta_rho=etacoord,xi_rho=xicoord)
                      dists = model_d2coasts_dum.isel(eta_rho=etacoord,xi_rho=xicoord)
                      # put into dictionary
                      extremes_dict[config][scenario][varia][threshold_type][lab]['Hplus'] = Hplus # np.array(temps)
                      extremes_dict[config][scenario][varia][threshold_type][lab]['Hplus_anom_rel_to_thresh'] = Hplus_anom_rel_to_thresh # np.array(temps_anom_rel_to_thresh)
                      extremes_dict[config][scenario][varia][threshold_type][lab]['Hplus_anom_rel_to_thresh_normalized'] = Hplus_anom_rel_to_thresh_normalized # np.array(temps_anom_rel_to_thresh)
                      extremes_dict[config][scenario][varia][threshold_type][lab]['area'] = areas # np.array(areas)
                      extremes_dict[config][scenario][varia][threshold_type][lab]['dist2coast'] = dists # np.array(dists)

# %%
print('save the dictionary.')
outdir = '/nfs/sea/work/fpfaeffli/extremes_char_dict/'
outfile = 'extremes_dict_oax_all_scenarios.pkl'
with open(f'{outdir}{outfile}', 'wb') as f:
    pickle.dump(extremes_dict, f)


#%% Compute the characteristics
print('go')
characteristics_dict = dict()
for config in configs:
    characteristics_dict[config] = dict()
    for scenario in scenarios:
         characteristics_dict[config][scenario] = dict()
         for varia in varias:
             characteristics_dict[config][scenario][varia] = dict()
             for threshold_type in threshold_types:
                 characteristics_dict[config][scenario][varia][threshold_type] = dict()
                 for lab in extremes_dict[config][scenario][varia][threshold_type].keys():
                      if np.mod(lab,10000)==0:
                          print(lab)
                      characteristics_dict[config][scenario][varia][threshold_type][lab] = dict()
                      # Get time information
                      choords_dum = extremes_dict[config][scenario][varia][threshold_type][lab]['coords']
                      timer = choords_dum[0,:]
                      
                      # Get the location information
                      eta_dum = choords_dum[1,0]
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['eta'] = eta_dum

                      xi_dum = choords_dum[2,0]
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['xi'] = xi_dum

                      characteristics_dict[config][scenario][varia][threshold_type][lab]['dist2coast'] = extremes_dict[config][scenario][varia][threshold_type][lab]['dist2coast'][0]

                      # get the lon/lat coordinates
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['lon'] = mask_dum.lon[eta_dum,xi_dum]
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['lat'] = mask_dum.lat[eta_dum,xi_dum]
                      
                      # Calculate the global start index
                      mintimeidx = np.min(timer)
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['time_index_start_global'] = mintimeidx
                      # Calculate the global end index
                      maxtimeidx = np.max(timer)
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['time_index_end_global'] = maxtimeidx
                      # Calculate the duration
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['duration_global'] = maxtimeidx-mintimeidx+1
                      # Maximum intensity absolute
                      intensity_max_abs        = np.max(extremes_dict[config][scenario][varia][threshold_type][lab]['Hplus_anom_rel_to_thresh'])
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['intensity_max_abs'] = intensity_max_abs
                      # Mean intensity absolute
                      intensity_mean_abs        = np.mean(extremes_dict[config][scenario][varia][threshold_type][lab]['Hplus_anom_rel_to_thresh'])
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['intensity_mean_abs'] = intensity_mean_abs                      
                      # Maximum intensity normalized
                      intensity_max_normalized = np.max(extremes_dict[config][scenario][varia][threshold_type][lab]['Hplus_anom_rel_to_thresh_normalized'])
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['intensity_max_normalized'] = intensity_max_normalized
                      # Mean intensity normalized
                      intensity_mean_normalized = np.mean(extremes_dict[config][scenario][varia][threshold_type][lab]['Hplus_anom_rel_to_thresh_normalized'])
                      characteristics_dict[config][scenario][varia][threshold_type][lab]['intensity_mean_normalized'] = intensity_mean_normalized                      
                      #Maximum footprint
                    #   footprint_area = []
                    #   for timeidx in np.arange(mintimeidx,maxtimeidx+1):
                    #       footprint_area.append(np.sum(extremes_dict[config][scenario][varia][threshold_type][lab]['area'][timer==timeidx]))
                    #   footprint_area = np.array(footprint_area)
                    #   characteristics_dict[config][scenario][varia][threshold_type][lab]['area_max'] = np.max(footprint_area)
                    #   characteristics_dict[config][scenario][varia][threshold_type][lab]['area_mean'] = np.mean(footprint_area)
                    #   characteristics_dict[config][scenario][varia][threshold_type][lab]['area_median'] = np.median(footprint_area)

# %%
print('save the dictionary.')
outdir = '/nfs/sea/work/fpfaeffli/extremes_char_dict/'
outfile = 'characteristics_dict_oax_all_scenarios.pkl'
with open(f'{outdir}{outfile}', 'wb') as f:
    pickle.dump(characteristics_dict, f)

#%% 
# LOAD THE CALCULATED DICTIONARIES
varias = ['Hplus']

# LOAD THE CHARACTERISTICS DICT
print('load the characteristics dictionary.')
outdir = '/nfs/sea/work/fpfaeffli/extremes_char_dict/'
outfile = 'characteristics_dict_oax_all_scenarios.pkl'
with open(f'{outdir}{outfile}', 'rb') as f:
    characteristics_dict = pickle.load(f)


# LOAD THE EXTREMES DICT
print('load the extremes dictionary.')
outdir = '/nfs/sea/work/fpfaeffli/extremes_char_dict/'
outfile = 'extremes_dict_oax_all_scenarios.pkl'
with open(f'{outdir}{outfile}', 'rb') as f:
    extremes_dict = pickle.load(f)

#%% TURN DIRCTIONARY INTO A VECTOR

characs = ['time_index_start_global','time_index_end_global','duration_global','intensity_max_abs','intensity_mean_abs','intensity_max_normalized','eta','xi','lon','lat','dist2coast']#,'area_max','area_mean','area_median']

print('go')
characteristics_vecs = dict()
for config in configs:
    characteristics_vecs[config] = dict()
    for scenario in scenarios:
         characteristics_vecs[config][scenario] = dict()
         for varia in varias:
             characteristics_vecs[config][scenario][varia] = dict()
             for threshold_type in threshold_types:
                 characteristics_vecs[config][scenario][varia][threshold_type] = dict()
                 for charac in characs:
                     print(config,scenario,varia,threshold_type,charac)
                     characteristics_vecs[config][scenario][varia][threshold_type][charac] = []
                     for lab in extremes_dict[config][scenario][varia][threshold_type].keys():
                         #if np.mod(lab,1000)==0:
                         #    print(lab)
                         try:
                             characteristics_vecs[config][scenario][varia][threshold_type][charac].append(characteristics_dict[config][scenario][varia][threshold_type][lab][charac].values)
                         except:
                             characteristics_vecs[config][scenario][varia][threshold_type][charac].append(characteristics_dict[config][scenario][varia][threshold_type][lab][charac])
                     characteristics_vecs[config][scenario][varia][threshold_type][charac] = np.array(characteristics_vecs[config][scenario][varia][threshold_type][charac])

#%%

###############################
#         PLOTTING            #
###############################


#%% Plots the  characteristics vs distance to coast

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# --- settings ---
save_arrays = False
ds = xr.Dataset()  # will hold all bin results

config_cho = 'romsoc_fully_coupled'
var_cho    = 'Hplus'
metrics    = ['mean']                     # or ['mean'], ['max','iqr'] etc.
char_cho   = 'duration_global'         # 'intensity_mean_abs' or 'duration_global'
scale_nmols = True                        # scale H+ from mol/L to nmol/L for plotting

# units / scaling
if char_cho.startswith('intensity'):
    scale = 1e9 if scale_nmols else 1.0
    char_cho_unit = 'nmol L$^{-1}$' if scale_nmols else 'mol L$^{-1}$'
elif char_cho == 'duration_global':
    scale = 1.0
    char_cho_unit = 'days'
else:
    scale = 1.0
    char_cho_unit = ''

fontsize = 14
plt.rcParams['font.size'] = fontsize

# colors
scenario_colors = {
    'present': "#ffab67",   # warm orange
    'ssp245':  "#65a342",   # green
    'ssp585':  "#9553e0",   # purple
}

# NOTE: no sharey -> dynamic, independent y-axis ranges per subplot
fig, ax = plt.subplots(1, 2, figsize=(8.5, 4))

bins_d2coast = np.arange(0, 600, 10)
bin_centers  = ((bins_d2coast[:-1] + bins_d2coast[1:]) / 2).astype(int)
ds['dist_to_coast_bin_edges']   = xr.DataArray(bins_d2coast,  dims='bin_edges',   attrs={'unit':'km'})
ds['dist_to_coast_bin_centers'] = xr.DataArray(bin_centers,   dims='bin_centers', attrs={'unit':'km'})

for tdx, threshold_type in enumerate(['present','present_plus_meandelta']):
    baseline = 'fixed' if threshold_type == 'present' else 'adjusted'
    # choose linestyle: fixed baseline -> dashed
    ls_for_baseline = '--' if baseline == 'fixed' else '-'

    for scenario in ['present','ssp245','ssp585']:
        x_vals   = characteristics_vecs[config_cho][scenario][var_cho][threshold_type]['dist2coast']
        char_raw = characteristics_vecs[config_cho][scenario][var_cho][threshold_type][char_cho] * scale

        # per-bin stats
        b_mean = []; b_median = []; b_p10 = []; b_p25 = []; b_p75 = []; b_p90 = []; b_max = []
        for bdx in range(len(bins_d2coast)-1):
            dist_cond = (x_vals >= bins_d2coast[bdx]) & (x_vals < bins_d2coast[bdx+1])
            vals = char_raw[dist_cond]
            if vals.size == 0:
                b_mean.append(np.nan); b_median.append(np.nan); b_p10.append(np.nan)
                b_p25.append(np.nan);  b_p75.append(np.nan);   b_p90.append(np.nan); b_max.append(np.nan)
            else:
                b_mean.append(np.nanmean(vals))
                b_median.append(np.nanmedian(vals))
                b_p10.append(np.nanpercentile(vals, 10))
                b_p25.append(np.nanpercentile(vals, 25))
                b_p75.append(np.nanpercentile(vals, 75))
                b_p90.append(np.nanpercentile(vals, 90))
                b_max.append(np.nanmax(vals))

        color = scenario_colors[scenario]

        if 'mean' in metrics:
            ax[tdx].plot(bin_centers, b_mean, color=color, lw=2.5, label=scenario, linestyle=ls_for_baseline)
            ds[f'mean_of_{char_cho}_{scenario}_{baseline}_baseline_vs_d2coast'] = xr.DataArray(
                b_mean, dims='bin_centers', attrs={'unit': char_cho_unit}
            )
        if 'max' in metrics:
            ax[tdx].plot(bin_centers, b_max, color=color, lw=2.5, label=None if 'mean' in metrics else scenario, linestyle=ls_for_baseline)
            ds[f'max_of_{char_cho}_{scenario}_{baseline}_baseline_vs_d2coast'] = xr.DataArray(
                b_max, dims='bin_centers', attrs={'unit': char_cho_unit}
            )
        if 'iqr' in metrics:
            ax[tdx].fill_between(bin_centers, b_p75, b_p25, color=color, alpha=0.25)
        if 'idr' in metrics:
            ax[tdx].fill_between(bin_centers, b_p90, b_p10, color=color, alpha=0.15)

    # axis cosmetics (per subplot)
    ax[tdx].set_title(f'{baseline} baseline', fontsize=fontsize)
    ax[tdx].grid(True, alpha=0.3)
    ax[tdx].set_xlabel('Distance to coast (km)')
    ax[tdx].set_xlim(0, 600)
    ax[tdx].set_xticks([0,100,200,300,400,500,600])

    # y-axis ticks/labels on both sides; 1 decimal
    ax[tdx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[tdx].tick_params(axis='y', which='both', labelleft=True, labelright=False)
    # vertical reference line at 50 km
    ax[tdx].axvline(50, color="k", linestyle="--", lw=1)

# y-label based on metric (apply to both subplots)
if char_cho == 'duration_global' and metrics == ['max']:
    ylabel = 'Max duration (days)'
elif char_cho == 'duration_global' and metrics == ['mean']:
    ylabel = 'Mean duration (days)'
elif char_cho == 'intensity_mean_abs':
    ylabel = f"Mean intensity ({char_cho_unit})"
elif char_cho == 'intensity_max_abs':
    ylabel = f"Max intensity ({char_cho_unit})"
elif char_cho == 'intensity_mean_normalized':
    ylabel = "Mean normalized intensity"
elif char_cho == 'intensity_max_normalized':
    ylabel = "Max normalized intensity"
else:
    ylabel = f"{char_cho} ({char_cho_unit})"

for a in ax:
    a.set_ylabel(ylabel)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    a.tick_params(axis='y', which='both', labelleft=True, labelright=False)

# legend (once)
handles, labels = ax[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, ['present','ssp245','ssp585'], loc='lower center', ncol=3, frameon=False)

plt.tight_layout(rect=(0,0.08,1,1))

# Save results if desired
if save_arrays:
    ds.to_netcdf(f'/path/to/save/{char_cho}_vs_d2coast_bins.nc')

plt.savefig(f'/nfs/sea/work/fpfaeffli/plots/extreme_char/d2coast_{char_cho}_both_baselines.png',
            dpi=200, transparent=True)
plt.show()



#%% Dummy maps to plot a map of the events
print('go')
dummy_maps = dict()
dummy_maps[config_cho] = dict()
for scenario in ['present','ssp245','ssp585']:
    dummy_maps[config_cho][scenario] = dict()
    dummy_maps[config_cho][scenario][var_cho] = dict()
    for tdx,threshold_type in enumerate(['present','present_plus_meandelta']):
        dummy_maps[config_cho][scenario][var_cho][threshold_type] = dict()
        dummy_maps[config_cho][scenario][var_cho][threshold_type][char_cho] = np.zeros((1000,model_area.shape[0],model_area.shape[1]))+np.NaN
        dummy_maps[config_cho][scenario][var_cho][threshold_type]['counter'] = np.zeros((model_area.shape[0],model_area.shape[1]))


print('go')
for scenario in ['present','ssp245','ssp585']:
    print(scenario)
    for tdx,threshold_type in enumerate(['present','present_plus_meandelta']):
        print(scenario,threshold_type)
        numevents = len(characteristics_vecs[config_cho][scenario][var_cho][threshold_type]['eta'])
        for edx in range(numevents):
            etaval = characteristics_vecs[config_cho][scenario][var_cho][threshold_type]['eta'][edx]
            xival = characteristics_vecs[config_cho][scenario][var_cho][threshold_type]['xi'][edx]
            varval = characteristics_vecs[config_cho][scenario][var_cho][threshold_type][char_cho][edx]
            counterval = int(dummy_maps[config_cho][scenario][var_cho][threshold_type]['counter'][etaval,xival])
            dummy_maps[config_cho][scenario][var_cho][threshold_type][char_cho][counterval,etaval,xival] = varval
            dummy_maps[config_cho][scenario][var_cho][threshold_type]['counter'][etaval,xival] += 1

#%%

# Plot maps of the H+ extreme durations
# --- settings ---
save_arrays = False
ds = xr.Dataset()  # will hold all bin results

config_cho = 'romsoc_fully_coupled'
var_cho    = 'Hplus'
metrics    = ['mean']                     # or ['mean'], ['max','iqr'] etc.
char_cho   = 'duration_global'         # 'intensity_mean_abs' or 'duration_global'
scale_nmols = True                       # scale H+ from mol/L to nmol/L for plotting

def plot_hplus_extreme_map(
    dummy_maps,
    ocean_mask,
    landmask_etopo,
    model_regions_dict,
    config_cho,
    var_cho,
    char_cho,
    metrics,
    char_cho_unit,
    scale_nmols=True
):
    scenarios    = ["present", "ssp245", "ssp585"]
    title_colors = {"present":"#f08c3a","ssp245":"#65a342","ssp585":"#9553e0"}
    lat_bounds   = [32, 46]
    lon_bounds   = [225, 245]

    # scaling for H+ intensities
    scale = 1e9 if (char_cho.startswith("intensity") and scale_nmols) else 1.0

    # ----- Determine global vmin/vmax for duration -----
    if char_cho == "duration_global":
        vals_all = []
        for scenario in scenarios:
            if 'mean' in metrics:
                vals = np.nanmean(dummy_maps[config_cho][scenario][var_cho]["present_plus_meandelta"][char_cho] * scale, axis=0)
            elif 'max' in metrics:
                vals = np.nanmax(dummy_maps[config_cho][scenario][var_cho]["present_plus_meandelta"][char_cho] * scale, axis=0)
            vals_all.append(vals)
        vals_all = np.concatenate([v.ravel() for v in vals_all])
        vals_all = vals_all[np.isfinite(vals_all)]

        # Round to nearest 10 days for cleaner ticks
        vmin = np.floor(vals_all.min() / 10) * 10
        vmax = np.ceil(vals_all.max() / 10) * 10

        # Levels and ticks
        nlevs = 20
        levels = np.linspace(vmin, vmax, nlevs)
        tick_step = 25 if (vmax - vmin) > 200 else 10
        ticks = np.arange(vmin, vmax + tick_step, tick_step)

        # Remove last tick if it’s too close to vmax
        if np.isclose(ticks[-1], vmax, rtol=1e-3, atol=1e-3) or ticks[-1] > vmax:
            ticks = ticks[:-1]


    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    for i, scenario in enumerate(scenarios):
        ax = axes[i]

        if 'mean' in metrics:
            plot_vals = np.nanmean(dummy_maps[config_cho][scenario][var_cho]["present_plus_meandelta"][char_cho] * scale, axis=0)
        elif 'max' in metrics:
            plot_vals = np.nanmax(dummy_maps[config_cho][scenario][var_cho]["present_plus_meandelta"][char_cho] * scale, axis=0)

        # Apply mask
        plot_vals = xr.DataArray(
            plot_vals, dims=("eta_rho", "xi_rho"),
            coords={"lat": ocean_mask.lat, "lon": ocean_mask.lon}
        ).where(ocean_mask == 1)

        # Levels for intensities if not duration
        if char_cho != "duration_global":
            finite_vals = plot_vals.values[np.isfinite(plot_vals.values)]
            if finite_vals.size > 0:
                vmin = np.percentile(finite_vals, 2)
                vmax = np.percentile(finite_vals, 98)
            else:
                vmin, vmax = 0, 1
            levels = np.linspace(vmin, vmax, 15)
            ticks = None

        # Plot
        cf = ax.contourf(
            plot_vals.lon, plot_vals.lat, plot_vals,
            levels=levels,
            cmap=plt.get_cmap('cmo.thermal', len(levels)-1),
            extend='both'
        )

        # Land mask
        ax.contourf(
            landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
            levels=[0, 1], colors=["black"], zorder=3
        )

        # Region contours
        for region, col in {
            "offshore_central": "black",
            "coastal_central": "black"
        }.items():
            mask2d = model_regions_dict[region]["mask"]
            mask_plot = xr.where(np.isnan(mask2d), 0, 1)
            ax.contour(plot_vals.lon, plot_vals.lat, mask_plot, levels=[0.5], colors=[col], linewidths=1, zorder=4)

        # Titles and limits
        ax.set_title(scenario, fontsize=16, color=title_colors[scenario])
        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_xlabel("Longitude", fontsize=13)
        if i == 0:
            ax.set_ylabel("Latitude", fontsize=13)

        yt = np.arange(lat_bounds[0], lat_bounds[1] + 1, 3)
        xt = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yt)
        ax.set_xticks(xt)
        ax.set_yticklabels([f"{y}°N" for y in yt])
        ax.set_xticklabels([f"{360-x}°W" for x in xt])

    # Colorbar
    cax = fig.add_axes([0.25, -0.04, 0.5, 0.02])
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal")
    if ticks is not None:
        cbar.set_ticks(ticks)

    # Label logic
    if char_cho == 'duration_global':
        label_prefix = "Mean" if metrics == ['mean'] else "Max"
        cbar.set_label(f"{label_prefix} duration of extremes (days)", fontsize=14)
    else:
        label_prefix = "Max" if metrics == ['max'] else "Mean"
        cbar.set_label(f"{label_prefix} intensity ({char_cho_unit})", fontsize=14)

    cbar.ax.tick_params(labelsize=12)

    out = f"/nfs/sea/work/fpfaeffli/plots/extreme_char/maps_{metrics[0]}_{char_cho}.png"
    plt.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()


# Example call:
plot_hplus_extreme_map(
    dummy_maps=dummy_maps,
    ocean_mask=model_regions['roms_only']['all_dists_all_lats']['mask'],
    landmask_etopo=landmask_etopo,
    model_regions_dict=model_regions['roms_only'],
    config_cho=config_cho,
    var_cho=var_cho,
    char_cho=char_cho,
    metrics=metrics,
    char_cho_unit=char_cho_unit
)


#%%
# Plot intensity maps

def _nice_ticks(vmin, vmax, max_ticks=9):
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return np.array([0, 1])
    span = vmax - vmin
    raw = span / max(2, max_ticks-1)
    exp = np.floor(np.log10(raw))
    base = raw / 10**exp
    for m in (1, 2, 2.5, 5, 10):
        if base <= m:
            step = m * 10**exp
            break
    tick_min = np.floor(vmin/step)*step
    tick_max = np.ceil(vmax/step)*step
    ticks = np.arange(tick_min, tick_max + 0.5*step, step)
    if ticks[-1] > vmax + 1e-9:
        ticks = ticks[:-1]
    return ticks

def plot_hplus_extreme_map(
    dummy_maps,
    ocean_mask,
    landmask_etopo,
    model_regions_dict,
    config_cho,
    var_cho,
    char_cho,
    metrics,            # e.g. ['mean'] or ['max']
    char_cho_unit,      # pass 'days' for duration, else 'mol L$^{-1}$' (we’ll override to nmol if scaling)
    scale_nmols=True
):
    scenarios    = ["present","ssp245","ssp585"]
    title_colors = {"present":"#f08c3a","ssp245":"#65a342","ssp585":"#9553e0"}
    lat_bounds   = [32,46]
    lon_bounds   = [225,245]
    metric = 'mean' if ('mean' in metrics and 'max' not in metrics) else 'max'

    # scale (only matters for intensity)
    is_duration = (char_cho == "duration_global")
    scale = 1.0 if is_duration else (1e9 if (scale_nmols and char_cho.startswith("intensity")) else 1.0)
    unit_str = "days" if is_duration else ("nmol L$^{-1}$" if scale == 1e9 else "mol L$^{-1}$")

    # ---------- GLOBAL RANGE (across scenarios) ----------
    vals_all = []
    for sc in scenarios:
        block = dummy_maps[config_cho][sc][var_cho]["present_plus_meandelta"][char_cho] * scale
        vals_all.append(np.nanmean(block, axis=0) if metric == 'mean' else np.nanmax(block, axis=0))
    flat = np.concatenate([v.ravel() for v in vals_all])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        vmin, vmax = 0, 1
    else:
        if is_duration:
            # true range, rounded a bit so ticks look nice
            vmin = np.floor(flat.min()/10)*10
            vmax = np.ceil(flat.max()/10)*10
        else:
            # robust range for intensity
            vmin = np.percentile(flat, 2)
            vmax = np.percentile(flat, 98)

    ticks  = _nice_ticks(vmin, vmax, max_ticks=9)
    levels = np.linspace(ticks[0], ticks[-1], 21)  # many levels → smoother contours

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    cf = None

    for i, sc in enumerate(scenarios):
        ax = axes[i]
        block = dummy_maps[config_cho][sc][var_cho]["present_plus_meandelta"][char_cho] * scale
        field = np.nanmean(block, axis=0) if metric == 'mean' else np.nanmax(block, axis=0)

        da = xr.DataArray(field, dims=("eta_rho","xi_rho"),
                          coords={"lat": ocean_mask.lat, "lon": ocean_mask.lon}).where(ocean_mask==1)

        cf = ax.contourf(da.lon, da.lat, da, levels=levels,
                         cmap=plt.get_cmap('cmo.thermal', len(levels)-1), extend='both')

        # land + regions
        ax.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
                    levels=[0,1], colors=["black"], zorder=3)
        for region in ("offshore_central","coastal_central"):
            mask2d = model_regions_dict[region]["mask"]
            ax.contour(da.lon, da.lat, xr.where(np.isnan(mask2d),0,1),
                       levels=[0.5], colors=["black"], linewidths=1, zorder=4)

        ax.set_title(sc, fontsize=16, color=title_colors[sc])
        ax.set_xlim(lon_bounds); ax.set_ylim(lat_bounds)
        ax.set_xlabel("Longitude", fontsize=13)
        if i == 0: ax.set_ylabel("Latitude", fontsize=13)
        yt = np.arange(lat_bounds[0], lat_bounds[1]+1, 3)
        xt = np.arange(lon_bounds[0], lon_bounds[1]+1, 5)
        ax.set_yticks(yt); ax.set_xticks(xt)
        ax.set_yticklabels([f"{y}°N" for y in yt])
        ax.set_xticklabels([f"{360-x}°W" for x in xt])

    # ---------- Colorbar ----------
    cax = fig.add_axes([0.2, -0.06, 0.6, 0.03])
    fmt = ScalarFormatter(useMathText=True); fmt.set_powerlimits((-2, 3))
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks, format=fmt)

    # tidy tick label precision
    step = ticks[1]-ticks[0] if len(ticks)>1 else 1
    if step >= 1:
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
    elif step >= 0.1:
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))
    else:
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))

    label_prefix = metric.capitalize()
    if is_duration:
        cbar.set_label(f"{label_prefix} duration of extremes ({unit_str})", fontsize=14)
    else:
        cbar.set_label(f"{label_prefix} intensity ({unit_str})", fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    out = f"/nfs/sea/work/fpfaeffli/plots/extreme_char/maps_{metric}_{char_cho}.png"
    plt.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()


metrics = ['max']  # or ['max'] if you want max intensity
char_cho = 'intensity_max_abs'  # or 'intensity_max_abs' if you have it
char_cho_unit = 'mol L$^{-1}$'   # the function will auto-convert to nmol L⁻¹ if scale_nmols=True

plot_hplus_extreme_map(
    dummy_maps=dummy_maps,
    ocean_mask=model_regions['roms_only']['all_dists_all_lats']['mask'],
    landmask_etopo=landmask_etopo,
    model_regions_dict=model_regions['roms_only'],
    config_cho=config_cho,
    var_cho=var_cho,
    char_cho=char_cho,
    metrics=metrics,
    char_cho_unit=char_cho_unit,
    scale_nmols=True  # True → converts mol L⁻¹ to nmol L⁻¹ in plot + label
)





#%%
# Plot PDFs of characteristics, split by coastal vs offshore


from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def _clean_finite(a):
    a = np.asarray(a).ravel()
    return a[np.isfinite(a)]

def _split_by_region(dist2coast, coastal_km=100.0):
    mask_finite = np.isfinite(dist2coast)
    coastal_mask  = np.zeros_like(mask_finite, dtype=bool)
    offshore_mask = np.zeros_like(mask_finite, dtype=bool)
    coastal_mask[mask_finite]  = dist2coast[mask_finite] <  coastal_km
    offshore_mask[mask_finite] = dist2coast[mask_finite] >= coastal_km
    return coastal_mask, offshore_mask

def _kde_curve(values, n=512, pad_frac=0.05, bw_scale=1.0, jitter=0.0):
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    if v.size < 5:
        return None, None
    if jitter > 0:
        v = v + np.random.uniform(-jitter, jitter, size=v.size)
    lo, hi = v.min(), v.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None, None
    span = hi - lo
    lo -= pad_frac * span
    hi += pad_frac * span
    x = np.linspace(lo, hi, n)
    kde = gaussian_kde(v, bw_method=lambda s: s.scotts_factor() * bw_scale)
    y = kde(x)
    return x, y

def plot_pdfs_split_regions_y_sync_per_metric(
    characteristics_vecs,
    config="romsoc_fully_coupled",
    scenarios=("present", "ssp245", "ssp585"),
    varia="Hplus",
    threshold_type="present",
    coastal_km=50.0,
    scale_intensity_to_nmol=True,
    figsize=(10, 7.2)
):
    scen_colors = {"present":"#f08c3a","ssp245":"#65a342","ssp585":"#9553e0"}
    # Panels define the two metrics (metric index = 0 intensity, 1 duration)
    panels = [
        ("intensity_mean_abs", "Mean intensity (nmol L$^{-1}$)"),
        ("duration_global",    "Mean duration (days)"),
    ]

    # Map metric & region -> subplot position (row, col)
    # Desired layout:
    # upper-left: coastal intensity      -> (0,0)
    # upper-right: offshore intensity    -> (0,1)
    # lower-left: coastal duration       -> (1,0)
    # lower-right: offshore duration     -> (1,1)
    mapping = {
        "intensity_mean_abs": {"coastal": (0, 0), "offshore": (0, 1)},
        "duration_global":    {"coastal": (1, 0), "offshore": (1, 1)},
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    legend_handles, legend_labels = {}, {}
    col_ymax = np.zeros(len(panels), dtype=float)

    for col, (char_key, xlabel) in enumerate(panels):
        for region in ("coastal", "offshore"):
            pos = mapping[char_key][region]
            ax = axes[pos[0], pos[1]]

            for scen in scenarios:
                V = characteristics_vecs[config][scen][varia][threshold_type]
                d2c = np.asarray(V["dist2coast"])
                coastal_mask, offshore_mask = _split_by_region(d2c, coastal_km=coastal_km)
                mask = coastal_mask if region == "coastal" else offshore_mask

                vals = (np.asarray(V[char_key]) *
                        (1e9 if char_key.startswith("intensity") and scale_intensity_to_nmol else 1.0))
                vals_reg = vals[mask]
                x, y = _kde_curve(vals_reg)
                color = scen_colors.get(scen, "k")

                if x is None:
                    v = _clean_finite(vals_reg)
                    if v.size > 0:
                        bins = np.arange(0, 101)
                        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=color, label=scen)
                else:
                    line, = ax.plot(x, y, lw=2.0, color=color, label=scen)
                    col_ymax[col] = max(col_ymax[col], float(np.nanmax(y)))
                    legend_handles.setdefault(scen, line)
                    legend_labels.setdefault(scen, scen)

            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.tick_params(labelsize=14)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # force one decimal


            # add "Coastal"/"Offshore" with baseline type on top of EACH panel
            if threshold_type == "present":
                region_label = f"{region} (fixed baseline)"
            elif threshold_type == "present_plus_meandelta":
                region_label = f"{region} (adjusted baseline)"
            else:
                region_label = region
            ax.set_title(region_label, fontsize=14, pad=6)

            if char_key == "duration_global":
                ax.set_xlim(left=0, right=30)
            if char_key == "intensity_mean_abs":
                ax.set_xlim(left=-0.5, right=4)

    # synchronize y-limits per metric across the two subplots that show that metric
    for col, (char_key, _) in enumerate(panels):
        ymax = col_ymax[col]
        if np.isfinite(ymax) and ymax > 0:
            ytop = ymax * 1.05
            positions = list(mapping[char_key].values())
            for pos in positions:
                axes[pos[0], pos[1]].set_ylim(0, ytop)

    if legend_handles:
        ordered = [legend_handles[s] for s in scenarios if s in legend_handles]
        ordered_labels = [legend_labels[s] for s in scenarios if s in legend_labels]
        fig.legend(ordered, ordered_labels, loc="lower center",
                   ncol=len(ordered_labels), frameon=False, fontsize=12,
                   bbox_to_anchor=(0.5, -0.08))

    out = f'/nfs/sea/work/fpfaeffli/plots/extreme_char/pdfs_split_regions_{varia}_{threshold_type}.png'
    plt.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# ---- Run it ----
plot_pdfs_split_regions_y_sync_per_metric(
    characteristics_vecs,
    config="romsoc_fully_coupled",
    scenarios=("present", "ssp245", "ssp585"),
    varia="Hplus",
    threshold_type="present",
    coastal_km=50.0,
    scale_intensity_to_nmol=True
)
# %%

# Plot PDF's fixed and adjusted baseline in one plot for coastal and offshore baseline


def plot_pdfs_coastal_offshore_both_baselines(
    characteristics_vecs,
    config="romsoc_fully_coupled",
    scenarios=("present", "ssp245", "ssp585"),
    varia="Hplus",
    coastal_km=50.0,
    scale_intensity_to_nmol=True,
    figsize=(10, 7.2)
):
    scen_colors = {"present":"#f08c3a","ssp245":"#65a342","ssp585":"#9553e0"}
    panels = [
        ("intensity_mean_abs", "Mean intensity (nmol L$^{-1}$)"),
        ("duration_global",    "Mean duration (days)"),
    ]
    mapping = {
        "intensity_mean_abs": {"coastal": (0, 0), "offshore": (0, 1)},
        "duration_global":    {"coastal": (1, 0), "offshore": (1, 1)},
    }
    baselines = [
        ("present_plus_meandelta", "adjusted", "-"),  # solid
        ("present",                "fixed",    "--"), # dashed
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    col_ymax = np.zeros(len(panels), dtype=float)

    scenario_handles, style_handles = {}, {}

    for col, (char_key, xlabel) in enumerate(panels):
        for region in ("coastal", "offshore"):
            r, c = mapping[char_key][region]
            ax = axes[r, c]

            for scen in scenarios:
                # for present → only adjusted baseline
                blist = baselines if scen != "present" else [baselines[0]]

                for (thresh_type, baseline_name, ls) in blist:
                    V = characteristics_vecs[config][scen][varia][thresh_type]
                    d2c = np.asarray(V["dist2coast"])
                    coastal_mask, offshore_mask = _split_by_region(d2c, coastal_km=coastal_km)
                    mask = coastal_mask if region == "coastal" else offshore_mask

                    vals = np.asarray(V[char_key]) * (1e9 if char_key.startswith("intensity") and scale_intensity_to_nmol else 1.0)
                    vals_reg = vals[mask]

                    if char_key == "duration_global":
                        v = _clean_finite(vals_reg)
                        if v.size > 0:
                            h = ax.hist(v, bins=np.arange(0, 31), density=True, histtype="step",
                                        lw=1.6, color=scen_colors.get(scen,"k"), linestyle=ls)
                            ymax_here = (h[0].max() if len(h[0]) else 0.0)
                            col_ymax[col] = max(col_ymax[col], ymax_here)
                            style_handles.setdefault(ls, Line2D([0],[0], color="gray", lw=2.0, linestyle=ls))
                            scenario_handles.setdefault(scen, Line2D([0],[0], color=scen_colors[scen], lw=2.0))
                    else:
                        x, y = _kde_curve(vals_reg)
                        color = scen_colors.get(scen, "k")
                        if x is not None:
                            line, = ax.plot(x, y, lw=2.0, color=color, linestyle=ls)
                            col_ymax[col] = max(col_ymax[col], float(np.nanmax(y)))
                            style_handles.setdefault(ls, Line2D([0],[0], color="gray", lw=2.0, linestyle=ls))
                            scenario_handles.setdefault(scen, Line2D([0],[0], color=color, lw=2.0))

            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.tick_params(labelsize=14, labelleft=True, labelright=False)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.set_title(region, fontsize=14, pad=6)

            if char_key == "duration_global":
                ax.set_xlim(left=0, right=30)
            if char_key == "intensity_mean_abs":
                ax.set_xlim(left=-0.5, right=4)

    for col in range(len(panels)):
        ymax = col_ymax[col]
        if np.isfinite(ymax) and ymax > 0:
            ytop = ymax * 1.05
            for region in ("coastal", "offshore"):
                r, c = mapping[panels[col][0]][region]
                axes[r, c].set_ylim(0, ytop)

    scen_order = [s for s in scenarios if s in scenario_handles]
    scen_h = [scenario_handles[s] for s in scen_order]
    scen_l = scen_order

    baseline_entries = [("adjusted baseline", "-"), ("fixed baseline", "--")]
    style_h = [style_handles[ls] for _, ls in baseline_entries if ls in style_handles]
    style_l = [name for name, ls in baseline_entries if ls in style_handles]

    if scen_h:
        fig.legend(scen_h, scen_l, loc="lower center", ncol=len(scen_h),
                   frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.06))
    if style_h:
        fig.legend(style_h, style_l, loc="lower center", ncol=len(style_h),
                   frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.12))

    out = f'/nfs/sea/work/fpfaeffli/plots/extreme_char/pdfs_split_regions_{varia}_both_baselines.png'
    plt.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()


# ---- Run it ----
plot_pdfs_coastal_offshore_both_baselines(
    characteristics_vecs,
    config="romsoc_fully_coupled",
    scenarios=("present", "ssp245", "ssp585"),
    varia="Hplus",
    coastal_km=50.0,
    scale_intensity_to_nmol=True
)

#%%

# %%
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

def _clean_finite(a):
    a = np.asarray(a).ravel()
    return a[np.isfinite(a)]

def _kde_curve(values, n=512, pad_frac=0.05, bw_scale=1.0, jitter=0.0):
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    if v.size < 5:
        return None, None
    if jitter > 0:
        v = v + np.random.uniform(-jitter, jitter, size=v.size)
    lo, hi = v.min(), v.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None, None
    span = hi - lo
    lo -= pad_frac * span
    hi += pad_frac * span
    x = np.linspace(lo, hi, n)
    kde = gaussian_kde(v, bw_method=lambda s: s.scotts_factor() * bw_scale)
    y = kde(x)  # raw density
    return x, y

def plot_pdfs_allpoints_both_baselines(
    characteristics_vecs,
    config="romsoc_fully_coupled",
    scenarios=("present","ssp245","ssp585"),
    varia="Hplus",
    scale_intensity_to_nmol=True,
    figsize=(12, 4.5)
):
    scen_colors = {"present":"#f08c3a","ssp245":"#65a342","ssp585":"#9553e0"}
    panels = [
        ("intensity_mean_abs", "Mean intensity (nmol L$^{-1}$)"),
        ("duration_global",    "Mean duration (days)"),
    ]
    baselines = [
        ("present_plus_meandelta", "adjusted", "-"),   # solid
        ("present",                "fixed",    "--"),  # dashed
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    col_ymax = np.zeros(len(panels), dtype=float)

    scenario_handles, style_handles = {}, {}

    for col, (char_key, xlabel) in enumerate(panels):
        ax = axes[col]

        for scen in scenarios:
            # For present → only adjusted baseline
            blist = baselines if scen != "present" else [baselines[0]]

            for (thresh_type, baseline_name, ls) in blist:
                V = characteristics_vecs[config][scen][varia][thresh_type]

                # pool ALL points (no coastal/offshore split)
                vals = np.asarray(V[char_key]) * (1e9 if char_key.startswith("intensity") and scale_intensity_to_nmol else 1.0)

                # KDE for both metrics
                x, y = _kde_curve(vals)
                color = scen_colors.get(scen, "k")
                if x is not None:
                    line, = ax.plot(x, y, lw=2.0, color=color, linestyle=ls)
                    col_ymax[col] = max(col_ymax[col], float(np.nanmax(y)))
                    style_handles.setdefault(ls, Line2D([0],[0], color="gray", lw=2.0, linestyle=ls))
                    scenario_handles.setdefault(scen, Line2D([0],[0], color=color, lw=2.0))

        # cosmetics
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=12, labelleft=True, labelright=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if char_key == "duration_global":
            ax.set_xlim(left=0, right=30)
        if char_key == "intensity_mean_abs":
            ax.set_xlim(left=-0.5, right=4)

        # set y-limit per panel, based on max seen
        ymax = col_ymax[col]
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, ymax * 1.05)

    # Build legends: scenarios (colors) and baselines (styles)
    scen_order = [s for s in scenarios if s in scenario_handles]
    scen_h = [scenario_handles[s] for s in scen_order]
    scen_l = scen_order

    baseline_entries = [("adjusted baseline", "-"), ("fixed baseline", "--")]
    style_h = [style_handles[ls] for _, ls in baseline_entries if ls in style_handles]
    style_l = [name for name, ls in baseline_entries if ls in style_handles]

    if scen_h:
        fig.legend(scen_h, scen_l, loc="lower center", ncol=len(scen_h),
                   frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.08))
    if style_h:
        fig.legend(style_h, style_l, loc="lower center", ncol=len(style_h),
                   frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.16))

    plt.savefig(f'/nfs/sea/work/fpfaeffli/plots/extreme_char/pdfs_allregions_gaussian_kde_{varia}_both_baselines.png')
    plt.show()

# ---- Run it ----
plot_pdfs_allpoints_both_baselines(
    characteristics_vecs,
    config="romsoc_fully_coupled",
    scenarios=("present","ssp245","ssp585"),
    varia="Hplus",
    scale_intensity_to_nmol=True
)

#%%

# with discrete KDE
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

def _clean_finite(a):
    a = np.asarray(a).ravel()
    return a[np.isfinite(a)]

def _kde_curve(values, n=512, pad_frac=0.05, bw_scale=1.0):
    """Continuous KDE for intensity."""
    v = _clean_finite(values)
    if v.size < 5:
        return None, None
    lo, hi = v.min(), v.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None, None
    span = hi - lo
    lo -= pad_frac * span
    hi += pad_frac * span
    x = np.linspace(lo, hi, n)
    kde = gaussian_kde(v, bw_method=lambda s: s.scotts_factor() * bw_scale)
    y = kde(x)  # raw density
    return x, y

def _discrete_kde_integer(values, grid_min=None, grid_max=None, sigma_days=0.5, normalize=True):
    """
    Discrete KDE for integer-valued data (e.g., duration in days).
    1) Build integer counts on grid,
    2) Smooth with Gaussian kernel on the integer grid (sigma in days),
    3) Return (grid, smoothed_pmf).
    """
    v = _clean_finite(values)
    if v.size == 0:
        return None, None
    # Ensure integers (your duration is already integer by construction)
    v = np.asarray(v, dtype=int)
    if grid_min is None: grid_min = int(v.min())
    if grid_max is None: grid_max = int(v.max())
    if grid_max < grid_min:
        return None, None
    grid = np.arange(grid_min, grid_max + 1, dtype=int)

    # Histogram on integer grid
    counts = np.bincount(v - grid_min, minlength=grid.size).astype(float)

    # Smooth on integer grid with Gaussian kernel
    smoothed = gaussian_filter1d(counts, sigma=sigma_days, mode="constant")

    if normalize:
        total = smoothed.sum()
        if total > 0:
            smoothed = smoothed / total  # PMF over integer grid (Δx = 1)
        else:
            return None, None

    return grid, smoothed

#%%
def plot_pdfs_allpoints_both_baselines_discrete_duration(
    characteristics_vecs,
    config="romsoc_fully_coupled",
    scenarios=("present","ssp245","ssp585"),
    varia="Hplus",
    scale_intensity_to_nmol=True,
    duration_xlim=(0, 30),
    duration_sigma_days=0.5,   # <-- controls smoothness on integer grid
    figsize=(10, 4.8)
):
    scen_colors = {"present":"#f08c3a","ssp245":"#65a342","ssp585":"#9553e0"}
    panels = [
        ("intensity_mean_abs", "Mean intensity (nmol L$^{-1}$)"),
        ("duration_global",    "Mean duration (days)"),
    ]
    baselines = [
        ("present_plus_meandelta", "adjusted", "-"),   # solid
        ("present",                "fixed",    "--"),  # dashed
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    col_ymax = np.zeros(len(panels), dtype=float)
    scenario_handles, style_handles = {}, {}

    for col, (char_key, xlabel) in enumerate(panels):
        ax = axes[col]

        for scen in scenarios:
            # For present → only adjusted baseline
            blist = baselines if scen != "present" else [baselines[0]]

            for (thresh_type, baseline_name, ls) in blist:
                V = characteristics_vecs[config][scen][varia][thresh_type]
                vals = np.asarray(V[char_key]) * (1e9 if char_key.startswith("intensity") and scale_intensity_to_nmol else 1.0)
                color = scen_colors.get(scen, "k")

                if char_key == "duration_global":
                    # Discrete KDE on integer grid
                    grid_min = duration_xlim[0]
                    grid_max = max(duration_xlim[1], int(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else duration_xlim[1])
                    xg, yg = _discrete_kde_integer(vals, grid_min=grid_min, grid_max=grid_max, sigma_days=duration_sigma_days, normalize=True)
                    if xg is not None:
                        line, = ax.plot(xg, yg, lw=2.5, color=color, linestyle=ls)
                        col_ymax[col] = max(col_ymax[col], float(np.nanmax(yg)))
                        style_handles.setdefault(ls, Line2D([0],[0], color="gray", lw=2.5, linestyle=ls))
                        scenario_handles.setdefault(scen, Line2D([0],[0], color=color, lw=2.5))
                else:
                    # Continuous KDE for intensity
                    x, y = _kde_curve(vals)
                    if x is not None:
                        line, = ax.plot(x, y, lw=1.5, color=color, linestyle=ls)
                        col_ymax[col] = max(col_ymax[col], float(np.nanmax(y)))
                        style_handles.setdefault(ls, Line2D([0],[0], color="gray", lw=2.5, linestyle=ls))
                        scenario_handles.setdefault(scen, Line2D([0],[0], color=color, lw=2.5))

        # cosmetics
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel("Density", fontsize=20)
        ax.grid(True, alpha=0.25, linestyle="-")
        ax.tick_params(labelsize=20, labelleft=True, labelright=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if char_key == "duration_global":
            ax.set_xlim(left=duration_xlim[0], right=duration_xlim[1])
        if char_key == "intensity_mean_abs":
            ax.set_xlim(left=-0.5, right=4)

        if char_key == "intensity_mean_abs":
            ax.set_xlim(left=-0.5, right=4)
            ax.set_ylim(0, 4.0)      # <-- fixed range for intensity PDFs

        if char_key == "duration_global":
            ax.set_xlim(left=duration_xlim[0], right=duration_xlim[1])
            ax.set_ylim(0, 0.2)      # <-- fixed range for duration PDFs


    # Legends: scenarios (colors) and baselines (styles)
    scen_order = [s for s in scenarios if s in scenario_handles]
    scen_h = [scenario_handles[s] for s in scen_order]
    scen_l = scen_order

    baseline_entries = [("adjusted baseline", "-"), ("fixed baseline", "--")]
    style_h = [style_handles[ls] for _, ls in baseline_entries if ls in style_handles]
    style_l = [name for name, ls in baseline_entries if ls in style_handles]

    if scen_h:
        fig.legend(scen_h, scen_l, loc="lower center", ncol=len(scen_h),
                   frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.08))
    if style_h:
        fig.legend(style_h, style_l, loc="lower center", ncol=len(style_h),
                   frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.16))
    plt.savefig(f'/nfs/sea/work/fpfaeffli/plots/extreme_char/smoothed_pdfs_allregions_gaussian_kde_{varia}_both_baselines.png')
    plt.show()

# ---- Run it ----
plot_pdfs_allpoints_both_baselines_discrete_duration(
    characteristics_vecs,
    config="romsoc_fully_coupled",
    scenarios=("present","ssp245","ssp585"),
    varia="Hplus",
    scale_intensity_to_nmol=True,
    duration_xlim=(0, 30),       # <- change window if needed
    duration_sigma_days=0.5       # <- smaller = more discrete-looking; larger = smoother
)

# %%
