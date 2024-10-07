"""
author: Eike Köhn
date: June 18, 2024
description: This file plots a snapshot of the present day and future blob for the different thresholds
"""

#%% enable the visibility of the modules for the import functions
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/'))
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
from get_model_datasets import ModelGetter as ModelGetter
from get_obs_datasets import ObsGetter as ObsGetter
from plotting_functions_extremes import ExtAnalysisPlotter as ExtAnalysisPlotter # FIXME : where is this file?


import multiprocessing
from tqdm import tqdm

from importlib import reload  # Python 3.4+
import plotting_functions_extremes
reload(plotting_functions_extremes)
from plotting_functions_extremes import ExtAnalysisPlotter as ExtAnalysisPlotter

import matplotlib.patheffects as pe


import get_obs_datasets 
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter

sys.path.append('/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/climatologies_and_thresholds')
from set_thresh_and_clim_params import ThresholdParameters
params = ThresholdParameters.standard_instance()

from funcs_for_clim_thresh import ThreshClimFuncs

import funcs_for_clim_thresh
reload(funcs_for_clim_thresh)
from funcs_for_clim_thresh import ThreshClimFuncs

import xesmf as xe

from plotting_functions_general import PlotFuncs as PlotFuncs


#%%

model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present','ssp245','ssp585'] # ,'ssp245'
configs = ['roms_only'] #['romsoc_fully_coupled'] # 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'zavg'    # 'avg'

#%% Get the model datasets for the oceanic and atmospheric variables

ocean_ds = dict()
atmosphere_ds = dict()
pressure_ds = dict()
cloud_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     atmosphere_ds[config] = dict()
     pressure_ds[config] = dict()
     cloud_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)
        #   print('atmosphere...')
        #   atmosphere_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,'avg',vtype='atmospheric',parent_model=parent_model,vtype_extra=None)
        #   atmosphere_ds[config][scenario]['lon'] = atmosphere_ds[config][scenario]['lon']+360.
        #   print('pressure...')
        #   pressure_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,'avg',vtype='atmospheric',parent_model=parent_model,vtype_extra='pressure')
        #   pressure_ds[config][scenario]['lon'] = pressure_ds[config][scenario]['lon']+360.
        #   print('clouds...')
        #   cloud_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,'avg',vtype='atmospheric',parent_model=parent_model,vtype_extra='clouds')
        #   cloud_ds[config][scenario]['lon'] = cloud_ds[config][scenario]['lon']+360.

#%% load the data at the respective location

variables = dict()
for config in configs:
     variables[config] = dict()
     for scenario in scenarios:
          print(f'Getting the variables for config {config} and scenrio {scenario}.')
          #
          # oceanic variables
          variables[config][scenario] = dict()
          print('sst...')
          variables[config][scenario]['sst'] = ocean_ds[config][scenario].temp.isel(depth=0).load() #.isel(depth=0,eta_rho=ocean_li[0],xi_rho=ocean_li[1]

#%%
varias = ['sst']

#%% GET THE CLIMATOLOGIES FOR EACH VARIABLE
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

#%% GET THE THRESHOLD FOR EACH VARIABLE
print('Get the present day threshold')
varia = 'sst'
thresholds = dict()
for config in configs:
     thresholds[config] = dict()
     for scenario in ['present']:#,'ssp245','ssp585']:
          thresholds[config][scenario] = dict()
          print(f'{config}, {scenario}')
          threshold, threshold_366 = ModelGetter.get_threshold('temp',0,'relative',90,config,scenario)
          thresholds[config][scenario][varia] = ModelGetter.concatenate_yearly_arrays(threshold,threshold_366,start_year=2011,end_year=2021)

#%%Adjust the thresholds
thresholds_mult = dict()
for config in configs:
     thresholds_mult[config] = dict()
     for scenario in ['present','ssp245','ssp585']:
          thresholds_mult[config][scenario] = dict()
          print(f'{config}, {scenario}')
          thresholds_mult[config][scenario]['present'] = thresholds[config]['present'][varia]
          thresholds_mult[config][scenario]['present_plus_meandelta'] = thresholds[config]['present'][varia] + (clims[config][scenario][varia].mean(dim='time') - clims[config]['present'][varia].mean(dim='time'))


# %%

#% Get the distance to coast file from ROMS and the regions over which to calculate the statistics
model_d2coasts = dict()
model_d2coasts['roms_only'] = ModelGetter.get_distance_to_coast(vtype='oceanic')

# Get the model regions
model_regions = dict()
model_regions['roms_only'] = GetRegions.define_CalCS_regions(model_d2coasts['roms_only'].lon, model_d2coasts['roms_only'].lat, model_d2coasts['roms_only'])

#%%
model_area = ModelGetter.get_model_area()


#%% 2D heatmaps for future vs present anomalies (future anomalies are on the xaxis)

configo = configs[0]

present = variables[configo]['present']['sst'] - thresholds[configo]['present']['sst']

for scenario in ['ssp245','ssp585']:
    for threshold_type in ['present','present_plus_meandelta']:#,]: #= 'present_plus_climdelta'

        #if threshold_type == 'present_plus_climdelta':
        #    future = variables[configo]['ssp585']['sst'] - thresholds_mult[configo]['ssp585']['present_plus_climdelta']#- (thresholds[configo]['present']['sst'] + clims[configo]['ssp585']['sst'] - clims[configo]['present']['sst'])
        if threshold_type == 'present_plus_meandelta':
            future = variables[configo][scenario]['sst'] - thresholds_mult[configo][scenario]['present_plus_meandelta']#(thresholds[configo]['present']['sst'] + variables[configo]['ssp585']['sst'].mean(dim='time') - variables[configo]['present']['sst'].mean(dim='time'))
        elif threshold_type == 'present':
            future = variables[configo][scenario]['sst'] - thresholds_mult[configo][scenario]['present']#- (thresholds[configo]['present']['sst'])

        #%
        for region_cho in ['coastal_all_lats']:#,'offshore_all_lats','coastal_northern','coastal_central','coastal_southern']:
            present_masked = present*model_regions['roms_only'][region_cho]['mask']
            future_masked = future*model_regions['roms_only'][region_cho]['mask']

            #
            bins = [np.linspace(-20,20,401),np.linspace(-20,21,411)]
            hist2d,xedges,yedges = np.histogram2d(future_masked.values.flatten(),present_masked.values.flatten(),bins=bins)
            hist2d[hist2d==0]=np.NaN

            #
            totnum = np.nansum(hist2d)
            future_extremes  = np.nansum(hist2d[xedges[:-1]>=0,:])/totnum
            present_extremes = np.nansum(hist2d[:,yedges[:-1]>=0])/totnum

            non_extremes = np.nansum(hist2d[(xedges[:-1]<0)[:,None]*(yedges[:-1]<0)])/totnum * 100
            new_extremes = np.nansum(hist2d[(xedges[:-1]>=0)[:,None]*(yedges[:-1]<0)])/totnum * 100
            disappearing_extremes = np.nansum(hist2d[(xedges[:-1]<0)[:,None]*(yedges[:-1]>=0)])/totnum * 100

            upper_right_array = hist2d[xedges[:-1]>=0,:][:,yedges[:-1]>=0]
            weakening_extremes_dum = np.triu(upper_right_array,k=1)
            weakening_extremes_dum[weakening_extremes_dum==0]=np.NaN
            intensifying_extremes_dum = np.tril(upper_right_array,k=0)
            intensifying_extremes_dum[intensifying_extremes_dum==0]=np.NaN

            intensifying_extremes = np.nansum(intensifying_extremes_dum)/totnum * 100
            weakening_extremes = np.nansum(weakening_extremes_dum)/totnum * 100


            if threshold_type == 'present':
                xmin = -9
                xmax = 12
            else:
                xmin = -10
                xmax = 10

            print('Make the plot')
            #% MAKE THE PLOT
            fontsize=12
            plt.rcParams['font.size']=fontsize
            fig, ax = plt.subplots(2, 2, figsize=(7.6, 7), gridspec_kw={'hspace': 0, 'wspace': 0, 'width_ratios': [5, 1], 'height_ratios': [1, 5]})
            # Turn off axes for specific subplots
            ax[0, 0].axis("off")#spines[['right', 'top','bottom']].set_visible(False)
            ax[0, 1].axis("off")
            ax[1, 1].axis("off")

            c10 = ax[1,0].pcolormesh(xedges[:-1],yedges[:-1],np.log10(hist2d.T/totnum*100),cmap=plt.get_cmap('cmo.ice',10),vmin=-6,vmax=-1)
            ax[1,0].axhline(0,color='C1')
            ax[1,0].axvline(0,color='C1')
            ax[1,0].plot([0,xmax],[0,xmax],color='C1')
            if threshold_type == 'present':
                ax[1,0].text(8.5,8.5,'1:1',color='C1',ha='right',va='top',bbox={'boxstyle':'round','fc':'w','alpha':0.75})
            else:
                ax[1,0].text(0.98,0.98,'1:1',transform=ax[1,0].transAxes,color='C1',ha='right',va='top',bbox={'boxstyle':'round','fc':'w','alpha':0.75})
            ax[1,0].set_ylim([-11,9])
            ax[1,0].set_xlim([xmin,xmax])
            ax[1,0].set_ylabel('Present-day SST anomalies rel. to thresh. in °C')
            if threshold_type == 'present_plus_meandelta':
                ax[1,0].set_xlabel(f'{scenario.upper()} SST anomalies rel. to threshold in °C\n(moving baseline, i.e., mean warming adjusted present-day threshold)')
            elif threshold_type == 'present_plus_climdelta':
                ax[1,0].set_xlabel(f'{scenario.upper()} SST anomalies rel. to threshold in °C\n(clim. warming adjusted present-day threshold)')         
            elif threshold_type == 'present':
                ax[1,0].set_xlabel(f'{scenario.upper()} SST anomalies rel. to threshold in °C\n(fixed baseline, i.e., present-day threshold)')    
            #ax[0,0].set_title(region_cho,loc='left')

            ax[0,0].plot(yedges[:-1],np.nansum(hist2d,axis=0)/totnum,color='k',linewidth=0.5)
            ax[0,0].plot(xedges[:-1],np.nansum(hist2d,axis=1)/totnum,color='r')
            ax[0,0].fill_between(xedges[:-1],np.nansum(hist2d,axis=1)/totnum,0,alpha=0.3,color='r')
            ax[0,0].set_xlim([xmin,xmax])
            ax[0,0].set_ylim([0,0.03])
            ax[0,0].plot([0,0],[0,0.03],color='C1')
            ax[0,0].plot([xmin+2.52,xmax-7],[0.01,0.01],'#888888',linestyle='--',linewidth=0.5)
            ax[0,0].plot([xmin+2.52,xmax-7],[0.02,0.02],'#888888',linestyle='--',linewidth=0.5)
            ax[0,0].text(xmin+2.5,0.01,'0.01',ha='right',va='center',color='#888888')
            ax[0,0].text(xmin+2.5,0.02,'0.02',ha='right',va='center',color='#888888')
            ax[0,0].text(xmin+1-0.02,0.015,'  Density',ha='right',va='center',color='#888888',rotation=90)
            ax[0,0].text(xmax-6,0.015,f'Future extreme\npercentage:\n{100*future_extremes:.2f}%',color='r',va='center')

            ax[1,1].fill_betweenx(yedges[:-1],np.nansum(hist2d,axis=0)/totnum,0,color='k',alpha=0.3)
            ax[1,1].plot(np.nansum(hist2d,axis=0)/totnum,yedges[:-1],color='k')
            ax[1,1].set_ylim([-11,9])
            ax[1,1].set_xlim([0,0.03])
            ax[1,1].plot([0,0.029],[0,0],color='C1')
            ax[1,1].plot([0.01,0.01],[-9,4],'#888888',linestyle='--',linewidth=0.5)
            ax[1,1].plot([0.02,0.02],[-8,3],'#888888',linestyle='--',linewidth=0.5)
            ax[1,1].text(0.01,-9.02,'0.01',ha='center',va='top',color='#888888')
            ax[1,1].text(0.02,-8.02,'0.02',ha='center',va='top',color='#888888')
            ax[1,1].text(0.002,6,f'Present-day\nextreme\npercentage:\n{100*present_extremes:.2f}%',va='center')
            cbax = fig.add_axes([0.15,0.79,0.3,0.025])
            plt.colorbar(c10,cax=cbax,orientation='horizontal',label=r'log$_{10}$ %',extend='min')

            plt.tight_layout()
            future_extreme_increase = (future_extremes-present_extremes)/present_extremes*100
            fig.text(0.84,0.85,f'Relative\nfuture\nextreme\nincrease:\n{future_extreme_increase:.2f}%',bbox={'boxstyle':'round','fc':'w'},color='C0')

            ax[1,0].text(0.25,0.3,f'non-\nextremes:\n{non_extremes:.2f}%',color='C1',ha='center',va='center',transform=ax[1,0].transAxes,fontsize=fontsize+4,fontweight='bold')
            ax[1,0].text(0.75,0.15,f'new\nextremes:\n{new_extremes:.2f}%',color='C1',ha='center',va='center',transform=ax[1,0].transAxes,fontsize=fontsize+4,fontweight='bold')
            ax[1,0].text(0.18,0.72,f'disappeared\nextremes:\n{disappearing_extremes:.2f}%',color='C1',ha='center',va='center',transform=ax[1,0].transAxes,fontsize=fontsize+4,fontweight='bold')

            if threshold_type == 'present':
                ax[1,0].text(0.72,0.65,f'intensified:\n{intensifying_extremes:.2f}%',color='C1',ha='center',va='center',transform=ax[1,0].transAxes,fontsize=fontsize+4,fontweight='bold')
                ax[1,0].text(0.57,0.88,f'weaker:\n{weakening_extremes:.2f}%',color='C1',ha='center',va='center',transform=ax[1,0].transAxes,fontsize=fontsize+4,fontweight='bold')
            else:
                ax[1,0].text(0.77,0.63,f'intensified:\n{intensifying_extremes:.2f}%',color='C1',ha='center',va='center',transform=ax[1,0].transAxes,fontsize=fontsize+4,fontweight='bold')#,rotation=45)
                ax[1,0].text(0.62,0.9,f'weaker:\n{weakening_extremes:.2f}%',color='C1',ha='center',va='center',transform=ax[1,0].transAxes,fontsize=fontsize+4,fontweight='bold')#,rotation=45)

            savedir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/histograms/'
            filename = f'future_vs_present_anomalies_rel_to_thresh_{threshold_type}_for_region_{region_cho}_flipped_{scenario}_{configo}.png'
            plt.savefig(savedir+filename,dpi=200,transparent=True)
            plt.show()

#%%  PLOT THE TIMESERIES OF THE AREA AFFECTED BY THE DIFFERENT TYPES OF EXTREME COMBINATIONS

present = variables['romsoc_fully_coupled']['present']['sst'] - thresholds['romsoc_fully_coupled']['present']['sst']

for scenario in ['ssp245','ssp585']:
    for threshold_type in ['present','present_plus_meandelta']: #= 'present_plus_climdelta'

        #if threshold_type == 'present_plus_climdelta':
        #    future = variables['romsoc_fully_coupled']['ssp585']['sst'] - thresholds_mult['romsoc_fully_coupled']['ssp585']['present_plus_climdelta']#- (thresholds['romsoc_fully_coupled']['present']['sst'] + clims['romsoc_fully_coupled']['ssp585']['sst'] - clims['romsoc_fully_coupled']['present']['sst'])
        if threshold_type == 'present_plus_meandelta':
            future = variables['romsoc_fully_coupled'][scenario]['sst'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_meandelta']#(thresholds['romsoc_fully_coupled']['present']['sst'] + variables['romsoc_fully_coupled']['ssp585']['sst'].mean(dim='time') - variables['romsoc_fully_coupled']['present']['sst'].mean(dim='time'))
        elif threshold_type == 'present':
            future = variables['romsoc_fully_coupled'][scenario]['sst'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present']#- (thresholds['romsoc_fully_coupled']['present']['sst'])

        for region_cho in ['coastal_all_lats']:#,'offshore_all_lats','coastal_northern','coastal_central','coastal_southern']:

            non_extremes_mask = (future<=0)*(present<=0)
            new_extremes_mask = (future>0)*(present<=0)
            disappearing_extremes_mask = (future<=0)*(present>0)
            intensifying_extremes_mask = (future>0)*(present>0)*(future>=present)
            weakening_extremes_mask = (future>0)*(present>0)*(future<present)

            #%
            region_area = model_regions['roms_only'][region_cho]['mask'].weighted(model_area.fillna(0)).sum(('eta_rho','xi_rho'))

            #%
            non_extremes_masked_area = (non_extremes_mask*model_regions['roms_only'][region_cho]['mask']).weighted(model_area.fillna(0)).sum(('eta_rho','xi_rho'))/region_area
            new_extremes_masked_area = (new_extremes_mask*model_regions['roms_only'][region_cho]['mask']).weighted(model_area.fillna(0)).sum(('eta_rho','xi_rho'))/region_area
            disappearing_extremes_masked_area = (disappearing_extremes_mask*model_regions['roms_only'][region_cho]['mask']).weighted(model_area.fillna(0)).sum(('eta_rho','xi_rho'))/region_area
            intensifying_extremes_masked_area = (intensifying_extremes_mask*model_regions['roms_only'][region_cho]['mask']).weighted(model_area.fillna(0)).sum(('eta_rho','xi_rho'))/region_area
            weakening_extremes_masked_area = (weakening_extremes_mask*model_regions['roms_only'][region_cho]['mask']).weighted(model_area.fillna(0)).sum(('eta_rho','xi_rho'))/region_area


            #%
            fontsize=12
            plt.rcParams['font.size']=12
            fix, ax = plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'width_ratios':[2.5,1]})
            labels = ['intensified','weaker','new','disappeared']
            colors = ['#d62728','#1f77b4','#ff7f0e','#CCCCCC']#,'#2ca02c']
            ax[0].stackplot(non_extremes_masked_area.time,intensifying_extremes_masked_area,weakening_extremes_masked_area,new_extremes_masked_area,disappearing_extremes_masked_area,labels=labels,colors=colors)
            #ax.plot(non_extremes_masked_area.time,new_extremes_masked_area,label='new',color='C2')
            # ax.plot(non_extremes_masked_area.time,intensifying_extremes_masked_area,label='intensifying extremes')
            # ax.plot(non_extremes_masked_area.time,weakening_extremes_masked_area,label='weakening extremes')
            ax[0].set_xlim([pd.to_datetime('2011-01-01'),pd.to_datetime('2022-01-01')])
            xticks = pd.date_range('2011-01-01','2022-01-01',freq='1Ys')
            ax[0].set_xticks(xticks)
            xticklabs = [xtick.strftime('%Y/%m') for xtick in xticks]
            xticklabs2 = []
            for i in range(len(xticklabs)):
                if np.mod(i,2)==0:
                    xticklabs2.append(xticklabs[i])
                else:
                    xticklabs2.append('')
            ax[0].set_xticklabels(xticklabs2)
            ax[0].set_ylim([0,1])
            ax[0].legend(loc='lower left',ncol=4,bbox_to_anchor=(0.01,1),handlelength=1,handletextpad=0.4)
            ax[0].set_xlabel('Time')
            ax[0].set_ylabel('Area fraction')
            ax[0].grid(linestyle='--',color='#888888',alpha=0.5)

            offsets = [-0.3,-0.1,0.1,0.3]
            for adx,arr in enumerate([intensifying_extremes_masked_area,weakening_extremes_masked_area,new_extremes_masked_area,disappearing_extremes_masked_area]):

                for month in range(1,13):
                    #monthly_arr = arr.resample(time="1MS").mean(dim="time")
                    monthly_data = arr.sel(time=(arr['time.month']==month))
                    ax[1].boxplot(monthly_data.values,
                                positions=[month+offsets[adx]],
                                patch_artist=True,
                                boxprops=dict(facecolor=colors[adx], color=colors[adx],linewidth=0),
                                showcaps=False,
                                showfliers=False,
                                showmeans=False,
                                whis=0,
                                widths=0.2,
                                medianprops=dict(color='k'),
                                )
            ax[1].set_xticks(np.arange(1,13))
            ax[1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            ax[1].set_xlabel('Month')
            ax[1].set_xlim([0.5,12.5])
            ax[1].grid(linestyle='--',color='#888888',alpha=0.5)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.15)
            savedir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/histograms/'
            filename = f'future_vs_present_anomalies_rel_to_thresh_{threshold_type}_for_region_{region_cho}_timeseries_{scenario}.png'
            plt.savefig(savedir+filename,dpi=200,transparent=True)
            plt.show()

#%%

present = variables['romsoc_fully_coupled']['present']['sst'] - thresholds['romsoc_fully_coupled']['present']['sst']

for threshold_type in ['present','present_plus_climdelta','present_plus_meandelta']: #= 'present_plus_climdelta'
    scenario = 'ssp585'
    #if threshold_type == 'present_plus_climdelta':
    #    future = variables['romsoc_fully_coupled']['ssp585']['sst'] - thresholds_mult['romsoc_fully_coupled']['ssp585']['present_plus_climdelta']#- (thresholds['romsoc_fully_coupled']['present']['sst'] + clims['romsoc_fully_coupled']['ssp585']['sst'] - clims['romsoc_fully_coupled']['present']['sst'])
    if threshold_type == 'present_plus_meandelta':
        future = variables['romsoc_fully_coupled'][scenario]['sst'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_meandelta']#(thresholds['romsoc_fully_coupled']['present']['sst'] + variables['romsoc_fully_coupled']['ssp585']['sst'].mean(dim='time') - variables['romsoc_fully_coupled']['present']['sst'].mean(dim='time'))
    elif threshold_type == 'present':
        future = variables['romsoc_fully_coupled'][scenario]['sst'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present']#- (thresholds['romsoc_fully_coupled']['present']['sst'])
    non_extremes_mask = (future<=0)*(present<=0)
    new_extremes_mask = (future>0)*(present<=0)
    disappearing_extremes_mask = (future<=0)*(present>0)
    intensifying_extremes_mask = (future>0)*(present>0)*(future>=present)
    weakening_extremes_mask = (future>0)*(present>0)*(future<present)
    
    #%
    fig, ax = plt.subplots(1,4,figsize=(15,5),sharey=True)
    if threshold_type == 'present':
        vmax = 250
    else:
        vmax = 50
    cmap = plt.get_cmap('cmo.amp',10)
    c0 = ax[0].pcolormesh(new_extremes_mask.lon,new_extremes_mask.lat,new_extremes_mask.sum(dim='time')/(2021-2011+1),vmin=0,vmax=vmax,cmap=cmap)#vmin=170,vmax=200)
    ax[0].set_title('new')
    ax[1].pcolormesh(disappearing_extremes_mask.lon,disappearing_extremes_mask.lat,disappearing_extremes_mask.sum(dim='time')/(2021-2011+1),vmin=0,vmax=vmax,cmap=cmap)#vmin=170,vmax=200)
    ax[1].set_title('disappear')
    ax[2].pcolormesh(intensifying_extremes_mask.lon,intensifying_extremes_mask.lat,intensifying_extremes_mask.sum(dim='time')/(2021-2011+1),vmin=0,vmax=vmax,cmap=cmap)#vmin=170,vmax=200)
    ax[2].set_title('intens')
    ax[3].pcolormesh(weakening_extremes_mask.lon,weakening_extremes_mask.lat,weakening_extremes_mask.sum(dim='time')/(2021-2011+1),vmin=0,vmax=vmax,cmap=cmap)#vmin=170,vmax=200)
    ax[3].set_title('weakening')
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    cbax = fig.add_axes([0.91,0.2,0.025,0.6])
    plt.colorbar(c0,cax=cbax,extend='max')
    cbax.set_title('days\n    per year',pad=15)
    if threshold_type == 'present':
        yticks = np.array([0,50,100,150,200,250])
    else:
        yticks = np.array([0,10,20,30,40,50])
    cbax.set_yticks(yticks)
    for axi in ax:
        axi.set_xlim([230,245])
        axi.set_ylim([30,50])
    yticks = np.arange(30,55,5)
    xticks = np.arange(230,245,5)
    for adx,axi in enumerate(ax):
        axi.set_yticks(yticks)
        axi.set_xticks(xticks)
        axi.set_yticklabels([str(val)+'°N' for val in yticks])
        axi.set_xticklabels([str(360-val)+'°W' for val in xticks])

    # add the continent
    landmask_etopo = PlotFuncs.get_etopo_data()
    for axi in ax.flatten():
        axi.contourf(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo,colors='#555555')

    savedir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/histograms/'
    filename = f'future_vs_present_anomalies_rel_to_thresh_{threshold_type}_for_region_{region_cho}_map_number_of_days.png'
    #plt.savefig(savedir+filename,dpi=200,transparent=True)
    plt.show()
    

# %%

present = variables['romsoc_fully_coupled']['present']['sst'] - thresholds['romsoc_fully_coupled']['present']['sst']

for region_cho in ['all_dists_all_lats']:
    for threshold_type in ['present','present_plus_meandelta']:#,'present_plus_climdelta']: #= 'present_plus_climdelta'
        
        if threshold_type == 'present':
            sharey=False
        else:
            sharey=True
        fig, ax = plt.subplots(1,4,figsize=(11,4),sharey=sharey)
        
        for scenario in ['ssp245','ssp585']:

            #if threshold_type == 'present_plus_climdelta':
            #    future = variables['romsoc_fully_coupled']['ssp585']['sst'] - thresholds_mult['romsoc_fully_coupled']['ssp585']['present_plus_climdelta']#- (thresholds['romsoc_fully_coupled']['present']['sst'] + clims['romsoc_fully_coupled']['ssp585']['sst'] - clims['romsoc_fully_coupled']['present']['sst'])
            if threshold_type == 'present_plus_meandelta':
                future = variables['romsoc_fully_coupled'][scenario]['sst'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_meandelta']#(thresholds['romsoc_fully_coupled']['present']['sst'] + variables['romsoc_fully_coupled']['ssp585']['sst'].mean(dim='time') - variables['romsoc_fully_coupled']['present']['sst'].mean(dim='time'))
            elif threshold_type == 'present':
                future = variables['romsoc_fully_coupled'][scenario]['sst'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present']#- (thresholds['romsoc_fully_coupled']['present']['sst'])

            non_extremes_mask = (future<=0)*(present<=0)                          *model_regions['roms_only'][region_cho]['mask']
            new_extremes_mask = (future>0)*(present<=0)                           *model_regions['roms_only'][region_cho]['mask']
            disappearing_extremes_mask = (future<=0)*(present>0)                  *model_regions['roms_only'][region_cho]['mask']
            intensifying_extremes_mask = (future>0)*(present>0)*(future>=present) *model_regions['roms_only'][region_cho]['mask']
            weakening_extremes_mask = (future>0)*(present>0)*(future<present)     *model_regions['roms_only'][region_cho]['mask']
            
            #% sum the arrays
            new_extremes_mask_sum = new_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)
            disappearing_extremes_mask_sum = disappearing_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)
            intensifying_extremes_mask_sum = intensifying_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)
            weakening_extremes_mask_sum = weakening_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)

            # flatten the arrays
            new_flat          =          new_extremes_mask_sum.values.flatten()
            disappear_flat          =          disappearing_extremes_mask_sum.values.flatten()
            intensify_flat          =          intensifying_extremes_mask_sum.values.flatten()
            weakening_flat          =          weakening_extremes_mask_sum.values.flatten()

            # flatten the distance to coast array
            d2coast_flat = model_d2coasts['roms_only'].values.flatten()

            # drop the nan entries
            nanmask = np.isnan(d2coast_flat)+np.isnan(new_flat)+np.isnan(disappear_flat)+np.isnan(intensify_flat)+np.isnan(weakening_flat)
            new_nonan = new_flat[~nanmask]
            disappear_nonan = disappear_flat[~nanmask]
            intensify_nonan = intensify_flat[~nanmask]
            weakening_nonan = weakening_flat[~nanmask]
            d2coast_flat_nonan = d2coast_flat[~nanmask]

            # compute the mean and the IQR for bins as a function of distance to coast
            new_vs_d2coast_mean = []
            new_vs_d2coast_p25 = []
            new_vs_d2coast_p75 = []

            disappear_vs_d2coast_mean = []
            disappear_vs_d2coast_p25 = []
            disappear_vs_d2coast_p75 = []

            intensify_vs_d2coast_mean = []
            intensify_vs_d2coast_p25 = []
            intensify_vs_d2coast_p75 = []

            weakening_vs_d2coast_mean = []
            weakening_vs_d2coast_p25 = []
            weakening_vs_d2coast_p75 = []

            bins_d2coast = np.arange(0,380,10)
            for bdx,binn in enumerate(bins_d2coast[:-1]):
                dist_cond = (d2coast_flat_nonan>=bins_d2coast[bdx])*(d2coast_flat_nonan<bins_d2coast[bdx+1]) 
                new_vs_d2coast_mean.append(np.mean(new_nonan[dist_cond]))
                new_vs_d2coast_p25.append(np.percentile(new_nonan[dist_cond],25))
                new_vs_d2coast_p75.append(np.percentile(new_nonan[dist_cond],75))

                disappear_vs_d2coast_mean.append(np.mean(disappear_nonan[dist_cond]))
                disappear_vs_d2coast_p25.append(np.percentile(disappear_nonan[dist_cond],25))
                disappear_vs_d2coast_p75.append(np.percentile(disappear_nonan[dist_cond],75))
                
                intensify_vs_d2coast_mean.append(np.mean(intensify_nonan[dist_cond]))
                intensify_vs_d2coast_p25.append(np.percentile(intensify_nonan[dist_cond],25))
                intensify_vs_d2coast_p75.append(np.percentile(intensify_nonan[dist_cond],75))

                weakening_vs_d2coast_mean.append(np.mean(weakening_nonan[dist_cond]))
                weakening_vs_d2coast_p25.append(np.percentile(weakening_nonan[dist_cond],25))
                weakening_vs_d2coast_p75.append(np.percentile(weakening_nonan[dist_cond],75))
            #%
            if scenario == 'ssp245':
                color = 'y'
            elif scenario == 'ssp585':
                color = 'r'
            vmax = 300
            cmap = plt.get_cmap('cmo.amp',11)
            c0 = ax[0].plot(bins_d2coast[:-1],new_vs_d2coast_mean,color=color,alpha=1,linewidth=2)#vmin=170,vmax=200)
            ax[0].fill_between(bins_d2coast[:-1],new_vs_d2coast_p75,new_vs_d2coast_p25,color=color,alpha=0.35)#vmin=170,vmax=200)
            #ax[0].plot(bins_d2coast[:-1],new_vs_d2coast_p75,color=color,alpha=0.35)#vmin=170,vmax=200)
            ax[0].set_title('New',loc='left')
            ax[1].plot(bins_d2coast[:-1],disappear_vs_d2coast_mean,color=color,alpha=1,linewidth=2,label=scenario)#vmin=170,vmax=200)
            ax[1].fill_between(bins_d2coast[:-1],disappear_vs_d2coast_p75,disappear_vs_d2coast_p25,color=color,alpha=0.35)#vmin=170,vmax=200)
            #ax[1].plot(bins_d2coast[:-1],disappear_vs_d2coast_p75,color=color,alpha=0.35)#vmin=170,vmax=200)    
            ax[1].set_title('Disappeared',loc='left')
            ax[2].plot(bins_d2coast[:-1],intensify_vs_d2coast_mean,color=color,alpha=1,linewidth=2)#vmin=170,vmax=200)
            ax[2].fill_between(bins_d2coast[:-1],intensify_vs_d2coast_p75,intensify_vs_d2coast_p25,color=color,alpha=0.35)#vmin=170,vmax=200)
            #ax[2].plot(bins_d2coast[:-1],intensify_vs_d2coast_p75,color=color,alpha=0.35)#vmin=170,vmax=200)    
            ax[2].set_title('Intensified',loc='left')
            ax[3].plot(bins_d2coast[:-1],weakening_vs_d2coast_mean,color=color,alpha=1,linewidth=2)#vmin=170,vmax=200)
            ax[3].fill_between(bins_d2coast[:-1],weakening_vs_d2coast_p75,weakening_vs_d2coast_p25,color=color,alpha=0.35)#vmin=170,vmax=200)
            #ax[3].plot(bins_d2coast[:-1],weakening_vs_d2coast_p75,color=color,alpha=0.35)#vmin=170,vmax=200)    
            ax[3].set_title('Weaker',loc='left')
            ax[0].set_ylabel('Extreme days per year\nfor respective extreme type')
        for axi in ax:
            axi.set_xlim([0,371])
            axi.grid(linestyle='--',alpha=0.25)
            axi.spines['top'].set_visible(False)
            axi.spines['right'].set_visible(False)
            axi.axvline(100,color='C0',linestyle='--',linewidth=2)
            axi.set_xlabel('Dist. to coast in km')
        ax[0].legend()
        plt.tight_layout()

        savedir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/histograms/'
        filename = f'future_vs_present_anomalies_rel_to_thresh_{threshold_type}_for_region_{region_cho}_dist2coast_number_of_days.png'
        plt.savefig(savedir+filename,dpi=200,transparent=True)
        plt.show()

# %% SAME BUT ONLY FOR NEW AND INTENSIFIED EXTREMES

present = variables['romsoc_fully_coupled']['present']['sst'] - thresholds['romsoc_fully_coupled']['present']['sst']

for region_cho in ['all_dists_all_lats']:
    for threshold_type in ['present','present_plus_meandelta']:#,'present_plus_climdelta']: #= 'present_plus_climdelta'
        
        if threshold_type == 'present':
            sharey=False
        else:
            sharey=False # True

        fig, ax = plt.subplots(1,2,figsize=(6,4),sharey=sharey)
        
        for scenario in ['ssp245','ssp585']:

            #if threshold_type == 'present_plus_climdelta':
            #    future = variables['romsoc_fully_coupled']['ssp585']['sst'] - thresholds_mult['romsoc_fully_coupled']['ssp585']['present_plus_climdelta']#- (thresholds['romsoc_fully_coupled']['present']['sst'] + clims['romsoc_fully_coupled']['ssp585']['sst'] - clims['romsoc_fully_coupled']['present']['sst'])
            if threshold_type == 'present_plus_meandelta':
                future = variables['romsoc_fully_coupled'][scenario]['sst'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_meandelta']#(thresholds['romsoc_fully_coupled']['present']['sst'] + variables['romsoc_fully_coupled']['ssp585']['sst'].mean(dim='time') - variables['romsoc_fully_coupled']['present']['sst'].mean(dim='time'))
            elif threshold_type == 'present':
                future = variables['romsoc_fully_coupled'][scenario]['sst'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present']#- (thresholds['romsoc_fully_coupled']['present']['sst'])

            #non_extremes_mask = (future<=0)*(present<=0)                          *model_regions['roms_only'][region_cho]['mask']
            new_extremes_mask = (future>0)*(present<=0)                           *model_regions['roms_only'][region_cho]['mask']
            #disappearing_extremes_mask = (future<=0)*(present>0)                  *model_regions['roms_only'][region_cho]['mask']
            intensifying_extremes_mask = (future>0)*(present>0)*(future>=present) *model_regions['roms_only'][region_cho]['mask']
            #weakening_extremes_mask = (future>0)*(present>0)*(future<present)     *model_regions['roms_only'][region_cho]['mask']
            
            #% sum the arrays
            new_extremes_mask_sum = new_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)
            #disappearing_extremes_mask_sum = disappearing_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)
            intensifying_extremes_mask_sum = intensifying_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)
            #weakening_extremes_mask_sum = weakening_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)

            # flatten the arrays
            new_flat          =          new_extremes_mask_sum.values.flatten()
            #disappear_flat          =          disappearing_extremes_mask_sum.values.flatten()
            intensify_flat          =          intensifying_extremes_mask_sum.values.flatten()
            #weakening_flat          =          weakening_extremes_mask_sum.values.flatten()

            # flatten the distance to coast array
            d2coast_flat = model_d2coasts['roms_only'].values.flatten()

            # drop the nan entries
            nanmask = np.isnan(d2coast_flat)+np.isnan(new_flat)+np.isnan(intensify_flat)#+np.isnan(weakening_flat)+np.isnan(disappear_flat)
            new_nonan = new_flat[~nanmask]
            #disappear_nonan = disappear_flat[~nanmask]
            intensify_nonan = intensify_flat[~nanmask]
            #weakening_nonan = weakening_flat[~nanmask]
            d2coast_flat_nonan = d2coast_flat[~nanmask]

            # compute the mean and the IQR for bins as a function of distance to coast
            new_vs_d2coast_mean = []
            new_vs_d2coast_p10 = []
            new_vs_d2coast_p25 = []
            new_vs_d2coast_p75 = []
            new_vs_d2coast_p90 = []

            #disappear_vs_d2coast_mean = []
            #disappear_vs_d2coast_p25 = []
            #disappear_vs_d2coast_p75 = []

            intensify_vs_d2coast_mean = []
            intensify_vs_d2coast_p10 = []
            intensify_vs_d2coast_p25 = []
            intensify_vs_d2coast_p75 = []
            intensify_vs_d2coast_p90 = []

            #weakening_vs_d2coast_mean = []
            #weakening_vs_d2coast_p25 = []
            #weakening_vs_d2coast_p75 = []

            bins_d2coast = np.arange(0,380,10)
            for bdx,binn in enumerate(bins_d2coast[:-1]):
                dist_cond = (d2coast_flat_nonan>=bins_d2coast[bdx])*(d2coast_flat_nonan<bins_d2coast[bdx+1]) 
                new_vs_d2coast_mean.append(np.mean(new_nonan[dist_cond]))
                new_vs_d2coast_p25.append(np.percentile(new_nonan[dist_cond],25))
                new_vs_d2coast_p75.append(np.percentile(new_nonan[dist_cond],75))
                new_vs_d2coast_p10.append(np.percentile(new_nonan[dist_cond],10))
                new_vs_d2coast_p90.append(np.percentile(new_nonan[dist_cond],90))
                #disappear_vs_d2coast_mean.append(np.mean(disappear_nonan[dist_cond]))
                #disappear_vs_d2coast_p25.append(np.percentile(disappear_nonan[dist_cond],25))
                #disappear_vs_d2coast_p75.append(np.percentile(disappear_nonan[dist_cond],75))
                
                intensify_vs_d2coast_mean.append(np.mean(intensify_nonan[dist_cond]))
                intensify_vs_d2coast_p25.append(np.percentile(intensify_nonan[dist_cond],25))
                intensify_vs_d2coast_p75.append(np.percentile(intensify_nonan[dist_cond],75))
                intensify_vs_d2coast_p10.append(np.percentile(intensify_nonan[dist_cond],10))
                intensify_vs_d2coast_p90.append(np.percentile(intensify_nonan[dist_cond],90))
                #weakening_vs_d2coast_mean.append(np.mean(weakening_nonan[dist_cond]))
                #weakening_vs_d2coast_p25.append(np.percentile(weakening_nonan[dist_cond],25))
                #weakening_vs_d2coast_p75.append(np.percentile(weakening_nonan[dist_cond],75))
            #%
            if scenario == 'ssp245':
                color = 'y'
            elif scenario == 'ssp585':
                color = 'r'
            vmax = 300
            cmap = plt.get_cmap('cmo.amp',11)
            c0 = ax[0].plot(bins_d2coast[:-1],new_vs_d2coast_mean,color=color,alpha=1,linewidth=2)#vmin=170,vmax=200)
            ax[0].fill_between(bins_d2coast[:-1],new_vs_d2coast_p75,new_vs_d2coast_p25,color=color,alpha=0.35)#vmin=170,vmax=200)
            ax[0].fill_between(bins_d2coast[:-1],new_vs_d2coast_p90,new_vs_d2coast_p10,color=color,alpha=0.15)#vmin=170,vmax=200)
            #ax[0].plot(bins_d2coast[:-1],new_vs_d2coast_p75,color=color,alpha=0.35)#vmin=170,vmax=200)
            ax[0].set_title('New',loc='left')
            #ax[1].plot(bins_d2coast[:-1],disappear_vs_d2coast_mean,color=color,alpha=1,linewidth=2,label=scenario)#vmin=170,vmax=200)
            #ax[1].fill_between(bins_d2coast[:-1],disappear_vs_d2coast_p75,disappear_vs_d2coast_p25,color=color,alpha=0.35)#vmin=170,vmax=200)
            #ax[1].plot(bins_d2coast[:-1],disappear_vs_d2coast_p75,color=color,alpha=0.35)#vmin=170,vmax=200)    
            #ax[1].set_title('Disappeared',loc='left')
            if threshold_type == 'present_plus_meandelta':
                ax[1].plot(bins_d2coast[:-1],intensify_vs_d2coast_mean,color=color,alpha=1,linewidth=2,label=scenario)#vmin=170,vmax=200)
            else:
                ax[1].plot(bins_d2coast[:-1],intensify_vs_d2coast_mean,color=color,alpha=1,linewidth=2)#vmin=170,vmax=200)                
            ax[1].fill_between(bins_d2coast[:-1],intensify_vs_d2coast_p75,intensify_vs_d2coast_p25,color=color,alpha=0.35)#vmin=170,vmax=200)
            ax[1].fill_between(bins_d2coast[:-1],intensify_vs_d2coast_p90,intensify_vs_d2coast_p10,color=color,alpha=0.15)#vmin=170,vmax=200)
            #ax[2].plot(bins_d2coast[:-1],intensify_vs_d2coast_p75,color=color,alpha=0.35)#vmin=170,vmax=200)    
            ax[1].set_title('Intensified',loc='left')
            #ax[3].plot(bins_d2coast[:-1],weakening_vs_d2coast_mean,color=color,alpha=1,linewidth=2)#vmin=170,vmax=200)
            #ax[3].fill_between(bins_d2coast[:-1],weakening_vs_d2coast_p75,weakening_vs_d2coast_p25,color=color,alpha=0.35)#vmin=170,vmax=200)
            #ax[3].plot(bins_d2coast[:-1],weakening_vs_d2coast_p75,color=color,alpha=0.35)#vmin=170,vmax=200)    
            #ax[3].set_title('Weaker',loc='left')
            ax[0].set_ylabel('Days per year')#\nfor respective extreme type')
        for axi in ax:
            axi.set_xlim([0,371])
            axi.grid(linestyle='--',alpha=0.25)
            axi.spines['top'].set_visible(False)
            axi.spines['right'].set_visible(False)
            axi.axvline(100,color='#888888',linestyle='-',linewidth=1)
            axi.set_xlabel('Dist. to coast in km')
        if threshold_type == 'present_plus_meandelta':
            ax[1].legend(loc='upper right')
        plt.tight_layout()

        savedir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/histograms/'
        filename = f'future_vs_present_anomalies_rel_to_thresh_{threshold_type}_for_region_{region_cho}_dist2coast_number_of_days_new_and_intensify_only.png'
        plt.savefig(savedir+filename,dpi=200,transparent=True)
        plt.show()

# %% Same but as a violinplot

present_thresh = thresholds['romsoc_fully_coupled']['present']['sst']
present = variables['romsoc_fully_coupled']['present']['sst'] - present_thresh
present_extreme = (present>0)
present_nonex = (present<=0)
mean_warming_ssp245 = variables['romsoc_fully_coupled']['ssp245']['sst'].mean(dim='time') - variables['romsoc_fully_coupled']['present']['sst'].mean(dim='time')
mean_warming_ssp585 = variables['romsoc_fully_coupled']['ssp585']['sst'].mean(dim='time') - variables['romsoc_fully_coupled']['present']['sst'].mean(dim='time')

#%%
flat_data_arrays = dict()
for region_cho in ['coastal_all_lats','offshore_all_lats']:#'coastal_northern','coastal_central','coastal_southern','offshore_northern','offshore_central','offshore_southern']:
    flat_data_arrays[region_cho] = dict()
    for threshold_type in ['present','present_plus_meandelta']:#,'present_plus_climdelta']: #= 'present_plus_climdelta'
        flat_data_arrays[region_cho][threshold_type] = dict()        
        for scenario in ['ssp245','ssp585']:
            print(region_cho,threshold_type,scenario)
            flat_data_arrays[region_cho][threshold_type][scenario] = dict()        
            # if threshold_type == 'present_plus_climdelta':
            #     future = variables['romsoc_fully_coupled'][scenario]['sst'] - (thresholds['romsoc_fully_coupled']['present']['sst'] + clims['romsoc_fully_coupled'][scenario]['sst'] - clims['romsoc_fully_coupled']['present']['sst'])
            if threshold_type == 'present_plus_meandelta':
                if scenario == 'ssp245':
                    future = variables['romsoc_fully_coupled'][scenario]['sst'] - (present_thresh + mean_warming_ssp245)
                elif scenario == 'ssp585':
                    future = variables['romsoc_fully_coupled'][scenario]['sst'] - (present_thresh + mean_warming_ssp585)
            elif threshold_type == 'present':
                future = variables['romsoc_fully_coupled'][scenario]['sst'] - present_thresh

            future_extreme = (future>0)
            future_nonex = (future<=0)

            #non_extremes_mask          = (future<=0) * (present<=0)                        *model_regions['roms_only'][region_cho]['mask']
            new_extremes_mask          = future_extreme* present_nonex                     #*model_regions['roms_only'][region_cho]['mask']
            disappearing_extremes_mask = future_nonex  * present_extreme                    #*model_regions['roms_only'][region_cho]['mask']
            intensifying_extremes_mask = future_extreme*present_extreme*(future>=present)   #*model_regions['roms_only'][region_cho]['mask']
            weakening_extremes_mask    = future_extreme*present_extreme*(future<present)    #*model_regions['roms_only'][region_cho]['mask']
            
            #% sum the arrays
            new_extremes_mask_sum          =          new_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)*model_regions['roms_only'][region_cho]['mask']
            disappearing_extremes_mask_sum = disappearing_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)*model_regions['roms_only'][region_cho]['mask']
            intensifying_extremes_mask_sum = intensifying_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)*model_regions['roms_only'][region_cho]['mask']
            weakening_extremes_mask_sum    =    weakening_extremes_mask.sum(dim='time',skipna=True,min_count=1)/(2021-2011+1)*model_regions['roms_only'][region_cho]['mask']

            # flatten the arrays
            flat_data_arrays[region_cho][threshold_type][scenario]['new']                =          new_extremes_mask_sum#.values.flatten()
            flat_data_arrays[region_cho][threshold_type][scenario]['disappear']          =          disappearing_extremes_mask_sum#.values.flatten()
            flat_data_arrays[region_cho][threshold_type][scenario]['intensify']          =          intensifying_extremes_mask_sum#.values.flatten()
            flat_data_arrays[region_cho][threshold_type][scenario]['weakening']          =          weakening_extremes_mask_sum#.values.flatten()

#%%
flat_data_arrays2 = dict()
for region_cho in ['coastal_all_lats','offshore_all_lats']:#['coastal_northern','coastal_central','coastal_southern','offshore_northern','offshore_central','offshore_southern']:
    flat_data_arrays2[region_cho] = dict()
    for threshold_type in ['present','present_plus_meandelta']:#,'present_plus_climdelta']: #= 'present_plus_climdelta'
        flat_data_arrays2[region_cho][threshold_type] = dict()        
        for scenario in ['ssp245','ssp585']:
            flat_data_arrays2[region_cho][threshold_type][scenario] = dict()        
            for etype in ['new','disappear','intensify','weakening']:
                print(region_cho,threshold_type,scenario,etype)
                dummy_data = flat_data_arrays[region_cho][threshold_type][scenario][etype]
                dummy_data_stacked = dummy_data.stack(dim_0=('eta_rho', 'xi_rho')).reset_index('dim_0').drop(['eta_rho', 'xi_rho']) 
                flat_dum = dummy_data_stacked.where(np.isfinite(dummy_data_stacked), drop=True)
                flat_data_arrays2[region_cho][threshold_type][scenario][etype] = flat_dum        


#%%
fig, ax = plt.subplots(2,1,figsize=(8,10),sharex=True)
threshold_type = 'present'#_plus_meandelta'
scenario = 'ssp585'
colors = ['C1','C3','C0','#CCCCCC']
etypes = ['new','intensify','weakening','disappear']
yticks = np.arange(len(etypes))
linalphs = [1,0.25]
#latbands = ['northern','central','southern']
latband = 'all_lats'
panlabs = ['a) ','b) ']
for ldx,scenario in enumerate(['ssp585','ssp245']):
    for edx,etype in enumerate(etypes):
        ax[ldx].axhline(edx,color='#555555',linewidth=1,alpha=0.5)
        for tdx,threshold_type in enumerate(['present','present_plus_meandelta']):
            # plot the coastal part
            region_cho = f'coastal_{latband}'
            parts = ax[ldx].violinplot(flat_data_arrays2[region_cho][threshold_type][scenario][etype],positions=[edx],showmeans=False, showmedians=False,
            showextrema=False,widths=0.9,vert=False)
            ax[ldx].plot([np.median(flat_data_arrays2[region_cho][threshold_type][scenario][etype])]*2,[edx,edx-0.1],color='k',alpha=linalphs[tdx])
            for pc in parts['bodies']:
                pc.set_facecolor(colors[edx])
                pc.set_edgecolor('k')
                if tdx == 0:
                    pc.set_alpha(1)
                else:
                    pc.set_alpha(0.25)
                #m = np.mean(pc.get_paths()[0].vertices[:, 0]) # get center
                #pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m) # modify the paths to not go further right than the center
                m = np.mean(pc.get_paths()[0].vertices[:, 1]) # get center
                pc.get_paths()[0].vertices[:, 1] = np.clip(pc.get_paths()[0].vertices[:, 1], -np.inf, m) # modify the paths to not go further right than the center

            # # plot the offshore part
            region_cho = f'offshore_{latband}'
            parts = ax[ldx].violinplot(flat_data_arrays2[region_cho][threshold_type][scenario][etype],positions=[edx],showmeans=False, showmedians=False,
            showextrema=False,widths=0.9,vert=False)
            ax[ldx].plot([np.median(flat_data_arrays2[region_cho][threshold_type][scenario][etype])]*2,[edx,edx+0.1],color='k',alpha=linalphs[tdx])
            for pc in parts['bodies']:
                pc.set_facecolor(colors[edx])
                pc.set_edgecolor('k')
                if tdx == 0:
                    pc.set_alpha(1)
                else:
                    pc.set_alpha(0.25)
                #m = np.mean(pc.get_paths()[0].vertices[:, 0]) # get center
                #pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf) # modify the paths to not go further right than the center
                m = np.mean(pc.get_paths()[0].vertices[:, 1]) # get center
                pc.get_paths()[0].vertices[:, 1] = np.clip(pc.get_paths()[0].vertices[:, 1], m, np.inf) # modify the paths to not go further right than the center
    ax[ldx].set_title(f'{panlabs[ldx]}{scenario} (opaque: fixed basel., transparent: moving basel.)',loc='left')
    ax[ldx].set_yticks(yticks)#,2,3])
    ax[ldx].set_yticklabels(etypes)#,rotation=90)
    ax[ldx].invert_yaxis()
    ax[ldx].grid(which='both',alpha=0.2)
    ax[ldx].axvline(10,color='k',alpha=0.5)
    ax[ldx].axvline(100,color='k',alpha=0.5)
    for ytick in yticks:
        if ytick == 0:
            ax[ldx].text(1.2,ytick+0.22,'offshore',va='bottom')
            ax[ldx].text(1.2,ytick-0.22,'coastal',va='top')
        else:
            ax[ldx].text(2.7*10**2,ytick+0.22,'offshore',va='bottom',ha='right')
            ax[ldx].text(2.7*10**2,ytick-0.22,'coastal',va='top',ha='right')            
ax[-1].set_xlabel('Local extreme days per year')
ax[-1].set_xscale('log')
ax[-1].set_xlim([10**0,3*10**2])
plt.tight_layout()
savedir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/histograms/'
filename = f'future_vs_present_anomalies_onshore_vs_offshore_number_of_days_extreme_types.png'
plt.savefig(savedir+filename,dpi=200,transparent=True)
plt.show()


# %%

# %%

print('Get variables present and futures')
present       = variables[config]['present']['sst']
future_ssp245 = variables[config]['ssp245']['sst']
future_ssp585 = variables[config]['ssp585']['sst']

baseline = 'moving'
if baseline == 'moving':
    present_ex =       present >      thresholds[config]['present']['sst']
    ssp245_ex  = future_ssp245 > thresholds_mult[config]['ssp245']['present_plus_meandelta']
    ssp585_ex  = future_ssp585 > thresholds_mult[config]['ssp585']['present_plus_meandelta']
elif baseline == 'fixed':
    present_ex = (      present > thresholds[config]['present']['sst'])
    ssp245_ex  = (future_ssp245 > thresholds[config]['present']['sst'])
    ssp585_ex  = (future_ssp585 > thresholds[config]['present']['sst'])

print('Masking')
regionalized_data = dict()
region_choices = ['all_dists_northern','all_dists_central','all_dists_southern'] # 'all_dists_all_lats',
for region_cho in region_choices:
    regionalized_data[region_cho] = dict()

    timecond              = (present_ex.time.dt.month>-8)
    present_extremes_mask = (present_ex.sel(time=timecond)==True)*model_regions['roms_only'][region_cho]['mask']

    timecond              = (ssp245_ex.time.dt.month>-8)
    ssp245_extremes_mask  = ( ssp245_ex.sel(time=timecond)==True)*model_regions['roms_only'][region_cho]['mask']

    timecond              = (ssp585_ex.time.dt.month>-8)
    ssp585_extremes_mask  = ( ssp585_ex.sel(time=timecond)==True)*model_regions['roms_only'][region_cho]['mask']

    masked_present = xr.where( present_extremes_mask==1,        present.sel(time=timecond),np.NaN)
    masked_ssp245  = xr.where(  ssp245_extremes_mask==1,  future_ssp245.sel(time=timecond),np.NaN)
    masked_ssp585  = xr.where(  ssp585_extremes_mask==1,  future_ssp585.sel(time=timecond),np.NaN)

    print('Flatten present')
    flattened_present = masked_present.stack(z=("time", "eta_rho", "xi_rho")).dropna(dim="z")
    flattened_present_data = flattened_present.values

    print('Flatten 245')
    flattened_ssp245 = masked_ssp245.stack(z=("time", "eta_rho", "xi_rho")).dropna(dim="z")
    flattened_ssp245_data = flattened_ssp245.values

    print('Flatten 585')
    flattened_ssp585 = masked_ssp585.stack(z=("time", "eta_rho", "xi_rho")).dropna(dim="z")
    flattened_ssp585_data = flattened_ssp585.values

    print('D2coast')
    d2coast_present_flat = model_d2coasts['roms_only'].values[flattened_present.eta_rho,flattened_present.xi_rho]
    d2coast_ssp245_flat = model_d2coasts['roms_only'].values[flattened_ssp245.eta_rho,flattened_ssp245.xi_rho]
    d2coast_ssp585_flat = model_d2coasts['roms_only'].values[flattened_ssp585.eta_rho,flattened_ssp585.xi_rho]

    print('Put into dictionary')
    regionalized_data[region_cho]['d2coast_present'] = d2coast_present_flat
    regionalized_data[region_cho]['d2coast_ssp245'] = d2coast_ssp245_flat
    regionalized_data[region_cho]['d2coast_ssp585'] = d2coast_ssp585_flat
    regionalized_data[region_cho]['present'] = flattened_present_data
    regionalized_data[region_cho]['ssp245'] = flattened_ssp245_data
    regionalized_data[region_cho]['ssp585'] = flattened_ssp585_data


#%% compute the mean and the IQR for bins as a function of distance to coast
import matplotlib.patches as mpatches

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

colors = ['k','y','r']
labs = ['a) ','b) ','c) ']
fig, ax = plt.subplots(3,1,figsize=(4,9),sharex=True)
for rdx,region_cho in enumerate(region_choices):
    offsets = [-30,0,30]
    labels = []
    for fdx,scenario in enumerate(scenarios):
        bins_d2coast = [0,100,370]#np.arange(0,380,25)
        bin_centers = [50,150]#235]
        binned_data = []
        for bdx,binn in enumerate(bins_d2coast[:-1]):
            dist_cond = (regionalized_data[region_cho][f'd2coast_{scenario}']>=bins_d2coast[bdx])*(regionalized_data[region_cho][f'd2coast_{scenario}']<bins_d2coast[bdx+1]) 
            binned_data.append(regionalized_data[region_cho][scenario][dist_cond])
        color = colors[fdx]
        cmap = plt.get_cmap('cmo.amp',11)    
        parts = ax[rdx].violinplot(binned_data,vert=True,positions=np.array(bin_centers)+offsets[fdx],widths=30, showmeans=False, showmedians=False,
        showextrema=False)
        labels.append((mpatches.Patch(color=colors[fdx]), scenario))
        for pc in parts['bodies']:
            pc.set_facecolor(colors[fdx])
            pc.set_edgecolor(colors[fdx])
            pc.set_alpha(1)
        for idx in range(len(binned_data)):
            quartile1, medians, quartile3 = np.percentile(binned_data[idx], [25, 50, 75])
            inds = [(np.array(bin_centers)+offsets[fdx])[idx]]
            ax[rdx].scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
            if fdx == 0:
                colo = '#888888'
            else:
                colo = '#888888'
            linelab = ax[rdx].vlines(inds, quartile1, quartile3, color=colo, linestyle='-', lw=5)
        ax[rdx].axvline(100,color='C0',alpha=0.2)
        ax[rdx].set_title(labs[rdx]+region_cho.split('_')[-1]+' CalCS',loc='left')
        ax[rdx].set_ylabel('SST (°C) during extremes')
        ax[rdx].set_xlim([0,200])
        ax[rdx].grid(linestyle='--',alpha=0.25)
        ax[rdx].spines['top'].set_visible(False)
        ax[rdx].spines['right'].set_visible(False)
        ax[rdx].set_xticks([50,150])
        ax[rdx].set_xticklabels(['Coastal','Offshore'])

labels.append((mpatches.Patch(color=colo), 'IQR'))
ax[-1].legend(*zip(*labels),ncol=4,columnspacing=1,handletextpad=0.2,handlelength=1,loc='lower left')
# ax[0].set_ylim([7,23])
# ax[1].set_ylim([11,27])
# ax[2].set_ylim([16,32])
ax[0].set_ylim([7,25])
ax[1].set_ylim([9,27])
ax[2].set_ylim([10,31])

ax[-1].set_xlabel('Distance to coast')
plt.tight_layout()
savedir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/violinplots/'
filename = f'future_vs_present_violin_abs_temp_for_all_regions_{config}_{baseline}_baseline.png'
#plt.savefig(savedir+filename,dpi=200,transparent=True)
plt.show()

#%% compute the mean and the IQR for bins as a function of distance to coast
import matplotlib.patches as mpatches

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

colors = ['k','y','r']
labs = ['a) ','b) ','c) ']
fontsize=12
plt.rcParams['font.size']=fontsize
fig, ax = plt.subplots(1,3,figsize=(9,4),sharey=True)
for rdx,region_cho in enumerate(region_choices):
    offsets = [-30,0,30]
    labels = []
    for fdx,scenario in enumerate(scenarios):
        bins_d2coast = [0,100]#np.arange(0,380,25)
        bin_centers = [50]#235]
        binned_data = []
        for bdx,binn in enumerate(bins_d2coast[:-1]):
            dist_cond = (regionalized_data[region_cho][f'd2coast_{scenario}']>=bins_d2coast[bdx])*(regionalized_data[region_cho][f'd2coast_{scenario}']<bins_d2coast[bdx+1]) 
            binned_data.append(regionalized_data[region_cho][scenario][dist_cond])
        color = colors[fdx]
        cmap = plt.get_cmap('cmo.amp',11)    
        parts = ax[rdx].violinplot(binned_data,vert=True,positions=np.array(bin_centers)+offsets[fdx],widths=30, showmeans=False, showmedians=False,
        showextrema=False)
        labels.append((mpatches.Patch(color=colors[fdx]), scenario))
        for pc in parts['bodies']:
            pc.set_facecolor(colors[fdx])
            pc.set_edgecolor(colors[fdx])
            pc.set_alpha(1)
        for idx in range(len(binned_data)):
            quartile1, medians, quartile3 = np.percentile(binned_data[idx], [25, 50, 75])
            inds = [(np.array(bin_centers)+offsets[fdx])[idx]]
            ax[rdx].scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
            if fdx == 0:
                colo = '#888888'
            else:
                colo = '#888888'
            linelab = ax[rdx].vlines(inds, quartile1, quartile3, color=colo, linestyle='-', lw=5)
        #ax[rdx].axvline(100,color='C0',alpha=0.2)
        ax[rdx].set_title(labs[rdx]+region_cho.split('_')[-1]+' coastal CalCS',loc='left')
        ax[rdx].set_xlim([0,100])
        ax[rdx].grid(linestyle='--',alpha=0.25)
        ax[rdx].spines['top'].set_visible(False)
        ax[rdx].spines['right'].set_visible(False)
        ax[rdx].spines['bottom'].set_visible(False)
        #if rdx > 0:
        #    ax[rdx].spines['left'].set_visible(False)
        ax[rdx].set_xticks([])
        #ax[rdx].set_xticklabels(['Coastal','Offshore'])
        #ax[rdx].set_xlabel('Coastal')

labels.append((mpatches.Patch(color=colo), 'IQR'))
ax[0].legend(*zip(*labels),ncol=2,columnspacing=1,handletextpad=0.2,handlelength=1,loc='upper left')
ax[0].set_ylabel('SST (°C) during extremes')
plt.tight_layout()
savedir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/violinplots/'
filename = f'future_vs_present_violin_abs_temp_for_all_regions_{config}_{baseline}_baseline_horizontal.png'
plt.savefig(savedir+filename,dpi=200,transparent=True)
plt.show()





# %%
