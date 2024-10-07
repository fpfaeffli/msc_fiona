#%%

# enable the visibility of the modules for the import functions
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/'))
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from get_model_datasets import ModelGetter as ModelGetter
from get_obs_datasets import ObsGetter as ObsGetter
from plotting_functions_general import PlotFuncs as PlotFuncs
import cmocean as cmo
import pandas as pd
from regridding_tools import Regridder as Regridder

#%%
# Contains:
#   01. plot map of the annual means
#   02. plot area averaged time series
#   03. plot area averaged climatology time series
#   04. plot vertical profiles
#   05. plot time vs. depth sections
#   06. plot time vs. depth climatology sections
#   07. plot depth vs. distance to coast sections
#   08. plot autocorrelation time scales map
#   09. plot map of standard devation
#   10. choosing colormaps and colorbars

#%%
class Plotter():
    """
    author: Eike E. Koehn
    date: June 7, 2024
    description: This class contains a number of functions that enable standardized plotting for the model evaluation. 
    The functions include:
    1. a map of the annual mean field
    2. area averaged timeseries over the full model period
    3. area averaged climatologies over the full model period
    """ 

    @staticmethod
    def plot_full_map_annual_mean(varia,depth,obs_da,model_da,obs_regions_dict,model_regions_dict,regional_data=None,regional_data_plottype='pcolmesh',savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: Plotting maps for the annual mean fields of a given variable. 
        The plot structure is as follows: 2 rows, 3 columns. 
        ax[0,0] = Obs           |  ax[0,1] = ROMSOC    | ax[0,2] = ROMSOC-Obs
        ax[1,0] = ROMSOC-ROMS   |  ax[1,1] = ROMS-only | ax[1,2] = ROMS-Obs
        """

        panel_labels = ['a)','b)','c)','d)','e)','f)']

        obs_time_dim_name = obs_da.dims[0]    
        assert obs_time_dim_name == 'time' or obs_time_dim_name == 'month'
        annual_mean_Obs = obs_da.mean(dim=obs_time_dim_name)

        model_time_dim_name = model_da['roms_only'].dims[0]
        assert model_time_dim_name == 'time' or model_time_dim_name == 'month'
        annual_mean_roms = model_da['roms_only'].mean(dim=model_time_dim_name)
        annual_mean_roms = xr.where(ModelGetter.get_model_mask()==0,np.NaN,annual_mean_roms)
        
        model_time_dim_name = model_da['romsoc_fully_coupled'].dims[0]
        assert model_time_dim_name == 'time' or model_time_dim_name == 'month'
        annual_mean_romsoc = model_da['romsoc_fully_coupled'].mean(dim=model_time_dim_name)
        annual_mean_romsoc = xr.where(ModelGetter.get_model_mask()==0,np.NaN,annual_mean_romsoc)

        # calculate the difference between model and obs - need to regrid first here (regrid obs to model?)
        annual_mean_Obs_on_model_grid = Regridder.regrid_original_to_target(annual_mean_Obs.values,annual_mean_Obs.lon.values,annual_mean_Obs.lat.values,annual_mean_roms.lon.values,annual_mean_roms.lat.values)
        
        # set up the plot
        col = Plotter.get_color_maps_and_ranges(varia,depth)
        fontsize=12
        plt.rcParams['font.size']=fontsize
        fig, ax = plt.subplots(2,3,figsize=(10,8),sharex=True,sharey=True)

        # Plot ax[0,0], i.e. the observations
        c00 = ax[0,0].pcolormesh(annual_mean_Obs.lon,annual_mean_Obs.lat,annual_mean_Obs,vmin=col['obs']['minval'],vmax=col['obs']['maxval'],cmap=col['obs']['cmap_pcmesh'])
        ax[0,0].set_title('Observations')
        ax[0,0].set_xlim([obs_regions_dict['full_map']['minlon'],obs_regions_dict['full_map']['maxlon']])
        ax[0,0].set_ylim([obs_regions_dict['full_map']['minlat'],obs_regions_dict['full_map']['maxlat']])
        cbar00 = plt.colorbar(c00,ax=ax[0,0],extend='both')
        cbar00.ax.set_title(col['obs']['unit'],pad=15)

        # Plot ax[0,1], i.e. ROMSOC
        ax[0,1].set_title('ROMSOC')
        c01 = ax[0,1].pcolormesh(annual_mean_romsoc.lon,annual_mean_romsoc.lat,annual_mean_romsoc,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
        cbar01 = plt.colorbar(c01,ax=ax[0,1],extend='both')
        cbar01.ax.set_title(col['mod']['unit'],pad=15)

        # Plot ax[1,1], i.e. ROMS-only
        ax[1,1].set_title('ROMS-only')
        c11 = ax[1,1].pcolormesh(annual_mean_roms.lon,annual_mean_roms.lat,annual_mean_roms,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
        cbar11 = plt.colorbar(c11,ax=ax[1,1],extend='both')
        cbar11.ax.set_title(col['mod']['unit'],pad=15)
       
        # Plot ax[1,0], i.e. ROMSOC-ROMS
        ax[1,0].set_title('ROMSOC minus ROMS')
        c10 = ax[1,0].pcolormesh(annual_mean_roms.lon,annual_mean_roms.lat,annual_mean_romsoc-annual_mean_roms,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        cbar10 = plt.colorbar(c10,ax=ax[1,0],extend='both')
        cbar10.ax.set_title(col['mod-mod']['unit'],pad=15)

        # Plot ax[0,2], i.e. ROMSOC minus Obs
        ax[0,2].set_title('ROMSOC minus Obs.')
        c02 = ax[0,2].pcolormesh(annual_mean_romsoc.lon,annual_mean_romsoc.lat,annual_mean_romsoc-annual_mean_Obs_on_model_grid,vmin=col['mod-obs']['minval'],vmax=col['mod-obs']['maxval'],cmap=col['mod-obs']['cmap_pcmesh'])
        cbar02 = plt.colorbar(c02,ax=ax[0,2],extend='both')
        cbar02.ax.set_title(col['mod-obs']['unit'],pad=15)

        # Plot ax[1,2], i.e. ROMS-only minus Obs
        ax[1,2].set_title('ROMS-only minus Obs.')
        c12 = ax[1,2].pcolormesh(annual_mean_roms.lon,annual_mean_roms.lat,annual_mean_roms-annual_mean_Obs_on_model_grid,vmin=col['mod-obs']['minval'],vmax=col['mod-obs']['maxval'],cmap=col['mod-obs']['cmap_pcmesh'])
        cbar12 = plt.colorbar(c12,ax=ax[1,2],extend='both')
        cbar12.ax.set_title(col['mod-obs']['unit'],pad=15)

        # add the continent
        landmask_etopo = PlotFuncs.get_etopo_data()
        for axi in ax.flatten():
            axi.contourf(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo,colors='#555555')

        # set the plot extent and labels
        for adx,axi in enumerate(ax.flatten()):
            axi.set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
            axi.set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
            yticks = np.arange(20,70,10)
            xticks = np.arange(230,260,10)
            axi.set_yticks(yticks)
            axi.set_yticklabels([str(val)+'°N' for val in yticks],fontsize=fontsize-2)
            axi.set_xticks(xticks)
            axi.set_xticklabels([str(360-val)+'°W' for val in xticks],fontsize=fontsize-2)
            axi.text(0.05,0.97,panel_labels[adx],ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=axi.transAxes)
        plt.tight_layout()

        if regional_data != None:
            regional_data_obs,regional_data_model = regional_data
            regions_to_plot = ['offshore_all_lats','coastal_all_lats'] # 'transition_all_lats',
            regions_to_plot_short = [reg.upper()[0] for reg in regions_to_plot]
            regions_to_plot_dummy = [reg.replace('_all_lats','') for reg in regions_to_plot]
            print('adding the monthly data timeseries for the regions')
            print(regions_to_plot)

            ax_insets = np.empty_like(ax)
            for rdx in range(np.shape(ax)[0]):
                for cdx in range(np.shape(ax)[1]):
                    if regional_data_plottype == 'pcolmesh':
                        ax_insets[rdx,cdx] = ax[rdx,cdx].inset_axes([.78, .3, .2, .68])
                    elif regional_data_plottype == 'lines':
                        ax_insets[rdx,cdx] = ax[rdx,cdx].inset_axes([.58, .75, .4, .22])   # [.58, .5, .4, .48]

            # concatenate all the data in the regions for the model and obs
            concat_obs = np.concatenate(tuple([regional_data_obs[regi].values[:,None] for regi in regions_to_plot]),axis=1)
            concat_romsoc = np.concatenate(tuple([regional_data_model['romsoc_fully_coupled'][regi].values[:,None] for regi in regions_to_plot]),axis=1)
            concat_roms = np.concatenate(tuple([regional_data_model['roms_only'][regi].values[:,None] for regi in regions_to_plot]),axis=1)

            if regional_data_plottype == 'pcolmesh':
                ax_insets[0,0].pcolormesh(concat_obs,cmap=col['obs']['cmap_pcmesh'],vmin=col['obs']['minval'],vmax=col['obs']['maxval'],edgecolor='k',linewidth=0.125)
                ax_insets[0,1].pcolormesh(concat_romsoc,cmap=col['mod']['cmap_pcmesh'],vmin=col['mod']['minval'],vmax=col['mod']['maxval'],edgecolor='k',linewidth=0.125)
                ax_insets[0,2].pcolormesh(concat_romsoc-concat_obs,cmap=col['mod-obs']['cmap_pcmesh'],vmin=col['mod-obs']['minval'],vmax=col['mod-obs']['maxval'],edgecolor='k',linewidth=0.125)
                ax_insets[1,0].pcolormesh(concat_romsoc-concat_roms,cmap=col['mod-mod']['cmap_pcmesh'],vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],edgecolor='k',linewidth=0.125)
                ax_insets[1,1].pcolormesh(concat_roms,cmap=col['mod']['cmap_pcmesh'],vmin=col['mod']['minval'],vmax=col['mod']['maxval'],edgecolor='k',linewidth=0.125)
                ax_insets[1,2].pcolormesh(concat_roms-concat_obs,cmap=col['mod-obs']['cmap_pcmesh'],vmin=col['mod-obs']['minval'],vmax=col['mod-obs']['maxval'],edgecolor='k',linewidth=0.125)
            elif regional_data_plottype == 'lines':
                ax_insets[0,0].plot(np.arange(1,13),concat_obs,'.-')
                ax_insets[0,1].plot(np.arange(1,13),concat_romsoc,'.-')
                lineObjs = ax_insets[0,2].plot(np.arange(1,13),concat_romsoc-concat_obs,'.-')
                ax_insets[1,0].plot(np.arange(1,13),concat_romsoc-concat_roms,'.-')
                ax_insets[1,1].plot(np.arange(1,13),concat_roms,'.-')
                ax_insets[1,2].plot(np.arange(1,13),concat_roms-concat_obs,'.-')
                ax_insets[0,2].legend(iter(lineObjs),tuple(regions_to_plot_dummy),loc='lower left',bbox_to_anchor=(0,-1.4),fontsize=fontsize-4,handletextpad = 0.4, handlelength=2)               

            for rdx in range(np.shape(ax)[0]):
                for cdx in range(np.shape(ax)[1]):
                    if regional_data_plottype == 'pcolmesh':
                        ax_insets[rdx,cdx].set_yticks(np.arange(0.5,12.5))
                        ax_insets[rdx,cdx].set_yticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],color='w',fontsize=plt.rcParams['font.size']-1)
                        ax_insets[rdx,cdx].set_xticks(np.arange(len(regions_to_plot))+0.5)
                        ax_insets[rdx,cdx].set_xticklabels(regions_to_plot_short,color='w',fontsize=plt.rcParams['font.size']-1)
                        ax_insets[rdx,cdx].invert_yaxis()
                    elif regional_data_plottype == 'lines':
                        ax_insets[rdx,cdx].set_xticks(np.arange(1,13,2))
                        ax_insets[rdx,cdx].set_xticklabels(['J','M','M','J','S','N'],color='w',fontsize=plt.rcParams['font.size']-1)
                        #yticklabs = ax_insets[rdx,cdx].yaxis.label.set_color('w')
                        ax_insets[rdx,cdx].tick_params(axis='y', colors='w')
                        #ax_insets[rdx,cdx].set_yticklabels(yticklabs,color='w',fontsize=plt.rcParams['font.size']-1)
                        ax_insets[rdx,cdx].grid(color='#EEEEEE',linewidth=0.5)
                        ax_insets[rdx,cdx].set_ylabel(col['mod']['unit'],color='w')    
                        ax_insets[rdx,cdx].set_xlabel('Month',color='w')    


        # add the mask regions in each panel (first the Obs. mask in ax[0,0], then the Model mask for all others)
        if regional_data == None:
            for region in obs_regions_dict.keys():
                if region != 'full_map' and 'all_lats' not in region:
                    region_mask_dummy = xr.where(np.isnan(obs_regions_dict[region]['mask']),0,1)
                    ax[0,0].contour(annual_mean_Obs.lon,annual_mean_Obs.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)
            for axi in ax.flatten()[1:]:
                for region in model_regions_dict.keys():
                    if region != 'full_map' and 'all_lats' not in region:
                        region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                        axi.contour(annual_mean_roms.lon,annual_mean_roms.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)

        elif regional_data != None:
            for region in regions_to_plot:
                region_mask_dummy = xr.where(np.isnan(obs_regions_dict[region]['mask']),0,1)
                ax[0,0].contour(annual_mean_Obs.lon,annual_mean_Obs.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)
            for axi in ax.flatten()[1:]:
                for region in regions_to_plot:
                    region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                    axi.contour(annual_mean_roms.lon,annual_mean_roms.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)            


        # put the plotted values into a Dataset 
        plotted_values = xr.Dataset()
        plotted_values['obs'] = annual_mean_Obs
        plotted_values['romsoc'] = annual_mean_romsoc
        plotted_values['roms'] = annual_mean_roms
        plotted_values['romsoc_minus_roms'] = annual_mean_romsoc-annual_mean_roms
        plotted_values['romsoc_minus_obs'] = annual_mean_romsoc-annual_mean_Obs_on_model_grid
        plotted_values['roms_minus_obs'] = annual_mean_roms-annual_mean_Obs_on_model_grid

        if savefig == True:
            outpath = f'/nfs/sea/work/fpfaeffli/plots/maps_timeseries_eike/map_annual_means/'
            figname = f'{varia}_{depth}m.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values
    

    @staticmethod
    def plot_area_averaged_timeseries(varia,depth,obs_da,obs_area,model_da,model_area,obs_regions_dict,model_regions_dict,plot_resolution,savefig=False):

        col = Plotter.get_color_maps_and_ranges(varia,depth)
        row_names = ['all_lats','northern','central','southern']
        col_names = ['all_dists','offshore','coastal'] # 'transition'
        fig,ax = plt.subplots(len(row_names),len(col_names),sharex=True,sharey=True,figsize=(12,10))
        for rdx, rn in enumerate(row_names):
            for cdx, cn in enumerate(col_names):
                region_name = f'{cn}_{rn}'

                # get the observations and compute the regional average
                obs_reg = obs_regions_dict[region_name]['mask']
                #obs_regional_mean = (obs_da*obs_area*obs_reg).sum(dim=('lat','lon')) / (obs_area*obs_reg).sum(dim=('lat','lon'))
                obs_regional_mean = obs_da.weighted((obs_area*obs_reg).fillna(0)).mean(dim=('lat','lon'))

                # get the model and compute the regional average
                model_reg = model_regions_dict[region_name]['mask']
                #roms_regional_mean = (model_da['roms_only']*model_area*model_reg).sum(dim=('eta_rho','xi_rho')) / (model_area*model_reg).sum(dim=('eta_rho','xi_rho')) 
                roms_regional_mean = model_da['roms_only'].weighted((model_area*model_reg).fillna(0)).mean(dim=('eta_rho','xi_rho'))
                #romsoc_regional_mean = (model_da['romsoc_fully_coupled']*model_area*model_reg).sum(dim=('eta_rho','xi_rho')) / (model_area*model_reg).sum(dim=('eta_rho','xi_rho')) 
                romsoc_regional_mean = model_da['romsoc_fully_coupled'].weighted((model_area*model_reg).fillna(0)).mean(dim=('eta_rho','xi_rho'))

                # compute monthly means for the model data if they are given in daily values
                if pd.infer_freq(roms_regional_mean.time) == 'D':
                    roms_regional_mean = roms_regional_mean.resample(time='1M').mean(dim='time')
                if pd.infer_freq(romsoc_regional_mean.time) == 'D':
                    romsoc_regional_mean = romsoc_regional_mean.resample(time='1M').mean(dim='time')                

                # set the spine color
                for spine in ax[rdx,cdx].spines.values():
                    spine.set_edgecolor(model_regions_dict[region_name]['id_colour'])
                ax[rdx,cdx].tick_params(color=model_regions_dict[region_name]['id_colour'])

                # check whether the observational time dimension is called "time" or "month"
                obs_time_dim_name = obs_da.dims[0]    
                if obs_time_dim_name == 'month':
                    obs_regional_mean = obs_regional_mean.rename({'month': 'time'}) # .rename_dims({'month': 'time'})
                    obs_regional_mean['time'] = roms_regional_mean.time[:12]
                elif obs_time_dim_name == 'time':
                    obs_regional_mean['time'] = obs_regional_mean.time
                else:
                    raise Exception('A time dimension in the observations that is not called "time" or "month" cannot be handled, yet.')
                
                # make sure that the time series resolution corresponds to the plot resolution
                if plot_resolution == 'monthly':
                    obs_regional_mean = obs_regional_mean.resample(time='1M').mean(dim='time')
                    obs_regional_mean['time'] = [pd.Timestamp(t).replace(day=1) for t in obs_regional_mean['time'].values]

                l0 = ax[rdx,cdx].plot(obs_regional_mean.time, obs_regional_mean,color='k',linewidth=2,label=f"Obs. ({col['obs']['unit']})".replace('  ',''))
                l1 = ax[rdx,cdx].plot(roms_regional_mean.time, roms_regional_mean,color='purple',linewidth=0.75,label=f"ROMS ({col['mod']['unit']})".replace('  ',''))
                l2 = ax[rdx,cdx].plot(romsoc_regional_mean.time, romsoc_regional_mean,color='fuchsia',linewidth=2,label=f"ROMSOC ({col['mod']['unit']})".replace('  ',''))
        for adx,axi in enumerate(ax[:,0]):
            axi.set_ylabel(row_names[adx],rotation='horizontal',fontweight='bold',ha='right')
        for adx,axi in enumerate(ax[0,:]):
            axi.set_title(col_names[adx],fontweight='bold')
        ax[-1,1].set_xlabel('Time')
        ax[-1,-1].legend(loc = 'center left',bbox_to_anchor=(1,0.5))

        plt.tight_layout()

        # put the plotted values into a Dataset      
        plotted_values = xr.Dataset()
        plotted_values['obs'] = obs_regional_mean
        plotted_values['romsoc'] = roms_regional_mean
        plotted_values['roms'] = romsoc_regional_mean

        if savefig == True:
            outpath = f'/nfs/sea/work/fpfaeffli/plots/maps_timeseries_eike/avg_timeseries_regions/'
            figname = f'{varia}_{depth}m.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values
    

    @staticmethod
    def plot_area_averaged_climatology_timeseries(varia,depth,obs_da,obs_area,model_da,model_area,obs_regions_dict,model_regions_dict,plot_resolution,savefig=False):

        col = Plotter.get_color_maps_and_ranges(varia,depth)
        row_names = ['all_lats','northern','central','southern']
        col_names = ['all_dists','offshore','coastal'] # 'transition'
        fig,ax = plt.subplots(len(row_names),len(col_names),sharex=True,sharey=True,figsize=(12,10))
        for rdx, rn in enumerate(row_names):
            for cdx, cn in enumerate(col_names):
                region_name = f'{cn}_{rn}'

                # get the observations and compute the regional average
                obs_reg = obs_regions_dict[region_name]['mask']
                #obs_regional_mean = (obs_da*obs_area*obs_reg).sum(dim=('lat','lon')) / (obs_area*obs_reg).sum(dim=('lat','lon'))

                # check whether the observational time dimension is called "time" or "month" (see if it is a climatology, or a full timeseries)
                obs_time_dim_name = obs_da.dims[0]    
                if obs_time_dim_name == 'month':
                    obs_regional_mean_clim_mean = obs_da.weighted((obs_area*obs_reg).fillna(0)).mean(dim=('lat','lon'))
                    obs_regional_mean_clim_min = obs_regional_mean_clim_mean #* 0.90 # substract 10 percent for visibility
                    obs_regional_mean_clim_max = obs_regional_mean_clim_mean #*1.1 # add 10 percent for visibility
                elif obs_time_dim_name == 'time':
                    obs_grouper = obs_da.weighted((obs_area*obs_reg).fillna(0)).mean(dim=('lat','lon')).groupby('time.month')
                    obs_regional_mean_clim_mean = obs_grouper.mean("time")
                    obs_regional_mean_clim_min = obs_grouper.min("time")
                    obs_regional_mean_clim_max = obs_grouper.max("time")
                else:
                    raise Exception('A time dimension in the observations that is not called "time" or "month" cannot be handled, yet.')

                # get the model and compute the regional average
                model_reg = model_regions_dict[region_name]['mask']
                #roms_regional_mean = (model_da['roms_only']*model_area*model_reg).sum(dim=('eta_rho','xi_rho')) / (model_area*model_reg).sum(dim=('eta_rho','xi_rho')) 
                roms_grouper = model_da['roms_only'].weighted((model_area*model_reg).fillna(0)).mean(dim=('eta_rho','xi_rho')).groupby('time.month')
                roms_regional_mean_clim_mean = roms_grouper.mean("time")
                roms_regional_mean_clim_min = roms_grouper.min("time")
                roms_regional_mean_clim_max = roms_grouper.max("time")                
                #romsoc_regional_mean = (model_da['romsoc_fully_coupled']*model_area*model_reg).sum(dim=('eta_rho','xi_rho')) / (model_area*model_reg).sum(dim=('eta_rho','xi_rho')) 
                romsoc_grouper = model_da['romsoc_fully_coupled'].weighted((model_area*model_reg).fillna(0)).mean(dim=('eta_rho','xi_rho')).groupby('time.month')
                romsoc_regional_mean_clim_mean = romsoc_grouper.mean("time")
                romsoc_regional_mean_clim_min = romsoc_grouper.min("time")
                romsoc_regional_mean_clim_max = romsoc_grouper.max("time") 

                # set the spine color
                for spine in ax[rdx,cdx].spines.values():
                    spine.set_edgecolor(model_regions_dict[region_name]['id_colour'])
                ax[rdx,cdx].tick_params(color=model_regions_dict[region_name]['id_colour'])
                ax[rdx,cdx].set_xticks(np.arange(1,13))

                # Plot the climatology
                bar_width = 0.3
                months = np.arange(1,13)

                bars_obs = ax[rdx,cdx].bar(months - bar_width, obs_regional_mean_clim_max - obs_regional_mean_clim_min, bar_width,
                            bottom=obs_regional_mean_clim_min, color='k',label=f"Obs. ({col['obs']['unit']})".replace('  ',''))
                bars_roms = ax[rdx,cdx].bar(months, roms_regional_mean_clim_max - roms_regional_mean_clim_min, bar_width,
                            bottom=roms_regional_mean_clim_min, color='purple',label=f"ROMS ({col['mod']['unit']})".replace('  ',''))
                bars_romsoc = ax[rdx,cdx].bar(months + bar_width, romsoc_regional_mean_clim_max - romsoc_regional_mean_clim_min, bar_width,
                            bottom=romsoc_regional_mean_clim_min, color='fuchsia', label=f"ROMSOC ({col['mod']['unit']})".replace('  ',''))
                marker_mean_obs = ax[rdx,cdx].scatter(months-bar_width, obs_regional_mean_clim_mean, color='#CCCCCC', marker='_',label='Mean')
                marker_mean_roms = ax[rdx,cdx].scatter(months, roms_regional_mean_clim_mean, color='#CCCCCC',marker='_')
                marker_mean_romsoc = ax[rdx,cdx].scatter(months+bar_width, romsoc_regional_mean_clim_mean, color='#CCCCCC',marker='_')

        for adx,axi in enumerate(ax[:,0]):
            axi.set_ylabel(row_names[adx],rotation='horizontal',fontweight='bold',ha='right')
        for adx,axi in enumerate(ax[0,:]):
            axi.set_title(col_names[adx],fontweight='bold')

        for axi in ax.flatten():
            axi.set_xlim([0.5,12.5])
            for val in np.arange(1.5,12.5,1):
                axi.axvline(val,color='#EEEEEE')

        ax[-1,1].set_xlabel('Month')
        ax[-1,-1].legend(loc = 'center left',bbox_to_anchor=(1,0.5))

        plt.tight_layout()

        # put the plotted mean values into a Dataset      
        plotted_values = xr.Dataset()
        plotted_values['obs'] = obs_regional_mean_clim_mean
        plotted_values['roms'] = roms_regional_mean_clim_mean
        plotted_values['romsoc'] = romsoc_regional_mean_clim_mean

        if savefig == True:
            outpath = f'/nfs/sea/work/fpfaeffli/plots/maps_timeseries_eike/avg_timeseries_regions_clim/'
            figname = f'{varia}_{depth}m.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values
    

    @staticmethod
    def plot_vertical_profiles(varia,obs_da,obs_mean_profiles,model_da,model_mean_profiles,model_regions_dict,plot_resolution,savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: This function allows for the plotting of vertical profiles of properties averaged in different regions.      
        """
        plotted_values = xr.Dataset()

        col = Plotter.get_color_maps_and_ranges(varia)
        row_names = ['all_lats','northern','central','southern']
        col_names = ['all_dists','offshore','coastal'] # 'transition'
        fontsize=12
        plt.rcParams['font.size']=fontsize
        fig,ax = plt.subplots(len(row_names),len(col_names),sharex=False,sharey=False,figsize=(15,10))
        maxvals = []
        minvals = []
        for rdx, rn in enumerate(row_names):
            for cdx, cn in enumerate(col_names):
                region_name = f'{cn}_{rn}'

                obs_regional_mean = obs_mean_profiles[region_name]
                roms_regional_mean = model_mean_profiles['roms_only'][region_name]
                romsoc_regional_mean = model_mean_profiles['romsoc_fully_coupled'][region_name]

                maxvals.append(np.maximum(np.max(obs_regional_mean),np.maximum(np.max(roms_regional_mean),np.max(romsoc_regional_mean)))*1.05)
                minvals.append(np.minimum(np.min(obs_regional_mean),np.minimum(np.min(roms_regional_mean),np.min(romsoc_regional_mean))))

                # set the spine color
                for spine in ax[rdx,cdx].spines.values():
                    spine.set_edgecolor(model_regions_dict[region_name]['id_colour'])
                ax[rdx,cdx].tick_params(color=model_regions_dict[region_name]['id_colour'])

                # assert that the depth vectors are both negatively defined: 
                if np.any(np.sign(obs_da.depth)==1):
                    obs_da['depth'] = obs_da.depth*-1

                l1 = ax[rdx,cdx].plot(roms_regional_mean,roms_regional_mean.depth,color='purple',linewidth=1,linestyle=':',label=f"ROMS-only".replace('  ',''))
                l2 = ax[rdx,cdx].plot(romsoc_regional_mean,romsoc_regional_mean.depth,color='fuchsia',linewidth=2,label=f"ROMSOC")
                l0 = ax[rdx,cdx].plot(obs_regional_mean,obs_da.depth,color='k',linewidth=2,label=f"Obs.".replace('  ',''))

                plotted_values[f'{region_name}_obs'] = obs_regional_mean.rename({"depth": "depth_obs"})
                plotted_values[f'{region_name}_roms'] = roms_regional_mean.rename({"depth": "depth_roms"})
                plotted_values[f'{region_name}_romsoc'] = romsoc_regional_mean.rename({"depth": "depth_romsoc"})

                ax[rdx,cdx].set_xlabel(col['obs']['unit'])
                ax[rdx,cdx].set_ylabel('Depth')

        for adx,axi in enumerate(ax[:,0]):
            axi.text(-0.5,0.5,row_names[adx],rotation='horizontal',fontweight='bold',ha='right',transform=axi.transAxes)
        for adx,axi in enumerate(ax[0,:]):
            axi.set_title(col_names[adx],fontweight='bold')
        ax[-1,-1].legend(loc = 'center left',bbox_to_anchor=(1,0.5))
        for axi in ax.flatten():
            axi.set_ylim([-500,0])
            axi.grid()
            axi.set_xlim([np.min(minvals),np.max(maxvals)])

        plt.tight_layout()

        if savefig == True:
            outpath = f'/nfs/sea/work/fpfaeffli/plots/verticals_eike/vertical_profiles/'
            figname = f'{varia}_{plot_resolution}.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values


    @staticmethod
    def plot_time_vs_depth_sections(varia,obs_da,obs_area,model_da,model_area,obs_regions_dict,model_regions_dict,plot_resolution,savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: This function allows for the plotting of time vs. depth sections averaged in different regions.      
        """

        plotted_values = xr.Dataset()

        col = Plotter.get_color_maps_and_ranges(varia)
        region_choice = 'all_lats'#,'northern','central','southern']
        row_names = ['Obs.','ROMSOC','ROMS']
        col_names = ['all_dists','offshore','coastal'] # 'transition'
        fig,ax = plt.subplots(len(row_names),len(col_names),sharex=True,sharey=True,figsize=(12,10))
        for cdx, cn in enumerate(col_names):
            region_name = f'{cn}_{region_choice}'

            # get the observations and compute the regional average
            obs_reg = obs_regions_dict[region_name]['mask']
            #obs_regional_mean = (obs_da*obs_area*obs_reg).sum(dim=('lat','lon')) / (obs_area*obs_reg).sum(dim=('lat','lon'))
            obs_regional_mean = obs_da.weighted((obs_area*obs_reg).fillna(0)).mean(dim=('lat','lon'))

            # get the model and compute the regional average
            model_reg = model_regions_dict[region_name]['mask']
            #roms_regional_mean = (model_da['roms_only']*model_area*model_reg).sum(dim=('eta_rho','xi_rho')) / (model_area*model_reg).sum(dim=('eta_rho','xi_rho')) 
            roms_regional_mean = model_da['roms_only'].weighted((model_area*model_reg).fillna(0)).mean(dim=('eta_rho','xi_rho'))
            #romsoc_regional_mean = (model_da['romsoc_fully_coupled']*model_area*model_reg).sum(dim=('eta_rho','xi_rho')) / (model_area*model_reg).sum(dim=('eta_rho','xi_rho')) 
            romsoc_regional_mean = model_da['romsoc_fully_coupled'].weighted((model_area*model_reg).fillna(0)).mean(dim=('eta_rho','xi_rho'))

            # set the spine color
            for axi in ax[:,cdx]:
                for spine in axi.spines.values():
                    spine.set_edgecolor(model_regions_dict[region_name]['id_colour'])
                axi.tick_params(color=model_regions_dict[region_name]['id_colour'])

            # assert that the depth vectors are both negatively defined: 
            if np.any(np.sign(obs_da.depth)==1):
                obs_da['depth'] = obs_da.depth*-1

            # check whether the observational time dimension is called "time" or "month"/"day" (depending on whether this is a climatological or interannualy varying dataset)
            print(roms_regional_mean.time)
            if 'month' in obs_regional_mean.dims:
                print('rename month to time')
                obs_regional_mean = obs_regional_mean.rename({'month':'time'})
                print(obs_regional_mean.time)
            elif 'day' in obs_regional_mean.dims:
                print('rename day to time')
                obs_regional_mean = obs_regional_mean.rename({'day':'time'})
            elif 'time' in obs_regional_mean.dims:
                print("Nothing to do. Time dimension 'time' already exists.")
            else:
                print('Generate the time axis.')
                obs_regional_mean = obs_regional_mean.expand_dims(dim={'time':roms_regional_mean.time})
                #raise Exception('A non-existant time dimension, or a time dimension that is not called any of the following cannot be handled yet: month, day, time.')

            if np.size(obs_regional_mean.time) != np.size(roms_regional_mean.time):
                print('Repeat the time vector and data of the observations x times.')
                x = int(np.size(roms_regional_mean.time)/np.size(obs_regional_mean.time))
                print(f'x={x}')
                repeated_data_list = [obs_regional_mean for _ in range(x)]
                obs_regional_mean = xr.concat(repeated_data_list, dim="time")
                obs_regional_mean = obs_regional_mean.assign_coords(time=roms_regional_mean.time)

            print(obs_regional_mean.shape)
            print(obs_regional_mean.time.shape)
            print(obs_da.depth.shape)
            # Make sure that the order here matches the order in the list "row_names"
            l0 = ax[0,cdx].contourf(obs_regional_mean.time,obs_da.depth,obs_regional_mean.T,levels=np.linspace(col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs']),cmap=col['obs']['cmap_contourf'],extend='both')
            l1 = ax[1,cdx].contourf(romsoc_regional_mean.time,romsoc_regional_mean.depth,romsoc_regional_mean.T,levels=np.linspace(col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs']),cmap=col['mod']['cmap_contourf'],extend='both')
            l2 = ax[2,cdx].contourf(roms_regional_mean.time,roms_regional_mean.depth,roms_regional_mean.T,levels=np.linspace(col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs']),cmap=col['mod']['cmap_contourf'],extend='both')

            plotted_values[f'{region_name}_obs'] = obs_regional_mean.rename({"depth": "depth_obs"})
            plotted_values[f'{region_name}_roms'] = roms_regional_mean.rename({"depth": "depth_roms"})
            plotted_values[f'{region_name}_romsoc'] = romsoc_regional_mean.rename({"depth": "depth_romsoc"})

        for adx,axi in enumerate(ax[:,0]):
            axi.set_ylabel(row_names[adx],rotation='horizontal',fontweight='bold',ha='right')
        for adx,axi in enumerate(ax[0,:]):
            axi.set_title(col_names[adx],fontweight='bold')
        ax[0,1].set_ylabel('Depth')
        #ax[-1,-1].legend(loc = 'center left',bbox_to_anchor=(1,0.5))

        for axi in ax.flatten():
            axi.set_ylim([-500,0])

        for axi in ax[-1,:]:
            xticklabels = axi.get_xticklabels()
            axi.set_xticklabels(xticklabels, rotation = 45)

        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        cbax = fig.add_axes([0.91,0.2,0.025,0.6])
        cbar = plt.colorbar(l0,cax=cbax)
        cbar.ax.set_title(col['obs']['unit'])

        if savefig == True:
            outpath = f'/nfs/sea/work/fpfaeffli/plots/verticals_eike/time_vs_depth/'
            figname = f'{varia}.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values


    @staticmethod
    def plot_time_vs_depth_sections_climatology(varia,obs_da,obs_area,model_da,model_area,obs_regions_dict,model_regions_dict,plot_resolution,savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: This function allows for the plotting of time vs. depth sections averaged in different regions.      
        """

        plotted_values = xr.Dataset()

        col = Plotter.get_color_maps_and_ranges(varia)
        region_choice = 'all_lats'#,'northern','central','southern']
        row_names = ['Obs.','ROMSOC','ROMS']
        col_names = ['all_dists','offshore','coastal'] # 'transition',
        fig,ax = plt.subplots(len(row_names),len(col_names),sharex=True,sharey=True,figsize=(12,10))
        for cdx, cn in enumerate(col_names):
            region_name = f'{cn}_{region_choice}'

            # get the observations and compute the regional average
            obs_reg = obs_regions_dict[region_name]['mask']
            #obs_regional_mean = (obs_da*obs_area*obs_reg).sum(dim=('lat','lon')) / (obs_area*obs_reg).sum(dim=('lat','lon'))
            obs_regional_mean = obs_da.weighted((obs_area*obs_reg).fillna(0)).mean(dim=('lat','lon'))
            if 'month' in obs_regional_mean.dims: 
                obs_regional_mean.rename({"month": "time"})
            elif 'day' in obs_regional_mean.dims:
                obs_regional_mean.rename({"day": "time"})
            
            # get the model and compute the regional average
            model_reg = model_regions_dict[region_name]['mask']
            #roms_regional_mean = (model_da['roms_only']*model_area*model_reg).sum(dim=('eta_rho','xi_rho')) / (model_area*model_reg).sum(dim=('eta_rho','xi_rho')) 
            roms_regional_mean = model_da['roms_only'].weighted((model_area*model_reg).fillna(0)).mean(dim=('eta_rho','xi_rho'))
            #romsoc_regional_mean = (model_da['romsoc_fully_coupled']*model_area*model_reg).sum(dim=('eta_rho','xi_rho')) / (model_area*model_reg).sum(dim=('eta_rho','xi_rho')) 
            romsoc_regional_mean = model_da['romsoc_fully_coupled'].weighted((model_area*model_reg).fillna(0)).mean(dim=('eta_rho','xi_rho'))

            # set the spine color
            for axi in ax[:,cdx]:
                for spine in axi.spines.values():
                    spine.set_edgecolor(model_regions_dict[region_name]['id_colour'])
                axi.tick_params(color=model_regions_dict[region_name]['id_colour'])

            # assert that the depth vectors are both negatively defined: 
            if np.any(np.sign(obs_da.depth)==1):
                obs_da['depth'] = obs_da.depth*-1

            # check whether the observational time dimension is called "time" or "month"/"day" (depending on whether this is a climatological or interannualy varying dataset)
            if 'month' in obs_regional_mean.dims:
                print('rename month to time')
                obs_regional_mean = obs_regional_mean.rename({'month':'time'})
                print(obs_regional_mean.time)
            elif 'day' in obs_regional_mean.dims:
                print('rename day to time')
                obs_regional_mean = obs_regional_mean.rename({'day':'time'})
            elif 'time' in obs_regional_mean.dims:
                print("Nothing to do. Time dimension 'time' already exists.")
            else:
                print('Generate the time axis.')
                obs_regional_mean = obs_regional_mean.expand_dims(dim={'time':roms_regional_mean.time})
                #raise Exception('A non-existant time dimension, or a time dimension that is not called any of the following cannot be handled yet: month, day, time.')

            # compute the seasonal climatology
            if plot_resolution == 'monthly':
                sampler = 'month'
            elif plot_resolution == 'daily':
                sampler = 'day'

            if sampler == 'month' and np.size(obs_regional_mean.time)!= 12:
                obs_regional_mean = obs_regional_mean.groupby(f"time.{sampler}").mean("time")
            elif sampler == 'day' and np.size(obs_regional_mean.time)!= 365:
                obs_regional_mean = obs_regional_mean.groupby(f"time.{sampler}").mean("time")
            else:
                obs_regional_mean = obs_regional_mean.rename({'time':sampler})
            roms_regional_mean = roms_regional_mean.groupby(f"time.{sampler}").mean("time")
            romsoc_regional_mean = romsoc_regional_mean.groupby(f"time.{sampler}").mean("time")

            # Make sure that the order here matches the order in the list "row_names"
            l0 = ax[0,cdx].contourf(obs_regional_mean.month,obs_da.depth,obs_regional_mean.T,levels=np.linspace(col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs']),cmap=col['obs']['cmap_contourf'],extend='both')
            l1 = ax[1,cdx].contourf(romsoc_regional_mean.month,romsoc_regional_mean.depth,romsoc_regional_mean.T,levels=np.linspace(col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs']),cmap=col['mod']['cmap_contourf'],extend='both')
            l2 = ax[2,cdx].contourf(roms_regional_mean.month,roms_regional_mean.depth,roms_regional_mean.T,levels=np.linspace(col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs']),cmap=col['mod']['cmap_contourf'],extend='both')

            plotted_values[f'{region_name}_obs'] = obs_regional_mean.rename({"depth": "depth_obs"})
            plotted_values[f'{region_name}_roms'] = roms_regional_mean.rename({"depth": "depth_roms"})
            plotted_values[f'{region_name}_romsoc'] = romsoc_regional_mean.rename({"depth": "depth_romsoc"})

        for adx,axi in enumerate(ax[:,0]):
            axi.set_ylabel(row_names[adx],rotation='horizontal',fontweight='bold',ha='right')
        for adx,axi in enumerate(ax[0,:]):
            axi.set_title(col_names[adx],fontweight='bold')
        ax[0,1].set_ylabel('Depth')

        for axi in ax.flatten():
            axi.set_ylim([-500,0])

        for axi in ax[-1,:]:
            if 'month' in obs_regional_mean.dims:
                xticks = axi.set_xticks(obs_regional_mean['month'])
                xlabel = axi.set_xlabel('Month of year')
            elif 'day' in obs_regional_mean.dims:
                xticks = axi.set_xticks(obs_regional_mean['day'])
                xlabel = axi.set_xlabel('Day of year')                
            xticklabels = axi.get_xticklabels()
            axi.set_xticklabels(xticklabels, rotation = 45)

        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        cbax = fig.add_axes([0.91,0.2,0.025,0.6])
        cbar = plt.colorbar(l0,cax=cbax)
        cbar.ax.set_title(col['obs']['unit'])

        if savefig == True:
            outpath = f'/nfs/sea/work/fpfaeffli/plots/verticals_eike/time_vs_depth_clim/'
            figname = f'{varia}_{plot_resolution}_clim.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)
        
        return plotted_values


    @staticmethod
    def plot_depth_vs_dist2coast_transect(varia,target_lats,obs_da_interp,obs_d2coast_interp,model_da_interp,model_d2coast_interp,savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: This function allows for the plotting of time vs. dist2coast sections averaged in different regions.      
        """
        #print('Generate plot: Depth vs dist2coast transects annual mean')

        plotted_values = xr.Dataset()
        col = Plotter.get_color_maps_and_ranges(varia)
        row_names = ['Obs.','ROMSOC','ROMS']
        col_names = [f'{target_lat}°N' for target_lat in target_lats]
        fig,ax = plt.subplots(len(row_names),len(col_names),sharex=True,sharey=True,figsize=(12,10))

        for cdx, targ_lat in enumerate(target_lats):

            # assert that the depth vectors are both negatively defined: 
            if np.any(np.sign(obs_da_interp.depth)==1):
                obs_da_interp['depth'] = obs_da_interp.depth*-1
            
            # Make sure that the order here matches the order in the list "row_names"
            obs_d2c = -1*obs_d2coast_interp.sel(lat=targ_lat)
            obs_d2c[np.isnan(obs_d2c)]=0.
            mod_d2c = -1*model_d2coast_interp.sel(lat=targ_lat)
            mod_d2c[np.isnan(mod_d2c)]=0.
            l0 = ax[0,cdx].contourf(obs_d2c,obs_da_interp.depth,obs_da_interp.sel(lat=targ_lat),levels=np.linspace(col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs']),cmap=col['obs']['cmap_contourf'],extend='both')
            l1 = ax[1,cdx].contourf(mod_d2c,model_da_interp['romsoc_fully_coupled'].depth,model_da_interp['romsoc_fully_coupled'].sel(lat=targ_lat),levels=np.linspace(col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs']),cmap=col['mod']['cmap_contourf'],extend='both')
            l2 = ax[2,cdx].contourf(mod_d2c,model_da_interp['roms_only'].depth,model_da_interp['roms_only'].sel(lat=targ_lat),levels=np.linspace(col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs']),cmap=col['mod']['cmap_contourf'],extend='both')

            plotted_values[f'obs'] = obs_da_interp
            plotted_values[f'roms'] = model_da_interp['roms_only']
            plotted_values[f'romsoc'] = model_da_interp['romsoc_fully_coupled']

        for adx,axi in enumerate(ax[:,0]):
            axi.set_ylabel(row_names[adx],rotation='horizontal',fontweight='bold',ha='right')
        for adx,axi in enumerate(ax[0,:]):
            axi.set_title(col_names[adx],fontweight='bold')
        ax[0,1].set_ylabel('Depth')

        for axi in ax.flatten():
            axi.set_ylim([-500,0])
            axi.set_xlim([-1000,0])

        for axi in ax[-1,:]:
            axi.set_xlabel('Dist. to coast in km')

        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        cbax = fig.add_axes([0.91,0.2,0.025,0.6])
        cbar = plt.colorbar(l0,cax=cbax)
        cbar.ax.set_title(col['obs']['unit'])

        if savefig == True:
            outpath = f'/nfs/sea/work/fpfaeffli/plots/verticals_eike/depth_vs_dist2coast_transect/'
            figname = f'{varia}.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values

    @staticmethod
    def plot_autocorrelation_timescales_map(varia,depth,obs_da,model_da,obs_regions_dict,model_regions_dict,savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: Plotting maps for the annual mean fields of a given variable. 
        The plot structure is as follows: 2 rows, 3 columns. 
        ax[0,0] = Obs           |  ax[0,1] = ROMSOC    | ax[0,2] = ROMSOC-Obs
        ax[1,0] = ROMSOC-ROMS   |  ax[1,1] = ROMS-only | ax[1,2] = ROMS-Obs
        """

        annual_mean_Obs = obs_da
        annual_mean_roms = model_da['roms_only']
        annual_mean_romsoc = model_da['romsoc_fully_coupled']

        #annual_mean_roms = xr.where(ModelGetter.get_model_mask()==0,np.NaN,annual_mean_roms)
        #annual_mean_romsoc = xr.where(ModelGetter.get_model_mask()==0,np.NaN,annual_mean_romsoc)

        # calculate the difference between model and obs - need to regrid first here (regrid obs to model?)
        #annual_mean_Obs_on_model_grid = Regridder.regrid_original_to_target(annual_mean_Obs.values,annual_mean_Obs.lon.values,annual_mean_Obs.lat.values,annual_mean_roms.lon.values,annual_mean_roms.lat.values)
        
        # set up the plot
        col = Plotter.get_color_maps_and_ranges(varia,depth)
        fontsize=12
        plt.rcParams['font.size']=fontsize
        fig, ax = plt.subplots(2,3,figsize=(10,8),sharex=True,sharey=True)

        # Plot ax[0,0], i.e. the observations
        c00 = ax[0,0].pcolormesh(annual_mean_Obs.lon,annual_mean_Obs.lat,annual_mean_Obs,vmin=col['obs']['minval'],vmax=col['obs']['maxval'],cmap=col['obs']['cmap_pcmesh'])
        ax[0,0].set_title('Observations')
        ax[0,0].set_xlim([obs_regions_dict['full_map']['minlon'],obs_regions_dict['full_map']['maxlon']])
        ax[0,0].set_ylim([obs_regions_dict['full_map']['minlat'],obs_regions_dict['full_map']['maxlat']])
        cbar00 = plt.colorbar(c00,ax=ax[0,0],extend='both')
        cbar00.ax.set_title(col['obs']['unit'],pad=15)

        # Plot ax[0,1], i.e. ROMSOC
        ax[0,1].set_title('ROMSOC')
        c01 = ax[0,1].pcolormesh(annual_mean_romsoc.lon,annual_mean_romsoc.lat,annual_mean_romsoc,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
        cbar01 = plt.colorbar(c01,ax=ax[0,1],extend='both')
        cbar01.ax.set_title(col['mod']['unit'],pad=15)

        # Plot ax[1,1], i.e. ROMS-only
        ax[1,1].set_title('ROMS-only')
        c11 = ax[1,1].pcolormesh(annual_mean_roms.lon,annual_mean_roms.lat,annual_mean_roms,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
        cbar11 = plt.colorbar(c11,ax=ax[1,1],extend='both')
        cbar11.ax.set_title(col['mod']['unit'],pad=15)
       
        # Plot ax[1,0], i.e. ROMSOC-ROMS
        ax[1,0].set_title('ROMSOC minus ROMS')
        c10 = ax[1,0].pcolormesh(annual_mean_roms.lon,annual_mean_roms.lat,annual_mean_romsoc-annual_mean_roms,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        cbar10 = plt.colorbar(c10,ax=ax[1,0],extend='both')
        cbar10.ax.set_title(col['mod-mod']['unit'],pad=15)

        # Plot ax[0,2], i.e. ROMSOC minus Obs
        ax[0,2].set_title('ROMSOC minus Obs.')
        c02 = ax[0,2].pcolormesh(annual_mean_romsoc.lon,annual_mean_romsoc.lat,annual_mean_romsoc-annual_mean_Obs,vmin=col['mod-obs']['minval'],vmax=col['mod-obs']['maxval'],cmap=col['mod-obs']['cmap_pcmesh'])
        cbar02 = plt.colorbar(c02,ax=ax[0,2],extend='both')
        cbar02.ax.set_title(col['mod-obs']['unit'],pad=15)

        # Plot ax[1,2], i.e. ROMS-only minus Obs
        ax[1,2].set_title('ROMS-only minus Obs.')
        c12 = ax[1,2].pcolormesh(annual_mean_roms.lon,annual_mean_roms.lat,annual_mean_roms-annual_mean_Obs,vmin=col['mod-obs']['minval'],vmax=col['mod-obs']['maxval'],cmap=col['mod-obs']['cmap_pcmesh'])
        cbar12 = plt.colorbar(c12,ax=ax[1,2],extend='both')
        cbar12.ax.set_title(col['mod-obs']['unit'],pad=15)

        # add the mask regions in each panel (first the Obs. mask in ax[0,0], then the Model mask for all others)
        for region in obs_regions_dict.keys():
            if region != 'full_map' and 'all_lats' not in region:
                region_mask_dummy = xr.where(np.isnan(obs_regions_dict[region]['mask']),0,1)
                ax[0,0].contour(annual_mean_Obs.lon,annual_mean_Obs.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)
        for axi in ax.flatten()[1:]:
            for region in model_regions_dict.keys():
                if region != 'full_map' and 'all_lats' not in region:
                    region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                    axi.contour(annual_mean_roms.lon,annual_mean_roms.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)

        # add the continent
        landmask_etopo = PlotFuncs.get_etopo_data()
        for axi in ax.flatten():
            axi.contourf(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo,colors='#555555')

        # set the plot extent and labels
        for axi in ax.flatten():
            axi.set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
            axi.set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
            yticks = np.arange(20,70,10)
            xticks = np.arange(230,260,10)
            axi.set_yticks(yticks)
            axi.set_yticklabels([str(val)+'°N' for val in yticks],fontsize=fontsize-2)
            axi.set_xticks(xticks)
            axi.set_xticklabels([str(360-val)+'°W' for val in xticks],fontsize=fontsize-2)

        plt.tight_layout()

        # put the plotted values into a Dataset 
        plotted_values = xr.Dataset()
        plotted_values['obs'] = annual_mean_Obs
        plotted_values['romsoc'] = annual_mean_romsoc
        plotted_values['roms'] = annual_mean_roms
        plotted_values['romsoc_minus_roms'] = annual_mean_romsoc-annual_mean_roms
        plotted_values['romsoc_minus_obs'] = annual_mean_romsoc-annual_mean_Obs#_on_model_grid
        plotted_values['roms_minus_obs'] = annual_mean_roms-annual_mean_Obs#_on_model_grid

        if savefig == True:
            outpath = f'/nfs/sea/work/fpfaeffli/plots/verticals_eike/autocorrelation_timescales_map/'
            figname = f'{varia}_{depth}m.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values

    @staticmethod
    def plot_autocorrelation_function_regionalized(varia_name,dep,region_names,model_acf,obs_acf,model_regions_dict,obs_regions_dict,configs,savefig=False):
        fontsize=12
        numrows = 4
        numcols = 3
        plt.rcParams['font.size']=fontsize
        fig, ax = plt.subplots(numrows,numcols,figsize=(15,10),sharey=False,sharex=False)
        print('-------------')
        print('working on {}'.format(varia_name))
        fig.suptitle(f"ACF {varia_name} ({dep}m)",ha='center',fontweight='bold',va='center')
        for rdx in range(len(region_names)):
            for cdx in range(len(region_names[rdx])):
                region = region_names[rdx][cdx]
                ax[rdx,cdx].set_title(region,fontweight='bold',color=model_regions_dict[region]['id_colour'])
                for config in configs:
                    if config == 'romsoc_fully_coupled':
                        linestyle = '-'
                        lc = 'fuchsia'
                    elif config == 'roms_only':
                        linestyle = ':'
                        lc = 'purple'
                    data = model_acf[config][region]
                    lags = np.arange(np.size(data.lag))
                    if rdx == 0 and cdx == numcols-1:
                        label = config.replace('romsoc_fully_coupled','ROMSOC').replace('roms_only','ROMS-only')
                        ax[rdx,cdx].plot(lags,data,color=lc,linestyle=linestyle,label=label)
                    else:
                        ax[rdx,cdx].plot(lags,data,color=lc,linestyle=linestyle)
                # add the observational acf
                data = obs_acf[region]
                lags = np.arange(np.size(data.lag))
                if rdx == 0 and cdx == numcols-1:
                    label = 'Obs.'
                    ax[rdx,cdx].plot(lags,data,color='k',linestyle=linestyle,label=label)
                else:
                    ax[rdx,cdx].plot(lags,data,color='k',linestyle=linestyle)                
                ax[rdx,cdx].axvline(0,color='C0')
                ax[rdx,cdx].spines['right'].set_visible(False)
                ax[rdx,cdx].spines['top'].set_visible(False)
                ax[rdx,cdx].spines['left'].set_visible(False)
                ax[rdx,cdx].spines['bottom'].set_visible(False)
                ax[rdx,cdx].grid(axis='both',linestyle='--',alpha=0.25)
                ax[rdx,cdx].set_xlim([0,60]) # np.max(lags)])
                ax[rdx,cdx].plot([0,38],[1/np.exp(1)]*2,color='#777777',linewidth=1)
                ax[rdx,cdx].text(39,1/np.exp(1),'1/e',ha='left',va='center',color='k')
            ax[rdx,0].set_ylabel('Autocorrelation')
        ax[-1,1].set_xlabel('Lag in days')
        ax[0,-1].legend(loc = 'center left',bbox_to_anchor=(1,0.5))
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plotted_values = xr.Dataset()
        if savefig == True:
            outpath = f'/nfs/sea/work/fpfaeffli/plots/verticals_eike/autocorrelation_function_regionalized/'
            figname = f'{varia_name}_acf_{dep}m_regional_means.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)
        plt.show()
        return plotted_values
    

    @staticmethod
    def plot_full_map_annual_mean_std(varia,depth,obs_ds,obs_da,model_ds,model_da,obs_regions_dict,model_regions_dict,plot_resolution,regional_data=None,regional_data_plottype='pcolmesh',savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: Plotting maps for the annual mean fields of a given variable. 
        The plot structure is as follows: 2 rows, 3 columns. 
        ax[0,0] = Obs           |  ax[0,1] = ROMSOC    | ax[0,2] = ROMSOC-Obs
        ax[1,0] = ROMSOC-ROMS   |  ax[1,1] = ROMS-only | ax[1,2] = ROMS-Obs
        """

        panel_labels = ['d)','e)','f)','g)','h)','i)']

        annual_mean_Obs = obs_da
        annual_mean_roms = model_da['roms_only']['present']
        annual_mean_romsoc = model_da['romsoc_fully_coupled']['present']

        # calculate the difference between model and obs - need to regrid first here (regrid obs to model?)
        #annual_mean_Obs_on_model_grid = Regridder.regrid_original_to_target(annual_mean_Obs.values,annual_mean_Obs.lon.values,annual_mean_Obs.lat.values,annual_mean_roms.lon.values,annual_mean_roms.lat.values)
        
        # set up the plot
        col = Plotter.get_color_maps_and_ranges(varia,depth)
        fontsize=12
        plt.rcParams['font.size']=fontsize
        fig, ax = plt.subplots(2,3,figsize=(10,8),sharex=True,sharey=True)

        # Plot ax[0,0], i.e. the observations
        c00 = ax[0,0].pcolormesh(annual_mean_Obs.lon,annual_mean_Obs.lat,annual_mean_Obs,vmin=col['obs']['minval'],vmax=col['obs']['maxval'],cmap=col['obs']['cmap_pcmesh'])
        ax[0,0].set_title('Observations')
        ax[0,0].set_xlim([obs_regions_dict['full_map']['minlon'],obs_regions_dict['full_map']['maxlon']])
        ax[0,0].set_ylim([obs_regions_dict['full_map']['minlat'],obs_regions_dict['full_map']['maxlat']])
        cbar00 = plt.colorbar(c00,ax=ax[0,0],extend='both')
        cbar00.ax.set_title(col['obs']['unit'],pad=15)

        # Plot ax[0,1], i.e. ROMSOC
        ax[0,1].set_title('ROMSOC')
        c01 = ax[0,1].pcolormesh(annual_mean_romsoc.lon,annual_mean_romsoc.lat,annual_mean_romsoc,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
        cbar01 = plt.colorbar(c01,ax=ax[0,1],extend='both')
        cbar01.ax.set_title(col['mod']['unit'],pad=15)

        # Plot ax[1,1], i.e. ROMS-only
        ax[1,1].set_title('ROMS-only')
        c11 = ax[1,1].pcolormesh(annual_mean_roms.lon,annual_mean_roms.lat,annual_mean_roms,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
        cbar11 = plt.colorbar(c11,ax=ax[1,1],extend='both')
        cbar11.ax.set_title(col['mod']['unit'],pad=15)
       
        # Plot ax[1,0], i.e. ROMSOC-ROMS
        ax[1,0].set_title('ROMSOC minus ROMS')
        c10 = ax[1,0].pcolormesh(annual_mean_roms.lon,annual_mean_roms.lat,annual_mean_romsoc-annual_mean_roms,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        cbar10 = plt.colorbar(c10,ax=ax[1,0],extend='both')
        cbar10.ax.set_title(col['mod-mod']['unit'],pad=15)

        # Plot ax[0,2], i.e. ROMSOC minus Obs
        ax[0,2].set_title('ROMSOC minus Obs.')
        c02 = ax[0,2].pcolormesh(annual_mean_romsoc.lon,annual_mean_romsoc.lat,annual_mean_romsoc-annual_mean_Obs,vmin=col['mod-obs']['minval'],vmax=col['mod-obs']['maxval'],cmap=col['mod-obs']['cmap_pcmesh'])
        cbar02 = plt.colorbar(c02,ax=ax[0,2],extend='both')
        cbar02.ax.set_title(col['mod-obs']['unit'],pad=15)

        # Plot ax[1,2], i.e. ROMS-only minus Obs
        ax[1,2].set_title('ROMS-only minus Obs.')
        c12 = ax[1,2].pcolormesh(annual_mean_roms.lon,annual_mean_roms.lat,annual_mean_roms-annual_mean_Obs,vmin=col['mod-obs']['minval'],vmax=col['mod-obs']['maxval'],cmap=col['mod-obs']['cmap_pcmesh'])
        cbar12 = plt.colorbar(c12,ax=ax[1,2],extend='both')
        cbar12.ax.set_title(col['mod-obs']['unit'],pad=15)

        # add the continent
        landmask_etopo = PlotFuncs.get_etopo_data()
        for axi in ax.flatten():
            axi.contourf(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo,colors='#555555')

        # set the plot extent and labels
        for adx,axi in enumerate(ax.flatten()):
            axi.set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
            axi.set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
            yticks = np.arange(20,70,10)
            xticks = np.arange(230,260,10)
            axi.set_yticks(yticks)
            axi.set_yticklabels([str(val)+'°N' for val in yticks],fontsize=fontsize-2)
            axi.set_xticks(xticks)
            axi.set_xticklabels([str(360-val)+'°W' for val in xticks],fontsize=fontsize-2)
            axi.text(0.05,0.97,panel_labels[adx],ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=axi.transAxes)
        plt.tight_layout()

        if regional_data != None:
            regional_data_obs,regional_data_model = regional_data
            regions_to_plot = ['offshore_all_lats','coastal_all_lats'] # 'transition_all_lats',
            regions_to_plot_short = [reg.upper()[0] for reg in regions_to_plot]
            regions_to_plot_dummy = [reg.replace('_all_lats','') for reg in regions_to_plot]
            print('adding the monthly data timeseries for the regions')
            print(regions_to_plot)

            ax_insets = np.empty_like(ax)
            for rdx in range(np.shape(ax)[0]):
                for cdx in range(np.shape(ax)[1]):
                    if regional_data_plottype == 'pcolmesh':
                        ax_insets[rdx,cdx] = ax[rdx,cdx].inset_axes([.78, .3, .2, .68])
                    elif regional_data_plottype == 'lines':
                        ax_insets[rdx,cdx] = ax[rdx,cdx].inset_axes([.58, .75, .4, .22])   # [.58, .5, .4, .48]

            # concatenate all the data in the regions for the model and obs
            concat_obs = np.concatenate(tuple([regional_data_obs[regi].values[:,None] for regi in regions_to_plot]),axis=1)
            concat_romsoc = np.concatenate(tuple([regional_data_model['romsoc_fully_coupled'][regi].values[:,None] for regi in regions_to_plot]),axis=1)
            concat_roms = np.concatenate(tuple([regional_data_model['roms_only'][regi].values[:,None] for regi in regions_to_plot]),axis=1)

            if regional_data_plottype == 'pcolmesh':
                ax_insets[0,0].pcolormesh(concat_obs,cmap=col['obs']['cmap_pcmesh'],vmin=col['obs']['minval'],vmax=col['obs']['maxval'],edgecolor='k',linewidth=0.125)
                ax_insets[0,1].pcolormesh(concat_romsoc,cmap=col['mod']['cmap_pcmesh'],vmin=col['mod']['minval'],vmax=col['mod']['maxval'],edgecolor='k',linewidth=0.125)
                ax_insets[0,2].pcolormesh(concat_romsoc-concat_obs,cmap=col['mod-obs']['cmap_pcmesh'],vmin=col['mod-obs']['minval'],vmax=col['mod-obs']['maxval'],edgecolor='k',linewidth=0.125)
                ax_insets[1,0].pcolormesh(concat_romsoc-concat_roms,cmap=col['mod-mod']['cmap_pcmesh'],vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],edgecolor='k',linewidth=0.125)
                ax_insets[1,1].pcolormesh(concat_roms,cmap=col['mod']['cmap_pcmesh'],vmin=col['mod']['minval'],vmax=col['mod']['maxval'],edgecolor='k',linewidth=0.125)
                ax_insets[1,2].pcolormesh(concat_roms-concat_obs,cmap=col['mod-obs']['cmap_pcmesh'],vmin=col['mod-obs']['minval'],vmax=col['mod-obs']['maxval'],edgecolor='k',linewidth=0.125)
            elif regional_data_plottype == 'lines':
                ax_insets[0,0].plot(np.arange(1,13),concat_obs,'.-')
                ax_insets[0,1].plot(np.arange(1,13),concat_romsoc,'.-')
                lineObjs = ax_insets[0,2].plot(np.arange(1,13),concat_romsoc-concat_obs,'.-')
                ax_insets[1,0].plot(np.arange(1,13),concat_romsoc-concat_roms,'.-')
                ax_insets[1,1].plot(np.arange(1,13),concat_roms,'.-')
                ax_insets[1,2].plot(np.arange(1,13),concat_roms-concat_obs,'.-')
                ax_insets[0,2].legend(iter(lineObjs),tuple(regions_to_plot_dummy),loc='lower left',bbox_to_anchor=(0,-1.4),fontsize=fontsize-4,handletextpad = 0.4, handlelength=2)               

            for rdx in range(np.shape(ax)[0]):
                for cdx in range(np.shape(ax)[1]):
                    if regional_data_plottype == 'pcolmesh':
                        ax_insets[rdx,cdx].set_yticks(np.arange(0.5,12.5))
                        ax_insets[rdx,cdx].set_yticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],color='w',fontsize=plt.rcParams['font.size']-1)
                        ax_insets[rdx,cdx].set_xticks(np.arange(len(regions_to_plot))+0.5)
                        ax_insets[rdx,cdx].set_xticklabels(regions_to_plot_short,color='w',fontsize=plt.rcParams['font.size']-1)
                        ax_insets[rdx,cdx].invert_yaxis()
                    elif regional_data_plottype == 'lines':
                        ax_insets[rdx,cdx].set_xticks(np.arange(1,13,2))
                        ax_insets[rdx,cdx].set_xticklabels(['J','M','M','J','S','N'],color='w',fontsize=plt.rcParams['font.size']-1)
                        #yticklabs = ax_insets[rdx,cdx].yaxis.label.set_color('w')
                        ax_insets[rdx,cdx].tick_params(axis='y', colors='w')
                        #ax_insets[rdx,cdx].set_yticklabels(yticklabs,color='w',fontsize=plt.rcParams['font.size']-1)
                        ax_insets[rdx,cdx].grid(color='#EEEEEE',linewidth=0.5)
                        ax_insets[rdx,cdx].set_ylabel(col['mod']['unit'],color='w')    
                        ax_insets[rdx,cdx].set_xlabel('Month',color='w')    


        # add the mask regions in each panel (first the Obs. mask in ax[0,0], then the Model mask for all others)
        if regional_data == None:
            for region in obs_regions_dict.keys():
                if region != 'full_map' and 'all_lats' not in region:
                    region_mask_dummy = xr.where(np.isnan(obs_regions_dict[region]['mask']),0,1)
                    ax[0,0].contour(annual_mean_Obs.lon,annual_mean_Obs.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)
            for axi in ax.flatten()[1:]:
                for region in model_regions_dict.keys():
                    if region != 'full_map' and 'all_lats' not in region:
                        region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                        axi.contour(annual_mean_roms.lon,annual_mean_roms.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)

        elif regional_data != None:
            for region in regions_to_plot:
                region_mask_dummy = xr.where(np.isnan(obs_regions_dict[region]['mask']),0,1)
                ax[0,0].contour(annual_mean_Obs.lon,annual_mean_Obs.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)
            for axi in ax.flatten()[1:]:
                for region in regions_to_plot:
                    region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                    axi.contour(annual_mean_roms.lon,annual_mean_roms.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)      

        # put the plotted values into a Dataset 
        plotted_values = xr.Dataset()
        plotted_values['obs'] = annual_mean_Obs
        plotted_values['romsoc'] = annual_mean_romsoc
        plotted_values['roms'] = annual_mean_roms
        plotted_values['romsoc_minus_roms'] = annual_mean_romsoc-annual_mean_roms
        plotted_values['romsoc_minus_obs'] = annual_mean_romsoc-annual_mean_Obs#_on_model_grid
        plotted_values['roms_minus_obs'] = annual_mean_roms-annual_mean_Obs#_on_model_grid

        if savefig == True:
            outpath = f'/nfs/sea/work/fpfaeffli/plots/verticals_eike/map_annual_mean_std/'
            figname = f'{varia}_{depth}m.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values
    


    @staticmethod
    def get_color_maps_and_ranges(varia = 'salt', depth = 0):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: This function sets the color ranges and maps for the different variables and depths.
        """

        col = dict()
        col['obs'] = dict()
        col['mod'] = dict()
        col['mod-obs'] = dict()
        col['mod-mod'] = dict()

        if varia == 'temp' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          7,22,16,'RdYlBu_r',plt.get_cmap('RdYlBu_r',15) ,'°C'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          7,22,16,'RdYlBu_r',plt.get_cmap('RdYlBu_r',15) ,'°C'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -3,3,13,'cmo.balance',plt.get_cmap('cmo.balance',12) ,'°C'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -3,3,13,'cmo.balance',plt.get_cmap('cmo.balance',12) ,'°C'

        if varia == 'salt' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                           32,36,13,'cmo.haline',plt.get_cmap('cmo.haline',12) ,'-'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                           32,36,13,'cmo.haline',plt.get_cmap('cmo.haline',12) ,'-'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] = -.8,.8,17,'cmo.balance',plt.get_cmap('cmo.balance',16) ,'-'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] = -.8,.8,17,'cmo.balance',plt.get_cmap('cmo.balance',16) ,'-'

        if varia == 'NO3' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          0,30,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'  mmol m-3'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,30,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'  mmol m-3'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -25,25,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'  mmol m-3'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -25,25,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'  mmol m-3'

        if varia == 'PO4' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          0,2,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'  mmol m-3'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,2,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'  mmol m-3'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -1,1,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'  mmol m-3'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -1,1,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'  mmol m-3'

        if varia == 'DIC' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          1800,2300,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'  mmol m-3'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          1800,2300,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'  mmol m-3'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -200,200,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'  mmol m-3'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -200,200,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'  mmol m-3'

        if varia == 'Alk' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          1900,2400,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'  mmol m-3'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          1900,2400,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'  mmol m-3'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -400,400,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'  mmol m-3'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -400,400,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'  mmol m-3'

        if varia == 'pH_offl' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          7.9,8.1,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'-'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          7.9,8.1,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'-'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -.1,.1,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'-'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -.1,.1,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'-'

        if varia == 'omega_arag_offl' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          1,4,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'-'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          1,4,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'-'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -.5,.5,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'-'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -.5,.5,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'-'

        if varia == 'zeta' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          -.5,.5,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'m'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          -.5,.5,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'m'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -.5,.5,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'m'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -.1,.1,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'m'

        if varia == 'O2' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          0,300,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'  mmol m-3'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,300,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'  mmol m-3'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -50,50,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'  mmol m-3'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -50,50,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'  mmol m-3'

        if varia == 'mld_holte' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          -80,0,17,'cmo.tempo_r',plt.get_cmap('cmo.tempo_r',16) ,' m'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          -80,0,17,'cmo.tempo_r',plt.get_cmap('cmo.tempo_r',16) ,' m'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -50,50,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,' m'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -50,50,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,' m'

        # if varia == 'temp_acf' and depth == 0:
        #     col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          0,400,21,'cmo.deep_r',plt.get_cmap('cmo.deep_r',20) ,'    days'
        #     col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,400,21,'cmo.deep_r',plt.get_cmap('cmo.deep_r',20) ,'    days'
        #     col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -100,100,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'    days'
        #     col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -100,100,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'    days'

        if varia == 'temp_acf' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          10,40,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'    days'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          10,40,16,'cmo.deep_r',plt.get_cmap('cmo.deep_r',15) ,'    days'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -20,20,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'    days'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mkdir -p /nfs/sea/work/fpfaeffli/...../mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -20,20,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'    days'

        if varia == 'temp_std' and depth == 0:
            col['obs']['minval'],col['obs']['maxval'],col['obs']['numlevs'],col['obs']['cmap_contourf'],col['obs']['cmap_pcmesh'],col['obs']['unit'] =                          0.4,2,17,'cmo.deep_r',plt.get_cmap('cmo.deep_r',16) ,'    °C'
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0.4,2,17,'cmo.deep_r',plt.get_cmap('cmo.deep_r',16) ,'    °C'
            col['mod-obs']['minval'],col['mod-obs']['maxval'],col['mod-obs']['numlevs'],col['mod-obs']['cmap_contourf'],col['mod-obs']['cmap_pcmesh'],col['mod-obs']['unit'] =   -0.5,0.5,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'    °C'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -0.5,0.5,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'    °C'

        return col
    

# %%
