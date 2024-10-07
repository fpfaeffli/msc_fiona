"""
author: Eike E Köhn
date: Jun 21, 2024
description: collection of functions used for the plotting of some analyses regarding the model extremes
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from plotting_functions_general import PlotFuncs as PlotFuncs
import cmocean as cmo
import pandas as pd
#sys.path.append('/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/')

#%%
class ExtAnalysisPlotter():
    """
    author: Eike E. Koehn
    date: June 7, 2024
    description: This class contains a number of functions that enable standardized plotting for the variability analysis across the different model simulations 
    The functions include:
    1. plotting maps of the annual mean field
    """  

    @staticmethod
    def plot_maps_of_blob_snapshot(model_da,extreme_anomalies,config_of_choice,scenarios,times_of_choice,varia_name,depth,model_regions_dict,savefig=False):
        lon = model_da[config_of_choice]['present'].lon
        lat = model_da[config_of_choice]['present'].lat

        col = ExtAnalysisPlotter.get_color_maps_and_ranges(varia_name,depth)
        # Figure design (4 rows, 3 columns)
        # present SST | ssp245 SST   | ssp585 SST   |
        # SST' (pres.)| SST' (pres.) | SST' (pres.) |
        # SST' (mcor.)| SST' (mcor.) | SST' (mcor.) |
        # SST' (ccor.)| SST' (ccor.) | SST' (ccor.) |
        ylabs = ['SST',"SST'\n(rel. to\npresent thresh)","SST'\n(rel. to\npresent thresh\n+ meandelta)","SST'\n(rel. to\npresent thresh\n+ climdelta)"]


        for tdx,time_of_choice in enumerate(times_of_choice):

            fig, ax = plt.subplots(4,3,sharex=True,sharey=True,figsize=(10,15))

            for sdx,scenario in enumerate(scenarios):
                c0 = ax[0,sdx].pcolormesh(lon,lat,model_da[config_of_choice][scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])

                c1 = ax[1,sdx].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
                ax[1,sdx].contour(lon,lat,extreme_anomalies[config_of_choice][scenario]['present'].sel(time=time_of_choice),[0],colors='k',linewidths=0.5)

                ax[2,sdx].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_meandelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
                ax[2,sdx].contour(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_meandelta'].sel(time=time_of_choice),[0],colors='k',linewidths=0.5)


                ax[3,sdx].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_climdelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
                ax[3,sdx].contour(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_climdelta'].sel(time=time_of_choice),[0],colors='k',linewidths=0.5)


            # set the plot extent and labels
            for axi in ax.flatten():
                axi.set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
                axi.set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
                yticks = np.arange(20,70,10)
                xticks = np.arange(230,260,10)
                axi.set_yticks(yticks)
                axi.set_xticks(xticks)
            for adx,axi in enumerate(ax[:,0]):
                axi.set_yticklabels([str(val)+'°N' for val in yticks])
                axi.set_ylabel(ylabs[adx],fontweight='bold',ha='right',rotation=0)
            for axi in ax[-1,:]:
                axi.set_xticklabels([str(360-val)+'°W' for val in xticks])
            for adx,axi in enumerate(ax[0,:]):
                axi.set_title(scenarios[adx])

            # add the continent
            landmask_etopo = PlotFuncs.get_etopo_data()
            for axi in ax.flatten():
                axi.contourf(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo,colors='#555555')
                
            cbax0 = fig.add_axes([0.91,0.75,0.025,0.22])
            cbar0 = plt.colorbar(c0,cax=cbax0,extend='both')
            cbar0.ax.set_title(col['mod']['unit'])

            cbax1 = fig.add_axes([0.91,0.125,0.025,0.5])
            cbar1 = plt.colorbar(c1,cax=cbax1,extend='both')
            cbar1.ax.set_title(col['mod']['unit'])

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05,hspace=0.05,right=0.9)
            
            fig.text(0.03,0.98,time_of_choice.strftime('%B %d, %Y'),fontweight='bold')
        
            # put the plotted values into a Dataset 
            plotted_values = xr.Dataset() 
            # put varia_dict into this one

            if savefig == True and np.size(times_of_choice) == 1:
                outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/snapshots/'
                figname = f'{varia_name}_{depth}m_{config_of_choice}.png'
            elif savefig == True and np.size(times_of_choice) > 1:
                outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/snapshots/animation_frames_long/'
                figname_prefix = f'{varia_name}_{depth}m_{config_of_choice}_frame_'
                figname_suffix = '.png'
                figname = figname_prefix+f'{int(tdx+1):04d}'+figname_suffix
            plt.savefig(outpath+figname,dpi=200,transparent=True)            

            plt.show()

        if savefig == True and np.size(times_of_choice) > 1:
            frames_per_second = 12
            if os.path.isfile(f'{outpath}{figname_prefix}animation.mp4'):
                os.system(f'rm {outpath}{figname_prefix}animation.mp4')
            os.system(f'ffmpeg -r {frames_per_second} -i {outpath}{figname_prefix}%04d{figname_suffix} -f mp4 -vcodec libx264 -pix_fmt yuv420p {outpath}{figname_prefix}animation.mp4')


        return plotted_values

    @staticmethod
    def plot_maps_of_blob_snapshot_with_obs(model_da,extreme_anomalies,obs_da,obs_extreme_anomaly,applied_thresholds,regional_sst_data,regional_applied_thresholds,temp_2m,temp_2m_clim_anomalies,obs_t2m_da,obs_t2m_clim_anomalies,config_of_choice,scenarios,times_of_choice,varia_name,depth,model_regions_dict,savefig=False):
        lon = model_da[config_of_choice]['present'].lon
        lat = model_da[config_of_choice]['present'].lat
        lon_2m = temp_2m_clim_anomalies['present']['present'].lon
        lat_2m = temp_2m_clim_anomalies['present']['present'].lat
        #
        col = ExtAnalysisPlotter.get_color_maps_and_ranges(varia_name,depth)
        #col = get_color_maps_and_ranges(varia_name,depth)
        # Figure design (4 rows, 4 columns)
        # SST | SST' (full)   | SST' (mean warming adjusted)   | SST' (clim. warming adjusted) |
        # Obs               | -                 | -                 | -                |
        # ROMSOC present    | ROMSOC present    | -                 | -                |
        # ROMSOC ssp585     | ROMSOC ssp585     | ROMSOC ssp585     | ROMSOC ssp585    |
        # ROMS-only ssp585  | ROMS-only ssp585  | ROMS-only ssp585  | ROMS-only ssp585 |
        #
        panel_labels = ['a)','b)',' ',' ','c)','d)',' ',' ','e)','f)','g)','h)','i)','j)','k)','l)']
        ylabs = ['Obs','ROMSOC\npresent','ROMSOC\nssp585','ROMS-only\nssp585']
        titles = ['SST & 2mT',"SST' (rel. to pres. thresh.)\n& 2mT' (rel. to pres. clim.)","SST' (rel. to pres. thresh.)\n& 2mT' (rel. to pres. clim.)\nmean warming adjusted","SST' (rel. to pres. thresh.)\n& 2mT' (rel. to pres. clim.)\nclim. warming adjusted"]
        #
        for tdx,time_of_choice in enumerate(times_of_choice):
            #
            fig, ax = plt.subplots(4,4,sharex=True,sharey=True,figsize=(12,15))
            #
            # Obs plot:
            config_of_choice = 'obs'
            scenario = 'present'
            c00 = ax[0,0].pcolormesh(obs_da.lon,obs_da.lat,obs_da,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
            c00_2m = ax[0,0].pcolormesh(obs_t2m_da.longitude,obs_t2m_da.latitude,obs_t2m_da.sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'],zorder=4)     
            c01 = ax[0,1].pcolormesh(obs_da.lon,obs_da.lat,obs_extreme_anomaly,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c01_2m = ax[0,1].pcolormesh(obs_t2m_da.longitude,obs_t2m_da.latitude,obs_t2m_clim_anomalies.sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'],zorder=4)     
            #
            # ROMSOC present plots:
            config_of_choice = 'romsoc_fully_coupled'
            scenario = 'present'
            c10 = ax[1,0].pcolormesh(lon,lat,model_da[config_of_choice][scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
            c10_2m = ax[1,0].pcolormesh(lon_2m,lat_2m,temp_2m[scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'],zorder=4)     
            c11 = ax[1,1].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c11_2m = ax[1,1].pcolormesh(lon_2m,lat_2m,temp_2m_clim_anomalies[scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'],zorder=4)            
            #
            # ROMSOC ssp585 plots:
            config_of_choice = 'romsoc_fully_coupled'
            scenario = 'ssp585'            
            c20 = ax[2,0].pcolormesh(lon,lat,model_da[config_of_choice][scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
            c20_2m = ax[2,0].pcolormesh(lon_2m,lat_2m,temp_2m[scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'],zorder=4)     
            c21 = ax[2,1].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c21_2m = ax[2,1].pcolormesh(lon_2m,lat_2m,temp_2m_clim_anomalies[scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'],zorder=4)    
            c22 = ax[2,2].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_meandelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c22_2m = ax[2,2].pcolormesh(lon_2m,lat_2m,temp_2m_clim_anomalies[scenario]['present_plus_meandelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'],zorder=4)  
            c23 = ax[2,3].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_climdelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c23_2m = ax[2,3].pcolormesh(lon_2m,lat_2m,temp_2m_clim_anomalies[scenario]['present_plus_climdelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'],zorder=4)  
            #
            # ROMS-only ssp585 plots:
            config_of_choice = 'roms_only'
            scenario = 'ssp585'            
            c30 = ax[3,0].pcolormesh(lon,lat,model_da[config_of_choice][scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
            c31 = ax[3,1].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c32 = ax[3,2].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_meandelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c33 = ax[3,3].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_climdelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            #
            # set the plot extent and labels
            for bdx,axi in enumerate(ax.flatten()):
                #axi.set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
                #axi.set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
                axi.set_xlim([230,245])
                axi.set_ylim([30,50])
                #yticks = np.arange(20,70,10) # np.arange(30,60,10)#
                #xticks = np.arange(230,260,10) # np.arange(230,250,5)#
                yticks = np.arange(30,55,5)
                xticks = np.arange(230,245,5)
                axi.set_yticks(yticks)
                axi.set_xticks(xticks)
                axi.text(0.05,0.97,panel_labels[bdx],ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=axi.transAxes)
            for adx,axi in enumerate(ax[:,0]):
                axi.set_yticklabels([str(val)+'°N' for val in yticks])
                axi.set_ylabel(ylabs[adx],fontweight='bold',ha='right',rotation=0)
            for axi in ax[-1,:]:
                axi.set_xticklabels([str(360-val)+'°W' for val in xticks])
            for adx,axi in enumerate(ax[0,:]):
                axi.set_title(titles[adx])
            for adx,axi in enumerate(ax[2,:]):
                if adx >= 2:
                    axi.set_title(titles[adx])
            #
            # add the continent
            landmask_etopo = PlotFuncs.get_etopo_data()
            landmask_etopo2 = xr.where(landmask_etopo==1,landmask_etopo,0)
            for axi in ax.flatten():
                axi.contourf(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo,colors='#555555')
                axi.contour(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo2,[0.5],colors='k',linewidths=1,zorder=5)
            #
            # remove some subplots
            fig.delaxes(ax[0,2])
            fig.delaxes(ax[0,3])
            fig.delaxes(ax[1,2])
            fig.delaxes(ax[1,3])
            #    
            cbax10 = fig.add_axes([0.57,0.94,0.2,0.02])
            cbar10 = plt.colorbar(c10,cax=cbax10,extend='both',orientation='horizontal')
            cbar10.ax.set_title(col['mod']['unit'])
            #
            cbax11 = fig.add_axes([0.79,0.94,0.2,0.02])
            cbar11 = plt.colorbar(c11,cax=cbax11,extend='both',orientation='horizontal')
            cbar11.ax.set_title(col['mod']['unit'])
            #
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05,hspace=0.05,right=0.99)
            #
            fig.text(0.03,0.94,time_of_choice.strftime('%B %d, %Y'),fontweight='bold')
            #
            #
            ## add some time series in upper right corner
            reg_of_choice = 'coastal_central'
            time_cho_start = 730#1500
            time_cho_end = 2555#2000
            time_cho = slice(time_cho_start,time_cho_end)
            time_vector = model_da['roms_only']['present'].time
            #
            # add some axes for time series
            sst_ax = fig.add_axes([0.62,0.76,0.37,0.14])
            #sst_ax.plot(time_vector.isel(time=time_cho),model_da['roms_only']['present'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='purple',linewidth=0.75,label='ROMS-only present')
            sst_ax.plot(time_vector.isel(time=time_cho),regional_sst_data['romsoc_fully_coupled']['ssp585'][reg_of_choice].isel(time=time_cho),color='r',linewidth=1,label='ROMSOC ssp585')            
            sst_ax.plot(time_vector.isel(time=time_cho),regional_sst_data['romsoc_fully_coupled']['present'][reg_of_choice].isel(time=time_cho),color='k',linewidth=1,label='ROMSOC present',linestyle='-')
            #
            sst_ax.plot(time_vector.isel(time=time_cho),regional_applied_thresholds['romsoc_fully_coupled']['present'][reg_of_choice]['present'].isel(time=time_cho),color='k',linewidth=3,alpha=0.5,label='present thresh.',linestyle='-')
            #
            sst_ax.plot(time_vector.isel(time=time_cho),regional_applied_thresholds['romsoc_fully_coupled']['ssp585'][reg_of_choice]['present_plus_climdelta'].isel(time=time_cho),color='r',linewidth=3,alpha=0.5,label='clim adj. thresh.',linestyle='-')
            #
            sst_ax.grid(color='#CCCCCC',linestyle='--')
            #
            maxval = np.max(regional_sst_data['romsoc_fully_coupled']['ssp585'][reg_of_choice].isel(time=time_cho))
            sst_ax.scatter(times_of_choice[0],1.05*maxval,50,marker='v',linestyle='-',color='k',clip_on=False)
            sst_ax.set_xlim(time_vector.isel(time=time_cho)[0],time_vector.isel(time=time_cho)[-1])
            sst_ax.set_ylabel('SST in °C')
            xticks = [pd.to_datetime('2013-01-01'),
                      pd.to_datetime('2014-01-01'),
                      pd.to_datetime('2015-01-01'),
                      pd.to_datetime('2016-01-01'),
                      pd.to_datetime('2017-01-01')]
            sst_ax.set_xticks(xticks)
            sst_ax.set_xticklabels([xtick.strftime("%Y/%m") for xtick in xticks])
            #sst_ax.legend()
            sst_ax.set_ylim([7,1.05*maxval])
            sst_ax.text(0.02,0.95,'m)',ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=sst_ax.transAxes)
            #
            ## add some time series avergaed over the coastal center region
            eta_cho = 495#495
            xi_cho = 210#200
            time_cho_start = 1500
            time_cho_end = 2000
            time_cho = slice(time_cho_start,time_cho_end)
            time_vector = model_da['roms_only']['present'].time
            #
            #add vertical lines to sst_ax panel to indicate zoom in
            sst_ax.fill_between(time_vector.isel(time=time_cho).values,7,maxval*1.05,color='#CCCCCC',alpha=0.2)
            sst_ax.text(0.98,0.01,'area averaged',ha='right',va='bottom',transform=sst_ax.transAxes)
            #
            # add some axes for time series
            sst_ax2 = fig.add_axes([0.62,0.58,0.37,0.14])
            #sst_ax.plot(time_vector.isel(time=time_cho),model_da['roms_only']['present'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='purple',linewidth=0.75,label='ROMS-only present')
            sst_ax2.plot(time_vector.isel(time=time_cho),model_da['romsoc_fully_coupled']['ssp585'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='r',linewidth=1,label='ROMSOC ssp585')            
            sst_ax2.plot(time_vector.isel(time=time_cho),model_da['romsoc_fully_coupled']['present'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='k',linewidth=1,label='ROMSOC present',linestyle='-')
            #
            sst_ax2.plot(time_vector.isel(time=time_cho),applied_thresholds['romsoc_fully_coupled']['present']['present'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='k',linewidth=3,alpha=0.5,label='present thresh.',linestyle='-')
            #
            sst_ax2.plot(time_vector.isel(time=time_cho),applied_thresholds['romsoc_fully_coupled']['ssp585']['present_plus_climdelta'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='r',linewidth=3,alpha=0.5,label='clim adj. thresh.',linestyle='-')
            #
            sst_ax2.grid(color='#CCCCCC',linestyle='--')
            #
            maxval = np.max(model_da['romsoc_fully_coupled']['ssp585'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho))
            sst_ax2.scatter(times_of_choice[0],1.05*maxval,50,marker='v',linestyle='-',color='k',clip_on=False)
            sst_ax2.set_xlim(time_vector.isel(time=time_cho)[0],time_vector.isel(time=time_cho)[-1])
            sst_ax2.set_ylabel('SST in °C')
            xticks = [pd.to_datetime('2015-05-01'),
                      pd.to_datetime('2015-08-01'),
                      pd.to_datetime('2015-11-01'),
                      pd.to_datetime('2016-02-01'),
                      pd.to_datetime('2016-05-01')]
            sst_ax2.set_xticks(xticks)
            sst_ax2.set_xticklabels([xtick.strftime("%Y/%m") for xtick in xticks])
            sst_ax2.legend()
            sst_ax2.set_ylim([7,1.05*maxval])
            sst_ax2.text(0.02,0.95,'n)',ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=sst_ax2.transAxes)
            sst_ax2.text(0.98,0.01,'at location x',ha='right',va='bottom',transform=sst_ax2.transAxes)
            #
            #
            # mark the timeseries location in the maps
            region_mask_dummy = xr.where(np.isnan(model_regions_dict[reg_of_choice]['mask']),0,1)
            for axi in ax.flatten():
                axi.scatter(lon[eta_cho,xi_cho],lat[eta_cho,xi_cho],50,'w',marker='x')
                axi.contour(lon,lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)
            #
            #
            # put the plotted values into a Dataset 
            plotted_values = xr.Dataset() 
            # put varia_dict into this one
            #
            if savefig == True and np.size(times_of_choice) == 1:
                outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/snapshots/'
                figname = f'{varia_name}_{depth}m_model_and_obs.png'
            elif savefig == True and np.size(times_of_choice) > 1:
                outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/snapshots/animation_frames_long/'
                figname_prefix = f'{varia_name}_{depth}m_{config_of_choice}_frame_'
                figname_suffix = '.png'
                figname = figname_prefix+f'{int(tdx+1):04d}'+figname_suffix
            #
            if savefig:
                plt.savefig(outpath+figname,dpi=200,transparent=True)            
            #
            plt.show()
        #
        if savefig == True and np.size(times_of_choice) > 1:
            frames_per_second = 12
            if os.path.isfile(f'{outpath}{figname_prefix}animation.mp4'):
                os.system(f'rm {outpath}{figname_prefix}animation.mp4')
            os.system(f'ffmpeg -r {frames_per_second} -i {outpath}{figname_prefix}%04d{figname_suffix} -f mp4 -vcodec libx264 -pix_fmt yuv420p {outpath}{figname_prefix}animation.mp4')
        #
        return plotted_values

    @staticmethod
    def plot_maps_of_blob_snapshot_with_obs2(model_da,extreme_anomalies,obs_da,obs_extreme_anomaly,applied_thresholds,regional_sst_data,regional_applied_thresholds,temp_2m,temp_2m_clim_anomalies,obs_t2m_da,obs_t2m_clim_anomalies,config_of_choice,scenarios,times_of_choice,varia_name,depth,model_regions_dict,savefig=False):
        lon = model_da[config_of_choice]['present'].lon
        lat = model_da[config_of_choice]['present'].lat
        lon_2m = temp_2m_clim_anomalies['present']['present'].lon
        lat_2m = temp_2m_clim_anomalies['present']['present'].lat
        #
        col = ExtAnalysisPlotter.get_color_maps_and_ranges(varia_name,depth)
        #col = get_color_maps_and_ranges(varia_name,depth)
        # Figure design (4 rows, 4 columns)
        # SST | SST' (full)   | SST' (mean warming adjusted)   
        # Obs               | Obs               | -                 | - |
        # ROMSOC present    | ROMSOC present    | -                 | - |
        # ROMSOC ssp585     | ROMSOC ssp585     | ROMSOC ssp585     | - |
        # ROMS-only ssp585  | ROMS-only ssp585  | ROMS-only ssp585  | - |
        #
        panel_labels = ['a)','b)',' ',' ','c)','d)',' ',' ','e)','f)','g)',' ','h)','i)','j)',' ','k)','l)']
        ylabs = ['Obs','ROMSOC\npresent','ROMSOC\nssp585','ROMS-only\nssp585']
        titles = ['SST & 2mT',"Anomalies (fixed baseline):\nSST' (rel. to thresh.)\n& 2mT' (rel. to clim.)","Anomalies (moving baseline):\nSST' (rel. to thresh.)\n& 2mT' (rel. to clim.)","SST' (rel. to pres. thresh.)\n& 2mT' (rel. to pres. clim.)\nclim. warming adjusted"]
        #
        fontsize=10
        plt.rcParams['font.size'] = fontsize # 10
        for tdx,time_of_choice in enumerate(times_of_choice):
            #
            fig, ax = plt.subplots(4,4,sharex=True,sharey=True,figsize=(12,15))
            #
            # Obs plot:
            config_of_choice = 'obs'
            scenario = 'present'
            c00 = ax[0,0].pcolormesh(obs_da.lon,obs_da.lat,obs_da,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
            c00_2m = ax[0,0].pcolormesh(obs_t2m_da.longitude,obs_t2m_da.latitude,obs_t2m_da.sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'],zorder=4)     
            c01 = ax[0,1].pcolormesh(obs_da.lon,obs_da.lat,obs_extreme_anomaly,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c01_2m = ax[0,1].pcolormesh(obs_t2m_da.longitude,obs_t2m_da.latitude,obs_t2m_clim_anomalies.sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'],zorder=4)     
            #
            # ROMSOC present plots:
            config_of_choice = 'romsoc_fully_coupled'
            scenario = 'present'
            c10 = ax[1,0].pcolormesh(lon,lat,model_da[config_of_choice][scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
            c10_2m = ax[1,0].pcolormesh(lon_2m,lat_2m,temp_2m[scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'],zorder=4)     
            c11 = ax[1,1].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c11_2m = ax[1,1].pcolormesh(lon_2m,lat_2m,temp_2m_clim_anomalies[scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'],zorder=4)            
            #
            # ROMSOC ssp585 plots:
            config_of_choice = 'romsoc_fully_coupled'
            scenario = 'ssp585'            
            c20 = ax[2,0].pcolormesh(lon,lat,model_da[config_of_choice][scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
            c20_2m = ax[2,0].pcolormesh(lon_2m,lat_2m,temp_2m[scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'],zorder=4)     
            c21 = ax[2,1].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c21_2m = ax[2,1].pcolormesh(lon_2m,lat_2m,temp_2m_clim_anomalies[scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'],zorder=4)    
            c22 = ax[2,2].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_meandelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c22_2m = ax[2,2].pcolormesh(lon_2m,lat_2m,temp_2m_clim_anomalies[scenario]['present_plus_meandelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'],zorder=4)  
            #c23 = ax[2,3].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_climdelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            #c23_2m = ax[2,3].pcolormesh(lon_2m,lat_2m,temp_2m_clim_anomalies[scenario]['present_plus_climdelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'],zorder=4)  
            #
            # ROMS-only ssp585 plots:
            config_of_choice = 'roms_only'
            scenario = 'ssp585'            
            c30 = ax[3,0].pcolormesh(lon,lat,model_da[config_of_choice][scenario].sel(time=time_of_choice),vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
            c31 = ax[3,1].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            c32 = ax[3,2].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_meandelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            #c33 = ax[3,3].pcolormesh(lon,lat,extreme_anomalies[config_of_choice][scenario]['present_plus_climdelta'].sel(time=time_of_choice),vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
            #
            # set the plot extent and labels
            for bdx,axi in enumerate(ax.flatten()):
                #axi.set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
                #axi.set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
                axi.set_xlim([230,245])
                axi.set_ylim([30,50])
                #yticks = np.arange(20,70,10) # np.arange(30,60,10)#
                #xticks = np.arange(230,260,10) # np.arange(230,250,5)#
                yticks = np.arange(30,55,5)
                xticks = np.arange(230,245,5)
                axi.set_yticks(yticks)
                axi.set_xticks(xticks)
                axi.text(0.05,0.97,panel_labels[bdx],ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=axi.transAxes)
            for adx,axi in enumerate(ax[:,0]):
                axi.set_yticklabels([str(val)+'°N' for val in yticks])
                axi.set_ylabel(ylabs[adx],fontweight='bold',ha='right',rotation=0)
            for axi in ax[-1,:]:
                axi.set_xticklabels([str(360-val)+'°W' for val in xticks])
            for adx,axi in enumerate(ax[0,:]):
                axi.set_title(titles[adx])
            for adx,axi in enumerate(ax[2,:]):
                if adx >= 2:
                    axi.set_title(titles[adx])
            #
            # add the continent
            landmask_etopo = PlotFuncs.get_etopo_data()
            landmask_etopo2 = xr.where(landmask_etopo==1,landmask_etopo,0)
            for axi in ax.flatten():
                axi.contourf(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo,colors='#555555')
                axi.contour(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo2,[0.5],colors='k',linewidths=1,zorder=5)
            #
            # remove some subplots
            fig.delaxes(ax[0,2])
            fig.delaxes(ax[0,3])
            fig.delaxes(ax[1,2])
            fig.delaxes(ax[1,3])
            fig.delaxes(ax[2,3])
            fig.delaxes(ax[3,3])
            #    
            # cbax10 = fig.add_axes([0.57,0.94,0.2,0.02])
            # cbar10 = plt.colorbar(c10,cax=cbax10,extend='both',orientation='horizontal')
            # cbar10.ax.set_title(col['mod']['unit'])
            # #
            # cbax11 = fig.add_axes([0.79,0.94,0.2,0.02])
            # cbar11 = plt.colorbar(c11,cax=cbax11,extend='both',orientation='horizontal')
            # cbar11.ax.set_title(col['mod']['unit'])
            cbax10 = fig.add_axes([0.8,0.27,0.025,0.2])
            cbar10 = plt.colorbar(c10,cax=cbax10,extend='both')
            cbar10.ax.set_title(col['mod']['unit'],pad=15,fontsize=fontsize+2)
            #
            cbax11 = fig.add_axes([0.89,0.27,0.025,0.2])
            cbar11 = plt.colorbar(c11,cax=cbax11,extend='both')
            cbar11.ax.set_title(col['mod']['unit'],pad=15,fontsize=fontsize+2)
            #
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05,hspace=0.05,right=0.99)
            #
            fig.text(0.03,0.98,time_of_choice.strftime('%B %d, %Y'),fontweight='bold')
            #
            #
            ## add some time series in upper right corner
            reg_of_choice = 'coastal_central'
            time_cho_start = 730#1500
            time_cho_end = 2555#2000
            time_cho = slice(time_cho_start,time_cho_end)
            time_vector = model_da['roms_only']['present'].time
            #
            # add some axes for time series
            sst_ax = fig.add_axes([0.62,0.75,0.33,0.14])
            #sst_ax.plot(time_vector.isel(time=time_cho),model_da['roms_only']['present'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='purple',linewidth=0.75,label='ROMS-only present')       
            sst_ax.plot(time_vector.isel(time=time_cho),regional_sst_data['romsoc_fully_coupled']['present'][reg_of_choice].isel(time=time_cho),color='k',linewidth=1,label='ROMSOC present',linestyle='-')
            sst_ax.plot(time_vector.isel(time=time_cho),regional_sst_data['romsoc_fully_coupled']['ssp585'][reg_of_choice].isel(time=time_cho),color='r',linewidth=1,label='ROMSOC ssp585',zorder=2)     
            #
            sst_ax.plot(time_vector.isel(time=time_cho),regional_applied_thresholds['romsoc_fully_coupled']['present'][reg_of_choice]['present'].isel(time=time_cho),color='k',linewidth=3,alpha=0.5,label='present day threshold',linestyle='-')
            #
            sst_ax.plot(time_vector.isel(time=time_cho),regional_applied_thresholds['romsoc_fully_coupled']['ssp585'][reg_of_choice]['present_plus_meandelta'].isel(time=time_cho),color='r',linewidth=3,alpha=0.5,label='mean warming adj. threshold',linestyle='-',zorder=2)
            #
            sst_ax.grid(color='#CCCCCC',linestyle='--')
            #
            maxval = np.max(regional_sst_data['romsoc_fully_coupled']['ssp585'][reg_of_choice].isel(time=time_cho))
            sst_ax.scatter(times_of_choice[0],1.05*maxval,50,marker='v',linestyle='-',color='k',clip_on=False)
            sst_ax.set_xlim(time_vector.isel(time=time_cho)[0],time_vector.isel(time=time_cho)[-1])
            sst_ax.set_ylabel('SST in °C')
            xticks = [pd.to_datetime('2013-01-01'),
                      pd.to_datetime('2014-01-01'),
                      pd.to_datetime('2015-01-01'),
                      pd.to_datetime('2016-01-01'),
                      pd.to_datetime('2017-01-01')]
            sst_ax.set_xticks(xticks)
            sst_ax.set_xticklabels([xtick.strftime("%Y/%m") for xtick in xticks])
            sst_ax.legend(ncol=2,loc='lower left',bbox_to_anchor=(0,1.18),handlelength=1,handletextpad=0.4,columnspacing=1.5)
            sst_ax.set_ylim([7,1.05*maxval])
            sst_ax.text(0.02,0.95,'h)',ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=sst_ax.transAxes)
            sst_ax.set_title('Central coastal area averaged SST & thresholds',loc='left')
            #
            ## add some time series avergaed over the coastal center region
            eta_cho = 492#495
            xi_cho = 205#210
            time_cho_start = 1500
            time_cho_end = 2000
            time_cho = slice(time_cho_start,time_cho_end)
            time_vector = model_da['roms_only']['present'].time
            #
            #add vertical lines to sst_ax panel to indicate zoom in
            sst_ax.fill_between(time_vector.isel(time=time_cho).values,7,maxval*1.05,color='#CCCCCC',alpha=0.2)
            #sst_ax.text(0.98,0.01,'area averaged',ha='right',va='bottom',transform=sst_ax.transAxes)
            #
            # add some axes for time series
            sst_ax2 = fig.add_axes([0.62,0.56,0.33,0.14])
            #sst_ax.plot(time_vector.isel(time=time_cho),model_da['roms_only']['present'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='purple',linewidth=0.75,label='ROMS-only present')
            sst_ax2.plot(time_vector.isel(time=time_cho),model_da['romsoc_fully_coupled']['ssp585'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='r',linewidth=1,label='ROMSOC ssp585')            
            sst_ax2.plot(time_vector.isel(time=time_cho),model_da['romsoc_fully_coupled']['present'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='k',linewidth=1,label='ROMSOC present',linestyle='-')
            #
            sst_ax2.plot(time_vector.isel(time=time_cho),applied_thresholds['romsoc_fully_coupled']['present']['present'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='k',linewidth=3,alpha=0.5,label='present thresh.',linestyle='-')
            #
            sst_ax2.plot(time_vector.isel(time=time_cho),applied_thresholds['romsoc_fully_coupled']['ssp585']['present_plus_meandelta'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho),color='r',linewidth=3,alpha=0.5,label='mean adj. thresh.',linestyle='-')
            #
            sst_ax2.grid(color='#CCCCCC',linestyle='--')
            #
            maxval = np.max(model_da['romsoc_fully_coupled']['ssp585'].isel(eta_rho=eta_cho,xi_rho=xi_cho,time=time_cho))
            sst_ax2.scatter(times_of_choice[0],1.05*maxval,50,marker='v',linestyle='-',color='k',clip_on=False)
            sst_ax2.set_xlim(time_vector.isel(time=time_cho)[0],time_vector.isel(time=time_cho)[-1])
            sst_ax2.set_ylabel('SST in °C')
            xticks = [pd.to_datetime('2015-05-01'),
                      pd.to_datetime('2015-08-01'),
                      pd.to_datetime('2015-11-01'),
                      pd.to_datetime('2016-02-01'),
                      pd.to_datetime('2016-05-01')]
            sst_ax2.set_xticks(xticks)
            sst_ax2.set_xticklabels([xtick.strftime("%Y/%m") for xtick in xticks])
            #sst_ax2.legend()
            sst_ax2.set_ylim([7,1.05*maxval])
            sst_ax2.text(0.02,0.95,'i)',ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=sst_ax2.transAxes)
            sst_ax2.set_title(r'SST & thresholds at 40°N, 125°W (location $\times$)',loc='left')
            #sst_ax2.text(0.98,0.01,'at location x',ha='right',va='bottom',transform=sst_ax2.transAxes)
            #
            #
            # mark the timeseries location in the maps
            region_mask_dummy = xr.where(np.isnan(model_regions_dict[reg_of_choice]['mask']),0,1)
            for axi in ax.flatten():
                axi.scatter(lon[eta_cho,xi_cho],lat[eta_cho,xi_cho],50,'w',marker='x')
                axi.contour(lon,lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)
            #
            #
            # put the plotted values into a Dataset 
            plotted_values = xr.Dataset() 
            # put varia_dict into this one
            #
            if savefig == True and np.size(times_of_choice) == 1:
                outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/snapshots/'
                figname = f'{varia_name}_{depth}m_model_and_obs_new.png'
            elif savefig == True and np.size(times_of_choice) > 1:
                outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/extreme_analysis/future_changes/snapshots/animation_frames_long/'
                figname_prefix = f'{varia_name}_{depth}m_{config_of_choice}_frame_'
                figname_suffix = '.png'
                figname = figname_prefix+f'{int(tdx+1):04d}'+figname_suffix
            #
            if savefig:
                plt.savefig(outpath+figname,dpi=200,transparent=True)            
            #
            plt.show()
        #
        if savefig == True and np.size(times_of_choice) > 1:
            frames_per_second = 12
            if os.path.isfile(f'{outpath}{figname_prefix}animation.mp4'):
                os.system(f'rm {outpath}{figname_prefix}animation.mp4')
            os.system(f'ffmpeg -r {frames_per_second} -i {outpath}{figname_prefix}%04d{figname_suffix} -f mp4 -vcodec libx264 -pix_fmt yuv420p {outpath}{figname_prefix}animation.mp4')
        #
        return plotted_values

    @staticmethod
    def get_color_maps_and_ranges(varia = 'temp', depth = 0):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: This function sets the color ranges and maps for the different variables and depths.
        """
        #
        col = dict()
        col['mod'] = dict()
        col['mod-mod'] = dict()
        #
        if varia == 'temp' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          10,30,21,'cmo.thermal',plt.get_cmap('cmo.thermal',20) ,'°C' # 'RdYlBu_r'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -10,10,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'°C'
        #
        return col
    
# %%
