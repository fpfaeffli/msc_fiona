"""
author: Eike E Köhn
date: Jun 21, 2024
description: collection of functions used for the plotting of some analyses regarding the model variability
"""

#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
#from plotting_functions_general import PlotFuncs as PlotFuncs
import cmocean as cmo
import pandas as pd
#sys.path.append('/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/')

#%%
# Contains:
#   01. plotting mapped future changes in annual mean -> plot_future_changes_map()
#   02. plotting future changes in climatology time series -> plot_future_changes_climatology_timeseries()
#   03. plotting autocorrelation functions in specified regions -> plot_autocorrelation_function_regionalized()
#   04. compare model future delta with parent model future delta -> plot_delta_comparison_with_parent_model()
#   05. plotting future changes in wind speed and direction -> plot_future_wind_changes_map_quiver()
#   06. plotting other atmospheric future changes -> plot_future_romsoc_atm_changes_map()
#   07. setting colormaps and colorbars -> get_color_maps_and_ranges()

#%%
class VariaAnalysisPlotter():
    """
    author: Eike E. Koehn
    date: June 7, 2024
    description: This class contains a number of functions that enable standardized plotting for the variability analysis across the different model simulations 
    The functions include:
    1. plotting maps of the annual mean field
    """  

    @staticmethod
    def plot_future_changes_map(varia_dict,varia_name,depth,model_regions_dict,regional_data=None,regional_data_plottype='pcolmesh',savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: Plotting maps of a given variable. The variable must be stored in a dict of the sturcture varia_dict[config][scenario]
        The plot structure is as follows: 2 rows, 3 columns. 
        ax[0,0] = ROMSOC present           |  ax[0,1] = ROMSOC ssp245 - present    | ax[0,2] = ROMSOC ssp585 - present
        ax[1,0] = ROMS-only present        |  ax[1,1] = ROMS-only ssp245 - present | ax[1,2] = ROMS-only ssp585 - present


        Argument "regional_data_plottype" can either be 'pcolmesh' or 'lines'
        """

        panel_labels = ['a)','b)','c)','d)','e)','f)']

        # set up the plot
        col = VariaAnalysisPlotter.get_color_maps_and_ranges(varia_name,depth)
        fontsize=12
        plt.rcParams['font.size']=12
        fig, ax = plt.subplots(2,3,figsize=(11,8),sharex=True,sharey=True)

        # Plot ax[0,0], i.e. ROMSOC present
        vdum = varia_dict['romsoc_fully_coupled']['present'] * model_regions_dict['full_map']['mask'] 
        c00 = ax[0,0].pcolormesh(vdum.lon,vdum.lat,vdum,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
        ax[0,0].set_title('present')
        ax[0,0].set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
        ax[0,0].set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
        cbar00 = plt.colorbar(c00,ax=ax[0,0],extend='both')
        cbar00.ax.set_title(col['mod']['unit'],pad=15)

        # Plot ax[0,1], i.e. ROMSOC ssp245 - present
        ax[0,1].set_title('ssp245 - present')
        vdum = varia_dict['romsoc_fully_coupled']['ssp245'] - varia_dict['romsoc_fully_coupled']['present']
        c01 = ax[0,1].pcolormesh(vdum.lon,vdum.lat,vdum,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        cbar01 = plt.colorbar(c01,ax=ax[0,1],extend='both')
        cbar01.ax.set_title(col['mod-mod']['unit'],pad=15)

        # Plot ax[0,2], i.e. ROMSOC minus Obs
        ax[0,2].set_title('ssp585 - present')
        vdum = varia_dict['romsoc_fully_coupled']['ssp585'] - varia_dict['romsoc_fully_coupled']['present']
        c02 = ax[0,2].pcolormesh(vdum.lon,vdum.lat,vdum,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        cbar02 = plt.colorbar(c02,ax=ax[0,2],extend='both')
        cbar02.ax.set_title(col['mod-mod']['unit'],pad=15)


        # Plot ax[1,0], i.e. roms-only present
        vdum = varia_dict['roms_only']['present']* model_regions_dict['full_map']['mask'] 
        c10 = ax[1,0].pcolormesh(vdum.lon,vdum.lat,vdum,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
        ax[1,0].set_title('present')
        ax[1,0].set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
        ax[1,0].set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
        cbar10 = plt.colorbar(c10,ax=ax[1,0],extend='both')
        cbar10.ax.set_title(col['mod']['unit'],pad=15)

        # Plot ax[1,1], i.e. ROMSOC ssp245 - present
        ax[1,1].set_title('ssp245 - present')
        vdum = varia_dict['roms_only']['ssp245'] - varia_dict['roms_only']['present']
        c11 = ax[1,1].pcolormesh(vdum.lon,vdum.lat,vdum,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        cbar11 = plt.colorbar(c11,ax=ax[1,1],extend='both')
        cbar11.ax.set_title(col['mod-mod']['unit'],pad=15)

        # Plot ax[0,2], i.e. ROMSOC minus Obs
        ax[1,2].set_title('ssp585 - present')
        vdum = varia_dict['roms_only']['ssp585'] - varia_dict['roms_only']['present']
        c12 = ax[1,2].pcolormesh(vdum.lon,vdum.lat,vdum,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        cbar12 = plt.colorbar(c12,ax=ax[1,2],extend='both')
        cbar12.ax.set_title(col['mod-mod']['unit'],pad=15)

        # add the continent
        landmask_etopo = PlotFuncs.get_etopo_data()
        for axi in ax.flatten():
            axi.contourf(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo,colors='#555555')

        # set the plot extent and labels
        for adx,axi in enumerate(ax.flatten()):
            axi.set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
            axi.set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
            yticks = np.arange(model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat'],10)+3
            xticks = np.arange(model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon'],10)
            axi.set_yticks(yticks)
            axi.set_yticklabels([str(val)+'°N' for val in yticks])
            axi.set_xticks(xticks)
            axi.set_xticklabels([str(360-val)+'°W' for val in xticks])
            axi.text(0.05,0.97,panel_labels[adx],ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=axi.transAxes)

        ax[0,0].set_ylabel('ROMSOC',fontweight='bold',ha='right',rotation=0)
        ax[1,0].set_ylabel('ROMS',fontweight='bold',ha='right',rotation=0)

        plt.tight_layout()

        if regional_data != None:
            regions_to_plot = ['offshore_all_lats','coastal_all_lats'] # ,'transition_all_lats'
            regions_to_plot_short = [reg.upper()[0] for reg in regions_to_plot]
            regions_to_plot_dummy = [reg.replace('_all_lats','') for reg in regions_to_plot]
            print('adding the monthly data timeseries for the regions')
            print(regions_to_plot)
            for cdx,config in enumerate(['romsoc_fully_coupled','roms_only']):
                for sdx,scenario in enumerate(['present','ssp245','ssp585']):
                    if scenario == 'present':
                        dd = regional_data[config][scenario]
                        to_concat = []
                        for regi in regions_to_plot:
                            to_concat.append(dd[regi].values[:,None])
                        concat_dd = np.concatenate(tuple(to_concat),axis=1)
                        if regional_data_plottype == 'pcolmesh':
                            axx = ax[cdx,sdx].inset_axes([.78, .3, .2, .68])
                            axx.pcolormesh(concat_dd,cmap=col['mod']['cmap_pcmesh'],vmin=col['mod']['minval'],vmax=col['mod']['maxval'],edgecolor='k',linewidth=0.125)
                        elif regional_data_plottype == 'lines':
                            axx = ax[cdx,sdx].inset_axes([.58, .75, .4, .22])
                            lineObjs = axx.plot(np.arange(1,13),concat_dd,'.-')
                            if cdx == 0 and sdx == 2:
                                axx.legend(iter(lineObjs),tuple(regions_to_plot_dummy),loc='lower left',bbox_to_anchor=(0,-1.4),fontsize=fontsize-4,handletextpad = 0.4, handlelength=2)    
                    elif scenario != 'present':
                        dd_sce = regional_data[config][scenario]
                        dd_ref = regional_data[config]['present']
                        to_concat = []
                        for regi in regions_to_plot:
                            dd_anom = dd_sce[regi] - dd_ref[regi]
                            to_concat.append(dd_anom.values[:,None])
                        concat_dd = np.concatenate(tuple(to_concat),axis=1)
                        if regional_data_plottype == 'pcolmesh':
                            axx = ax[cdx,sdx].inset_axes([.78, .3, .2, .68])
                            axx.pcolormesh(concat_dd,cmap=col['mod-mod']['cmap_pcmesh'],vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],edgecolor='k',linewidth=0.125)
                        elif regional_data_plottype == 'lines':
                            axx = ax[cdx,sdx].inset_axes([.58, .75, .4, .22])
                            lineObjs = axx.plot(np.arange(1,13),concat_dd,'.-')
                            if cdx == 0 and sdx == 2:
                                axx.legend(iter(lineObjs),tuple(regions_to_plot_dummy),loc='lower left',bbox_to_anchor=(-0.1,-1.4),fontsize=fontsize-4,handletextpad = 0.4, handlelength=2)    
                    if regional_data_plottype == 'pcolmesh':
                        axx.set_yticks(np.arange(0.5,12.5))
                        axx.set_yticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],color='w',fontsize=plt.rcParams['font.size']-1)
                        axx.set_xticks(np.arange(len(regions_to_plot))+0.5)
                        axx.set_xticklabels(regions_to_plot_short,color='w',fontsize=plt.rcParams['font.size']-1)
                        axx.invert_yaxis()
                    elif regional_data_plottype == 'lines':
                        axx.set_xticks(np.arange(1,13,2))
                        axx.set_xticklabels(['J','M','M','J','S','N'],color='w',fontsize=plt.rcParams['font.size']-1)
                        #yticklabs = ax_insets[rdx,cdx].yaxis.label.set_color('w')
                        axx.tick_params(axis='y', colors='w')
                        #ax_insets[rdx,cdx].set_yticklabels(yticklabs,color='w',fontsize=plt.rcParams['font.size']-1)
                        axx.grid(color='#EEEEEE',linewidth=0.5)
                        axx.set_ylabel(col['mod']['unit'],color='w')    
                        axx.set_xlabel('Month',color='w')                           

        if regional_data == None:
            # add the mask regions in each panel (first the Obs. mask in ax[0,0], then the Model mask for all others)
            for axi in ax.flatten():
                for region in model_regions_dict.keys():
                    if region != 'full_map' and 'all_lats' not in region:
                        region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                        axi.contour(vdum.lon,vdum.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)
        elif regional_data != None:
            # add the mask regions in each panel (first the Obs. mask in ax[0,0], then the Model mask for all others)
            for axi in ax.flatten():
                for region in regions_to_plot:
                    region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                    axi.contour(vdum.lon,vdum.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)            

        # put the plotted values into a Dataset 
        plotted_values = xr.Dataset() 
        # put varia_dict into this one

        if savefig == True:
            outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/variability_analysis/future_changes/maps/'
            figname = f'{varia_name}_{depth}m.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values
    

    @staticmethod
    def plot_future_changes_climatology_timeseries(regional_data,varia_name,depth,model_regions_dict,num_regions=3,savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: Plotting maps of a given variable. The variable must be stored in a dict of the sturcture varia_dict[config][scenario]
        The plot structure is as follows: 2 rows, 3 columns. 
        ax[0,0] = ROMSOC present           |  ax[0,1] = ROMSOC ssp245 - present    | ax[0,2] = ROMSOC ssp585 - present
        ax[1,0] = ROMS-only present        |  ax[1,1] = ROMS-only ssp245 - present | ax[1,2] = ROMS-only ssp585 - present


        Argument "regional_data_plottype" can either be 'pcolmesh' or 'lines'
        """

        # set up the plot
        configs = ['romsoc_fully_coupled','roms_only']
        scenarios = ['present','ssp245','ssp585']
        col = VariaAnalysisPlotter.get_color_maps_and_ranges(varia_name,depth)
        fig, ax = plt.subplots(2,3,figsize=(10,6),sharex=True)
        if num_regions == 3:
            regions_to_plot = ['all_dists_all_lats','offshore_all_lats','coastal_all_lats'] # ,'transition_all_lats'
        elif num_regions == 9:
            regions_to_plot = ['all_dists_northern','offshore_northern','coastal_northern', # ,'transition_northern'
                            'all_dists_central','offshore_central','coastal_central', # ,'transition_central'
                            'all_dists_southern','offshore_southern','coastal_southern']       #     ,'transition_southern'
        elif num_regions == 12:
            regions_to_plot = ['all_dists_all_lats','offshore_all_lats','coastal_all_lats',
                               'all_dists_northern','offshore_northern','coastal_northern', # ,'transition_northern'
                               'all_dists_central','offshore_central','coastal_central', # ,'transition_central'
                               'all_dists_southern','offshore_southern','coastal_southern']       #     ,'transition_southern'
                            #['offshore_all_lats','transition_all_lats','coastal_all_lats',
                            #'offshore_northern','transition_northern','coastal_northern',
                            #'offshore_central','transition_central','coastal_central',
                            #'offshore_southern','transition_southern','coastal_southern']
        # elif num_regions == 'northern':
        #     regions_to_plot = ['offshore_northern','transition_northern','coastal_northern']
        # elif num_regions == 'central':
        #     regions_to_plot = ['offshore_central','transition_central','coastal_central']
        # elif num_regions == 'southern':
        #     regions_to_plot = ['offshore_southern','transition_southern','coastal_southern']

        print('adding the monthly data timeseries for the regions')
        print(regions_to_plot)
        minpresent = 10**6
        maxpresent = -10**6
        minfuture = 0
        maxfuture = 0
        for cdx,config in enumerate(configs):
            for sdx,scenario in enumerate(scenarios):
                for region in regions_to_plot:
                    if scenario == 'present':
                        ax[cdx,sdx].plot(np.arange(1,13),regional_data[config][scenario][region],'.-',color=model_regions_dict[region]['id_colour'])
                        minpresent = np.min((minpresent,np.min(regional_data[config][scenario][region])))
                        maxpresent = np.max((maxpresent,np.max(regional_data[config][scenario][region])))
                    elif scenario != 'present':
                        anom = regional_data[config][scenario][region]-regional_data[config]['present'][region]
                        ax[cdx,sdx].plot(np.arange(1,13),anom,'.-',color=model_regions_dict[region]['id_colour'],label=region)
                        ax[cdx,sdx].axhline(0,color='k')
                        minfuture = np.min((minfuture,np.min(anom)))
                        maxfuture = np.max((maxfuture,np.max(anom)))

        ax[-1,-1].set_xticks(np.arange(1,13))
        ax[-1,-1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],color='k',fontsize=plt.rcParams['font.size']-1)
        for axi in ax.flatten():    
            axi.grid(color='#888888',alpha=0.2,linewidth=0.5)                        
        # put the plotted values into a Dataset 
        plotted_values = xr.Dataset() 
        # put regional_data into this one

        ax[0,0].set_ylabel('ROMSOC',fontweight='bold',ha='right',rotation=0)
        ax[1,0].set_ylabel('ROMS',fontweight='bold',ha='right',rotation=0)
        ax[0,0].set_title('present')
        ax[0,1].set_title('ssp245-present')
        ax[0,2].set_title('ssp585-present')

        ax[-1,-1].legend(loc='lower left',bbox_to_anchor=(1,0))

        plt.tight_layout()
        fig.text(0.07,0.95,col['mod']['unit'])
        for i in range(2):
            ax[i,0].set_ylim([minpresent,maxpresent])
            ax[i,1].set_ylim([minfuture,maxfuture])
            ax[i,2].set_ylim([minfuture,maxfuture])

        if savefig == True:
            outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/variability_analysis/future_changes/regional_means/climatologies/'
            figname = f'{varia_name}_{depth}m_{num_regions}_regions.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values
    
    
    @staticmethod
    def plot_autocorrelation_function_regionalized(varia_dict,varia_name,dep,region_names,model_regions_dict,configs,scenarios,savefig=False):
        fontsize=12
        numrows = 4
        numcols = 3
        plt.rcParams['font.size']=fontsize
        fig, ax = plt.subplots(numrows,numcols,figsize=(10,10),sharey=True,sharex=True)
        print('-------------')
        print('working on {}'.format(varia_name))
        fig.suptitle(f"{varia_name} ({dep}m)",ha='center',fontweight='bold',va='center')
        for rdx in range(len(region_names)):
            for cdx in range(len(region_names[rdx])):
                region = region_names[rdx][cdx]
                ax[rdx,cdx].set_title(region,fontweight='bold',color=model_regions_dict[region]['id_colour'])
                for config in configs:
                    if config == 'romsoc_fully_coupled':
                        linestyle = '-'
                    elif config == 'roms_only':
                        linestyle = ':'
                    for scenario in scenarios:
                        if scenario == 'present':
                            lc = 'k'
                        elif scenario == 'ssp245':
                            lc = 'y'
                        elif scenario == 'ssp585':
                            lc = 'r'
                        data = varia_dict[config][scenario][region]
                        lags = np.arange(np.size(data.lag))
                        if rdx == 0 and cdx == numcols-1 and scenario == 'present':
                            label = config.replace('romsoc_fully_coupled','ROMSOC').replace('roms_only','ROMS-only')
                            ax[rdx,cdx].plot(lags,data,color=lc,linestyle=linestyle,label=label)
                        if rdx == 0 and cdx == numcols-1 and config == 'romsoc_fully_coupled':
                            label = scenario
                            ax[rdx,cdx].plot(lags,data,color=lc,linestyle=linestyle,label=label)
                        else:
                            ax[rdx,cdx].plot(lags,data,color=lc,linestyle=linestyle)
                ax[rdx,cdx].axvline(0,color='C0')
                ax[rdx,cdx].spines['right'].set_visible(False)
                ax[rdx,cdx].spines['top'].set_visible(False)
                ax[rdx,cdx].spines['left'].set_visible(False)
                ax[rdx,cdx].spines['bottom'].set_visible(False)
                ax[rdx,cdx].grid(axis='both',linestyle='--',alpha=0.25)
                ax[rdx,cdx].set_xlim([0,300]) # np.max(lags)])
                ax[rdx,cdx].axhline(1/np.exp(1),color='C0',linewidth=1)
            ax[rdx,0].set_ylabel('Autocorrelation')
        ax[-1,1].set_xlabel('Lag in days')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        ax[0,-1].legend(fontsize=fontsize-2,ncols=1)
        plotted_values = xr.Dataset()
        if savefig == True:
            outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/variability_analysis/future_changes/regional_means/'
            figname = f'{varia_name}_{dep}m.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)
        plt.show()
        return plotted_values
    
    @staticmethod
    def plot_delta_comparison_with_parent_model(var,dep,recalc_regional_data,parent_regional_data,savefig=False):
        fontsize=12
        plt.rcParams['font.size']=fontsize
        col = VariaAnalysisPlotter.get_color_maps_and_ranges(var,dep)
        fig,ax = plt.subplots(2,2,figsize=(8,5),sharex=True,sharey=True)
        for sdx,scenario in enumerate(['ssp245','ssp585']):
            for cdx,config in enumerate(['roms_only','romsoc_fully_coupled']):
                for rdx,region in enumerate(['offshore_all_lats','coastal_all_lats']):
                        modelled_difference = (recalc_regional_data[config][scenario][region]-recalc_regional_data[config]['present'][region]).values
                        parent_difference = (parent_regional_data[scenario][region]).values
                        ax[sdx,cdx].plot(np.arange(1,13),modelled_difference-parent_difference,'.-',label=region.replace('_all_lats',''))
                        ax[sdx,cdx].set_title('{}: {}'.format(scenario,config.replace('_fully_coupled','').replace('_only',''))+r'$\Delta$ - parent$\Delta$')
                        ax[sdx,cdx].axhline(0,color='k',linestyle='-')
                        ax[sdx,cdx].set_xticks(np.arange(1,13,1))
                        ax[sdx,cdx].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
                        ax[sdx,cdx].grid('on',linestyle='--')
                        ax[sdx,cdx].spines[['right', 'top','bottom','left']].set_visible(False)
            ax[sdx,0].set_ylabel(col['mod']['unit'])
        ax[-1,0].set_xlabel('Month')
        ax[-1,-1].set_xlabel('Month')
        ax[-1,-1].legend(loc='lower left',bbox_to_anchor=(1,0))
        plt.tight_layout()
        if savefig == True:
            outdir = '/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/variability_analysis/future_changes/regional_means/parent_model_comparison/'
            filename = f'{outdir}{var}_{dep}m.png'
            plt.savefig(filename,dpi=200,transparent=True)
        plt.show()
        plotted_values = xr.Dataset()
        return plotted_values

    @staticmethod
    def plot_future_wind_changes_map_quiver(absstress,ustress,vstress,rmask,varia_name,depth,model_regions_dict,regional_data=None,regional_data_plottype='pcolmesh',savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: Plotting maps of a given variable. The variable must be stored in a dict of the sturcture varia_dict[config][scenario]
        The plot structure is as follows: 1 rows, 3 columns. 
        ax[0] = ROMSOC present           |  ax[1] = ROMSOC ssp245 - present    | ax[2] = ROMSOC ssp585 - present


        Argument "regional_data_plottype" can either be 'pcolmesh' or 'lines'
        """

        panel_labels = ['a)','b)','c)','d)','e)','f)']

        # set up the plot
        col = VariaAnalysisPlotter.get_color_maps_and_ranges(varia_name,depth)
        fontsize=12
        plt.rcParams['font.size']=12
        fig, ax = plt.subplots(1,3,figsize=(11,4),sharex=True,sharey=True)
        res = 35
        headlength = 4
        headwidth = 5
        width = 0.0075

        # Plot ax[0], i.e. ROMSOC present
        udum = ustress['romsoc_fully_coupled']['present'].where(rmask!=1)
        vdum = vstress['romsoc_fully_coupled']['present'].where(rmask!=1)
        absdum = absstress['romsoc_fully_coupled']['present'].where(rmask!=1)
        scale = 1
        c0 = ax[0].pcolormesh(udum.lon,udum.lat,absdum,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
        c00 = ax[0].quiver(udum.lon[::res,::res],udum.lat[::res,::res],udum[::res,::res],vdum[::res,::res],color='k',scale=scale,width=width,zorder=2,headlength=headlength,headwidth=headwidth)
        ax[0].set_title('present')
        ax[0].set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
        ax[0].set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
        cbar0 = plt.colorbar(c0,ax=ax[0],extend='both')
        ax[0].quiverkey(c00, X=0.7, Y=0.34, U=0.1, label=f"0.1\n{col['mod']['unit'].replace('        ','')}", labelpos='E', coordinates='axes',labelsep=0.01)
        cbar0.ax.set_title(col['mod']['unit'],pad=15)

        # Plot ax[1], i.e. ROMSOC ssp245 - present
        udum = ustress['romsoc_fully_coupled']['ssp245'].where(rmask!=1)-ustress['romsoc_fully_coupled']['present'].where(rmask!=1)
        vdum = vstress['romsoc_fully_coupled']['ssp245'].where(rmask!=1)-vstress['romsoc_fully_coupled']['present'].where(rmask!=1)
        absdum = absstress['romsoc_fully_coupled']['ssp245'].where(rmask!=1)-absstress['romsoc_fully_coupled']['present'].where(rmask!=1)
        ax[1].set_title('ssp245 - present')
        c1 = ax[1].pcolormesh(udum.lon,udum.lat,absdum,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        scale = .04
        c11 = ax[1].quiver(udum.lon[::res,::res],udum.lat[::res,::res],udum[::res,::res],vdum[::res,::res],color='k',scale=scale,width=width,zorder=2,headlength=headlength,headwidth=headwidth)
        cbar1 = plt.colorbar(c1,ax=ax[1],extend='both')
        ax[1].quiverkey(c11, X=0.7, Y=0.34, U=0.005, label=f"0.005\n{col['mod-mod']['unit'].replace('        ','')}", labelpos='E', coordinates='axes',labelsep=0.01)
        cbar1.ax.set_title(col['mod-mod']['unit'],pad=15)

        # Plot ax[2], i.e. ROMSOC ssp585 - present
        udum = ustress['romsoc_fully_coupled']['ssp585'].where(rmask!=1)-ustress['romsoc_fully_coupled']['present'].where(rmask!=1)
        vdum = vstress['romsoc_fully_coupled']['ssp585'].where(rmask!=1)-vstress['romsoc_fully_coupled']['present'].where(rmask!=1)
        absdum = absstress['romsoc_fully_coupled']['ssp585'].where(rmask!=1)-absstress['romsoc_fully_coupled']['present'].where(rmask!=1)
        ax[2].set_title('ssp585 - present')
        c2 = ax[2].pcolormesh(udum.lon,udum.lat,absdum,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        scale = .04
        c22 = ax[2].quiver(udum.lon[::res,::res],udum.lat[::res,::res],udum[::res,::res],vdum[::res,::res],color='k',scale=scale,width=width,zorder=2,headlength=headlength,headwidth=headwidth)
        cbar2 = plt.colorbar(c2,ax=ax[2],extend='both')
        ax[2].quiverkey(c22, X=0.7, Y=0.34, U=0.005, label=f"0.005\n{col['mod-mod']['unit'].replace('        ','')}", labelpos='E', coordinates='axes',labelsep=0.01)
        cbar2.ax.set_title(col['mod-mod']['unit'],pad=15)

        # add the continent
        landmask_etopo = PlotFuncs.get_etopo_data()
        for axi in ax:
            axi.contourf(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo,colors='#555555')

        # set the plot extent and labels
        for adx,axi in enumerate(ax):
            axi.set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
            axi.set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
            yticks = np.arange(30,60,10)#np.arange(20,70,10)
            xticks = np.arange(230,260,10)
            axi.set_yticks(yticks)
            axi.set_yticklabels([str(val)+'°N' for val in yticks])
            axi.set_xticks(xticks)
            axi.set_xticklabels([str(360-val)+'°W' for val in xticks])
            axi.text(0.05,0.97,panel_labels[adx],ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=axi.transAxes)

        ax[0].set_ylabel('ROMSOC',fontweight='bold',ha='right',rotation=0)

        plt.tight_layout()

        if regional_data != None:

            reg_u,reg_v = tuple(regional_data)
            print(reg_u,reg_v)

            regions_to_plot = ['offshore_all_lats','coastal_all_lats'] # ,'transition_all_lats'
            cols = ['C0','C1']
            regions_to_plot_short = [reg.upper()[0] for reg in regions_to_plot]
            regions_to_plot_dummy = [reg.replace('_all_lats','') for reg in regions_to_plot]
            print('adding the monthly data timeseries for the regions')
            print(regions_to_plot)
            for cdx,config in enumerate(['romsoc_fully_coupled']):
                for sdx,scenario in enumerate(['present','ssp245','ssp585']):
                        if scenario == 'present':
                            dd_u = reg_u[config][scenario]
                            dd_v = reg_v[config][scenario]
                            to_concat_u = []
                            to_concat_v = []
                            for regi in regions_to_plot:
                                to_concat_u.append(dd_u[regi].values[:,None])
                                to_concat_v.append(dd_v[regi].values[:,None])
                            concat_dd_u = np.concatenate(tuple(to_concat_u),axis=1)
                            concat_dd_v = np.concatenate(tuple(to_concat_v),axis=1)
                            if regional_data_plottype == 'pcolmesh':
                                axx = ax[sdx].inset_axes([.78, .3, .2, .68])
                                axx.pcolormesh(concat_dd,cmap=col['mod']['cmap_pcmesh'],vmin=col['mod']['minval'],vmax=col['mod']['maxval'],edgecolor='k',linewidth=0.125)
                            elif regional_data_plottype == 'lines':
                                axx = ax[sdx].inset_axes([.58, .75, .4, .22])
                                lineObjs = axx.plot(np.arange(1,13),concat_dd,'.-')
                                if cdx == 0 and sdx == 2:
                                    axx.legend(iter(lineObjs),tuple(regions_to_plot_dummy),loc='lower left',bbox_to_anchor=(0,-1.4),fontsize=fontsize-4,handletextpad = 0.4, handlelength=2)    
                            elif regional_data_plottype == 'quiver':
                                axx = ax[sdx].inset_axes([.26, .885, .73, .11])
                                yposis = [-0.95,-1.25]
                                qlabs = [' ',f"0.1\n{col['mod']['unit'].replace('        ','')}"]
                                for jk in range(np.shape(concat_dd_u)[1]):
                                    scale = .5
                                    lineObjs = axx.quiver(np.arange(1,13),np.zeros(12),concat_dd_u[:,jk],concat_dd_v[:,jk],color=cols[jk],scale=scale,width=width*1.5,zorder=2,headlength=headlength,headwidth=headwidth,label=regions_to_plot[jk].replace('_all_lats',''))
                                    axx.quiverkey(lineObjs,0.82,yposis[jk],0.1,qlabs[jk],labelpos='S',labelcolor=cols[jk])
                                axx.legend(fontsize=fontsize-4,bbox_to_anchor=(0.42,-4.2),loc='lower left',handletextpad=0.4,handlelength=1)
                                axx.set_yticklabels([])
                                axx.set_ylim([-0.125,0.025])
                                #axx.quiverkey(lineObjs,0.7,-2,0.1,f"0.1{col['mod']['unit'].replace('        ','')}\n{regions_to_plot_dummy[jk]}",labelpos='S',labelcolor=cols[jk])
                                # if cdx == 0 and sdx == 2:
                                #      axx.legend(iter(lineObjs),tuple(regions_to_plot_dummy),loc='lower left',bbox_to_anchor=(0,-1.4),fontsize=fontsize-4,handletextpad = 0.4, handlelength=2)                              
                        elif scenario != 'present':
                            dd_sce_u = reg_u[config][scenario]
                            dd_ref_u = reg_u[config]['present']
                            dd_sce_v = reg_v[config][scenario]
                            dd_ref_v = reg_v[config]['present']
                            to_concat_u = []
                            to_concat_v = []
                            for regi in regions_to_plot:
                                dd_anom_u = dd_sce_u[regi] - dd_ref_u[regi]
                                dd_anom_v = dd_sce_v[regi] - dd_ref_v[regi]
                                to_concat_u.append(dd_anom_u.values[:,None])
                                to_concat_v.append(dd_anom_v.values[:,None])
                            concat_dd_u = np.concatenate(tuple(to_concat_u),axis=1)
                            concat_dd_v = np.concatenate(tuple(to_concat_v),axis=1)
                            if regional_data_plottype == 'pcolmesh':
                                axx = ax[sdx].inset_axes([.78, .3, .2, .68])
                                axx.pcolormesh(concat_dd,cmap=col['mod-mod']['cmap_pcmesh'],vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],edgecolor='k',linewidth=0.125)
                            elif regional_data_plottype == 'lines':
                                axx = ax[sdx].inset_axes([.58, .75, .4, .22])
                                lineObjs = axx.plot(np.arange(1,13),concat_dd,'.-')
                                if cdx == 0 and sdx == 2:
                                    axx.legend(iter(lineObjs),tuple(regions_to_plot_dummy),loc='lower left',bbox_to_anchor=(-0.1,-1.4),fontsize=fontsize-4,handletextpad = 0.4, handlelength=2) 
                            elif regional_data_plottype == 'quiver':
                                axx = ax[sdx].inset_axes([.26, .885, .73, .11])
                                yposis = [-0.95,-1.25]
                                qlabs = ['',f"0.02\n{col['mod']['unit'].replace('        ','')}"]
                                for jk in range(np.shape(concat_dd_u)[1]):
                                    scale = 0.1
                                    lineObjs = axx.quiver(np.arange(1,13),np.zeros(12),concat_dd_u[:,jk],concat_dd_v[:,jk],color=cols[jk],scale=scale,width=width*1.5,zorder=2,headlength=headlength,headwidth=headwidth,label=regions_to_plot[jk].replace('_all_lats',''))
                                    axx.quiverkey(lineObjs,0.82,yposis[jk],0.02,qlabs[jk],labelpos='S',labelcolor=cols[jk])
                                    #axx.legend(fontsize=fontsize-4,bbox_to_anchor=(0.42,-4.2),loc='lower left',handletextpad=0.4,handlelength=1)

                                axx.set_yticklabels([])
                                axx.set_ylim([-0.01,0.01])
                        if regional_data_plottype == 'pcolmesh':
                            axx.set_yticks(np.arange(0.5,12.5))
                            axx.set_yticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],color='w',fontsize=plt.rcParams['font.size']-1)
                            axx.set_xticks(np.arange(len(regions_to_plot))+0.5)
                            axx.set_xticklabels(regions_to_plot_short,color='w',fontsize=plt.rcParams['font.size']-1)
                            axx.invert_yaxis()
                        elif regional_data_plottype == 'lines':
                            axx.set_xticks(np.arange(1,13,2))
                            axx.set_xticklabels(['J','M','M','J','S','N'],color='w',fontsize=plt.rcParams['font.size']-1)
                            axx.tick_params(axis='y', colors='w')
                            axx.grid(color='#EEEEEE',linewidth=0.5)
                            axx.set_ylabel(col['mod']['unit'].replace('        ',''),color='w')    
                            axx.set_xlabel('Month',color='w')
                        elif regional_data_plottype == 'quiver':
                            axx.set_xlim([0.5,13])
                            axx.set_xticks(np.arange(1,13,1))
                            axx.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],color='w',fontsize=plt.rcParams['font.size']-1)
                            axx.grid(color='#EEEEEE',linewidth=0.5)
                            axx.set_xlabel('Month',color='w')

        if regional_data == None:
            # add the mask regions in each panel (first the Obs. mask in ax[0,0], then the Model mask for all others)
            for axi in ax.flatten():
                for region in model_regions_dict.keys():
                        if region != 'full_map' and 'all_lats' not in region:
                            region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                            axi.contour(vdum.lon,vdum.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=1,linewidths=0.5)
        elif regional_data != None:
            # add the mask regions in each panel (first the Obs. mask in ax[0,0], then the Model mask for all others)
            for axi in ax.flatten():
                for region in regions_to_plot:
                        region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                        axi.contour(vdum.lon,vdum.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=1,linewidths=0.5)            

        # put the plotted values into a Dataset 
        plotted_values = xr.Dataset() 
        # put varia_dict into this one

        if savefig == True:
            outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/variability_analysis/future_changes/maps/'
            figname = f'{varia_name}_{depth}m.png'
            plt.savefig(outpath+figname,dpi=200,transparent=True)

        return plotted_values

    @staticmethod
    def plot_future_romsoc_atm_changes_map(varia_dict,varia_name,depth,model_regions_dict,regional_data=None,regional_data_plottype='pcolmesh',savefig=False):
        """
        author: Eike E. Köhn
        date: June 10, 2024
        description: Plotting maps of a given variable. The variable must be stored in a dict of the sturcture varia_dict[config][scenario]
        The plot structure is as follows: 1 rows, 3 columns. 
        ax[0] = ROMSOC present           |  ax[1] = ROMSOC ssp245 - present    | ax[2] = ROMSOC ssp585 - present


        Argument "regional_data_plottype" can either be 'pcolmesh' or 'lines'
        """

        panel_labels = ['d)','e)','f)']

        # set up the plot
        col = VariaAnalysisPlotter.get_color_maps_and_ranges(varia_name,depth)
        fontsize=12
        plt.rcParams['font.size']=12
        fig, ax = plt.subplots(1,3,figsize=(10.7,4),sharex=True,sharey=True) # (10,4) # # (11,4)

        # Plot ax[0], i.e. ROMSOC present
        vdum = varia_dict['romsoc_fully_coupled']['present']
        c0 = ax[0].pcolormesh(vdum.lon,vdum.lat,vdum,vmin=col['mod']['minval'],vmax=col['mod']['maxval'],cmap=col['mod']['cmap_pcmesh'])
        ax[0].set_title('present')
        ax[0].set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
        ax[0].set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
        cbar0 = plt.colorbar(c0,ax=ax[0],extend='both')
        cbar0.ax.set_title(col['mod']['unit'],pad=15)
        if varia_name == 'windstress_curl':
            cbar0.formatter.set_useMathText(True)
            cbar0.ax.yaxis.set_offset_position('left')


        # Plot ax[1], i.e. ROMSOC ssp245 - present
        ax[1].set_title('ssp245 - present')
        vdum = varia_dict['romsoc_fully_coupled']['ssp245'] - varia_dict['romsoc_fully_coupled']['present']
        c1 = ax[1].pcolormesh(vdum.lon,vdum.lat,vdum,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        cbar1 = plt.colorbar(c1,ax=ax[1],extend='both')
        cbar1.ax.set_title(col['mod-mod']['unit'],pad=15)
        if varia_name == 'windstress_curl':
            cbar1.formatter.set_useMathText(True)
            cbar1.ax.yaxis.set_offset_position('left')

        # Plot ax[2], i.e. ROMSOC ssp585 - present
        ax[2].set_title('ssp585 - present')
        vdum = varia_dict['romsoc_fully_coupled']['ssp585'] - varia_dict['romsoc_fully_coupled']['present']
        c2 = ax[2].pcolormesh(vdum.lon,vdum.lat,vdum,vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],cmap=col['mod-mod']['cmap_pcmesh'])
        cbar2 = plt.colorbar(c2,ax=ax[2],extend='both')
        cbar2.ax.set_title(col['mod-mod']['unit'],pad=15)
        if varia_name == 'windstress_curl':
            cbar2.formatter.set_useMathText(True)
            cbar2.ax.yaxis.set_offset_position('left')

        # add the continent
        landmask_etopo = PlotFuncs.get_etopo_data()
        for axi in ax:
            axi.contourf(landmask_etopo.lon,landmask_etopo.lat,landmask_etopo,colors='#555555')

        # set the plot extent and labels
        for adx,axi in enumerate(ax):
            axi.set_xlim([model_regions_dict['full_map']['minlon'],model_regions_dict['full_map']['maxlon']])
            axi.set_ylim([model_regions_dict['full_map']['minlat'],model_regions_dict['full_map']['maxlat']])
            yticks = np.arange(30,60,10)
            xticks = np.arange(230,260,10)
            axi.set_yticks(yticks)
            axi.set_yticklabels([str(val)+'°N' for val in yticks])
            axi.set_xticks(xticks)
            axi.set_xticklabels([str(360-val)+'°W' for val in xticks])
            axi.text(0.05,0.97,panel_labels[adx],ha='left',va='top',bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),transform=axi.transAxes)

        ax[0].set_ylabel('ROMSOC',fontweight='bold',ha='right',rotation=0)

        plt.tight_layout()

        if regional_data != None:
            regions_to_plot = ['offshore_all_lats','coastal_all_lats'] #  ['coastal_northern','coastal_central','coastal_southern']#['offshore_all_lats','coastal_all_lats'] # ,'transition_all_lats'
            regions_to_plot_short = [reg.upper()[0] for reg in regions_to_plot]
            regions_to_plot_dummy = [reg.replace('_all_lats','') for reg in regions_to_plot]
            print('adding the monthly data timeseries for the regions')
            print(regions_to_plot)
            for cdx,config in enumerate(['romsoc_fully_coupled']):
                for sdx,scenario in enumerate(['present','ssp245','ssp585']):
                    if scenario == 'present':
                        dd = regional_data[config][scenario]
                        to_concat = []
                        for regi in regions_to_plot:
                            to_concat.append(dd[regi].values[:,None])
                        concat_dd = np.concatenate(tuple(to_concat),axis=1)
                        if regional_data_plottype == 'pcolmesh':
                            axx = ax[sdx].inset_axes([.78, .3, .2, .68])
                            axx.pcolormesh(concat_dd,cmap=col['mod']['cmap_pcmesh'],vmin=col['mod']['minval'],vmax=col['mod']['maxval'],edgecolor='k',linewidth=0.125)
                        elif regional_data_plottype == 'lines':
                            if 'windstress' in varia_name:
                                axx = ax[sdx].inset_axes([.58, .7, .4, .22])
                            elif 'clouds' or 'pressure' in varia_name:
                                axx = ax[sdx].inset_axes([.58, .76, .4, .22])
                            lineObjs = axx.plot(np.arange(1,13),concat_dd,'.-')
                            if cdx == 0 and sdx == 2:
                                axx.legend(iter(lineObjs),tuple(regions_to_plot_dummy),loc='lower left',bbox_to_anchor=(0,-1.4),fontsize=fontsize-4,handletextpad = 0.4, handlelength=2)    
                    elif scenario != 'present':
                        dd_sce = regional_data[config][scenario]
                        dd_ref = regional_data[config]['present']
                        to_concat = []
                        for regi in regions_to_plot:
                            dd_anom = dd_sce[regi] - dd_ref[regi]
                            to_concat.append(dd_anom.values[:,None])
                        concat_dd = np.concatenate(tuple(to_concat),axis=1)
                        if regional_data_plottype == 'pcolmesh':
                            axx = ax[sdx].inset_axes([.78, .3, .2, .68])
                            axx.pcolormesh(concat_dd,cmap=col['mod-mod']['cmap_pcmesh'],vmin=col['mod-mod']['minval'],vmax=col['mod-mod']['maxval'],edgecolor='k',linewidth=0.125)
                        elif regional_data_plottype == 'lines':
                            if 'windstress' in varia_name:
                                axx = ax[sdx].inset_axes([.58, .7, .4, .22])
                            elif 'clouds' or 'pressure' in varia_name:
                                axx = ax[sdx].inset_axes([.58, .76, .4, .22])
                            lineObjs = axx.plot(np.arange(1,13),concat_dd,'.-')
                            if cdx == 0 and sdx == 2:
                                axx.legend(iter(lineObjs),tuple(regions_to_plot_dummy),loc='lower left',bbox_to_anchor=(-0.1,-1.4),fontsize=fontsize-4,handletextpad = 0.4, handlelength=2)    
                    if regional_data_plottype == 'pcolmesh':
                        axx.set_yticks(np.arange(0.5,12.5))
                        axx.set_yticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],color='w',fontsize=plt.rcParams['font.size']-1)
                        axx.set_xticks(np.arange(len(regions_to_plot))+0.5)
                        axx.set_xticklabels(regions_to_plot_short,color='w',fontsize=plt.rcParams['font.size']-1)
                        axx.invert_yaxis()
                    elif regional_data_plottype == 'lines':
                        axx.set_xticks(np.arange(1,13,2))
                        axx.set_xticklabels(['J','M','M','J','S','N'],color='w',fontsize=plt.rcParams['font.size']-1)
                        #yticklabs = ax_insets[rdx,cdx].yaxis.label.set_color('w')
                        axx.tick_params(axis='y', colors='w')
                        #ax_insets[rdx,cdx].set_yticklabels(yticklabs,color='w',fontsize=plt.rcParams['font.size']-1)
                        axx.grid(color='#EEEEEE',linewidth=0.5)
                        axx.set_ylabel(col['mod']['unit'].replace('        ',''),color='w')    
                        axx.set_xlabel('Month',color='w')                           

        if regional_data == None:
            # add the mask regions in each panel (first the Obs. mask in ax[0,0], then the Model mask for all others)
            for axi in ax.flatten():
                for region in model_regions_dict.keys():
                    if region != 'full_map' and 'all_lats' not in region:
                        region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                        axi.contour(vdum.lon,vdum.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)
        elif regional_data != None:
            # add the mask regions in each panel (first the Obs. mask in ax[0,0], then the Model mask for all others)
            for axi in ax.flatten():
                for region in regions_to_plot:
                    region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']),0,1)
                    axi.contour(vdum.lon,vdum.lat,region_mask_dummy,levels=[0.5],colors='k',zorder=2,linewidths=0.5)            

        # put the plotted values into a Dataset 
        plotted_values = xr.Dataset() 
        # put varia_dict into this one

        if savefig == True:
            outpath = f'/nfs/kryo/work/koehne/roms/analysis/pactcs30/future_sim/analysis_plots/variability_analysis/future_changes/maps/'
            figname = f'{varia_name}_{depth}m.png'
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
        col['mod'] = dict()
        col['mod-mod'] = dict()

        if varia == 'temp' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          8,22,15,'RdYlBu_r',plt.get_cmap('RdYlBu_r',14) ,'°C'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   0,3,13,'cmo.amp',plt.get_cmap('cmo.amp',12) ,'°C'

        if varia == 'salt' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          33,34.5,16,'cmo.haline',plt.get_cmap('cmo.haline',15) ,'-'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -0.3,0.3,13,'cmo.balance',plt.get_cmap('cmo.balance',12) ,'-'

        if varia == 'temp_acf' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,400,21,'cmo.deep_r',plt.get_cmap('cmo.deep_r',20) ,'days'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -100,100,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'days'

        if varia == 'temp_std' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0.5,2,16,'cmo.thermal',plt.get_cmap('cmo.thermal',25) ,'°C'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -0.3,0.3,13,'cmo.balance',plt.get_cmap('cmo.balance',12) ,'°C'

        if varia == 'mld_holte' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          -80,0,17,'cmo.tempo_r',plt.get_cmap('cmo.tempo_r',16) ,'m'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -6,6,13,'cmo.balance',plt.get_cmap('cmo.balance',12) ,'m'

        if varia == 'abs_windstress' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,0.15,16,'cmo.speed',plt.get_cmap('cmo.speed',15) ,r'        Nm$^{-2}$'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -0.015,0.015,16,'cmo.balance',plt.get_cmap('cmo.balance',15) ,r'        Nm$^{-2}$'

        if varia == 'zonal_windstress' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          -0.15,0.15,16,'cmo.delta',plt.get_cmap('cmo.delta',15) ,r'        Nm$^{-2}$'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -0.01,0.01,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,r'        Nm$^{-2}$'

        if varia == 'merid_windstress' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          -0.15,0.15,16,'cmo.delta',plt.get_cmap('cmo.delta',15) ,r'        Nm$^{-2}$'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -0.01,0.01,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,r'        Nm$^{-2}$'

        if varia == 'windstress' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,0.15,16,'cmo.speed',plt.get_cmap('cmo.speed',15) ,r'        Nm$^{-2}$'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -0.01,0.01,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,r'        Nm$^{-2}$'

        if varia == 'windstress_curl' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          -5e-7,5e-7,21,'cmo.delta',plt.get_cmap('cmo.delta',20) ,r'        Nm$^{-3}$'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -5e-8,5e-8,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,r'        Nm$^{-3}$'

        if varia == 'lower_clouds' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,.8,17,'cmo.ice',plt.get_cmap('cmo.ice',16) ,'-'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -0.08,0.08,17,'cmo.balance',plt.get_cmap('cmo.balance',16) ,'-'

        if varia == 'total_clouds' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,.8,17,'cmo.ice',plt.get_cmap('cmo.ice',16) ,'-'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -0.1,0.1,21,'cmo.balance',plt.get_cmap('cmo.balance',20) ,'-'

        if varia == 'air_pressure' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          1010,1025,16,'cmo.turbid_r',plt.get_cmap('cmo.turbid_r',15) ,'hPa'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -.6,.6,13,'cmo.balance',plt.get_cmap('cmo.balance',12) ,'hPa'

        if varia == 'air_pressure_at_msl' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          1010,1025,16,'cmo.turbid_r',plt.get_cmap('cmo.turbid_r',15) ,'hPa'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -.6,.6,13,'cmo.balance',plt.get_cmap('cmo.balance',12) ,'hPa'

        # if varia == 'eke' and depth == 0:
        #     col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,0.025,11,'cmo.haline',plt.get_cmap('cmo.haline',10) ,r'     m$^2$s$^{-2}$'
        #     col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -.008,.008,17,'cmo.balance',plt.get_cmap('cmo.balance',16) ,r'     m$^2$s$^{-2}$'
        if varia == 'eke' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          0,250,11,'cmo.haline',plt.get_cmap('cmo.haline',10) ,r'    cm$^2$s$^{-2}$'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -80,80,17,'cmo.balance',plt.get_cmap('cmo.balance',16) ,r'    cm$^2$s$^{-2}$'

        if varia == 'net_surface_heatflux' and depth == 0:
            col['mod']['minval'],col['mod']['maxval'],col['mod']['numlevs'],col['mod']['cmap_contourf'],col['mod']['cmap_pcmesh'],col['mod']['unit'] =                          -100,100,11,'cmo.balance',plt.get_cmap('cmo.balance',10) ,r'     Wm$^{-2}$'
            col['mod-mod']['minval'],col['mod-mod']['maxval'],col['mod-mod']['numlevs'],col['mod-mod']['cmap_contourf'],col['mod-mod']['cmap_pcmesh'],col['mod-mod']['unit'] =   -10,10,11,'cmo.balance',plt.get_cmap('cmo.balance',10) ,r'     Wm$^{-2}$'

        return col
    
# %%
