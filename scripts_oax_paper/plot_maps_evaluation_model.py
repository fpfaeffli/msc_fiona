"""
author: Eike Köhn
changes: Fiona Pfäffli
date: 22.10.2024
description: This file loads in the model data and compares it with the respective observational dataset. The comparison consists of:
1. map of annual mean
2. for subregions full timeseries of variable (calculating trends)
3. for subregions climatological timeseries
"""

#%% load packages
 
# enable the visibility of the modules for the import functions
import sys
import os
import glob
from pathlib import Path
#sys.path.append(os.path.join(os.path.dirname(__file__), '/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/'))
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')


# load the package
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

from regridding_tools import Regridder as Regridder

from get_auxiliary_data import AuxGetter as AuxGetter

#%% Setting the variable to compare and across which models to compare

var = 'DIC'  # 'omega_arag_offl', 'DIC', 'Alk'

dep = 0 # m
vert_struct = 'zavg'    # 'zavg'

plot_resolution = 'daily'
model_temp_resolution = 'daily' # 'daily'
obs_temp_resolution = 'monthly' # 'monthly'

scenario = 'present'
simulation_type = 'hindcast'
ensemble_run = '000'
vtype = 'oceanic'
configs = ['romsoc_fully_coupled']

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

#%% Getting observational dataset (low resolution)
# print("Getting observational dataset for variable {} at depth {}m.".format(var,dep))

# if var == 'temp' and dep == 0:
#     obs_ds, obs_da = ObsGetter.get_sst_data(res=obs_temp_resolution)
# elif var == 'salt' and dep == 0:
#     obs_ds, obs_da = ObsGetter.get_sss_data(res=obs_temp_resolution)
# elif var == 'NO3' and dep == 0:
#     obs_ds, obs_da = ObsGetter.get_no3_data(res=obs_temp_resolution+'_clim')
#     obs_da = obs_da.sel(depth=dep)
# elif var == 'PO4' and dep == 0:
#     obs_ds, obs_da = ObsGetter.get_po4_data(res=obs_temp_resolution+'_clim')
#     obs_da = obs_da.sel(depth=dep)
# elif var == 'DIC' and dep == 0:
#     obs_ds, obs_da = ObsGetter.get_dic_data(res=obs_temp_resolution+'_clim')
#     obs_da = obs_da.sel(depth=dep)
# elif var == 'Alk' and dep == 0:
#     obs_ds, obs_da = ObsGetter.get_alk_data(res=obs_temp_resolution+'_clim')
#     obs_da = obs_da.sel(depth=dep)
# elif var == 'pH_offl' and dep == 0:
#     obs_ds, obs_da = ObsGetter.get_ph_data(res=obs_temp_resolution)
#     obs_da = obs_da
# elif var == 'omega_arag_offl' and dep == 0:
#     obs_ds, obs_da = ObsGetter.get_omega_arag_data(res=obs_temp_resolution)
#     obs_da = obs_da
# elif var == 'zeta' and dep == 0:
#     obs_ds, obs_da = ObsGetter.get_ssh_data(res=obs_temp_resolution)
#     obs_da = obs_da
# elif var == 'mld_holte':
#     obs_ds, obs_da = ObsGetter.get_mld_data(res=obs_temp_resolution+'_clim')
#     obs_da = obs_da*-1

# # get the area and distance to coast fields
# obs_area = ObsGetter.get_obs_area(obs_ds)
# obs_d2coast = ObsGetter.get_distance_to_coast(model_d2coast,obs_da)   # note that there are some artifacts in the polar regions

#%%
# Getting observational data (HR)


# 1) Build your file list (one per 8-day step, 2011–2021)
ph_base = "/nfs/sea/work/public/shared/ESA-OHOA/OceanSODA_ETHZ_HR-v2023.01-full_carbsys/ph_total"
files = []
for y in range(2011, 2022):
    year_dir = os.path.join(ph_base, str(y))
    for fn in sorted(os.listdir(year_dir)):
        if fn.endswith(".nc"):
            files.append(os.path.join(year_dir, fn))

print(f"Found {len(files)} files.")

# 2) Open + manually decode time for each
ph_list = []
for path in files:
    ds = xr.open_dataset(path, decode_times=False)
    da = ds["ph_total"]

    # Inspect ds.time.attrs['units'], e.g. "days since 2011-01-01"
    units = da.time.attrs.get("units", "")
    if "since" in units:
        origin = units.split("since")[1].strip()
    else:
        # fallback: assume days since Jan1 of the folder name
        year = os.path.basename(os.path.dirname(path))
        origin = f"{year}-01-01"

    raw = da.time.values.astype(int)
    dates = pd.to_datetime(raw, unit="D", origin=origin)

    # assign real datetimes back onto the DataArray
    da = da.assign_coords(time=("time", dates))

    ph_list.append(da)

# 3) Concatenate all years into one big DataArray
obs_da = xr.concat(ph_list, dim="time")
# after you have obs_da(time,lat,lon) on –180..+180
obs_da = obs_da.assign_coords(
    lon = (obs_da.lon + 360) % 360
).sortby("lon")

print(obs_da)

obs_ds = xr.Dataset({"ph_total": obs_da})

print(obs_da.time.dtype)  # should now be datetime64[ns]
print(obs_da.time.values[:5], "...", obs_da.time.values[-5:])

# get the area and distance to coast fields
obs_area = ObsGetter.get_obs_area(obs_ds)
obs_d2coast = ObsGetter.get_distance_to_coast(model_d2coast,obs_da)   # note that there are some artifacts in the polar regions


#%% 

# Get regions over which to calculate the statistics and to plot
model_regions_dict = GetRegions.define_CalCS_regions(model_area.lon,model_area.lat,model_d2coast)
obs_regions_dict = GetRegions.define_CalCS_regions(obs_area.lon,obs_area.lat,obs_d2coast)

# Load in the model and observational data
print('Loading the observational data for var {} at depth {}m.'.format(var,dep))
obs_da.load()
print('Done')
for config in configs:
    print('Loading the corresponding model data for config {}.'.format(config))
    model_da[config].load()
    print('Done')

#%% Calculate regional means 

print('calculate the monthly means for the regions')
# print('------Low resolution Obs-------')
# regional_data_obs = dict()
# for region in obs_regions_dict.keys():
#     if region != 'full_map':
#         print(region)
#         intermediate = (obs_da*obs_regions_dict[region]['mask']).weighted(obs_area.fillna(0)).mean(("lat","lon"))
#         if "time" in obs_da.dims:
#             regional_data_obs[region] = intermediate.groupby("time.month").mean("time")
#         elif "month" in obs_da.dims and np.size(obs_da.month)==12:
#             regional_data_obs[region] = intermediate

#%%
print('------High resolution Obs (yearly means)-------')
print('Calculating monthly means for high resolution pH observations')
regional_monthly_ph = {}
for region, info in obs_regions_dict.items():
    if region == "full_map":
        continue

    masked   = obs_da * info["mask"]
    weighted = masked.weighted(obs_area.fillna(0))
    mean_xy  = weighted.mean(("lat", "lon"))

    monthly = mean_xy.groupby("time.month").mean("time")
    regional_monthly_ph[region] = monthly

#%%
print('------Model------')
regional_data_model = dict()

for config in configs:
    regional_data_model[config] = dict()
    for region in model_regions_dict.keys():
        if region != 'full_map':
            print(config, region)
            regional_mean = (
                (model_da[config] * model_regions_dict[region]['mask'])
                .weighted(model_area.fillna(0))
                .mean(("eta_rho", "xi_rho"))
                .groupby("time.month")
                .mean("time")
            )
            regional_data_model[config][region] = regional_mean

            # Print mean value across all months for summary
            print(f"Mean [{config} - {region}]: {regional_mean.mean().values:.2e}")

#%%
# Import CSV files from boje pH observations

#CCE1 data
path_cce1 = '/nfs/sea/work/fdesmet/Data/Obs/Mooring/CCE1'
dfs_cce1 = {
    csv_file: pd.read_csv(csv_file)
    for csv_file in glob.glob(f"{path_cce1}/*.csv")
}
print("Loaded CSV files:")
for fname in dfs_cce1:
    print("  •", fname)


# CCE2 data
path_cce2 = '/nfs/sea/work/fdesmet/Data/Obs/Mooring/CCE2'
dfs_cce2 = {
    csv_file: pd.read_csv(csv_file)
    for csv_file in glob.glob(f"{path_cce2}/*.csv")
}
print("Loaded CSV files:")
for fname in dfs_cce2:
    print("  •", fname)


#%%##################
# START THE PLOTTING#
#####################


# %% 
def plot_full_map_annual_mean_no_roms(varia, depth, obs_da, model_da, obs_regions_dict, model_regions_dict, regional_data=None, regional_data_plottype='pcolmesh', savefig=False):
    """
    Modified version: only plots Observations, ROMSOC, and ROMSOC - Obs.
    """

    panel_labels = ['a)', 'b)', 'c)']

    obs_time_dim_name = obs_da.dims[0]    
    assert obs_time_dim_name == 'time' or obs_time_dim_name == 'month'
    annual_mean_Obs = obs_da.mean(dim=obs_time_dim_name)

    model_time_dim_name = model_da['romsoc_fully_coupled'].dims[0]
    assert model_time_dim_name == 'time' or model_time_dim_name == 'month'
    annual_mean_romsoc = model_da['romsoc_fully_coupled'].mean(dim=model_time_dim_name)
    annual_mean_romsoc = xr.where(ModelGetter.get_model_mask()==0, np.NaN, annual_mean_romsoc)

    annual_mean_Obs_on_model_grid = Regridder.regrid_original_to_target(
        annual_mean_Obs.values, 
        annual_mean_Obs.lon.values, 
        annual_mean_Obs.lat.values,
        annual_mean_romsoc.lon.values, 
        annual_mean_romsoc.lat.values
    )

    col = Plotter.get_color_maps_and_ranges(varia, depth)
    fontsize = 14
    plt.rcParams['font.size'] = fontsize
    fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)

    # a) Observations
    c0 = ax[0].pcolormesh(annual_mean_Obs.lon, annual_mean_Obs.lat, annual_mean_Obs,
                          vmin=col['obs']['minval'], vmax=col['obs']['maxval'], cmap=col['obs']['cmap_pcmesh'])
    ax[0].set_title('Observations', fontsize=16)
    ax[0].set_xlim([obs_regions_dict['full_map']['minlon'], obs_regions_dict['full_map']['maxlon']])
    ax[0].set_ylim([obs_regions_dict['full_map']['minlat'], obs_regions_dict['full_map']['maxlat']])
    cbar0 = plt.colorbar(c0, ax=ax[0], extend='both')
    cbar0.ax.set_title('pH', pad=30)

    # b) ROMSOC
    c1 = ax[1].pcolormesh(annual_mean_romsoc.lon, annual_mean_romsoc.lat, annual_mean_romsoc,
                          vmin=col['mod']['minval'], vmax=col['mod']['maxval'], cmap=col['mod']['cmap_pcmesh'])
    ax[1].set_title('ROMSOC', fontsize=16)
    cbar1 = plt.colorbar(c1, ax=ax[1], extend='both')
    cbar1.ax.set_title('pH', pad=30)

    # c) ROMSOC - Obs
    c2 = ax[2].pcolormesh(annual_mean_romsoc.lon, annual_mean_romsoc.lat,
                          annual_mean_romsoc - annual_mean_Obs_on_model_grid,
                          vmin=col['mod-obs']['minval'], vmax=col['mod-obs']['maxval'], cmap=col['mod-obs']['cmap_pcmesh'])
    ax[2].set_title('ROMSOC - Obs.', fontsize=16)
    cbar2 = plt.colorbar(c2, ax=ax[2], extend='both')
    cbar2.ax.set_title('ΔpH', pad=30)


    
    # Add land mask
    landmask_etopo = AuxGetter.get_etopo_data()
    for a in ax:
        a.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='black', zorder=1)

    for i, a in enumerate(ax):
        # Set axis limits for study area
        a.set_xlim([225, 245])
        a.set_ylim([29, 48])

        # Panel labels
        a.text(0.04, 0.97, panel_labels[i], ha='left', va='top',
            bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),
            transform=a.transAxes)

        # Axis ticks and labels
        yticks = np.arange(30, 48, 5)
        xticks = np.arange(225, 246, 5)
        a.set_yticks(yticks)
        a.set_yticklabels([f'{val}°N' for val in yticks], fontsize=fontsize - 2)
        a.set_xticks(xticks)
        a.set_xticklabels([f'{360 - val}°W' for val in xticks], fontsize=fontsize - 2)

        if regional_data is not None:
            regional_data_obs, regional_data_model = regional_data
            regions_to_plot = ['offshore_central', 'coastal_central']
            region_labels  = ['offshore',          'coastal'         ]

            # Create insets
            ax_insets = []
            for a in ax:
                inset = a.inset_axes([.67, .77, .27, .2], facecolor='white')
                ax_insets.append(inset)

            # Extract and plot the timeseries data
            obs_concat = np.stack([regional_data_obs[r].values for r in regions_to_plot], axis=1)
            mod_concat = np.stack([
                regional_data_model[r].values
                for r in regions_to_plot
            ],axis=1)

            diff_concat = mod_concat - obs_concat

            months = np.arange(1, 13)
            colors = ['saddlebrown', 'coral']
            for idx, data in enumerate([obs_concat, mod_concat, diff_concat]):
                for i in range(len(region_labels)):
                    ax_insets[idx].plot(
                        months, data[:, i], '.-',
                        color=colors[i],
                        label=region_labels[i] if idx == 0 else None
                    )

            for axin in ax_insets:
                axin.set_xticks([1, 3, 5, 7, 9, 11])
                axin.set_xticklabels(['J', 'M', 'M', 'J', 'S', 'N'], color='white', fontsize=10)

                # Show & color y-axis ticks and labels
                axin.tick_params(axis='y', colors='white', labelsize=10)
                axin.tick_params(axis='x', colors='white', labelsize=10)

                # Set y-labels (if any) to white as well
                for label in axin.get_yticklabels():
                    label.set_color('white')

                # Set axis spines to white
                for spine in axin.spines.values():
                    spine.set_color('white')

                axin.grid(color='gray', linewidth=0.3, linestyle='--')

                for inset in ax_insets[:2]:
                    inset.set_ylim(7.96, 8.13)
                    inset.set_yticks([8.0, 8.05, 8.1])

            # Define colors and labels
            colors = ['saddlebrown', 'coral']
            labels = ['offshore', 'coastal']

            # Add legends manually to all insets
            for axin in ax_insets:
                # Plot dummy lines to generate legend entries
                handles = [axin.plot([], [], color=colors[i], label=labels[i])[0] for i in range(len(labels))]
                legend = axin.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, -0.8), fontsize=12, frameon=False)

                # Set legend text color to white
                for text in legend.get_texts():
                    text.set_color('white')

    # Add offshore vs. coastal boundaries with colored contour lines
    downsample_factor = 1 # Adjust as needed
    region_contour_colors = {
        'offshore_central': 'saddlebrown',
        'coastal_central': 'coral'
    }

    for a in ax:
        for region in region_contour_colors:
            if region in model_regions_dict:
                region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']), 0, 1)
                region_mask_dummy = region_mask_dummy[::downsample_factor, ::downsample_factor]
                lon_rho = annual_mean_romsoc.lon.values
                lat_rho = annual_mean_romsoc.lat.values
                a.contour(
                    lon_rho[::downsample_factor, ::downsample_factor],
                    lat_rho[::downsample_factor, ::downsample_factor],
                    region_mask_dummy,
                    levels=[0.5],
                    colors=region_contour_colors[region],
                    linewidths=2,
                    zorder=4
                )

    # === Add CCE station markers + labels only to panel (a): Observations ===
    cce_stations = {
        "CCE1": {"lat": 33.4300, "lon": 237.5200, "color": "saddlebrown"},
        "CCE2": {"lat": 34.3240, "lon": 239.1850, "color": "coral"}
    }

    for name, props in cce_stations.items():
        # Plot colored marker
        ax[0].plot(
            props["lon"], props["lat"],
            marker='o',
            markersize=8,
            markerfacecolor=props["color"],
            markeredgecolor='black',
            zorder=5
        )
        # Add label
        ax[0].text(
            props["lon"] + 0.4, props["lat"] + 0.3,
            name,
            fontsize=10,
            color='black',
            bbox=dict(facecolor=props["color"], edgecolor='none', boxstyle='round,pad=0.15'),
            zorder=6
        )

    plt.tight_layout()

    
    # Save figure if needed
    if savefig:
        outpath = '/nfs/sea/work/fpfaeffli/plots/model_eval/maps_model_eval'
        figname = f'_{varia}_{depth}m_noROMS.png'
        plt.savefig(outpath + figname, dpi=200)

    # Return plotted values
    plotted_values = xr.Dataset()
    plotted_values['obs'] = annual_mean_Obs
    plotted_values['romsoc'] = annual_mean_romsoc
    plotted_values['romsoc_minus_obs'] = annual_mean_romsoc - annual_mean_Obs_on_model_grid

    return plotted_values


#%%
print('Generate plot: Map for annual mean with low resolution Obs and ROMSOC')
plot_full_map_annual_mean_no_roms(var,dep,obs_da,model_da,obs_regions_dict,model_regions_dict,regional_data=[regional_data_obs,regional_data_model],regional_data_plottype='lines',savefig=True)



# %%
# Maps with high resolution observations 

def plot_full_map_annual_mean_no_roms(varia, depth, obs_da, model_da, obs_regions_dict, model_regions_dict, regional_data=None, regional_data_plottype='pcolmesh', savefig=False):
    """
    Modified version: only plots High-Resolution Observations, ROMSOC, and ROMSOC - Obs.
    """

    panel_labels = ['a)', 'b)', 'c)']

    obs_time_dim_name = obs_da.dims[0]    
    assert obs_time_dim_name == 'time' or obs_time_dim_name == 'month'
    # annual mean of the HIGH-RESOLUTION observations
    annual_mean_Obs = obs_da.mean(dim=obs_time_dim_name)

    model_time_dim_name = model_da['romsoc_fully_coupled'].dims[0]
    assert model_time_dim_name == 'time' or model_time_dim_name == 'month'
    annual_mean_romsoc = model_da['romsoc_fully_coupled'].mean(dim=model_time_dim_name)
    annual_mean_romsoc = xr.where(ModelGetter.get_model_mask()==0, np.NaN, annual_mean_romsoc)

    # regrid high-res Obs onto the ROMS grid
    annual_mean_Obs_on_model_grid = Regridder.regrid_original_to_target(
        annual_mean_Obs.values, 
        annual_mean_Obs.lon.values, 
        annual_mean_Obs.lat.values,
        annual_mean_romsoc.lon.values, 
        annual_mean_romsoc.lat.values
    )

    col = Plotter.get_color_maps_and_ranges(varia, depth)
    fontsize = 16
    plt.rcParams['font.size'] = fontsize
    fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)

    # a) High-Resolution Observations
    c0 = ax[0].pcolormesh(
        annual_mean_Obs.lon, annual_mean_Obs.lat, annual_mean_Obs,
        vmin=col['obs']['minval'], vmax=col['obs']['maxval'], cmap=col['obs']['cmap_pcmesh']
    )
    ax[0].set_title('OceanSODA', fontsize=18)
    ax[0].set_xlim([obs_regions_dict['full_map']['minlon'], obs_regions_dict['full_map']['maxlon']])
    ax[0].set_ylim([obs_regions_dict['full_map']['minlat'], obs_regions_dict['full_map']['maxlat']])
    cbar0 = plt.colorbar(c0, ax=ax[0], extend='both')
    cbar0.ax.set_title('pH', pad=30, fontsize=16)
    cbar0.ax.tick_params(labelsize=16)

    # b) ROMSOC
    c1 = ax[1].pcolormesh(
        annual_mean_romsoc.lon, annual_mean_romsoc.lat, annual_mean_romsoc,
        vmin=col['mod']['minval'], vmax=col['mod']['maxval'], cmap=col['mod']['cmap_pcmesh']
    )
    ax[1].set_title('ROMSOC', fontsize=18)
    cbar1 = plt.colorbar(c1, ax=ax[1], extend='both')
    cbar1.ax.set_title('pH', pad=30, fontsize=16)
    cbar1.ax.tick_params(labelsize=16)

    # c) ROMSOC - Obs
    c2 = ax[2].pcolormesh(
        annual_mean_romsoc.lon, annual_mean_romsoc.lat,
        annual_mean_romsoc - annual_mean_Obs_on_model_grid,
        vmin=col['mod-obs']['minval'], vmax=col['mod-obs']['maxval'], cmap=col['mod-obs']['cmap_pcmesh']
    )
    ax[2].set_title('ROMSOC - OceanSODA', fontsize=18)
    cbar2 = plt.colorbar(c2, ax=ax[2], extend='both')
    cbar2.ax.set_title('ΔpH', pad=30, fontsize=16)
    cbar2.ax.tick_params(labelsize=16)

    # Add land mask
    landmask_etopo = AuxGetter.get_etopo_data()
    for a in ax:
        a.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='black', zorder=1)

    for i, a in enumerate(ax):
        # Set axis limits for study area
        a.set_xlim([225, 245])
        a.set_ylim([29, 48])

        # Panel labels
        a.text(0.04, 0.97, panel_labels[i], ha='left', va='top',
            bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'),
            transform=a.transAxes)

        # Axis ticks and labels
        yticks = np.arange(30, 48, 5)
        xticks = np.arange(225, 246, 5)
        a.set_yticks(yticks)
        a.set_yticklabels([f'{val}°N' for val in yticks], fontsize=16)
        a.set_xticks(xticks)
        a.set_xticklabels([f'{360 - val}°W' for val in xticks], fontsize=16)

        if regional_data is not None:
            regional_data_obs, regional_data_model = regional_data
            regions_to_plot = ['offshore_central', 'coastal_central']
            region_labels  = ['offshore',          'coastal'         ]

            # Create insets
            ax_insets = []
            for a in ax:
                inset = a.inset_axes([.67, .77, .27, .2], facecolor='white')
                ax_insets.append(inset)

            # Extract and plot the timeseries data
            obs_concat = np.stack([regional_data_obs[r].values for r in regions_to_plot], axis=1)
            mod_concat = np.stack([
                regional_data_model[r].values
                for r in regions_to_plot
            ],axis=1)

            diff_concat = mod_concat - obs_concat

            months = np.arange(1, 13)
            colors = ['saddlebrown', 'coral']
            for idx, data in enumerate([obs_concat, mod_concat, diff_concat]):
                for i in range(len(region_labels)):
                    ax_insets[idx].plot(
                        months, data[:, i], '.-',
                        color=colors[i],
                        label=region_labels[i] if idx == 0 else None
                    )

            for axin in ax_insets:
                axin.set_xticks([1, 3, 5, 7, 9, 11])
                axin.set_xticklabels(['J', 'M', 'M', 'J', 'S', 'N'], color='white', fontsize=13)

                # Show & color y-axis ticks and labels
                axin.tick_params(axis='y', colors='white', labelsize=13)
                axin.tick_params(axis='x', colors='white', labelsize=13)

                # Set y-labels (if any) to white as well
                for label in axin.get_yticklabels():
                    label.set_color('white')

                # Set axis spines to white
                for spine in axin.spines.values():
                    spine.set_color('white')

                axin.grid(color='gray', linewidth=0.3, linestyle='-')

                for inset in ax_insets[:2]:
                    inset.set_ylim(7.89, 8.13)
                    inset.set_yticks([7.9, 8.0, 8.1])

            # Define colors and labels
            colors = ['saddlebrown', 'coral']
            labels = ['offshore', 'coastal']

            # Add legends manually to all insets
            for axin in ax_insets:
                # Plot dummy lines to generate legend entries
                handles = [axin.plot([], [], color=colors[i], label=labels[i])[0] for i in range(len(labels))]
                legend = axin.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, -0.8), fontsize=14, frameon=False)

                # Set legend text color to white
                for text in legend.get_texts():
                    text.set_color('white')

    # Add offshore vs. coastal boundaries with colored contour lines
    downsample_factor = 1 # Adjust as needed
    region_contour_colors = {
        'offshore_central': 'black',
        'coastal_central': 'black'
    }

    for a in ax:
        for region in region_contour_colors:
            if region in model_regions_dict:
                region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']), 0, 1)
                region_mask_dummy = region_mask_dummy[::downsample_factor, ::downsample_factor]
                lon_rho = annual_mean_romsoc.lon.values
                lat_rho = annual_mean_romsoc.lat.values
                a.contour(
                    lon_rho[::downsample_factor, ::downsample_factor],
                    lat_rho[::downsample_factor, ::downsample_factor],
                    region_mask_dummy,
                    levels=[0.5],
                    colors=region_contour_colors[region],
                    linewidths=2,
                    zorder=4
                )

    # === Add CCE station markers + labels only to panel (a): Observations ===
    cce_stations = {
        "CCE2": {"lat": 34.3240, "lon": 239.1850, "color": "coral"}
    }

    for name, props in cce_stations.items():
        # Plot colored marker
        ax[0].plot(
            props["lon"], props["lat"],
            marker='o',
            markersize=8,
            markerfacecolor=props["color"],
            markeredgecolor='black',
            zorder=5
        )
        # Add label
        ax[0].text(
            props["lon"] + 0.45, props["lat"] - 0.5,
            name,
            fontsize=16,
            color='black',
            bbox=dict(facecolor=props["color"], edgecolor='none', boxstyle='round,pad=0.15'),
            zorder=6
        )

    plt.tight_layout()

    
    # Save figure if needed
    if savefig:
        outpath = '/nfs/sea/work/fpfaeffli/plots/model_eval/maps_model_eval'
        figname = f'_{varia}_{depth}m_HR_Obs_black_contour_smaller_region.png'
        plt.savefig(outpath + figname, dpi=200)

    # Return plotted values
    plotted_values = xr.Dataset()
    plotted_values['obs'] = annual_mean_Obs
    plotted_values['romsoc'] = annual_mean_romsoc
    plotted_values['romsoc_minus_obs'] = annual_mean_romsoc - annual_mean_Obs_on_model_grid

    return plotted_values

# Plot high res Obs

plot_full_map_annual_mean_no_roms(
    var, dep, obs_da, model_da,
    obs_regions_dict, model_regions_dict,
    # <-- here, select the per‐region dict for your config:
    regional_data=[
        regional_monthly_ph,
        regional_data_model['romsoc_fully_coupled']
    ],
    regional_data_plottype='lines',
    savefig=True
)


# %%

# Averaged timeseries 

def plot_area_averaged_timeseries(varia, depth, obs_da, obs_area, model_da, model_area, obs_regions_dict, model_regions_dict, plot_resolution, savefig=False):
    """
    Plot area-averaged monthly time series of obs and ROMSOC for 'all_lats' only.
    """

    # Set consistent font sizes globally
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

    panel_labels = ['d)', 'e)', 'f)']

    col = Plotter.get_color_maps_and_ranges(varia, depth)

    row_names = ['all_lats']  # Only this row
    col_names = ['all_dists', 'offshore', 'coastal']  # Keep all 3 distance categories

    fig, ax = plt.subplots(1, len(col_names), sharex=True, sharey=True, figsize=(12, 3.5))

    for cdx, cn in enumerate(col_names):
        region_name = f'{cn}_all_lats'

        obs_reg = obs_regions_dict[region_name]['mask']
        model_reg = model_regions_dict[region_name]['mask']

        obs_regional_mean = obs_da.weighted((obs_area * obs_reg).fillna(0)).mean(dim=('lat', 'lon'))
        romsoc_regional_mean = model_da['romsoc_fully_coupled'].weighted((model_area * model_reg).fillna(0)).mean(dim=('eta_rho', 'xi_rho'))

        if pd.infer_freq(romsoc_regional_mean.time) == 'D':
            romsoc_regional_mean = romsoc_regional_mean.resample(time='1M').mean(dim='time')

        obs_time_dim_name = obs_da.dims[0]
        if obs_time_dim_name == 'month':
            obs_regional_mean = obs_regional_mean.rename({'month': 'time'})
            obs_regional_mean['time'] = romsoc_regional_mean.time[:12]
        elif obs_time_dim_name == 'time':
            obs_regional_mean['time'] = obs_regional_mean.time

        if plot_resolution == 'monthly':
            obs_regional_mean = obs_regional_mean.resample(time='1M').mean(dim='time')
            obs_regional_mean['time'] = [pd.Timestamp(t).replace(day=1) for t in obs_regional_mean['time'].values]

        ax_here = ax[cdx]

        # Plot time series
        ax_here.plot(obs_regional_mean.time, obs_regional_mean, color='black', linewidth=2, label='Observations')
        ax_here.plot(romsoc_regional_mean.time, romsoc_regional_mean, color='darkgrey', linewidth=2, label='ROMSOC')

        # X-axis formatting
        ax_here.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_here.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_here.tick_params(axis='both', labelsize=10)

        # Title mapping with color
        title_map = {
            'all_dists': ('whole region', 'black'),
            'offshore': ('offshore region', 'saddlebrown'),
            'coastal': ('coastal region', 'coral')
        }
        title_text, title_color = title_map[cn]
        ax_here.set_title(title_text, color=title_color)

        # Add panel label (d, e, f)
        ax_here.text(0.05, 0.94, panel_labels[cdx], ha='left', va='top', fontsize=9,
                    transform=ax_here.transAxes,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'))

        # Custom spine color
        if cn == 'offshore':
            spine_color = 'saddlebrown'
        elif cn == 'coastal':
            spine_color = 'coral'
        else:
            spine_color = 'black'

        for spine in ax_here.spines.values():
            spine.set_edgecolor(spine_color)
        ax_here.tick_params(color=spine_color)

    ax[1].set_xlabel('Time')
    ax[0].set_ylabel('°C')
    ax[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.tight_layout()

    # Output dataset
    plotted_values = xr.Dataset()
    plotted_values['obs'] = obs_regional_mean
    plotted_values['romsoc'] = romsoc_regional_mean

    if savefig:
        outpath = '/nfs/sea/work/fpfaeffli/plots/model_eval/maps_model_eval'
        figname = f'_timeseries_{varia}_{depth}m_romsoc.png'
        plt.savefig(outpath + figname, dpi=200, transparent=True)

    return plotted_values



print('Generate plot: Averaged timeseries in regions')
plotted_values = plot_area_averaged_timeseries(var,dep,obs_da,obs_area,model_da,model_area,obs_regions_dict,model_regions_dict,plot_resolution,savefig=True)

# %%
##########################
#  CCE1 and CCE2 pH obs  #
##########################

#%%
# Find out lat lon of CCE1 and CCE2 moorings in the model grid

# 1) CCE station coords
stations = {
    "CCE1": (33.4300, -122.4800),
    "CCE2": (34.3240, -120.8150),
}

# 2) pick your config key (you only have one in configs)
config = configs[0]   # e.g. 'romsoc_fully_coupled'
ds = model_ds[config]

lat_rho = ds["lat"]
lon_rho = ds["lon"]

# 4) turn to numpy for a simple nearest‐neighbour search
lat_vals = lat_rho.values
lon_vals = lon_rho.values

for name, (slat, slon) in stations.items():
    # compute squared distance
    dist2 = (lat_vals - slat)**2 + (lon_vals - slon)**2
    
    # find the linear index of the minimum distance
    flat_idx = np.nanargmin(dist2)
    # unravel into 2D grid indices (eta_rho, xi_rho)
    j, i = np.unravel_index(flat_idx, lat_vals.shape)
    
    print(f"{name}:")
    print(f"  grid indices → eta_rho = {j}, xi_rho = {i}")
    print(f"  model lat,lon → {lat_vals[j,i]:.4f}, {lon_vals[j,i]:.4f}\n")


# %%
# Plot pH of both CCE1 and CCE2 moorings over time

paths = {
    "CCE1": Path("/nfs/sea/work/fdesmet/Data/Obs/Mooring/CCE1"),
    "CCE2": Path("/nfs/sea/work/fdesmet/Data/Obs/Mooring/CCE2")
}

records = []
for label, folder in paths.items():
    for f in sorted(folder.glob("*.csv")):
        # find header row
        with f.open() as fh:
            header_row = next((i for i, line in enumerate(fh) if "Date" in line), 0)
        df = pd.read_csv(f, header=header_row)
        if {"Date", "Time"}.issubset(df.columns):
            df["Datetime"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                errors="coerce"
            )
            for col in df.columns:
                if "ph" in col.lower() and "qf" not in col.lower():  # skip quality flag cols
                    print(f"→ Found pH column: '{col}' in file {f.name}")
                    s = pd.to_numeric(df[col], errors="coerce")
                    mask = df["Datetime"].notna() & s.between(6, 9)
                    records.append(pd.DataFrame({
                        "Datetime": df.loc[mask, "Datetime"],
                        "pH": s[mask],
                        "Mooring": label
        }))


ph_df = pd.concat(records, ignore_index=True).sort_values("Datetime")

fig, ax = plt.subplots(figsize=(10, 5))
for mooring, grp in ph_df.groupby("Mooring"):
    ax.scatter(grp["Datetime"], grp["pH"], s=10, alpha=0.7, label=mooring)
ax.set(xlabel="Date", ylabel="pH", title="pH Time Series for CCE1 & CCE2")
ax.legend()
plt.tight_layout()
plt.show()


#%%
# Plots timeseries of ROMSOC output vs mooring observations 


# === 0) Import & Prepare Observation Data ===
from pathlib import Path

paths = {
    "CCE1": Path("/nfs/sea/work/fdesmet/Data/Obs/Mooring/CCE1"),
    "CCE2": Path("/nfs/sea/work/fdesmet/Data/Obs/Mooring/CCE2")
}

records = []

for label, folder in paths.items():
    for f in sorted(folder.glob("*.csv")):
        with f.open() as fh:
            header_row = next((i for i, line in enumerate(fh) if "Date" in line), 0)
        df = pd.read_csv(f, header=header_row)
        if {"Date", "Time"}.issubset(df.columns):
            df["Datetime"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                errors="coerce"
            )
            for col in df.columns:
                # include all possible pH columns except QF flags
                if "ph" in col.lower() and "qf" not in col.lower():
                    print(f"→ Found pH column: '{col}' in file {f.name}")
                    s = pd.to_numeric(df[col], errors="coerce")
                    mask = df["Datetime"].notna() & s.between(6, 9)
                    records.append(pd.DataFrame({
                        "Datetime": df.loc[mask, "Datetime"],
                        "pH": s[mask],
                        "Mooring": label
                    }))

ph_df = pd.concat(records, ignore_index=True).sort_values("Datetime")

# === 1) Define station locations and convert longitude ===
def convert_lon(lon):
    return lon + 360 if lon < 0 else lon

stations = {
    "CCE1": (33.4300, convert_lon(-122.4800)),  # → 237.52
    "CCE2": (34.3240, convert_lon(-120.8150)),  # → 239.185
}

# === 2) Pick model config and grab datasets ===
config = "romsoc_fully_coupled"
ds = model_ds[config]
da = model_da[config]  # pH_offl(time, eta_rho, xi_rho)

# === 3) Extract lat/lon fields ===
lat2d = ds["lat"].values
lon2d = ds["lon"].values

# === 4) Create ocean mask from first timestep ===
pH0 = da.isel(time=0).values
ocean_mask = np.isfinite(pH0) & (pH0 < 1e18)

# === 5) Nearest-ocean-cell finder ===
def find_nearest_ocean(lat2d, lon2d, mask, tlat, tlon):
    dist2 = (lat2d - tlat)**2 + (lon2d - tlon)**2
    dist2[~mask] = np.inf
    flat_idx = np.argmin(dist2)
    return np.unravel_index(flat_idx, lat2d.shape)

# === 6) Extract model time series for each station ===
time_dim = next(c for c in da.coords if "time" in c.lower())
model_ts = {}

for name, (plat, plon) in stations.items():
    η, ξ = find_nearest_ocean(lat2d, lon2d, ocean_mask, plat, plon)
    print(f"{name}: η={η}, ξ={ξ}, lat={lat2d[η,ξ]:.4f}, lon={lon2d[η,ξ]:.4f}")
    window = 2  # 2 on each side of center = 5×5 box
    pt = da.isel(
        eta_rho=slice(η - window, η + window + 1),
        xi_rho=slice(ξ - window, ξ + window + 1)
    ).mean(dim=("eta_rho", "xi_rho")) # average over 5x5 gridcells
    selected = da.isel(
    eta_rho=slice(η - window, η + window + 1),
    xi_rho=slice(ξ - window, ξ + window + 1)
    )
    n_valid_cells = np.isfinite(selected.isel(time=0)).sum().item()
    print(f"{name}: averaging over {n_valid_cells} valid cells")

    times = pd.to_datetime(pt[time_dim].values)
    model_ts[name] = pd.Series(pt.values, index=times)

# === 7) Extract obs time series from DataFrame ===
obs_ts = {
    name: ph_df[ph_df["Mooring"] == name]
               .set_index("Datetime")["pH"]
               .resample("1D")
               .mean()
    for name in stations
}

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Plot settings
colors = {
    "CCE1": {"model": "saddlebrown", "obs": "darkgrey"},
    "CCE2": {"model": "coral", "obs": "darkgrey"}
}
region_labels = {
    "CCE1": "offshore",
    "CCE2": "coastal"
}

for name in ["CCE1", "CCE2"]:
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot model
    ax.plot(
        model_ts[name].index,
        model_ts[name].values,
        color=colors[name]["model"],
        linestyle='-',
        label="ROMSOC"
    )

    # Plot observations
    ax.plot(
        obs_ts[name].index,
        obs_ts[name].values,
        color=colors[name]["obs"],
        linestyle='-',
        label="Obs"
    )

    ax.set_ylabel("pH")
    ax.set_xlabel("Date")
    ax.set_title(f"{name} vs. ROMSOC", fontsize=14)
    ax.legend(loc='lower right', fontsize=10)

    # Set x-axis ticks every 2 years
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(f'/nfs/sea/work/fpfaeffli/plots/model_eval/{name.lower()}_model_vs_obs_pH.png', dpi=200, transparent=True)
    plt.show()


# %%
# Plots timeseries of OceanSODA-ETHZ-HR vs. Mooring observations (CCE1 and CCE2)

# Load in situ mooring pH (you already have this)
paths = {
    "CCE1": Path("/nfs/sea/work/fdesmet/Data/Obs/Mooring/CCE1"),
    "CCE2": Path("/nfs/sea/work/fdesmet/Data/Obs/Mooring/CCE2")
}

records = []
for label, folder in paths.items():
    for f in sorted(folder.glob("*.csv")):
        with f.open() as fh:
            header_row = next((i for i, line in enumerate(fh) if "Date" in line), 0)
        df = pd.read_csv(f, header=header_row)
        if {"Date", "Time"}.issubset(df.columns):
            df["Datetime"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                errors="coerce"
            )
            for col in df.columns:
                if "ph" in col.lower() and "qf" not in col.lower():
                    s = pd.to_numeric(df[col], errors="coerce")
                    mask = df["Datetime"].notna() & s.between(6, 9)
                    records.append(pd.DataFrame({
                        "Datetime": df.loc[mask, "Datetime"],
                        "pH": s[mask],
                        "Mooring": label
                    }))

ph_df = pd.concat(records, ignore_index=True).sort_values("Datetime")


# === 1) Define station locations and convert longitude ===
def convert_lon(lon):
    return lon + 360 if lon < 0 else lon

stations = {
    "CCE1": (33.4300, convert_lon(-122.4800)),  # → 237.52
    "CCE2": (34.3240, convert_lon(-120.8150)),  # → 239.185
}


# === 3) Extract lat/lon fields ===
lat2d = obs_da.lat.values
lon2d = obs_da.lon.values

# === 4) Create ocean mask from first timestep ===
first = obs_da.isel(time=0).values
ocean_mask = np.isfinite(first) & (first < 1e18)

# === 5’) Revised nearest-ocean-cell finder (1D→2D via broadcasting) ===
def find_nearest_ocean(lat1d, lon1d, mask2d, tlat, tlon):
    # compute (M×N) distance array via broadcasting
    # lat1d[:, None] is shape (M,1), lon1d[None, :] is (1,N)
    dist2 = (lat1d[:, None] - tlat)**2 + (lon1d[None, :] - tlon)**2

    # mask out land
    dist2[~mask2d] = np.inf

    # find the linear index of the minimum
    flat_idx = np.nanargmin(dist2)
    return np.unravel_index(flat_idx, dist2.shape)

# === 6) Extract OceanSODA-HR time series for each station ===
oceansoda_ts = {}
for name, (plat, plon) in stations.items():
    j, i = find_nearest_ocean(lat2d, lon2d, ocean_mask, plat, plon)
    print(f"{name}: grid cell → lat idx {j}, lon idx {i}  (≈ {lat2d[j]:.2f}°, {lon2d[i]:.2f}°)")
    w = 2  # half-width of the box
    subset = obs_da.isel(
        lat=slice(j-w, j+w+1),
        lon=slice(i-w, i+w+1)
    )
    ts = subset.mean(dim=("lat","lon")).to_series()
    oceansoda_ts[name] = ts

# === 7) Extract obs time series from DataFrame and plot ===
obs_ts = {
    name: ph_df[ph_df["Mooring"] == name]
               .set_index("Datetime")["pH"]
               .resample("1D")
               .mean()
    for name in stations
}

colors = {
    "CCE1": {"osoda": "saddlebrown", "obs": "darkgrey"},
    "CCE2": {"osoda": "coral",      "obs": "darkgrey"}
}
region_labels = {
    "CCE1": "offshore",
    "CCE2": "coastal"
}

for name in ["CCE1", "CCE2"]:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        oceansoda_ts[name].index, oceansoda_ts[name].values,
        color=colors[name]["osoda"], linestyle='-',
        label="OceanSODA-ETHZ-HR"
    )
    ax.plot(
        obs_ts[name].index, obs_ts[name].values,
        color=colors[name]["obs"], linestyle='-',
        label="Obs"
    )

    ax.set_ylabel("pH")
    ax.set_xlabel("Date")
    ax.set_title(f"{name} vs. OceanSODA-HR", fontsize=14)
    ax.legend(loc='lower right', fontsize=10)

    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()

# %%
# All three in one plot


# === 0) Load in situ mooring pH (as before) ===
paths = {
    "CCE1": Path("/nfs/sea/work/fdesmet/Data/Obs/Mooring/CCE1"),
    "CCE2": Path("/nfs/sea/work/fdesmet/Data/Obs/Mooring/CCE2")
}

records = []
for label, folder in paths.items():
    for f in sorted(folder.glob("*.csv")):
        with f.open() as fh:
            header_row = next((i for i, line in enumerate(fh) if "Date" in line), 0)
        df = pd.read_csv(f, header=header_row)
        if {"Date", "Time"}.issubset(df.columns):
            df["Datetime"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                errors="coerce"
            )
            for col in df.columns:
                if "ph" in col.lower() and "qf" not in col.lower():
                    s = pd.to_numeric(df[col], errors="coerce")
                    mask = df["Datetime"].notna() & s.between(6, 9)
                    records.append(pd.DataFrame({
                        "Datetime": df.loc[mask, "Datetime"],
                        "pH": s[mask],
                        "Mooring": label
                    }))

ph_df = pd.concat(records, ignore_index=True).sort_values("Datetime")

# Resample to daily means
obs_ts = {
    name: ph_df[ph_df["Mooring"]==name]
               .set_index("Datetime")["pH"]
               .resample("1D")
               .mean()
    for name in paths
}


# === 1) Station coords ===
def convert_lon(lon):
    return lon + 360 if lon < 0 else lon

stations = {
    "CCE1": (33.4300, convert_lon(-122.4800)),  # → 237.52°E
    "CCE2": (34.3240, convert_lon(-120.8150)),  # → 239.185°E
}


# === 2) ROMSOC model DataArray (assumes these dicts exist) ===
config = "romsoc_fully_coupled"
ds_mod = model_ds[config]
da_mod = model_da[config]    # dims (time, eta_rho, xi_rho)

# === 3) OceanSODA-HR observations DataArray ===
# (assumes you've concatenated into `obs_da(time, lat, lon)`)
# e.g.
# obs_da = xr.open_dataset(".../OceanSODA_HR_concat.nc")["ph_total"]


# === 4) Prep masks ===
# ROMSOC grid coords & mask
lat_mod = ds_mod["lat"].values    # shape (M,N)
lon_mod = ds_mod["lon"].values
mask_mod = np.isfinite(da_mod.isel(time=0).values)

# OceanSODA grid coords & mask
lat_os = obs_da.lat.values        # (P,)
lon_os = obs_da.lon.values        # (Q,)
mask_os = np.isfinite(obs_da.isel(time=0).values) & (obs_da.isel(time=0).values<1e18)


# === 5) Nearest-ocean-cell finder for OceanSODA (1D→2D) ===
def find_nearest_os(lat1d, lon1d, mask2d, tlat, tlon):
    # lat1d:(P,), lon1d:(Q,), mask2d:(P,Q)
    dist2 = (lat1d[:,None] - tlat)**2 + (lon1d[None,:] - tlon)**2
    dist2[~mask2d] = np.inf
    idx = np.nanargmin(dist2)
    return np.unravel_index(idx, dist2.shape)  # → (j_os, i_os)


# === 6) Extract ROMSOC & OceanSODA time series ===
model_ts     = {}
oceansoda_ts = {}

for name, (plat, plon) in stations.items():
    # ---- ROMSOC nearest cell (flatten approach) ----
    M, N = lat_mod.shape
    lats_flat = lat_mod.ravel()
    lons_flat = lon_mod.ravel()
    mask_flat = mask_mod.ravel()

    dist2 = (lats_flat - plat)**2 + (lons_flat - plon)**2
    dist2[~mask_flat] = np.inf
    flat  = np.nanargmin(dist2)
    η, ξ   = divmod(flat, N)

    # average a 5×5 window
    w = 2
    sub_mod = da_mod.isel(
        eta_rho=slice(η-w, η+w+1),
        xi_rho =slice(ξ-w, ξ+w+1)
    )
    m_ts = sub_mod.mean(dim=("eta_rho","xi_rho"))
    model_ts[name] = pd.Series(
        m_ts.values,
        index=pd.to_datetime(m_ts.time.values)
    )

    # ---- OceanSODA nearest cell ----
    j_os, i_os = find_nearest_os(lat_os, lon_os, mask_os, plat, plon)
    sub_os = obs_da.isel(
        lat=slice(j_os-w, j_os+w+1),
        lon=slice(i_os-w, i_os+w+1)
    )
    os_ts = sub_os.mean(dim=("lat","lon")).to_series()
    oceansoda_ts[name] = os_ts


# === 7) Plot all three together ===
colors = {
    "CCE1": {"obs":"saddlebrown",   "mod":"grey", "os":"black"},
    "CCE2": {"obs":"coral",   "mod":"grey",       "os":"black"},
}
labels = {"CCE1":"offshore","CCE2":"coastal"}

for name in stations:
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(obs_ts[name].index, obs_ts[name].values,
            label=f'Mooring ({name})',    color=colors[name]["obs"], linewidth=1.5)
    ax.plot(model_ts[name].index, model_ts[name].values,
            label="ROMSOC",     color=colors[name]["mod"], linewidth=1.5)
    ax.plot(oceansoda_ts[name].index, oceansoda_ts[name].values,
            label="OceanSODA",color=colors[name]["os"],  linewidth=1.5)

    ax.set_xlabel("Year"); ax.set_ylabel("pH")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(f'/nfs/sea/work/fpfaeffli/plots/model_eval/model_eval_CCE_ROMSOC_SODA_timeseries.png', dpi=200, transparent=True)   
    plt.show()

# %%
