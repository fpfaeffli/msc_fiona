"""
author: Fiona Pfäffli
description: Basic plotting of the CaCS regions on a map for thesis.
"""


#%% enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

# load the packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import cmocean
from get_study_regions import GetRegions as GetRegions
from get_model_datasets_Hplus_surf_only import ModelGetter as ModelGetter
from matplotlib.colors import TwoSlopeNorm

from importlib import reload  # Python 3.4+

import get_obs_datasets 
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter

from func_for_clim_thresh import ThreshClimFuncs
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs

from get_auxiliary_data import AuxGetter as AuxGetter

from matplotlib.lines import Line2D
import glob

#Get the distance to coast file from ROMS and the regions over which to calculate the statistics
model_d2coasts = dict()
model_d2coasts['roms_only'] = ModelGetter.get_distance_to_coast(vtype='oceanic')

# Get the model regions
model_regions = dict()
model_regions['roms_only'] = GetRegions.define_CalCS_regions(model_d2coasts['roms_only'].lon, model_d2coasts['roms_only'].lat, model_d2coasts['roms_only'])

# Get the model area
model_area = ModelGetter.get_model_area()
model_regions_dict = GetRegions.define_CalCS_regions(model_area.lon, model_area.lat, model_d2coasts['roms_only'])


# Load landmask
landmask_etopo = AuxGetter.get_etopo_data()

#%% 

# Define the threshold 
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.Hplus_instance() #95th percentile threshold

# Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present'] # 
configs = ['romsoc_fully_coupled'] # [ms_only'] 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'zavg'    # 'avg'
depth = 0

#%% 

#################################
#          GETTING DATA         #
#################################

# Get the model datasets for the oceanic variables

ocean_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)


#%% Define variable
varias = ['pH_offl'] # 'pH_offl' , 'Hplus' , 'omega_arag_offl'

variables = dict()
for config in configs:
    variables[config] = dict()
    for scenario in scenarios:
        print(f'Getting the variables for config {config} and scenario {scenario}.')
        # Oceanic variables
        variables[config][scenario] = dict()
        for var in varias:
            if var == 'Hplus':
                print('Hplus')
                # Convert pH to [H+] concentration
                pH = ocean_ds[config][scenario].pH_offl.isel(depth=0).load()
                variables[config][scenario]['Hplus'] = 10 ** (-pH)
            elif var == 'pH_offl':
                print('pH_offl')
                # Directly take pH_offl without conversion
                variables[config][scenario]['pH_offl'] = ocean_ds[config][scenario].pH_offl.isel(depth=0).load()
            elif var == 'omega_arag_offl':
                print('omega_arag_offl')
                variables[config][scenario]['omega_arag_offl'] = ocean_ds[config][scenario].omega_arag_offl.isel(depth=0).load()

#%%
# Find coupling domain lat and lon from the raw model COSMO output

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# 1) Open COSMO grid
grid_ds = xr.open_dataset(
    '/nfs/meso/work/loher/ROMSOC/future_SSP245_2010/'
    'cosmo_output/lffd2010012712.nc'
)
lon2d = grid_ds.lon.values   # shape (rlat, rlon)
lat2d = grid_ds.lat.values

# 2) Extract the four edges in order: west edge (all rlat at rlon=0),
#    then north edge (all rlon at rlat=-1), then east edge, then south edge.
west_lon  = lon2d[:,    0]
west_lat  = lat2d[:,    0]

north_lon = lon2d[-1,  :]
north_lat = lat2d[-1,  :]

east_lon  = lon2d[::-1, -1]   # reverse so we go north→south
east_lat  = lat2d[::-1, -1]

south_lon = lon2d[0,   ::-1]  # reverse so we go east→west
south_lat = lat2d[0,   ::-1]

# 3) Concatenate into a single closed polygon
poly_lon = np.concatenate([west_lon, north_lon, east_lon, south_lon])
poly_lat = np.concatenate([west_lat, north_lat, east_lat, south_lat])


# 1) Wrap longitudes into 0–360° for a 0–360 plot:
poly_lon_wrapped = np.mod(poly_lon, 360)



##########################
#         PLOTTING            #
###############################

# %%
# Plotting a snapshot of the whole model domain (Pacific Ocean)

# Coupling‑zone extents (big black box)
couple_lon_min, couple_lon_max = 197.3, 249.6
couple_lat_min, couple_lat_max =  15.0,  60.0

# Study‑area extents (small indigo box)
study_lon_min, study_lon_max = 230.0, 240.0
study_lat_min, study_lat_max =  35.0,  43.0

def plot_model_snapshot_pH_with_grid(
    ocean_ds,
    landmask_etopo,
    date="2018-06-10",
    coupling_extent=None,
    config="romsoc_fully_coupled",
    scenario="present",
):
    """
    Plots a snapshot of surface pH, masks west of 100°E, draws:
     - big black rectangle for coupling zone,
     - small indigo rectangle for study area.
    """
    ds = ocean_ds[config][scenario]
    lat_rho = ds.lat_rho.isel(time=0)
    lon_rho = ds.lon_rho.isel(time=0)

    # pH snapshot + title
    pH = ds['pH_offl'].sel(time=date, method='nearest').isel(depth=0)
    title_time = f"Snapshot: {pH.time.dt.strftime('%Y-%m-%d').values}"

    # mask west of 100°E
    pH = pH.where(lon_rho >= 100)

    # figure
    fig, ax = plt.subplots(figsize=(12, 8))
    vmin, vmax = 7.6, 8.1
    levels = np.linspace(vmin, vmax, 40)
    norm   = plt.Normalize(vmin=vmin, vmax=vmax)

    cf = ax.contourf(
        lon_rho, lat_rho, pH,
        levels=levels, cmap='Spectral', norm=norm,
        zorder=1, extend='both'
    )
    plt.colorbar(
        cf, ax=ax, orientation='vertical',
        label='Surface pH', extend='both',
        ticks=np.arange(vmin, vmax+0.01, 0.1)
    )

    # gridlines
    stride = 10
    for i in range(0, lat_rho.shape[0], stride):
        ax.plot(lon_rho[i, :], lat_rho[i, :],
                'k-', lw=0.6, alpha=0.5, zorder=2)
    for j in range(0, lon_rho.shape[1], stride):
        ax.plot(lon_rho[:, j], lat_rho[:, j],
                'k-', lw=0.6, alpha=0.5, zorder=2)

    # landmask
    ax.contourf(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        levels=[0, 1], colors=['black'], zorder=3
    )

    outline = mpatches.Polygon(
    list(zip(poly_lon_wrapped, poly_lat)),
    closed=True,
    edgecolor='lightgrey',
    facecolor='none',
    linewidth=2,
    zorder=6,
    )
    ax.add_patch(outline)



    # small indigo study-area box
    study_rect = mpatches.Rectangle(
        (study_lon_min, study_lat_min),
        study_lon_max - study_lon_min,
        study_lat_max - study_lat_min,
        linewidth=2, edgecolor='indigo',
        facecolor='none', zorder=5
    )
    ax.add_patch(study_rect)

    # Create legend handles for the two boxes:
    legend_handles = [
        mpatches.Patch(edgecolor='lightgrey', facecolor='none', linewidth=2, label='Coupling zone'),
        mpatches.Patch(edgecolor='indigo', facecolor='none', linewidth=2, label='Study region')
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='white',
        fontsize=14,
        framealpha=1.0,  # Make the legend frame fully opaque
    )

    # keep full domain view
    ax.set_xlim(100, float(lon_rho.max()))
    ax.set_ylim(-90, float(lat_rho.max()))

    # ticks & labels
    yticks = np.arange(-90, np.ceil(float(lat_rho.max())/20)*20 + 1, 20)
    xticks = np.arange(100, np.ceil(float(lon_rho.max())/20)*20 + 1, 20)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels([f"{lat:.0f}°" + ("S" if lat < 0 else "N")
                        for lat in yticks], fontsize=13)
    ax.set_xticklabels([f"{lon:.0f}°E" for lon in xticks], fontsize=13)

    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.set_title(title_time, fontsize=16)
    plt.tight_layout()

    outpath = (
        f'/nfs/sea/work/fpfaeffli/plots/methods/'
        f'aa_finished_model_domain_snapshot_surface_pH_'
        f'couplingzone_overlay_{date}.png'
    )
    plt.savefig(outpath, dpi=150, transparent=True, bbox_inches='tight')
    plt.show()

    


# Usage:
plot_model_snapshot_pH_with_grid(
    ocean_ds,
    landmask_etopo,
    date="2013-07-06",
    coupling_extent=(couple_lon_min, couple_lon_max, couple_lat_min, couple_lat_max)
)

#%%
# Plot snapshot of CaCS region

def plot_CaCS_snapshot_pH_with_grid(
    ocean_ds,
    landmask_etopo,
    model_regions_dict,
    config="romsoc_fully_coupled",
    scenario="present",
    date="2018-06-10"
):
    """
    Plots a one-day snapshot of surface pH (pH_offl) over the CaCS domain,
    masking everything west of 100°E, cropping to CaCS bounds, and using a
    diverging Spectral_r colormap centered at pH=8.1.
    """
    # CaCS bounds in native 0–360 lon
    lat_bounds = (32, 46)
    lon_bounds = (225, 245)

    ds = ocean_ds[config][scenario]
    lat_rho = ds.lat_rho.isel(time=0)
    lon_rho = ds.lon_rho.isel(time=0)

    # Select single-day snapshot
    pH = ds['pH_offl'].sel(time=date, method='nearest').isel(depth=0)
    title_time = f"Snapshot: {pH.time.dt.strftime('%Y-%m-%d').values}"

    # Mask west of 100°E
    pH = pH.where(lon_rho >= 100)

    fig, ax = plt.subplots(figsize=(2.85, 3.77))
   

    # Linear levels & norm (no center)
    vmin, vmax = 7.6, 8.1
    levels = np.linspace(vmin, vmax, 40)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    cf = ax.contourf(
        lon_rho, lat_rho, pH,
        levels=levels,
        cmap='Spectral',
        norm=norm,
        zorder=1,
        extend='both'  # <-- Added extend='both'
    )

    

    # Overlay model gridlines
    stride = 10
    for i in range(0, lat_rho.shape[0], stride):
        ax.plot(lon_rho[i, :], lat_rho[i, :], 'k-', lw=0.6, alpha=0.5, zorder=2)
    for j in range(0, lon_rho.shape[1], stride):
        ax.plot(lon_rho[:, j], lat_rho[:, j], 'k-', lw=0.6, alpha=0.5, zorder=2)

    # Overlay landmask
    ax.contourf(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        levels=[0, 1], colors=['black'], zorder=3
    )

    # Offshore vs coastal contours
    region_contour_colors = {
        'offshore_central': 'black',
        'coastal_central': 'black'
    }
    for region, color in region_contour_colors.items():
        if region in model_regions_dict:
            mask = model_regions_dict[region]['mask']
            mask_dummy = xr.where(np.isnan(mask), 0, 1)
            ax.contour(
                lon_rho, lat_rho, mask_dummy,
                levels=[0.5],
                colors=[color],
                linewidths=2,
                zorder=4
            )

    # Crop to CaCS domain
    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)

    # Ticks every 5°
    yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 3)
    xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels([f"{lat:.0f}°N" for lat in yticks], fontsize=13)
    ax.set_xticklabels([f"{lon:.0f}°E" for lon in xticks], fontsize=13)

    # Thicken & recolor spines
    for spine in ax.spines.values():
        spine.set_color('indigo')
        spine.set_linewidth(2)
    ax.tick_params(axis='x', colors='indigo', width=3)
    ax.tick_params(axis='y', colors='indigo', width=3)

    plt.tight_layout()

    outpath = (
        f'/nfs/sea/work/fpfaeffli/plots/methods/'
        f'aa_finished_CaCS_snapshot_surface_pH_{date}.png'
    )
    
    plt.savefig(outpath, dpi=150, transparent=True, bbox_inches='tight')
    plt.show()


# Example call:
plot_CaCS_snapshot_pH_with_grid(
    ocean_ds, landmask_etopo, model_regions_dict,
    date="2013-07-06" )

# %%

# %%
# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def plot_CaCS_triplet_snapshot(
    ocean_ds,
    landmask_etopo,
    model_regions_dict,
    config="romsoc_fully_coupled",
    scenario="present",
    date="2018-06-10"
):
    """
    Plots three identical copies of the one-day surface pH snapshot
    over the CaCS domain, side by side, matching the 12.5×5″ three‑panel layout,
    with 16 pt titles.  Latitude label only on the first panel.
    """
    # 1) load data & timestamp
    ds      = ocean_ds[config][scenario]
    lat_rho = ds.lat_rho.isel(time=0)
    lon_rho = ds.lon_rho.isel(time=0)
    t0      = ds["pH_offl"].sel(time=date, method="nearest").time.values
    title_str = f"Snapshot: {np.datetime_as_string(t0, unit='D')}"
    pH_snap = (ds["pH_offl"]
               .sel(time=date, method="nearest")
               .isel(depth=0)
               .where(lon_rho >= 100))

    # 2) plotting parameters
    lat_bounds = (32, 46)
    lon_bounds = (225, 245)
    yticks     = np.arange(lat_bounds[0], lat_bounds[1]+1, 3)
    xticks     = np.arange(lon_bounds[0], lon_bounds[1]+1, 5)
    vmin, vmax = 7.6, 8.1
    levels     = np.linspace(vmin, vmax, 40)
    norm       = plt.Normalize(vmin=vmin, vmax=vmax)

    # 3) figure + axes
    fig, axes = plt.subplots(
        1, 3,
        figsize=(12, 4),
        constrained_layout=True
    )

    # 4) draw each panel
    for i, ax in enumerate(axes):
        cf = ax.contourf(
            lon_rho, lat_rho, pH_snap,
            levels=levels, cmap="Spectral",
            norm=norm, extend="both", zorder=1
        )

        # landmask
        ax.contourf(
            landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
            levels=[0,1], colors=["black"], zorder=3
        )

        # regions
        for region in ("offshore_central", "coastal_central"):
            if region in model_regions_dict:
                mask = model_regions_dict[region]["mask"]
                m2d  = xr.where(np.isnan(mask), 0, 1)
                ax.contour(
                    lon_rho, lat_rho, m2d,
                    levels=[0.5], colors=["black"],
                    linewidths=1, zorder=4
                )

        # limits
        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)

        # x‐ticks always
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.0f}°E" for x in xticks], color='indigo')
        ax.set_xlabel("Longitude", fontsize=12, color='indigo')

        # y‐ticks & label only on the first subplot
        if i == 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{y:.0f}°N" for y in yticks], color='indigo')
            ax.set_ylabel("Latitude", fontsize=12, color='indigo')
        else:
            ax.set_yticks(yticks)
            ax.set_yticklabels([], color='indigo')

        # Make spines and ticks indigo
        for spine in ax.spines.values():
            spine.set_color('indigo')
            spine.set_linewidth(1)
        ax.tick_params(axis='x', colors='indigo', width=1)
        ax.tick_params(axis='y', colors='indigo', width=1)

        # title
        ax.set_title(title_str, fontsize=16)

    # 5) shared colorbar
    cax = fig.add_axes([0.25, -0.04, 0.5, 0.02])
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label("Surface pH", fontsize=14)
    cbar.set_ticks(np.arange(vmin, vmax+0.01, 0.1))
    cbar.ax.tick_params(labelsize=12)

    # 6) save & show
    outpath = f"/nfs/sea/work/fpfaeffli/plots/methods/CaCS_triplet_{date}.png"
    plt.savefig(outpath, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# %%
plot_CaCS_triplet_snapshot(
    ocean_ds,
    landmask_etopo,
    model_regions_dict,
    date="2013-07-06"
)
# %%
