"""
author: Fiona Pfäffli
description: Basic plotting of the CaCS region to make a movie of pH.
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
import matplotlib.animation as animation


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
scenarios = ['ssp245', 'ssp585'] # 
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


################################
# Plotting the data            #
################################

#%%
# Plotting the daily snapshots of surface pH for the CaCS region 

def plot_CaCS_daily_snapshots_pH(
    ocean_ds,
    landmask_etopo,
    config="romsoc_fully_coupled",
    scenario="present",
    out_dir="/nfs/sea/work/fpfaeffli/plots/pH_movie"
):
    """
    Loops through every day in the hindcast and saves a CaCS-domain snapshot
    of surface pH (with landmask and box, no gridlines), one PNG per day.
    """
    os.makedirs(out_dir, exist_ok=True)

    lat_bounds = (29, 48)
    lon_bounds = (225, 245)

    ds    = ocean_ds[config][scenario]
    times = ds['pH_offl'].time.values

    # coords
    lat_rho = ds.lat_rho.isel(time=0)
    lon_rho = ds.lon_rho.isel(time=0)

    levels = np.linspace(7.8, 8.5, 20)

    for t in times:
        date_str = np.datetime_as_string(t, unit='D')
        pH = ds['pH_offl'].sel(time=t, method='nearest').isel(depth=0)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_aspect('equal')

        # pH contour
        cf = ax.contourf(lon_rho, lat_rho, pH,
                         levels=levels, cmap='cmo.ice_r', zorder=1)
        plt.colorbar(cf, ax=ax, orientation='vertical',
                     label='Surface pH', extend='both',
                     ticks=np.arange(7.8, 8.5, 0.1), shrink=0.8)

        # landmask
        ax.contourf(landmask_etopo.lon, landmask_etopo.lat,
                    landmask_etopo, levels=[0,1], colors=['black'], zorder=2)

        # --- create a fresh box for this figure ---
        rect = mpatches.Rectangle(
            (lon_bounds[0], lat_bounds[0]),
            lon_bounds[1] - lon_bounds[0],
            lat_bounds[1] - lat_bounds[0],
            linewidth=2, edgecolor='black',
            facecolor='none', linestyle='-',
            zorder=3
        )
        ax.add_patch(rect)

        # limits & ticks
        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        yticks = np.arange(lat_bounds[0], lat_bounds[1]+1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1]+1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat:.0f}°N" for lat in yticks], color='black')
        ax.set_xticklabels([f"{lon:.0f}°E" for lon in xticks], color='black')
        ax.tick_params(axis='both', color='black', width=1, length=6)

        # spines
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)

        ax.set_title(f"CaCS Surface pH: {date_str}", color='black', fontsize=16)
        plt.tight_layout()

        fname = f"{scenario}_CaCS_pH_{date_str}.png"
        fig.savefig(os.path.join(out_dir, fname),
                    dpi=150, transparent=True, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(times)} snapshots to {out_dir}/")


# Call the function to plot the daily snapshots
plot_CaCS_daily_snapshots_pH(
    ocean_ds,
    landmask_etopo,
    config="romsoc_fully_coupled",
    scenario="ssp585",
    out_dir="/nfs/sea/work/fpfaeffli/plots/pH_movie"
)

#%% Make the movie from the saved PNGs

%%bash
ffmpeg -y \
  -framerate 37 \
  -pattern_type glob -i '/nfs/sea/work/fpfaeffli/plots/pH_movie/CaCS_pH_*.png' \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p \
  /nfs/sea/work/fpfaeffli/plots/pH_movie/CaCS_pH_movie_fast.mp4
