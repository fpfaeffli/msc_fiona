"""
author: Fiona Pfäffli
description: Calculates the extreme H+ days per year for the different scenarios and plots them.
"""

#%% enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')
# sys.path.append('/home/koehne/Documents/supervision/fiona/msc_fiona/scripts/modules/')
# sys.path.append('/home/koehne/Documents/supervision/fiona/msc_fiona/scripts/evaluation/')
# sys.path.append('/home/koehne/Documents/supervision/fiona/msc_fiona/scripts/climatologies_and_thresholds/')

# load the packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
import pprint
from get_study_regions import GetRegions as GetRegions
from get_model_datasets_Hplus_surf_only import ModelGetter as ModelGetter
from matplotlib import colors as mcolors
from importlib import reload  # Python 3.4+
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates


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
scenarios = ['present', 'ssp245', 'ssp585'] # ,'ssp585'
configs = ['romsoc_fully_coupled'] # [ms_only'] 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'zavg'    # 'avg'
depth = 0

#%% 

################ Loading all the data ##################

# Get the model datasets for the oceanic variables

ocean_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)

#%%
# load the data at the respective location
def load_data_in(varia):
    vdict = dict()
    for config in configs:
        vdict[config] = dict()
        for scenario in scenarios:
            print(f'Getting the variable for config {config} and scenario {scenario}.')
            print(varia)
            if varia == 'Hplus':
                pH = ocean_ds[config][scenario].pH_offl.load()
                vdict[config][scenario] = np.power(10, -pH)
            
    return vdict

hplus = load_data_in('Hplus')



#%% LOAD in all the climatologies

def load_clim_in(varia):
    cdict = dict()
    for config in configs:
        cdict[config] = dict()
        for scenario in scenarios:
            print(f'Getting the variable for config {config} and scenario {scenario}.')
            clim_dir = f'/nfs/sea/work/fpfaeffli/future_sim/thresholds_and_climatologies/{config}/{scenario}/'
            clim_file = f'hobday2016_threshold_and_climatology_{varia}_95.0perc_2011-2021baseperiod_fixedbaseline_11aggregation_31smoothing_0depthlevelindex_ensemble000.nc'
            ds_dummy = xr.open_dataset(clim_dir+clim_file)
            clim_365 = ds_dummy.clim_smoothed
            clim_366 = ModelGetter.include_feb29(clim_365)
            concatenated_clim = ModelGetter.concatenate_yearly_arrays(clim_365, clim_366, start_year=2011, end_year=2021)
            if 'day_of_year_adjusted' in concatenated_clim.dims:
                concatenated_clim = concatenated_clim.rename({'day_of_year_adjusted': 'time'})
            elif 'day_of_year_adjusted' in concatenated_clim.coords:
                concatenated_clim = concatenated_clim.rename_coords({'day_of_year_adjusted': 'time'})
            cdict[config][scenario] = concatenated_clim
    return cdict

hplus_clim = load_clim_in('Hplus')



#%% LOAD in the threshold for Hplus

print("Loading the threshold for Hplus...")
def load_thresh_in(varia, thresh_value):
    print('Get the present day threshold')
    config = configs[0]
    threshold, threshold_366 = ModelGetter.get_threshold(varia, 0, 'relative', thresh_value, config, 'present')
    concatenated_threshold = ModelGetter.concatenate_yearly_arrays(threshold, threshold_366, start_year=2011, end_year=2021)
    if 'day_of_year_adjusted' in concatenated_threshold.dims:
        concatenated_threshold = concatenated_threshold.rename({'day_of_year_adjusted': 'time'})
    elif 'day_of_year_adjusted' in concatenated_threshold.coords:
        concatenated_threshold = concatenated_threshold.rename_coords({'day_of_year_adjusted': 'time'})
    thresholds_mult = dict()
    thresholds_mult[config] = dict()
    for scenario in ['present', 'ssp245', 'ssp585']:
        thresholds_mult[config][scenario] = dict()
        print(f'{config}, {scenario}')
        thresholds_mult[config][scenario]['present'] = concatenated_threshold
        thresholds_mult[config][scenario]['present_plus_meandelta'] = concatenated_threshold + (hplus_clim[config][scenario].mean(dim='time') - hplus_clim[config]['present'].mean(dim='time'))
        
    return thresholds_mult

thresholds_mult = load_thresh_in('Hplus', 95)


# %% COMPUTE THE THRESHOLD EXCEEDANCES (EXTREMES MASK) for Hplus for whole area

print("Calculating the extreme masks for Hplus...")
def calc_extreme_masks(inst_dict, thresh_dict):
    extreme_mask_dict = dict()
    for config in configs:
        extreme_mask_dict[config] = dict()
        for scenario in scenarios:
            extreme_mask_dict[config][scenario] = dict()
            for threshold_type in ['present', 'present_plus_meandelta']:
                extreme_mask_dict[config][scenario][threshold_type] = inst_dict[config][scenario] > thresh_dict[config][scenario][threshold_type]
    return extreme_mask_dict

extreme_mask_dict = calc_extreme_masks(hplus, thresholds_mult)


#%%

# %%
# Calculate the anomalies for the climatology and threshold based on the fixed and shifted baseline
anomalies_clim_fixed   = {}
anomalies_clim_moving  = {}
anomalies_thresh_fixed = {}
anomalies_thresh_moving= {}

#Present climatology
present_clim   = hplus_clim['romsoc_fully_coupled']['present']     # smoothed daily climatology of present run
present_mean   = present_clim.mean(dim='time')                    # mean over the base period

# Loop over scenarios
for scenario in scenarios:
    print(f"\nProcessing scenario: {scenario}")

    # Extract H+ timeseries and extreme masks
    hplus_da     = hplus['romsoc_fully_coupled'][scenario]
    mask_fixed   = extreme_mask_dict['romsoc_fully_coupled'][scenario]['present']
    mask_moving  = extreme_mask_dict['romsoc_fully_coupled'][scenario]['present_plus_meandelta']

    # Align the H+ data with the present climatology
    hplus_da, present_clim = xr.align(hplus_da, present_clim, join="exact")

    # Compute the scenario’s climatology differences (all scenarios reference the present day climatology)
    scen_clim    = hplus_clim['romsoc_fully_coupled'][scenario]
    scen_mean    = scen_clim.mean(dim='time')
    clim_offset  = scen_mean - present_mean

    # Anomalies relative to fixed (present) baseline 
    clim_fixed_da  = (hplus_da - present_clim).where(mask_fixed)

    # Anomalies relative to shifted baseline 
    shifted_clim   = present_clim + clim_offset
    clim_moving_da = (hplus_da - shifted_clim).where(mask_moving)

    # Threshold anomalies 
    thresh_fixed_da  = (hplus_da - thresholds_mult['romsoc_fully_coupled'][scenario]['present']).where(mask_fixed)
    thresh_moving_da = (hplus_da - thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_meandelta']).where(mask_moving)

    # Flatten to 1D numpy arrays, dropping NaNs 
    def flatten(da):
        arr = da.values
        return arr[~np.isnan(arr)]

    anomalies_clim_fixed[scenario]    = flatten(clim_fixed_da)
    anomalies_clim_moving[scenario]   = flatten(clim_moving_da)
    anomalies_thresh_fixed[scenario]  = flatten(thresh_fixed_da)
    anomalies_thresh_moving[scenario] = flatten(thresh_moving_da)

    # Sanity prints
    print("  fixed‐clim anomalies:",  anomalies_clim_fixed[scenario].size,  "points")
    print("  shifted‐clim anomalies:", anomalies_clim_moving[scenario].size, "points")
    print("  fixed‐thresh anomalies:", anomalies_thresh_fixed[scenario].size,  "points")
    print("  shifted‐thresh anomalies:",anomalies_thresh_moving[scenario].size, "points")


#%%
###############################
#     COUNTING EXTREMES       #
###############################

#%%

# Define total number of time steps based on any scenario's extreme mask
total_time_steps = extreme_mask_dict[configs[0]]["present"]["present"].shape[0]


# %%
print("Calculating regional mean extreme days per year (all_dists/coastal/offshore)...")

# Add 'all_dists_all_lats' to the regions
regional_extreme_days = {
    'all_dists_central': {},
    'coastal_central': {},
    'offshore_central': {}
}

# Make sure total_time_steps is defined
total_time_steps = extreme_mask_dict[configs[0]]["present"]["present"].shape[0]

# Loop over regions and compute the area-weighted mean
for region_key in regional_extreme_days.keys():
    print(f"\nRegion: {region_key}")
    region_mask = model_regions_dict[region_key]['mask']
    weighted_mask = (region_mask * model_area).fillna(0)

    for scenario in scenarios:
        for threshold_type in ['present', 'present_plus_meandelta']:
            # Get the binary extreme mask
            mask = extreme_mask_dict[configs[0]][scenario][threshold_type]

            # Count extreme days per grid cell
            extreme_days = mask.sum(dim='time')

            # Convert to mean per year
            extreme_days_per_year = (extreme_days / total_time_steps) * 365

            # Area-weighted mean in region
            regional_mean = (
                (extreme_days_per_year * weighted_mask).sum(dim=('eta_rho', 'xi_rho')) /
                weighted_mask.sum(dim=('eta_rho', 'xi_rho'))
            ).item()

            # Store and print
            regional_extreme_days[region_key][(scenario, threshold_type)] = regional_mean
            print(f"{scenario} | {threshold_type}: {regional_mean:.2f} days/year")

##############################
#   CALC EXTREME METRICS     #
################################

#%%



#%%
# Colors for the scenarios
# Fixed color assignment per scenario
scenarios = ['ssp585', 'ssp245', 'present']
colors = {
    'present': "#ffab67",          # warm orange

    'ssp585_light': "#b98bf0",     # lighter purple
    'ssp585':        "#9553e0",     # original
    'ssp585_dark':  "#6c2eb3",     # darker purple

    'ssp245_light': "#8ad764",     # lighter green
    'ssp245':        "#65a342",     # original
    'ssp245_dark':  "#3c7d1f"      # darker green
}

#%%
# extract the two central-latitude band masks
mask_central_coastal = model_regions_dict['coastal_central']['mask']
mask_central_offshore = model_regions_dict['offshore_central']['mask']


#%%
###############################
#         PLOTTING MAPS       #
###############################

#%%

# Plot surface pH comparison of the three scenarios:
# %%


def plot_surface_pH_comparison(
    ocean_ds,
    landmask_etopo,
    model_regions_dict,
    config="romsoc_fully_coupled"
):
    """
    Side-by-side contourf maps of mean surface pH for present, SSP245, and SSP585,
    but only plotting values in the [7.5, 8.5] range using the 'Spectral_r' colormap.
    """
    scenarios    = ["present", "ssp245", "ssp585"]
    title_colors = {"present":"#f08c3a","ssp245":"#65a342","ssp585":"#9553e0"}
    lat_bounds   = [32, 46]
    lon_bounds   = [225, 245]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)


    # define contour levels between 7.5 and 8.5
    levels = np.linspace(7.6, 8.1, 20)

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        ds = ocean_ds[config][scenario]

        # compute mean surface pH
        ph_surf = ds["pH_offl"].isel(depth=0).mean(dim="time")

        # mask outside [7.5,8.5]
        ph_surf = ph_surf.where((ph_surf >= 7.5) & (ph_surf <= 8.1))

        lat_rho = ds.lat_rho.isel(time=0)
        lon_rho = ds.lon_rho.isel(time=0)

        # contourf plot with Spectral_r
        cf = ax.contourf(
            lon_rho, lat_rho, ph_surf,
            levels=levels,
            cmap='Spectral',
            extend="both",
            zorder=1,
            linewidths=1
        )

        # land mask
        ax.contourf(
            landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
            levels=[0, 1], colors=["black"], zorder=3
        )

        # region contours
        for region, col in {
            "offshore_central":"black",
            "coastal_central":"black"
        }.items():
            mask2d = model_regions_dict[region]["mask"]
            mask_plot = xr.where(np.isnan(mask2d), 0, 1)
            ax.contour(
                lon_rho, lat_rho, mask_plot,
                levels=[0.5], colors=[col], linewidths=1, zorder=4
            )

        ax.set_title(scenario, fontsize=16, color=title_colors[scenario])
        ax.set_xlim(lon_bounds); ax.set_ylim(lat_bounds)
        ax.set_xlabel("Longitude", fontsize=13)
        if i == 0:
            ax.set_ylabel("Latitude", fontsize=13)
        yt = np.arange(lat_bounds[0], lat_bounds[1]+1, 3)
        xt = np.arange(lon_bounds[0], lon_bounds[1]+1, 5)
        ax.set_yticks(yt); ax.set_xticks(xt)
        ax.set_yticklabels([f"{y}°N" for y in yt])
        ax.set_xticklabels([f"{360-x}°W" for x in xt])

    # colorbar, fixed to 7.5–8.5
    cax = fig.add_axes([0.25, -0.04, 0.5, 0.02])
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label("Surface pH", fontsize=14)
    cbar.set_ticks(np.arange(7.6, 8.11, 0.05))
    cbar.ax.tick_params(labelsize=12)

    out = "/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/surface_pH_comparison.png"
    plt.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# Call it:
plot_surface_pH_comparison(ocean_ds, landmask_etopo, model_regions_dict, config="romsoc_fully_coupled")


# %%
# Plot extreme days per year comparison of the three scenarios:

def plot_extreme_days_comparison(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict, baseline_type, config="romsoc_fully_coupled"):
    """
    Plots a side-by-side comparison of mean extreme H+ days per year 
    for the scenarios: present, SSP245, and SSP585.
    """
    
    # Define the scenarios to plot
    scenarios = ["present", "ssp245", "ssp585"]

    # Define scenario colors for titles
    title_colors = {"present": "#f08c3a", "ssp245": "#65a342", "ssp585": "#9553e0"}

    
    # Set bounds
    lat_bounds = [32, 46]
    lon_bounds = [225, 245]
    
    # Create a single figure with 3 subplots (side by side)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    
    # Loop through scenarios and plot each one
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        dataset = ocean_ds[config][scenario]
        
        # Extract lat/lon coordinates
        lat_rho = dataset.lat_rho.isel(time=0)
        lon_rho = dataset.lon_rho.isel(time=0)
        
        # Compute mean extreme days per year per grid cell
        total_time_steps = extreme_mask_dict[config][scenario][baseline_type].shape[0]
        extreme_days_per_year = ((extreme_mask_dict[config][scenario][baseline_type].sum(dim="time") / total_time_steps) * 365).squeeze()
        print("Mean extreme days per year in the map:", np.nanmean(extreme_days_per_year.values))
        print("Min extreme days in map:", np.nanmin(extreme_days_per_year.values))
        print("Max extreme days in map:", np.nanmax(extreme_days_per_year.values))



        # Add solid black landmask
        ax.contourf(
            landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
            levels=[0, 1], colors=['black'], zorder=3
        )
        
        # Add contour levels
        contour_levels = np.linspace(0, 365, 40)  # Consistent scale for all scenarios

        
        # Create filled contour plot
        cf = ax.contourf(
            lon_rho, lat_rho, extreme_days_per_year, levels=contour_levels, cmap=cmocean.cm.amp, zorder=1, extend="both"
        )

                # Add coastal and offshore boundary contours
        region_contour_colors = {
            'offshore_central': 'black',
            'coastal_central': 'black'
        }
        downsample_factor = 1  # Optional for performance / clarity

        for region in region_contour_colors:
            if region in model_regions_dict:
                region_mask = model_regions_dict[region]['mask']
                region_mask_dummy = xr.where(np.isnan(region_mask), 0, 1)
                region_mask_dummy = region_mask_dummy[::downsample_factor, ::downsample_factor]

                lon_ds = lon_rho[::downsample_factor, ::downsample_factor]
                lat_ds = lat_rho[::downsample_factor, ::downsample_factor]

                ax.contour(
                    lon_ds,
                    lat_ds,
                    region_mask_dummy,
                    levels=[0.5],
                    colors=region_contour_colors[region],
                    linewidths=1,
                    zorder=4
                )

        
        
        # Set labels and titles
        ax.set_title(f"{scenario}", fontsize=16, color=title_colors[scenario])
        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_xlabel("Longitude", fontsize=13)
        if i == 0:  # Only set ylabel for first subplo
            ax.set_ylabel("Latitude", fontsize=13)
        
        # Format latitude & longitude labels
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 3)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticklabels([f"{xtick}°E" for xtick in xticks])
    
    # Add colorbar below the subplots
    cbar_ax = fig.add_axes([0.25, -0.04, 0.5, 0.02])  # Adjusted placement
    cbar = plt.colorbar(cf, cax=cbar_ax, extend="both", orientation="horizontal")
    
    # Set predefined tick values for the colorbar
    cbar.set_ticks([0, 50, 100, 150, 200, 250, 300, 350])
    cbar.set_ticklabels([str(tick) for tick in [0, 50, 100, 150, 200, 250, 300, 350]])
    
    cbar.set_label(f'Extreme [$H^+$] days per year', fontsize=16)
    
    # Add suptitle
    #if baseline_type == "present":
      #  fig.suptitle(f"Comparison of mean extreme H+ days per year for different scenarios (fixed baseline)", fontsize=16)
   # if baseline_type == "present_plus_meandelta":
       # fig.suptitle(f"Comparison of mean extreme H+ days per year for different scenarios (moving baseline)", fontsize=16)
    
    # Save 
    output_path = f'/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/extreme_Hplus_days_per_year_comparison_{baseline_type}.png'
    plt.savefig(output_path, dpi=150, transparent=True, bbox_inches='tight')
    plt.show()

# Call the function for fixed and moving baselines
plot_extreme_days_comparison(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict, "present", config="romsoc_fully_coupled")
plot_extreme_days_comparison(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict, "present_plus_meandelta", config="romsoc_fully_coupled")



#%%
# Plot fixed and moved baseline threshold exceedance comparison of the three scenarios under both baselines

def plot_threshold_exceedance_comparison(
    ocean_ds,
    landmask_etopo,
    model_regions_dict,
    inst_dict,
    thresh_dict,
    baseline_type,
    config="romsoc_fully_coupled"
):
    """
    Side‑by‑side maps of mean threshold exceedance (H+ – H+_thresh) 
    during extreme days for present, SSP245, SSP585.
    """

    scenarios    = ["present", "ssp245", "ssp585"]
    title_colors = {"present":"#f08c3a","ssp245":"#65a342","ssp585":"#9553e0"}
    lat_bounds   = [32, 46]
    lon_bounds   = [225, 245]

    # Compute per-scenario exceedance maps
    exceedance = {}
    for scenario in scenarios:
        h    = inst_dict[config][scenario]
        th   = thresh_dict[config][scenario][baseline_type]
        mask = h > th

        exc = (h - th).where(mask)
        if 'depth' in exc.dims:
            exc = exc.isel(depth=0)

        exceedance[scenario] = exc.mean(dim="time")

    # Common contour levels
    all_vals = np.concatenate([exceedance[s].values.flatten() for s in scenarios])
    vmin, vmax = 0.0, np.nanpercentile(all_vals, 98)
    levels = np.linspace(vmin, vmax, 30)

    # Create figure & subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        lon_rho = ocean_ds[config][scenario].lon_rho.isel(time=0)
        lat_rho = ocean_ds[config][scenario].lat_rho.isel(time=0)

        cf = ax.contourf(
            lon_rho, lat_rho, exceedance[scenario],
            levels=levels, cmap=cmocean.cm.matter, extend="both", zorder=1
        )
        # land mask
        ax.contourf(
            landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
            levels=[0, 1], colors=["black"], zorder=3
        )
        # region outlines
        for region, col in {"offshore_central":"black","coastal_central":"black"}.items():
            m2 = model_regions_dict[region]["mask"]
            mp = xr.where(np.isnan(m2), 0, 1)
            ax.contour(
                lon_rho, lat_rho, mp,
                levels=[0.5], colors=[col], linewidths=1, zorder=4
            )

        ax.set_title(scenario, fontsize=16, color=title_colors[scenario])
        ax.set_xlim(lon_bounds); ax.set_ylim(lat_bounds)
        ax.set_xlabel("Longitude", fontsize=13)
        if i == 0:
            ax.set_ylabel("Latitude", fontsize=13)
        yt = np.arange(lat_bounds[0], lat_bounds[1]+1, 2)
        xt = np.arange(lon_bounds[0], lon_bounds[1]+1, 5)
        ax.set_yticks(yt); ax.set_xticks(xt)
        ax.set_yticklabels([f"{y}°N" for y in yt])
        ax.set_xticklabels([f"{360-x}°W" for x in xt])

    if baseline_type == "present_plus_meandelta":
        # ─────── UPDATED COLORBAR BLOCK ───────
        # 1) create axes for the colorbar
        cax = fig.add_axes([0.25, -0.04, 0.5, 0.02])

        # 2) draw it
        cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", extend=cf.cmap.name if hasattr(cf, 'cmap') else None)
        cbar.set_label( r"Threshold exceedance of [H$^+$] ($10^{-9}$ mol L$^{-1}$)",
        fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    if baseline_type == "present":
        # ─────── UPDATED COLORBAR BLOCK ───────
        # 1) create axes for the colorbar
        cax = fig.add_axes([0.25, -0.04, 0.5, 0.02])

        # 2) draw it
        cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", extend=cf.cmap.name if hasattr(cf, 'cmap') else None)
        cbar.set_label( r"Threshold exceedance of [H$^+$] ($10^{-10}$ mol L$^{-1}$)",
        fontsize=14)
        cbar.ax.tick_params(labelsize=12)


    # Save & show
    out = (
        f"/nfs/sea/work/fpfaeffli/plots/"
        f"extreme_days_per_year/"
        f"threshold_exceedance_comparison_{baseline_type}.png"
    )
    plt.savefig(out, dpi=150, transparent=True, bbox_inches="tight")
    plt.show()

# Plot threshold exceedance comparison for the fixed baseline
plot_threshold_exceedance_comparison(
    ocean_ds,
    landmask_etopo,
    model_regions_dict,
    hplus,
    thresholds_mult,
    baseline_type="present",
    config="romsoc_fully_coupled"
)

# for moving baseline
plot_threshold_exceedance_comparison(
    ocean_ds,
    landmask_etopo,
    model_regions_dict,
    hplus,
    thresholds_mult,
    baseline_type="present_plus_meandelta",
    config="romsoc_fully_coupled"
)

#%%
# Dummy colorbar
# Define 30 discrete levels between 1 and 10
levels = np.linspace(1, 10, 30)

# Create a BoundaryNorm so the colormap shows 30 bins
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmocean.cm.matter.N, clip=True)

# Dummy mappable
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmocean.cm.matter)
mappable.set_array(levels)

# Common label
label = r"Threshold exceedance of [H$^+$] ($10^{-10}$ mol L$^{-1}$)"

# Figure
fig = plt.figure(figsize=(6, 3))
# Fixed-baseline colorbar
cax1 = fig.add_axes([0.1, 0.7, 0.8, 0.02])  # Make the colorbar thinner vertically
cbar1 = plt.colorbar(
    mappable,
    cax=cax1,
    orientation="horizontal",
    extend="both",
    ticks=np.arange(1, 11)
)
cbar1.set_label(label, fontsize=12)
cbar1.ax.tick_params(labelsize=10)
plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/"
        f"extreme_days_per_year/"
        f"dummy_colorbar_threshold_exceedance_movec.png")
plt.show()


#%%
###############################
#     PLOTTING HISTOGRAM      #
###############################

#%%

# Histogram of extreme H+ values for each scenario under both baselines over the whole domain

######### Start plotting #########
# Prepare data containers
extreme_values_fixed = {}
extreme_values_moving = {}

# Define consistent bin edges across all data
bin_min = -0.3e-8
bin_max = 3e-8
bins = np.linspace(bin_min, bin_max, 51)  # 50 equal-width bins


# Collect data
for scenario in scenarios:
    # Fixed baseline
    mask_fixed = extreme_mask_dict['romsoc_fully_coupled'][scenario]['present'].values
    hplus_values = hplus['romsoc_fully_coupled'][scenario].values
    extreme_values_fixed[scenario] = hplus_values[mask_fixed]

    # Moving baseline
    mask_moving = extreme_mask_dict['romsoc_fully_coupled'][scenario]['present_plus_meandelta'].values
    extreme_values_moving[scenario] = hplus_values[mask_moving]

# Define x-axis limit
xmax = 2.5e-8

# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# === Fixed baseline ===
for scenario in scenarios:
    ax1.hist(
        extreme_values_fixed[scenario], bins=bins, alpha=0.6,
        label=scenario, density=True,
        color=colors[scenario], edgecolor='black'
    )
ax1.set_title('Fixed baseline', fontsize=15)
ax1.set_xlabel('[H$^+$] during extremes [mol/L]', fontsize=16)
ax1.set_ylabel('Density', fontsize=13)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.tick_params(axis='both', labelsize=14)
ax1.set_xlim(0, xmax)

# === Moving baseline ===
for scenario in scenarios:
    ax2.hist(
        extreme_values_moving[scenario], bins=bins, alpha=0.6,
        label=scenario, density=True,
        color=colors[scenario], edgecolor='black'
    )
ax2.set_title('Shifted baseline', fontsize=15)
ax2.set_xlabel('[H$^+$] during extremes [mol/L]', fontsize=16)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_xlim(0, xmax)

# Add horizontal legend below the plots
custom_handles = [
    Line2D([0], [0], color=colors['present'], lw=5, label='present'),
    Line2D([0], [0], color=colors['ssp245'], lw=5, label='ssp245'),
    Line2D([0], [0], color=colors['ssp585'], lw=5, label='ssp585'),
]

fig.legend(custom_handles, ['present', 'ssp245', 'ssp585'],
           loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=3, fontsize=12, frameon=False)


# Final layout and save
plt.suptitle('Distributions of extreme [H$^+$] under different baselines', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
output_path = "/nfs/sea/work/fpfaeffli/plots/statistics/histogram_extreme_hplus_values_fixed_vs_moving_baseline.png"
plt.savefig(output_path, dpi=200, transparent=True, bbox_inches="tight")
plt.show()


# %%
# =============================================
# Plot histograms of extreme [H⁺] by region
# =============================================

# Define distance-to-coast masks
config = 'romsoc_fully_coupled'
d2c = model_d2coasts['roms_only']
coastal_mask = (d2c >= 0) & (d2c < 100)
offshore_mask = (d2c >= 100) & (d2c <= 600)

# Define regions and setup containers
regions = ['coastal', 'offshore']
baseline_types = ['present', 'present_plus_meandelta']
extreme_values = {region: {baseline: {} for baseline in baseline_types} for region in regions}

# Define bins and max for x-axis
bin_min = 0e-8
bin_max = 3e-8
bins = np.linspace(bin_min, bin_max, 51)
xmax = 3e-8
xmin = 0e-8

# === Fill extreme_values ===
for scenario in scenarios:
    for region in regions:
        mask_region = coastal_mask if region == "coastal" else offshore_mask

        for baseline in baseline_types:
            mask_extreme = extreme_mask_dict['romsoc_fully_coupled'][scenario][baseline]
            hplus_vals = hplus['romsoc_fully_coupled'][scenario]

            # Combined condition
            combined_mask = mask_extreme & mask_region

            # Extract values
            extreme_vals = hplus_vals.where(combined_mask).values
            extreme_values[region][baseline][scenario] = extreme_vals[~np.isnan(extreme_vals)]


# === Plotting function ===
def plot_extreme_histograms(region):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot for fixed baseline
    for scenario in scenarios:
        ax1.hist(
            extreme_values[region]["present"][scenario], bins=bins, alpha=0.6,
            label=scenario, density=True,
            color=colors[scenario], edgecolor='black'
        )
    ax1.set_title(f"{region.capitalize()} – Fixed baseline", fontsize=15)
    ax1.set_xlabel('[H$^+$] during extremes [mol/L]', fontsize=16)
    ax1.set_ylabel('Density', fontsize=13)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlim(xmin, xmax)

    # Plot for moving baseline
    for scenario in scenarios:
        ax2.hist(
            extreme_values[region]["present_plus_meandelta"][scenario], bins=bins, alpha=0.6,
            label=scenario, density=True,
            color=colors[scenario], edgecolor='black'
        )
    ax2.set_title(f"{region.capitalize()} – Shifted baseline", fontsize=15)
    ax2.set_xlabel('[H$^+$] during extremes [mol/L]', fontsize=16)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlim(xmin, xmax)

    # Shared legend
    custom_handles = [
        Line2D([0], [0], color=colors['present'], lw=5, label='present'),
        Line2D([0], [0], color=colors['ssp245'], lw=5, label='ssp245'),
        Line2D([0], [0], color=colors['ssp585'], lw=5, label='ssp585'),
    ]

    fig.legend(custom_handles, ['present', 'ssp245', 'ssp585'],
               loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=3, fontsize=12, frameon=False)

    plt.suptitle(f'Distributions of extreme [H$^+$] – {region.capitalize()}', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/statistics/extreme_hplus_distribution_{region}.png", dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# === Run plots ===
plot_extreme_histograms("coastal")
plot_extreme_histograms("offshore")

#%% 

# Plots histograms of the anomalies


# Plotting histogram of H+ anomalies for fixed and moving baselines

# Define bins and max for x-axis
bin_min = 0e-8
bin_max = 3e-8
bins = np.linspace(bin_min, bin_max, 51)

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
titles = [
    '[H$^+$] - climatology (fixed baseline)',
    '[H$^+$] - climatology (shifted baseline)',
    '[H$^+$] - threshold (fixed baseline)',
    '[H$^+$] - threshold (shifted baseline)'
]
data_sources = [
    anomalies_clim_fixed,
    anomalies_clim_moving,
    anomalies_thresh_fixed,
    anomalies_thresh_moving
]

# Flatten axs to loop easily
axs = axs.flatten()

# Plot each panel
for i, (title, data_dict) in enumerate(zip(titles, data_sources)):
    ax = axs[i]
    for scenario in scenarios:
        ax.hist(
            data_dict[scenario], bins=bins, alpha=0.6,
            label=scenario, density=True,
            color=colors[scenario], edgecolor='black'
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('[H$^+$] anomaly [mol/L]', fontsize=14)
    if i in [0, 2]:  # left column
        ax.set_ylabel('Density', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=14)

# Add horizontal legend below the plots
custom_handles = [
    Line2D([0], [0], color=colors['present'], lw=5, label='present'),
    Line2D([0], [0], color=colors['ssp245'], lw=5, label='ssp245'),
    Line2D([0], [0], color=colors['ssp585'], lw=5, label='ssp585'),
]

fig.legend(custom_handles, ['present', 'ssp245', 'ssp585'],
           loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=3, fontsize=12, frameon=False)


# Final layout
plt.suptitle('Anomalies of [H$^+$] during extremes', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("/nfs/sea/work/fpfaeffli/plots/statistics/hplus_anomalies_histograms_both_clim_thresh.png", dpi=200, transparent=True, bbox_inches="tight")
plt.show()

# %%
# Define consistent bin edges across all data
bin_min = -0.1e-8
bin_max = 3e-8
bins = np.linspace(bin_min, bin_max, 31)  # 30 equal-width bins

# === Plot anomalies relative to climatology ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

titles = ['Fixed baseline', 'Shifted baseline']
data_sources = [anomalies_clim_fixed, anomalies_clim_moving]

for ax, title, data_dict in zip([ax1, ax2], titles, data_sources):
    for scenario in scenarios:
        ax.hist(
            data_dict[scenario], bins=bins, alpha=0.6,
            label=scenario, density=True,
            color=colors[scenario], edgecolor='black'
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('[H$^+$] anomaly [mol/L]', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim(-0.1e-8, 3e-8)  # Set x-axis limit

ax1.set_ylabel('Density', fontsize=12)

# Add horizontal legend below the plots
custom_handles = [
    Line2D([0], [0], color=colors['present'], lw=5, label='present'),
    Line2D([0], [0], color=colors['ssp245'], lw=5, label='ssp245'),
    Line2D([0], [0], color=colors['ssp585'], lw=5, label='ssp585'),
]

fig.legend(custom_handles, ['present', 'ssp245', 'ssp585'],
           loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=3, fontsize=12, frameon=False)

plt.suptitle('[H$^+$] anomalies ([H$^+$] minus climatology)', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("/nfs/sea/work/fpfaeffli/plots/statistics/hplus_histogram_anomaly_climatology_fixed_vs_shifted.png", dpi=200, transparent=True, bbox_inches="tight")
plt.show()



#%%

######################
# Grouped barcharts #
######################

# Grouped bar charts for the extreme [H+] values by region and baseline

scenarios      = ["present", "ssp245", "ssp585"]
colors         = {"present":"#ffab67","ssp245":"#65a342","ssp585":"#9553e0"}
baseline_keys  = ["present", "present_plus_meandelta"]
regions        = ["coastal", "offshore"]

# histogram bins for grouped‐bar
bin_min, bin_max = 0.3e-8, 3e-8
bins = np.linspace(bin_min, bin_max, 31)


# model_d2coasts["roms_only"] is a DataArray of distance‐to‐coast in km
d2c = model_d2coasts["roms_only"]
coastal_mask  = (d2c >=   0) & (d2c <  50)
offshore_mask = (d2c >=  50) & (d2c <=600)
region_masks  = {"coastal": coastal_mask, "offshore": offshore_mask}

config = "romsoc_fully_coupled"
extreme_values = {
    region: {bl: {} for bl in baseline_keys}
    for region in regions
}

# Extract extreme [H⁺] values for each region and baseline
for scenario in scenarios:
    for region in regions:
        mask_region = region_masks[region]
        for baseline in baseline_keys:
            mask_extreme = extreme_mask_dict[config][scenario][baseline]
            hplus_vals   = hplus[config][scenario]
            combined = mask_extreme & mask_region

            # pull out the values & drop NaNs
            raw = hplus_vals.where(combined).values
            extreme_values[region][baseline][scenario] = raw[~np.isnan(raw)]

# Function to plot grouped bar charts
def plot_grouped_bar(ax, data_dict, title):
    """
    ax         : matplotlib Axes to draw into
    data_dict  : dict scenario → 1D array of values
    title      : string for this subplot
    """
    centers = 0.5 * (bins[:-1] + bins[1:])
    bar_w   = (centers[1] - centers[0]) * 0.7 / len(scenarios)
    offsets = np.linspace(-bar_w, +bar_w, len(scenarios))
    
    for scen, off in zip(scenarios, offsets):
        hist, _ = np.histogram(data_dict[scen], bins=bins, density=True)
        ax.bar(centers + off, hist,
               width=bar_w,
               color=colors[scen],
               edgecolor="black",
               alpha=0.7,
               label=scen)
    
    ax.set_title(title, fontsize=13)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("[H$^+$] during extremes [mol/L]")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=10)


fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axes = axes.flatten()

for i, region in enumerate(regions):
    for j, baseline in enumerate(baseline_keys):
        ax    = axes[i*2 + j]
        data  = extreme_values[region][baseline]
        title = (f"{region.capitalize()} – Fixed baseline"
                 if baseline == "present"
                 else f"{region.capitalize()} – Shifted baseline")
        plot_grouped_bar(ax, data, title)

# shared legend below
handles = [Line2D([0], [0], color=colors[s], lw=6, label=s) for s in scenarios]
fig.legend(handles, scenarios,
           loc="lower center",
           bbox_to_anchor=(0.5, -0.02),
           ncol=3,
           frameon=False,
           fontsize=12)

plt.suptitle("Distribution of extreme [H$^+$] values by region & baseline",
             fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 1])

outpath = "/nfs/sea/work/fpfaeffli/plots/statistics/" + \
          "extreme_hplus_distributions_coastal_offshore_grouped_barcharts.png"
plt.savefig(outpath, dpi=200, transparent=True, bbox_inches="tight")
plt.show()



#%%
# Grouped bar charts for the anomalies

# === Manually adjustable bin edges ===
bins_clim_fixed    = np.linspace(-0.1e-8, 1.5e-8, 41)
bins_clim_shifted  = np.linspace(-0.05e-8, 0.4e-8, 41)   

bins_thresh_fixed   = np.linspace(-0.1e-8, 1.3e-8, 41)
bins_thresh_shifted = np.linspace(-0.05e-8, 0.15e-8, 41)

plot_configs = [
    {
        'data_dict': anomalies_clim_fixed,
        'bin_edges': bins_clim_fixed,
        'title': 'Fixed baseline',
        'relative': 'climatology',
        'fname': 'hplus_grouped_bars_anomaly_climatology_fixed.png'
    },
    {
        'data_dict': anomalies_clim_moving,
        'bin_edges': bins_clim_shifted,
        'title': 'Shifted baseline',
        'relative': 'climatology',
        'fname': 'hplus_grouped_bars_anomaly_climatology_shifted.png'
    },
    {
        'data_dict': anomalies_thresh_fixed,
        'bin_edges': bins_thresh_fixed,
        'title': 'Fixed baseline',
        'relative': 'threshold',
        'fname': 'hplus_grouped_bars_anomaly_threshold_fixed.png'
    },
    {
        'data_dict': anomalies_thresh_moving,
        'bin_edges': bins_thresh_shifted,
        'title': 'Shifted baseline',
        'relative': 'threshold',
        'fname': 'hplus_grouped_bars_anomaly_threshold_shifted.png'
    },
]

def plot_grouped(ax, data_dict, bin_edges, title):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_width   = 0.75 * (bin_centers[1] - bin_centers[0]) / 3
    offsets     = [-bar_width, 0, bar_width]

    for scenario, offset in zip(scenarios, offsets):
        hist, _ = np.histogram(data_dict[scenario], bins=bin_edges, density=True)
        ax.bar(bin_centers + offset, hist,
               width=bar_width, label=scenario,
               color=colors[scenario], alpha=0.7)

    ax.set_title(title, fontsize=14)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel('[H$^+$] anomaly', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)

for cfg in plot_configs:
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_grouped(ax, cfg['data_dict'], cfg['bin_edges'], cfg['title'])

    fig.legend(
        [Line2D([0], [0], color=colors[s], lw=6) for s in scenarios],
        scenarios,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(scenarios),
        fontsize=12,
        frameon=False
    )

    plt.suptitle(f"[H$^+$] anomalies relative to {cfg['relative']}", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/statistics/firstdraft_{cfg['fname']}",
                dpi=200, transparent=True)
    plt.show()



#%% Zoomed tail for the shifted baseline anomalies relative to the threshold

# === Zoom into tail for shifted baseline (threshold) ===
fig, ax = plt.subplots(figsize=(7, 5))

# Plot the grouped bars as before
plot_grouped(
    ax,
    anomalies_thresh_moving,
    bins_thresh_shifted,
    ''
)

# Set the zoom window
ax.set_xlim(0.75e-9, 1.5e-9)   # x from 0.74e-9 to 1.5e-9
ax.set_ylim(0.0e10, 0.05e10)     # y from 0e10 to 0.1e10

# Re‑draw legend if desired
from matplotlib.lines import Line2D
fig.legend(
    [Line2D([0], [0], color=colors[s], lw=6) for s in scenarios],
    scenarios,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=len(scenarios),
    fontsize=12,
    frameon=False
)

plt.suptitle("[H$^+$] anomalies — Shifted baseline (threshold) (zoomed)", fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/statistics/firstdraft_hplus_grouped_bars_anomaly_threshold_shifted_zoomed.png")
plt.show()


###############################
#         STAIR PLOTS         #
###############################

# %%
# Stair plots of extreme [H⁺] values by region and baseline

# histogram bins
bin_min, bin_max = 0.3e-8, 3e-8
bins = np.linspace(bin_min, bin_max, 31)

def plot_grouped_steps(ax, data_dict, title):
    """
    ax         : matplotlib Axes to draw into
    data_dict  : dict scenario → 1D array of values
    title      : string for this subplot
    """
    for scen in scenarios:
        # compute density histogram
        hist, edges = np.histogram(
            data_dict[scen],
            bins=bins,
            density=True
        )
        # draw stairs; edges has length len(bins), hist has len(bins)-1
        ax.stairs(
            hist,
            edges,
            label=scen,
            linewidth=2,
            color=colors[scen]
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("[H$^+$] during extremes [mol/L]")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=10)


# Example of how you'd call it in your figure setup:
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axes = axes.flatten()

for i, region in enumerate(regions):
    for j, baseline in enumerate(baseline_keys):
        ax    = axes[i*2 + j]
        data  = extreme_values[region][baseline]
        title = (f"{region.capitalize()} – Fixed baseline"
                 if baseline == "present"
                 else f"{region.capitalize()} – Shifted baseline")
        plot_grouped_steps(ax, data, title)

# shared legend below
fig.legend(
    scenarios,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    frameon=False,
    fontsize=12
)

plt.savefig("/nfs/sea/work/fpfaeffli/plots/statistics/extreme_hplus_distributions_coastal_offshore_stair_plots.png",
            dpi=200, transparent=True, bbox_inches="tight")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()






#%%
#######################################
#  MORE STATISTICS ON H⁺ VALUES       #
#######################################


#%%
 # Plot some more statistics on H⁺ values

config_key   = 'romsoc_fully_coupled'
scenarios    = ['present', 'ssp245', 'ssp585']
baseline_map = {'fixed':'present', 'moved':'present_plus_meandelta'}
colors = {
    'present':      "#ffab67",
    'ssp585_light': "#b98bf0", 'ssp585': "#9553e0", 'ssp585_dark':  "#6c2eb3",
    'ssp245_light': "#8ad764", 'ssp245': "#65a342", 'ssp245_dark':  "#3c7d1f"
}

def get_color(scenario, kind):
    """kind='light' or 'dark' or None for base"""
    key = f"{scenario}_{kind}" if kind else scenario
    return colors.get(key, colors[scenario])


# Extract surface‐only extreme and all H⁺ values 

data_extremes = {bl: {} for bl in baseline_map}
for bl, mask_key in baseline_map.items():
    print(f"\nExtracting extremes for '{bl}' baseline:")
    for s in scenarios:
        print(f"  • {s} …", end="")
        arr4d  = hplus[config_key][s].values
        mask4d = extreme_mask_dict[config_key][s][mask_key].values
        surf   = arr4d[:, 0, :, :]
        smask  = mask4d[:, 0, :, :]
        vals   = surf[smask]
        data_extremes[bl][s] = vals
        print(f"{vals.size} points")

data_all = {bl: {} for bl in baseline_map}
for bl, mask_key in baseline_map.items():
    print(f"\nExtracting ALL surface H⁺ for '{bl}' baseline:")
    for s in scenarios:
        arr4d = hplus[config_key][s].values
        surf   = arr4d[:, 0, :, :].ravel()    # flatten all times & lats/lons
        data_all[bl][s] = surf
        print(f"  • {s}: {surf.size} points")

#%%


# 1) Tail percentiles & means (60th … 99th)

pctiles    = [60, 65, 70, 75, 80, 85, 90, 95, 99]
cutoffs    = {bl: {s: [] for s in scenarios} for bl in baseline_map}
tail_means = {bl: {s: [] for s in scenarios} for bl in baseline_map}

for bl in baseline_map:
    for s in scenarios:
        arr = data_all[bl][s]
        for p in pctiles:
            c = np.percentile(arr, p, method='lower')
            cutoffs[bl][s].append(c)
            tail = arr[arr >= c]
            tail_means[bl][s].append(tail.mean() if tail.size>0 else np.nan)


# 2) Plotting

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
x = pctiles

for s in scenarios:
    # fixed baseline: light colors + circle markers
    ax1.plot( x, cutoffs['fixed'][s],   'o-', color=get_color(s,'light'), 
              label=f"{s} fixed")
    ax2.plot( x, tail_means['fixed'][s], 'o-', color=get_color(s,'light'),
              label=f"{s} fixed")
    # moved baseline: dark colors + square markers
    ax1.plot( x, cutoffs['moved'][s],   's--', color=get_color(s,'dark'),  
              label=f"{s} moved")
    ax2.plot( x, tail_means['moved'][s], 's--', color=get_color(s,'dark'),  
              label=f"{s} moved")

# Primary axes labels
ax1.set_title('H⁺ cutoff at various percentiles')
ax1.set_xlabel('Percentile (%)')
ax1.set_ylabel('H⁺ (mol L⁻¹)')
ax1.set_xticks(pctiles)

ax2.set_title('Mean H⁺ above percentile cutoff')
ax2.set_xlabel('Percentile (%)')
ax2.set_ylabel('H⁺ (mol L⁻¹)')
ax2.set_xticks(pctiles)

# Secondary pH axes
def H_to_pH(x): return -np.log10(x)
def pH_to_H(x): return 10**(-x)

sec1 = ax1.secondary_yaxis('right', functions=(H_to_pH, pH_to_H))
sec1.set_ylabel('pH')
sec1.set_ylim(H_to_pH(ax1.get_ylim()[0]), H_to_pH(ax1.get_ylim()[1]))

sec2 = ax2.secondary_yaxis('right', functions=(H_to_pH, pH_to_H))
sec2.set_ylabel('pH')
sec2.set_ylim(H_to_pH(ax2.get_ylim()[0]), H_to_pH(ax2.get_ylim()[1]))

# Shared legend

plt.suptitle("Tail percentiles of surface [H⁺]", y=1.03, fontsize=16)

# Save & show
out = "/nfs/sea/work/fpfaeffli/plots/statistics/tail_percentiles_all_hplus_and_pH.png"
plt.savefig(out, dpi=200, transparent=True)
plt.show()

#%%
# ——————————————————————————————————————————————————————————————
# 2) Violin distributions
# ——————————————————————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(6, 4))
positions = np.arange(len(scenarios))
w = 0.3
for i, s in enumerate(scenarios):
    # fixed
    vp_f = ax.violinplot([data_extremes['fixed'][s]],
                         positions=[positions[i] - w/2],
                         widths=w, showextrema=False)
    for pc in vp_f['bodies']:
        pc.set_facecolor(get_color(s, 'light'))
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    # moved
    vp_m = ax.violinplot([data_extremes['moved'][s]],
                         positions=[positions[i] + w/2],
                         widths=w, showextrema=False)
    for pc in vp_m['bodies']:
        pc.set_facecolor(get_color(s, 'dark'))
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

ax.set_xticks(positions)
ax.set_xticklabels(scenarios)
ax.set_ylabel('H⁺ [mol L⁻¹]')
ax.set_title('Distribution of Extreme H⁺')
plt.tight_layout()
plt.show()

#%%
# ——————————————————————————————————————————————————————————————
# 3) Empirical CDFs
# ——————————————————————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(6, 4))
for s in scenarios:
    # fixed
    vals_f = np.sort(data_extremes['fixed'][s])
    y_f    = np.linspace(0, 1, vals_f.size)
    ax.plot(vals_f, y_f, '-', color=get_color(s, 'light'), label=f"{s} fixed")
    # moved
    vals_m = np.sort(data_extremes['moved'][s])
    y_m    = np.linspace(0, 1, vals_m.size)
    ax.plot(vals_m, y_m, '--', color=get_color(s, 'dark'), label=f"{s} moved")

ax.set_xlabel('H⁺ [mol L⁻¹]')
ax.set_ylabel('Cumulative Probability')
ax.set_title('CDF of extreme [H⁺]')
ax.legend()
plt.tight_layout()
plt.show()


#%%

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Scenario order and colors
scenarios = ['ssp585', 'ssp245', 'present']
colors = {
    'present': "#ffab67",
    'ssp245':  "#65a342",
    'ssp585':  "#9553e0",
}

# Choose anomaly dictionaries
fixed_dict  = anomalies_clim_fixed
moving_dict = anomalies_clim_moving

#%%
# Plot setup
plt.figure(figsize=(9, 5))

for scenario in scenarios:
    for baseline_type, anomaly_dict, linestyle in [
        ('fixed',  fixed_dict,  '-'),
        ('shifted', moving_dict, '--')
    ]:
        values = anomaly_dict[scenario]
        values = values[~np.isnan(values)]

        if len(values) > 50:
            kde = gaussian_kde(values, bw_method='scott')
            x_grid = np.linspace(-0.4e-8, 0.4e-8, 500)
            label = f"{scenario.upper()} ({baseline_type})"
            plt.plot(x_grid, kde(x_grid), label=label, color=colors[scenario], linestyle=linestyle, linewidth=2)
        else:
            print(f"Skipping {scenario} {baseline_type} due to insufficient data")

# Labels and formatting
plt.xlabel("[H⁺] anomaly (mol/L)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.title("PDF of extreme [H⁺] anomalies to climatology", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(fontsize=10, title="scenario + baseline", loc="best")
plt.xlim(-0.4e-8, 0.4e-8)
plt.tight_layout()

# Save or show
plt.savefig("/nfs/sea/work/fpfaeffli/plots/statistics/KDE_both_baselines_zoomed.png",
            dpi=200, transparent=True)
plt.show()



#%%
####################################
#           TIMESERIES             #
####################################

#%%
# Plot pH over time (coastal vs. offshore)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Scenario color dictionary
colors = {
    'present': "#ffab67",
    'ssp585':  "#9553e0",
    'ssp245':  "#65a342"
}

# Define distance‐to‐coast masks (0–50 km = coastal, 50–600 km = offshore)
config = 'romsoc_fully_coupled'
d2c = model_d2coasts['roms_only']  # DataArray (eta_rho, xi_rho)
coastal_mask  = (d2c >=   0) & (d2c <  50)
offshore_mask = (d2c >=  50) & (d2c <=600)

# Initialize figure
fig, ax = plt.subplots(figsize=(10, 5))

for scenario in ['present', 'ssp245', 'ssp585']:
    print(f"Processing scenario: {scenario}")
    
    # Extract surface pH DataArray and drop depth dimension
    ph = ocean_ds[config][scenario]['pH_offl'].isel(depth=0)
    
    # Compute spatial mean over each region
    ph_coastal  = ph.where(coastal_mask).mean(dim=('eta_rho','xi_rho'), skipna=True)
    ph_offshore = ph.where(offshore_mask).mean(dim=('eta_rho','xi_rho'), skipna=True)
    
    # Plot time series: solid for coastal, dashed for offshore
    ax.plot(ph_coastal.time,  ph_coastal,  color=colors[scenario],
            linestyle='-',  label=f"{scenario} – coastal")
    ax.plot(ph_offshore.time, ph_offshore, color=colors[scenario],
            linestyle='--', label=f"{scenario} – offshore")

# Formatting
ax.set_title("Mean Surface pH Over Time\ncoastal (0–50 km) vs offshore (50–600 km)", fontsize=14)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("pH", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.3)

# Yearly ticks
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Legend outside plot
ax.legend(title="Scenario & Region",
          bbox_to_anchor=(1.02, 1), loc='upper left',
          borderaxespad=0., fontsize=10)

plt.tight_layout()
plt.savefig(
    "/nfs/sea/work/fpfaeffli/plots/timeseries_variables/"
    "timeseries_pH_coastal50_offshore50_all_scenarios.png",
    dpi=200, transparent=True, bbox_inches="tight"
)
plt.show()



# %%
# %%

# Plot setup
fig, ax = plt.subplots(figsize=(10, 5))
colors = {
    'present': "#ffab67",
    'ssp585': "#9553e0",
    'ssp245': "#65a342"
}

for scenario in ['present', 'ssp245', 'ssp585']:
    print(f"Processing scenario: {scenario}")
    ph = ocean_ds[config][scenario]['pH_offl'].isel(depth=0)

    # Apply box masks
    ph_coastal_box = ph.where(coastal_box_mask).mean(dim=('eta_rho', 'xi_rho'), skipna=True)
    ph_offshore_box = ph.where(offshore_box_mask).mean(dim=('eta_rho', 'xi_rho'), skipna=True)

    # Plot both
    ax.plot(ph_coastal_box.time, ph_coastal_box, label=f"{scenario} – coastal box", color=colors[scenario], linestyle='-')
    ax.plot(ph_offshore_box.time, ph_offshore_box, label=f"{scenario} – offshore box", color=colors[scenario], linestyle='--')

# Format
ax.set_title("Mean Surface pH Over Time\n10×10 Box: Coastal vs Offshore (Central Region)")
ax.set_xlabel("Year")
ax.set_ylabel("pH")
ax.grid(True)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Legend outside
ax.legend(
    title="Scenario & Region",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.
)

plt.tight_layout()
plt.show()

# %%


# === SETTINGS ===
config = 'romsoc_fully_coupled'
threshold_type = 'present_plus_meandelta'  # or 'present'
scenario = 'ssp585'  # or 'ssp245', 'present'
box_size = 10
colors = {
    'hplus': 'black',
    'clim': 'blue',
    'thresh': 'red',
    'extreme': 'darkred'
}

# === GRID INFO ===
d2c = model_d2coasts['roms_only']
coastal_mask = (d2c >= 0) & (d2c < 100)
offshore_mask = (d2c >= 100) & (d2c <= 500)

def get_center_index(mask):
    i_inds, j_inds = np.where(mask)
    return i_inds[len(i_inds)//2], j_inds[len(j_inds)//2]

def box_mask(center_i, center_j, box_size, shape):
    mask = np.full(shape, False)
    mask[center_i:center_i+box_size, center_j:center_j+box_size] = True
    return xr.DataArray(mask, dims=('eta_rho', 'xi_rho'))

# === Define coastal/offshore boxes ===
i_c, j_c = get_center_index(coastal_mask.values)
i_o, j_o = get_center_index(offshore_mask.values)
coastal_box = box_mask(i_c, j_c, box_size, d2c.shape)
offshore_box = box_mask(i_o, j_o, box_size, d2c.shape)

# === Define function to plot for a box ===
def plot_timeseries_Hplus_components(mask, label, ax):
    # Extract components
    hplus_da = hplus[config][scenario].where(mask).mean(dim=('eta_rho', 'xi_rho'), skipna=True)
    clim_da = hplus_clim[config][scenario].where(mask).mean(dim=('eta_rho', 'xi_rho'), skipna=True)
    thresh_da = thresholds_mult[config][scenario][threshold_type].where(mask).mean(dim=('eta_rho', 'xi_rho'), skipna=True)
    mask_extreme = extreme_mask_dict[config][scenario][threshold_type].where(mask).mean(dim=('eta_rho', 'xi_rho')) > 0.5

    # Plot
    ax.plot(hplus_da.time, hplus_da, label=f'{label}: H⁺', color=colors['hplus'])
    ax.plot(clim_da.time, clim_da, label=f'{label}: Climatology', color=colors['clim'], linestyle='--')
    ax.plot(thresh_da.time, thresh_da, label=f'{label}: Threshold', color=colors['thresh'], linestyle=':')
    ax.scatter(
        hplus_da.time[mask_extreme],
        hplus_da[mask_extreme],
        color=colors['extreme'],
        s=15,
        label=f'{label}: Extreme'
    )

# === PLOT ===
fig, ax = plt.subplots(figsize=(12, 6))

plot_timeseries_Hplus_components(coastal_box, 'Coastal', ax)
plot_timeseries_Hplus_components(offshore_box, 'Offshore', ax)

# === Formatting ===
ax.set_title(f"[H⁺] Time Series with Climatology and Threshold\nScenario: {scenario}, Threshold: {threshold_type}")
ax.set_xlabel("Year")
ax.set_ylabel("[H⁺] [mol/L]")
ax.grid(True)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# === Legend ===
ax.legend(title="Component & Region", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %%
# Plotting timeseries of H⁺ with climatology and threshold for coastal vs offshore regions

# ——— Shared setup ———
config = 'romsoc_fully_coupled'
scenarios = ['present', 'ssp245', 'ssp585']
threshold_types = ['present', 'present_plus_meandelta']
box_size = 10

colors = {
    'hplus': 'black',
    'clim': 'cornflowerblue',
    'thresh': 'darkred',
}

# Distance-to-coast
d2c = model_d2coasts['roms_only']
coastal_mask = (d2c >= 0) & (d2c < 50)
offshore_mask = (d2c >= 50) & (d2c <= 500)

# Helpers
def get_center_index(mask):
    i_inds, j_inds = np.where(mask)
    return i_inds[len(i_inds)//2], j_inds[len(j_inds)//2]

def box_mask(center_i, center_j, box_size, shape):
    mask = np.full(shape, False)
    mask[center_i:center_i+box_size, center_j:center_j+box_size] = True
    return xr.DataArray(mask, dims=('eta_rho', 'xi_rho'))

# Define central boxes
i_c, j_c = get_center_index(coastal_mask.values)
i_o, j_o = get_center_index(offshore_mask.values)
coastal_box = box_mask(i_c, j_c, box_size, d2c.shape)
offshore_box = box_mask(i_o, j_o, box_size, d2c.shape)

def plot_single_timeseries(mask, box_label, scenario, threshold_type):
    hplus_da = hplus[config][scenario].where(mask).mean(dim=('eta_rho', 'xi_rho'), skipna=True)
    if threshold_type == 'present':
        clim_da = hplus_clim[config]['present'].where(mask).mean(dim=('eta_rho', 'xi_rho'), skipna=True)
    else:
        clim_da = hplus_clim[config][scenario].where(mask).mean(dim=('eta_rho', 'xi_rho'), skipna=True)

    thresh_da = thresholds_mult[config][scenario][threshold_type].where(mask).mean(dim=('eta_rho', 'xi_rho'), skipna=True)

    # Make sure all are 1D NumPy arrays
    time = hplus_da.time.values
    hplus_vals = hplus_da.values.squeeze()
    thresh_vals = thresh_da.values.squeeze()
    exceed_mask = hplus_vals > thresh_vals

    fig, ax = plt.subplots(figsize=(12, 5))

    # Fill exceedance area
    ax.fill_between(
        time,
        hplus_vals,
        thresh_vals,
        where=exceed_mask,
        facecolor='khaki',
        alpha=0.7,
        interpolate=True,
        label='[H⁺] > threshold'
    )

    # Plot lines
    ax.plot(time, hplus_vals, label="[H⁺]", color=colors['hplus'])
    ax.plot(time, clim_da.values.squeeze(), label="climatology", color=colors['clim'], linestyle='-')
    ax.plot(time, thresh_vals, label='threshold (95$^{\\mathrm{th}}$ percentile)', color=colors['thresh'], linestyle='-')

    # Formatting
    ax.set_title(f"[H⁺] timeseries ({box_label}) \nscenario: {scenario}, threshold: {threshold_type}")
    ax.set_xlabel("Year")
    ax.set_xlim(np.datetime64('2011-01-01'), np.datetime64('2022-01-01'))
    ax.set_ylabel("[H⁺] [mol/L]")
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(
        bbox_to_anchor=(1.05, 0.5),
        loc='center left',
        borderaxespad=0.
    )
    plt.tight_layout()
    plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/timeseries_variables/timeseries_hplus_clim_thresh_{box_label}_{scenario}_{threshold_type}_500km_offshore.png", dpi=200, transparent=True, bbox_inches="tight")
    plt.show()


# ——— Loop over scenarios and thresholds ———
for scenario in scenarios:
    for threshold_type in threshold_types:
        plot_single_timeseries(coastal_box, "coastal", scenario, threshold_type)
        plot_single_timeseries(offshore_box, "offshore", scenario, threshold_type)


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ——— PARAMETERS (assumes extreme_values & scenarios already in memory) ———
regions       = ['coastal', 'offshore']
baseline_keys = ['present', 'present_plus_meandelta']
scenarios     = ['present', 'ssp245', 'ssp585']
colors = {
    'present': '#ffab67',
    'ssp245':  '#65a342',
    'ssp585':  '#9553e0'
}

# Bins for histogram smoothing
bin_min, bin_max = 0.3e-8, 3e-8
hist_bins = np.linspace(bin_min, bin_max, 50)
centers   = 0.5*(hist_bins[:-1] + hist_bins[1:])

# ——— Plot one region, overlay fixed vs shifted ———
for region in regions:
    fig, ax = plt.subplots(figsize=(8, 5))

    for scen in scenarios:
        for baseline in baseline_keys:
            data = extreme_values[region][baseline][scen]
            # fast smooth: histogram + Gaussian blur
            hist, _ = np.histogram(data, bins=hist_bins, density=True)
            smooth = gaussian_filter1d(hist, sigma=2)

            # linestyle: solid for fixed, dashed for shifted
            ls = '-' if baseline == 'present' else '--'
            label = f"{scen} ({'fixed' if baseline=='present' else 'shifted'})"

            ax.plot(centers, smooth,
                    label=label,
                    color=colors[scen],
                    linestyle=ls,
                    linewidth=2)

    ax.set_title(f"{region.capitalize()} extremes: fixed vs shifted baseline", fontsize=14)
    ax.set_xlim(bin_min, bin_max)
    ax.set_xlabel('[H⁺] during extremes [mol/L]')
    ax.set_ylabel('Density')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=10)

    # legend outside
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=10,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/statistics/extreme_hplus_{region}_fixed_vs_shifted.png", dpi=200, transparent=True, bbox_inches="tight")
    plt.show()


#%%

# Plotting timeseries of h+ with climatology and threshold for CCE1 and CCE2

# 1) Station definitions + region labels
def convert_lon(lon):
    return lon + 360 if lon < 0 else lon

stations = {
    "CCE1": (33.4300, convert_lon(-122.4800)),
    "CCE2": (34.3240, convert_lon(-120.8150)),
}
region_labels = {
    "CCE1": "offshore",
    "CCE2": "coastal",
}

# 2) Model config, scenarios & thresholds
config          = "romsoc_fully_coupled"
scenarios       = ["present", "ssp245", "ssp585"]
all_thresholds  = ["present", "present_plus_meandelta"]

# 3) Grid lats/lons (static 2-D)
ds0   = ocean_ds[config][scenarios[0]]
lat2d = ds0["lat"].values    # (eta_rho, xi_rho)
lon2d = ds0["lon"].values

# 4) Nearest-cell finder
def find_ij_nearest(lat_arr, lon_arr, plat, plon):
    dist2 = (lat_arr - plat)**2 + (lon_arr - plon)**2
    flat  = dist2.argmin()
    return np.unravel_index(flat, lat_arr.shape)

station_idx = {
    name: find_ij_nearest(lat2d, lon2d, plat, plon)
    for name, (plat, plon) in stations.items()
}

# 5) Labels & colors
colors = {"hplus":"darkgrey", "clim":"cornflowerblue", "thresh":"darkred"}
threshold_labels = {
    "present": "fixed baseline",
    "present_plus_meandelta": "shifted baseline"
}

# 6) Plotting routine
def plot_point_timeseries(name, i0, j0, scenario, threshold_type):
    # extract data
    hpt = hplus[config][scenario].isel(eta_rho=i0, xi_rho=j0)
    if threshold_type == "present":
        cpt = hplus_clim[config]["present"].isel(eta_rho=i0, xi_rho=j0)
    else:
        cpt = hplus_clim[config][scenario].isel(eta_rho=i0, xi_rho=j0)
    tpt = thresholds_mult[config][scenario][threshold_type] \
              .isel(eta_rho=i0, xi_rho=j0)

    # to numpy
    time_vals = hpt.time.values
    hvals     = hpt.values.squeeze()
    cvals     = cpt.values.squeeze()
    tvals     = tpt.values.squeeze()
    exceed    = hvals > tvals

    # labels
    region          = region_labels[name]
    thresh_label    = threshold_labels[threshold_type]

    # make plot
    fig, ax = plt.subplots(figsize=(12,5))
    ax.fill_between(time_vals, hvals, tvals,
                    where=exceed, facecolor='khaki', alpha=0.7,
                    interpolate=True,
                    label=f"[H⁺] > {thresh_label}")
    ax.plot(time_vals, hvals, color=colors['hplus'], label='[H⁺]')
    ax.plot(time_vals, cvals, color=colors['clim'],   label='climatology')
    ax.plot(time_vals, tvals, color=colors['thresh'],
            label=f"threshold ({thresh_label})")

    ax.set_title(f"{name} ({region}) — {scenario}, {thresh_label}")
    ax.set_xlim(
        datetime.datetime(2011, 1, 1),
        datetime.datetime(2022, 1, 1)
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("[H⁺] [mol/L]")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05,0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/statistics/CCE/extreme_hplus_{name}_{region}_{scenario}_{thresh_label}.png", dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# 7) Loop: only "present" threshold for the present scenario
for name, (i0, j0) in station_idx.items():
    for scen in scenarios:
        thresholds = ["present"] if scen == "present" else all_thresholds
        for thr in thresholds:
            plot_point_timeseries(name, i0, j0, scen, thr)

# %%

# %%
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr

# 1) Station defs + regions (as before)
def convert_lon(lon):
    return lon + 360 if lon < 0 else lon

stations = {
    "CCE1": (33.4300, convert_lon(-122.4800)),
    "CCE2": (34.3240, convert_lon(-120.8150)),
}
region_labels = {"CCE1":"offshore","CCE2":"coastal"}

# 2) Config, scenarios, thresholds (as before)
config         = "romsoc_fully_coupled"
scenarios      = ["present","ssp245","ssp585"]
all_thresholds = ["present","present_plus_meandelta"]

# 3) Build station indices (as before)
ds0    = ocean_ds[config][scenarios[0]]
lat2d  = ds0["lat"].values
lon2d  = ds0["lon"].values

def find_ij(lat_arr, lon_arr, plat, plon):
    dist2 = (lat_arr-plat)**2 + (lon_arr-plon)**2
    idx   = dist2.argmin()
    return np.unravel_index(idx, lat_arr.shape)

station_idx = {
    name: find_ij(lat2d, lon2d, plat, plon)
    for name,(plat,plon) in stations.items()
}

# 4) Colors & labels
colors = {
    "hplus":        "darkgrey",
    "present_clim": "cornflowerblue",
    "scen_clim":    "orange",
    "thresh":       "darkred",
}
threshold_labels = {
    "present":                "fixed baseline",
    "present_plus_meandelta": "shifted baseline",
}

# 5) Plotting routine (fixed color keys)
def plot_point_timeseries(name, i0, j0, scenario, thr_key):
    # DataArrays
    hda = hplus[config][scenario].isel(eta_rho=i0, xi_rho=j0)
    pcd = hplus_clim[config]["present"].isel(eta_rho=i0, xi_rho=j0)
    scd = hplus_clim[config][scenario].isel(eta_rho=i0, xi_rho=j0)
    tda = thresholds_mult[config][scenario][thr_key].isel(eta_rho=i0, xi_rho=j0)

    # align to common time
    hda, pcd, scd, tda = xr.align(hda, pcd, scd, tda, join="inner")

    time = hda.time.values
    hval = hda.values.squeeze()
    pval = pcd.values.squeeze()
    sval = scd.values.squeeze()
    tval = tda.values.squeeze()
    exceed = hval > tval

    region   = region_labels[name]
    thr_lbl  = threshold_labels[thr_key]

    fig, ax = plt.subplots(figsize=(12,5))

    # exceedance fill
    ax.fill_between(time, hval, tval,
                    where=exceed, facecolor="khaki", alpha=0.7,
                    interpolate=True,
                    label=f"[H⁺] > {thr_lbl}")

    # timeseries
    ax.plot(time, hval, color=colors["hplus"],        label="[H⁺]")
    ax.plot(time, pval, color=colors["present_clim"], label="present climatology")
    ax.plot(time, sval, color=colors["scen_clim"], 
            linestyle="--", label=f"{scenario} climatology")
    ax.plot(time, tval, color=colors["thresh"],      label=f"Threshold ({thr_lbl})")

    # formatting
    ax.set_title(f"{name} ({region}) — {scenario}, {thr_lbl}", fontsize=14)
    ax.set_xlim(datetime.datetime(2011,1,1), datetime.datetime(2022,1,1))
    ax.set_xlabel("Year")
    ax.set_ylabel("[H⁺] (mol L⁻¹)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05,0.5), loc="center left")

    plt.tight_layout()
    out = (
        f"/nfs/sea/work/fpfaeffli/plots/statistics/CCE/"
        f"extreme_hplus_{name}_{region}_{scenario}_{thr_lbl.replace(' ','_')}.png"
    )
    plt.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# 6) Loop
for name,(i0,j0) in station_idx.items():
    for scen in scenarios:
        thr_list = ["present"] if scen=="present" else all_thresholds
        for thr in thr_list:
            plot_point_timeseries(name, i0, j0, scen, thr)













# %%
# timeseries with present and future clim / thresh
from datetime import datetime

# 1) Define your point
target_lat = 38.5
target_lon = 235
config     = "romsoc_fully_coupled"
ds0        = ocean_ds[config]["present"]
lat2d      = ds0["lat"].values
lon2d      = ds0["lon"].values
idx_flat   = ((lat2d - target_lat)**2 + (lon2d - target_lon)**2).argmin()
i0, j0     = np.unravel_index(idx_flat, lat2d.shape)

# 2) Color definitions—using cornflowerblue and darkred
colors = {
    "hplus":        "grey",
    "present_clim": "cornflowerblue",
    "present_thr":  "cornflowerblue",
    "future_clim":  "firebrick",
    "future_thr":   "firebrick",
    "fill_pres":    "cornflowerblue",
    "fill_future":  "firebrick"
}

def plot_timeseries_baseline_methods(i0, j0):
    scenario = "ssp245"

    # extract DataArrays
    hda      = hplus[config][scenario].isel(eta_rho=i0, xi_rho=j0)
    clim_pres= hplus_clim[config]["present"].isel(eta_rho=i0, xi_rho=j0)
    clim_ssp = hplus_clim[config][scenario].isel(eta_rho=i0, xi_rho=j0)
    thr_pres = thresholds_mult[config][scenario]["present"].isel(eta_rho=i0, xi_rho=j0)
    thr_mov  = thresholds_mult[config][scenario]["present_plus_meandelta"].isel(eta_rho=i0, xi_rho=j0)

    # align times
    hda, clim_pres, clim_ssp, thr_pres, thr_mov = xr.align(
        hda, clim_pres, clim_ssp, thr_pres, thr_mov, join="inner"
    )

    t = hda.time.values
    h = hda.values.squeeze()
    p = thr_pres.values.squeeze()
    f = thr_mov.values.squeeze()

    fig, ax = plt.subplots(figsize=(12,5))



    # 3) climatologies & thresholds
    ax.plot(t, clim_pres.values.squeeze(),
            color=colors["present_clim"],
            linewidth=1,
            label="present climatology")
    ax.plot(t, clim_ssp.values.squeeze(),
            color=colors["future_clim"],
            linewidth=1,
            label=f'{scenario} climatology')
    ax.plot(t, p,
            color=colors["present_thr"],
            linestyle="--",
            linewidth=1,
            label="present threshold")
    ax.plot(t, f,
            color=colors["future_thr"],
            linestyle="--",
            linewidth=1,
            label=f'{scenario} threshold')
    
    # 1) H+ line
    ax.plot(t, h, color=colors["hplus"], label=f'{scenario} [H⁺]', linewidth=1)

    # 2) exceedance shading
    mask_pres = h > p
    ax.fill_between(t, h, p,
                    where=mask_pres,
                    facecolor=colors["fill_pres"],
                    edgecolor="none",
                    alpha=0.4,
                    label="fixed baseline extremes")
    mask_fut = h > f
    ax.fill_between(t, h, f,
                    where=mask_fut,
                    facecolor=colors["fill_future"],
                    edgecolor="none",
                    alpha=0.4,
                    label="shifted baseline extremes")

    import datetime

    # define your highlight window
    start = datetime.datetime(2013, 8, 1)
    end   = datetime.datetime(2015, 8, 1, 23, 59)

    # draw a coral line on the BOTTOM spine between start and end
    ax.plot(
        [start, end], [0, 0],
        transform=ax.get_xaxis_transform(),   # y=0 in axis coords is the bottom spine
        color='coral',
        linewidth=3,
        clip_on=False
    )

    # and if you also want the TOP spine segment in coral, plot at y=1 in axis coords
    ax.plot(
        [start, end], [1, 1],
        transform=ax.get_xaxis_transform(),   # y=1 in axis coords is the top spine
        color='coral',
        linewidth=3,
        clip_on=False
    )


    # formatting
    ax.set_xlim(datetime.datetime(2011,1,1), datetime.datetime(2022,1,1))
    ax.set_xlabel("Year")
    ax.set_ylabel("[H⁺] (mol L⁻¹)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True)
    ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=4,
    frameon=False
    )


    plt.tight_layout()
    plt.savefig(
        f'/nfs/sea/work/fpfaeffli/plots/statistics/aa_finished_timseries_{scenario}_hplus_comparison_coastal_point_{i0}_{j0}.png',
        dpi=200, transparent=True, bbox_inches="tight"
    )
    plt.show()

# call it
plot_timeseries_baseline_methods(i0, j0)

# %%
# Zoomed in for negative anomalies
# 1) Define your point
target_lat = 38.5
target_lon = 235
config     = "romsoc_fully_coupled"
ds0        = ocean_ds[config]["present"]
lat2d      = ds0["lat"].values
lon2d      = ds0["lon"].values
idx_flat   = ((lat2d - target_lat)**2 + (lon2d - target_lon)**2).argmin()
i0, j0     = np.unravel_index(idx_flat, lat2d.shape)

def plot_ssp585_zoom_sep2011_to_sep2012(i0, j0):
    scenario = "ssp245"

    # extract DataArrays
    hda       = hplus[config][scenario].isel(eta_rho=i0, xi_rho=j0)
    clim_pres = hplus_clim[config]["present"].isel(eta_rho=i0, xi_rho=j0)
    thr_pres  = thresholds_mult[config][scenario]["present"].isel(eta_rho=i0, xi_rho=j0)
    thr_mov   = thresholds_mult[config][scenario]["present_plus_meandelta"].isel(eta_rho=i0, xi_rho=j0)

    # align to common time
    hda, clim_pres, clim_ssp, thr_pres, thr_mov = xr.align(
        hda, clim_pres, clim_ssp, thr_pres, thr_mov, join="inner"
    )

    t = hda.time.values
    h = hda.values.squeeze()
    p = thr_pres.values.squeeze()
    f = thr_mov.values.squeeze()

    fig, ax = plt.subplots(figsize=(8,4))

    # raw H+
    ax.plot(t, h, color=colors["hplus"], label=f'{scenario} [H⁺]', lw=1)

    # shaded extremes
    ax.fill_between(t, h, p, where=(h>p),
                    facecolor=colors["fill_pres"], alpha=0.4,
                    label="fixed baseline extremes")
    ax.fill_between(t, h, f, where=(h>f),
                    facecolor=colors["fill_ssp585"], alpha=0.4,
                    label="shifted baseline extremes")

    # climatologies & thresholds
    ax.plot(t, clim_pres.values.squeeze(),
            color=colors["present_clim"], lw=1, label="present climatology")
    ax.plot(t, p, color=colors["present_thr"], ls="--", lw=1, label="present threshold")
    ax.plot(t, f, color=colors["ssp585_thr"], ls="--", lw=1, label="ssp585 threshold")

    # zoom window: 
    ax.set_xlim(datetime.datetime(2011,10,1), datetime.datetime(2012,8,30,23,59))
    ax.set_ylim(1.9e-8, 2.1e-8)

    # formatting: monthly ticks at every month
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    ax.set_xlabel("")
    ax.set_ylabel("[H⁺] (mol L⁻¹)")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(
        "/nfs/sea/work/fpfaeffli/plots/statistics/aa_finished_timeseries_ssp585_hplus_zoomed.png",
        dpi=200, transparent=True, bbox_inches="tight"
    )
    plt.show()

# call it
plot_ssp585_zoom_sep2011_to_sep2012(i0, j0)

# %%

# Make a zoomed in version of ssp245 timeseris 
import datetime

# 1) Define your point
target_lat = 38.5
target_lon = 235
config     = "romsoc_fully_coupled"
ds0        = ocean_ds[config]["present"]
lat2d      = ds0["lat"].values
lon2d      = ds0["lon"].values
idx_flat   = ((lat2d - target_lat)**2 + (lon2d - target_lon)**2).argmin()
i0, j0     = np.unravel_index(idx_flat, lat2d.shape)

colors = {
    "hplus":        "grey",
    "present_clim": "cornflowerblue",
    "present_thr":  "cornflowerblue",
    "future_clim":  "firebrick",
    "future_thr":   "firebrick",
    "fill_pres":    "cornflowerblue",
    "fill_future":  "firebrick"
}

def plot_timeseries_zoom_2013_to_2015(i0, j0):
    scenario = "ssp245"

    # extract DataArrays
    hda       = hplus[config][scenario].isel(eta_rho=i0, xi_rho=j0)
    clim_pres = hplus_clim[config]["present"].isel(eta_rho=i0, xi_rho=j0)
    thr_pres  = thresholds_mult[config][scenario]["present"].isel(eta_rho=i0, xi_rho=j0)
    thr_mov   = thresholds_mult[config][scenario]["present_plus_meandelta"].isel(eta_rho=i0, xi_rho=j0)

    # align to common time
    hda, clim_pres, thr_pres, thr_mov = xr.align(
        hda, clim_pres, thr_pres, thr_mov, join="inner"
    )

    t = hda.time.values
    h = hda.values.squeeze()
    p = thr_pres.values.squeeze()
    f = thr_mov.values.squeeze()

    fig, ax = plt.subplots(figsize=(8,4))
    fig.subplots_adjust(bottom=0.20)  # room for legend

    # raw H+ line
    ax.plot(t, h, color=colors["hplus"], label=f'{scenario} [H⁺]', lw=1)

    # exceedance shading
    ax.fill_between(t, h, p, where=(h>p),
                    facecolor=colors["fill_pres"], alpha=0.4,
                    label="fixed baseline extremes")
    ax.fill_between(t, h, f, where=(h>f),
                    facecolor=colors["fill_future"], alpha=0.4,
                    label="shifted baseline extremes")

    # climatologies & thresholds
    ax.plot(t, clim_pres.values.squeeze(),
            color=colors["present_clim"], lw=1, label="present climatology")
    ax.plot(t, p, color=colors["present_thr"], ls="--", lw=1, label="present threshold")
    ax.plot(t, f, color=colors["future_thr"], ls="--", lw=1, label=f'{scenario} threshold')

    # zoom window: Jan 1, 2013 to Dec 31, 2015
    ax.set_xlim(
        datetime.datetime(2013, 8, 1),
        datetime.datetime(2015, 8, 1, 23, 59)
    )
    # (optional) tighten y‑limits around the data
    ylim_min = np.nanmin([h, p, f]) * 0.98
    ylim_max = np.nanmax([h, p, f]) * 1.02
    ax.set_ylim(ylim_min, ylim_max)

    # formatting: monthly ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlabel("")
    ax.set_ylabel("[H⁺] (mol L⁻¹)")
    ax.grid(True)



    plt.tight_layout()
    plt.savefig(
        f"/nfs/sea/work/fpfaeffli/plots/statistics/aa_finished_zoomed_timeseries_{scenario}_2013_2015_point_{i0}_{j0}.png",
        dpi=200, transparent=True, bbox_inches="tight"
    )
    plt.show()

# call it
plot_timeseries_zoom_2013_to_2015(i0, j0)








# Your color palette
colors = {
    'present':      "#ffab67",     # warm orange
    'ssp585_light': "#b98bf0",     # lighter purple
    'ssp585':       "#9553e0",     # original
    'ssp585_dark':  "#6c2eb3",     # darker purple
    'ssp245_light': "#8ad764",     # lighter green
    'ssp245':       "#65a342",     # original
    'ssp245_dark':  "#3c7d1f"      # darker green
}

# === Manually adjustable bin edges (just for defining x‐ranges) ===
bins_clim_fixed    = np.linspace(-0.1e-8, 1.5e-8, 41)
bins_clim_shifted  = np.linspace(-0.05e-8, 0.4e-8, 41)   
bins_thresh_fixed   = np.linspace(-0.1e-8, 1.3e-8, 41)
bins_thresh_shifted = np.linspace(-0.05e-8, 0.15e-8, 41)

plot_configs = [
    {
        'data_dict': anomalies_clim_fixed,
        'bin_edges': bins_clim_fixed,
        'title': 'Fixed baseline (climatology)',
        'relative': 'climatology'
    },
    {
        'data_dict': anomalies_clim_moving,
        'bin_edges': bins_clim_shifted,
        'title': 'Shifted baseline (climatology)',
        'relative': 'climatology'
    },
    {
        'data_dict': anomalies_thresh_fixed,
        'bin_edges': bins_thresh_fixed,
        'title': 'Fixed baseline (threshold)',
        'relative': 'threshold'
    },
    {
        'data_dict': anomalies_thresh_moving,
        'bin_edges': bins_thresh_shifted,
        'title': 'Shifted baseline (threshold)',
        'relative': 'threshold'
    },
]
for cfg in plot_configs:
    print(f"Plotting KDE for: {cfg['title']} (relative to {cfg['relative']})")
    fig, ax = plt.subplots(figsize=(7, 5))

    # set x_min according to fixed vs. shifted baseline
    if 'Fixed' in cfg['title']:
        x_min = -0.25e-8
    else:
        x_min = -0.25e-9
    x_max = cfg['bin_edges'][-1]
    x_vals = np.linspace(x_min, x_max, 300)

    # plot one KDE per scenario
    for scen in scenarios:
        print(f"  Scenario: {scen} — data size: {len(cfg['data_dict'][scen])}")
        data = cfg['data_dict'][scen]
        kde = gaussian_kde(data)
        ax.plot(
            x_vals,
            kde(x_vals),
            label=scen,
            color=colors[scen],
            lw=2
        )

    # Formatting
    ax.set_title(f"[H$^+$] anomalies relative to {cfg['relative']} — {cfg['title']}",
                 fontsize=14, pad=18)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('[H$^+$] anomaly', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.grid(linestyle='--', alpha=0.7)
    ax.tick_params(labelsize=10)

    # Legend centered below
    fig.legend(
        [Line2D([0], [0], color=colors[s], lw=2) for s in scenarios],
        scenarios,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(scenarios),
        fontsize=12,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fn = cfg['title'].replace(' ', '_').lower()
    print(f"  Saving to: /nfs/sea/work/fpfaeffli/plots/statistics/KDE_hplus_anomalies_{cfg['relative']}_{fn}.png")
    plt.savefig(f"/nfs/sea/work/fpfaeffli/plots/statistics/KDE_hplus_anomalies_{cfg['relative']}_{fn}.png")
    plt.show()


#%%
# only the legend

# %%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Define your color mapping
colors = {
    "hplus":        "grey",
    "present_clim": "cornflowerblue",
    "present_thr":  "cornflowerblue",
    "future_clim":  "firebrick",
    "future_thr":   "firebrick",
    "fill_pres":    "cornflowerblue",
    "fill_future":  "firebrick"
}

# Create legend handles (one per entry)
handles = [
    Line2D([0], [0], color=colors["present_clim"], lw=1, label="present climatology"),
    Line2D([0], [0], color=colors["present_thr"], linestyle="--", lw=1, label="present threshold"),
    Line2D([0], [0], color=colors["future_thr"], linestyle="--", lw=1, label="ssp245 threshold"),
    Line2D([0], [0], color=colors["hplus"], lw=1, label="ssp245 [H⁺]"),
    Patch(facecolor=colors["fill_pres"], edgecolor='none', alpha=0.4, label="fixed baseline extremes"),
    Patch(facecolor=colors["fill_future"], edgecolor='none', alpha=0.4, label="adjusted baseline extremes")
]

labels = [h.get_label() for h in handles]

# Create a figure just for the legend
fig = plt.figure(figsize=(4, 6))
# Add the legend in the center, one item per row
fig.legend(
    handles,
    labels,
    loc='center',
    ncol=1,
    frameon=True,
    fancybox=True,
    shadow=False,
    labelspacing=1.2  # increase space between lines if desired
)
plt.axis('off')
plt.tight_layout()
plt.savefig(
    "/nfs/sea/work/fpfaeffli/plots/statistics/legend_hplus_timeseries.png",
    dpi=200, transparent=True, bbox_inches="tight"
)
plt.show()

# %%
