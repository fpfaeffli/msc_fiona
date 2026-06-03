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
    lat_bounds   = [25, 50]
    lon_bounds   = [225, 245]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5), constrained_layout=True)

    # define contour levels between 7.5 and 8.5
    levels = np.linspace(7.5, 8.1, 30)

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
            zorder=1
        )

        # land mask
        ax.contourf(
            landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
            levels=[0, 1], colors=["black"], zorder=3
        )

        # region contours
        for region, col in {
            "offshore_central":"saddlebrown",
            "coastal_central":"coral"
        }.items():
            mask2d = model_regions_dict[region]["mask"]
            mask_plot = xr.where(np.isnan(mask2d), 0, 1)
            ax.contour(
                lon_rho, lat_rho, mask_plot,
                levels=[0.5], colors=[col], linewidths=2, zorder=4
            )

        ax.set_title(scenario, fontsize=16, color=title_colors[scenario])
        ax.set_xlim(lon_bounds); ax.set_ylim(lat_bounds)
        ax.set_xlabel("Longitude", fontsize=12)
        if i == 0:
            ax.set_ylabel("Latitude", fontsize=12)
        yt = np.arange(lat_bounds[0], lat_bounds[1]+1, 5)
        xt = np.arange(lon_bounds[0], lon_bounds[1]+1, 5)
        ax.set_yticks(yt); ax.set_xticks(xt)
        ax.set_yticklabels([f"{y}°N" for y in yt])
        ax.set_xticklabels([f"{360-x}°W" for x in xt])

    # colorbar, fixed to 7.5–8.5
    cax = fig.add_axes([0.25, -0.04, 0.5, 0.02])
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label("Mean surface pH", fontsize=14)
    cbar.set_ticks(np.linspace(7.5, 8.1, 11))
    cbar.ax.tick_params(labelsize=12)

    out = "/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/surface_pH_7p5_to_8p5_comparison.png"
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
    lat_bounds = [25, 50]
    lon_bounds = [225, 245]
    
    # Create a single figure with 3 subplots (side by side)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5), constrained_layout=True)
    
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
            'offshore_central': 'saddlebrown',
            'coastal_central': 'coral'
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
                    linewidths=2,
                    zorder=4
                )

        
        
        # Set labels and titles
        ax.set_title(f"{scenario}", fontsize=16, color=title_colors[scenario])
        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_xlabel("Longitude", fontsize=12)
        if i == 0:  # Only set ylabel for first subplo
            ax.set_ylabel("Latitude", fontsize=12)
        
        # Format latitude & longitude labels
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
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
###############################
#     PLOTTING DIST2COAST     #
###############################

#%%
# Distance to coast plots with the bin approach

def plot_extreme_days_vs_distance(
    ocean_ds,
    model_d2coasts,
    extreme_mask_dict,
    model_regions_dict,
    config="romsoc_fully_coupled"
):
    """
    Plots mean extreme H+ days/year vs. distance to the coast for
    two baselines, but *only* inside coastal_central and offshore_central.
    Each region gets its own set of curves.
    """
    scenarios      = ["present", "ssp245", "ssp585"]
    colors         = {"present":"#ffab67","ssp245":"#65a342","ssp585":"#9553e0"}
    bins_d2coast   = np.arange(0,600,10)
    regions        = ["coastal_central","offshore_central"]
    region_colors  = {"coastal_central":"coral","offshore_central":"saddlebrown"}
    baseline_types = ["present","present_plus_meandelta"]
    titles         = ["fixed baseline","shifted baseline"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
    
    for ax, baseline, title in zip(axes, baseline_types, titles):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Distance to coast (km)")
        ax.set_ylabel("Extreme [H$^+$] days/year")
        ax.set_xlim(0,600)
        ax.axvline(50, color="k", linestyle="--", lw=1.5)
        ax.grid(linestyle="--", alpha=0.3)
        
        for scenario in scenarios:
            # pull the mask‐array for this scenario & baseline
            data = extreme_mask_dict[config][scenario][baseline]
            total_steps = data.sizes["time"]
            yearly = (data.sum(dim="time")/total_steps*365).squeeze()
            
            # pull the d2coast array once
            d2c = model_d2coasts["roms_only"]
            
            for region in regions:
                # get the region mask
                rmask = model_regions_dict[region]["mask"]
                
                # flatten & mask
                ext_flat = yearly.values.ravel()
                d2c_flat = d2c.values.ravel()
                reg_flat = rmask.values.ravel().astype(bool)
                
                good = (~np.isnan(ext_flat)) & (~np.isnan(d2c_flat)) & reg_flat
                ext_flat = ext_flat[good]
                d2c_flat = d2c_flat[good]
                
                # debug: how many points?
                print(f"{baseline} / {scenario} / {region}: {ext_flat.size} points")
                
                # bin stats
                medians, p25, p75 = [], [], []
                for i in range(len(bins_d2coast)-1):
                    inbin = (d2c_flat>=bins_d2coast[i])&(d2c_flat<bins_d2coast[i+1])
                    if inbin.any():
                        medians.append(np.median(ext_flat[inbin]))
                        p25.append(np.percentile(ext_flat[inbin],25))
                        p75.append(np.percentile(ext_flat[inbin],75))
                    else:
                        medians.append(np.nan); p25.append(np.nan); p75.append(np.nan)
                
                # plot
                label = f"{scenario.upper()} – {region.split('_')[0]}"
                ax.plot(bins_d2coast[:-1], medians,
                        color=colors[scenario],
                        linestyle="-" if region=="coastal_central" else "--",
                        lw=2, label=label)
                ax.fill_between(bins_d2coast[:-1], p75, p25,
                                facecolor=colors[scenario], alpha=0.2)
                
        # custom x‐axis color bar
        ax.spines['bottom'].set_visible(False)
        ax.plot([0,50],[0,0], color="coral", lw=5,
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
        ax.plot([50,600],[0,0], color="saddlebrown", lw=5,
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
    
    # single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10)
    
    out = "/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/" + \
          "dist2coast_central_CaCS_both_baselines.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
    plt.show()


# --- call it ---
plot_extreme_days_vs_distance(
    ocean_ds=ocean_ds,
    model_d2coasts=model_d2coasts,
    extreme_mask_dict=extreme_mask_dict,
    model_regions_dict=model_regions_dict,
    config="romsoc_fully_coupled"
)

#%%

def plot_yaxis_cut_fixed_baseline(
    extreme_mask_dict,
    model_d2coasts,
    model_regions_dict,
    config="romsoc_fully_coupled"
):
    """
    Distance-to-coast for fixed baseline, broken y-axis, but only 
    inside coastal_central & offshore_central. Separate curves for each region.
    """
    scenarios     = ["present", "ssp245", "ssp585"]
    colors        = {"present":"#ffab67","ssp245":"#65a342","ssp585":"#9553e0"}
    bins_d2coast  = np.arange(0,600,10)
    regions       = ["coastal_central","offshore_central"]
    linestyles    = {"coastal_central":"-","offshore_central":"--"}
    region_labels = {"coastal_central":"coastal","offshore_central":"offshore"}

    # set up broken-y figure
    fig = plt.figure(figsize=(5.5,4))
    gs = gridspec.GridSpec(2,1, height_ratios=[1,1], hspace=0.25)
    ax_top    = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
    

    # loop over scenarios & regions
    for scenario in scenarios:
        mask = extreme_mask_dict[config][scenario]["present"]
        total_steps = mask.sizes["time"]
        yearly = (mask.sum(dim="time")/total_steps*365)

        # flatten once
        ext_flat_full = yearly.values.ravel()
        d2c_flat_full = model_d2coasts["roms_only"].values.ravel()

        for region in regions:
            rmask_flat = model_regions_dict[region]["mask"].values.ravel().astype(bool)

            # apply nan + region mask
            good = (~np.isnan(ext_flat_full)) & (~np.isnan(d2c_flat_full)) & rmask_flat
            ext_flat = ext_flat_full[good]
            d2c_flat = d2c_flat_full[good]

            print(f"{scenario} / {region}: {ext_flat.size} points")

            # bin stats
            medians, p25s, p75s = [], [], []
            for b in range(len(bins_d2coast)-1):
                inbin = (d2c_flat>=bins_d2coast[b])&(d2c_flat<bins_d2coast[b+1])
                if inbin.any():
                    medians.append(np.median(ext_flat[inbin]))
                    p25s.append(np.percentile(ext_flat[inbin],25))
                    p75s.append(np.percentile(ext_flat[inbin],75))
                else:
                    medians.append(np.nan); p25s.append(np.nan); p75s.append(np.nan)

            x = bins_d2coast[:-1] + 5

            # plot onto both axes
            for ax in (ax_top, ax_bottom):
                ax.plot(
                    x, medians,
                    color=colors[scenario],
                    linestyle=linestyles[region],
                    lw=2,
                    label=f"{scenario.upper()} – {region_labels[region]}"
                )
                ax.fill_between(
                    x, p25s, p75s,
                    facecolor=colors[scenario],
                    alpha=0.2
                )

    # set y-limits for the broken axes
    ax_top.set_ylim(300, 370)
    ax_bottom.set_ylim(0, 25)
    ax_bottom.set_yticks(np.arange(0, 26, 5))
    ax_bottom.set_xlim(0, 600)

    # Add vertical dashed line at x=50 km in both axes
    for ax in (ax_top, ax_bottom):
        ax.axvline(50, color="k", linestyle="--", lw=1.5)

    # hide spines & ticks where they overlap
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False)
    ax_bottom.xaxis.tick_bottom()

    # labels
    ax_bottom.set_xlabel("Distance to coast (km)", fontsize=11)
    fig.text(0.01, 0.5, "Extreme [H$^+$] days/year",
             va='center', rotation='vertical', fontsize=11)
    fig.suptitle("Fixed baseline", fontsize=14, y=0.94)

    # coral/saddlebrown x-axis coloring on the bottom panel
    for ax in (ax_bottom,):
        ax.plot([0, 50], [0, 0], color='coral', lw=3,
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=10)
        ax.plot([50, 600], [0, 0], color='saddlebrown', lw=3,
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=10)

    # diagonal “break” marks
    kwargs = dict(marker=[(-1, -1), (1, 1)], markersize=12,
                  linestyle='none', color='k', mec='k', mew=1, clip_on=False)
    ax_top.plot([0,1],[0,0], transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot([0,1],[1,1], transform=ax_bottom.transAxes, **kwargs)

    ax_top.grid(True, linestyle='--', alpha=0.3)
    ax_bottom.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0,0.05,1,1])
    out = "/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/" + \
          "dist2coast_fixed_baseline_broken_yaxis.png"
    plt.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()


# --- call it ---
plot_yaxis_cut_fixed_baseline(
    extreme_mask_dict=extreme_mask_dict,
    model_d2coasts=model_d2coasts,
    model_regions_dict=model_regions_dict,
    config="romsoc_fully_coupled"
)

#%%

def plot_shifted_baseline(extreme_mask_dict, model_d2coasts, config="romsoc_fully_coupled"):
    scenarios = ["present", "ssp245", "ssp585"]
    colors = {"present": "#ffab67", "ssp245": "#65a342", "ssp585": "#9553e0"}
    bins_d2coast = np.arange(0, 600, 10)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))

    for scenario in scenarios:
        mask = extreme_mask_dict[config][scenario]["present_plus_meandelta"]
        total_time_steps = mask.shape[0]
        extreme_days = (mask.sum(dim="time") / total_time_steps * 365).values.flatten()
        d2coast = model_d2coasts["roms_only"].values.flatten()

        valid = ~np.isnan(extreme_days) & ~np.isnan(d2coast)
        d2coast = d2coast[valid]
        extreme_days = extreme_days[valid]

        medians, p25s, p75s = [], [], []
        for b in range(len(bins_d2coast) - 1):
            cond = (d2coast >= bins_d2coast[b]) & (d2coast < bins_d2coast[b + 1])
            vals = extreme_days[cond]
            if vals.size > 0:
                medians.append(np.median(vals))
                p25s.append(np.percentile(vals, 25))
                p75s.append(np.percentile(vals, 75))
            else:
                medians.append(np.nan)
                p25s.append(np.nan)
                p75s.append(np.nan)

        x = bins_d2coast[:-1] + 5

        ax.plot(x, medians, color=colors[scenario], lw=2)
        ax.fill_between(x, p25s, p75s, color=colors[scenario], alpha=0.4)

    ax.set_xlim(0, 600)
    ax.set_xlabel("Distance to coast (km)", fontsize=11)
    ax.set_ylabel("Extreme [H$^+$] days/year", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Coral/saddlebrown x-axis coloring
    ax.spines['bottom'].set_visible(False)
    ax.plot([0, 50], [0, 0], color='coral', lw=3,
            transform=ax.get_xaxis_transform(), clip_on=False)
    ax.plot([50, 600], [0, 0], color='saddlebrown', lw=3,
            transform=ax.get_xaxis_transform(), clip_on=False)
    ax.axvline(50, color="k", linestyle="--", lw=1.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=colors["present"], lw=2, label="present"),
        Line2D([0], [0], color=colors["ssp245"], lw=2, label="ssp245"),
        Line2D([0], [0], color=colors["ssp585"], lw=2, label="ssp585"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=10)
    fig.suptitle("shifted baseline", fontsize=14, y=0.94)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/dist2coast_shifted_baseline.png", dpi=200, transparent=True)
    plt.show()


# Call the function
plot_shifted_baseline(extreme_mask_dict, model_d2coasts)

# %%
# Present and two future scenarios in different subplots

def plot_single_baseline(extreme_mask_dict, model_d2coasts, config, scenarios, baseline_type, title, filename):
    colors = {"present": "#ffab67", "ssp245": "#65a342", "ssp585": "#9553e0"}
    bins_d2coast = np.arange(0, 500, 10)
    
    fig, ax = plt.subplots(figsize=(4, 6))
    
    for scenario in scenarios:
        mask = extreme_mask_dict[config][scenario][baseline_type]
        total_time_steps = mask.shape[0]
        extreme_days = (mask.sum(dim="time") / total_time_steps * 365).values.flatten()
        d2coast = model_d2coasts["roms_only"].values.flatten()
        
        valid = ~np.isnan(extreme_days) & ~np.isnan(d2coast)
        d2coast = d2coast[valid]
        extreme_days = extreme_days[valid]
        
        medians, p25s, p75s = [], [], []
        for b in range(len(bins_d2coast) - 1):
            cond = (d2coast >= bins_d2coast[b]) & (d2coast < bins_d2coast[b + 1])
            if np.any(cond):
                medians.append(np.median(extreme_days[cond]))
                p25s.append(np.percentile(extreme_days[cond], 25))
                p75s.append(np.percentile(extreme_days[cond], 75))
            else:
                medians.append(np.nan)
                p25s.append(np.nan)
                p75s.append(np.nan)
        
        x = bins_d2coast[:-1]
        ax.plot(x, medians, color=colors[scenario], lw=2, label=scenario.upper())
        ax.fill_between(x, p25s, p75s, color=colors[scenario], alpha=0.5)

    ax.set_xlim([0, 500])
    ax.grid(linestyle="--", alpha=0.25)
    ax.axvline(100, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Distance to coast (km)", fontsize=10)
    ax.set_ylabel('Extreme [H$^+$] days/year', fontsize=10)
    ax.set_title(title, fontsize=14)
    
    ax.spines['bottom'].set_visible(False)
    ax.plot([0, 100], [0, 0], color='coral', lw=3,
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=10)
    ax.plot([100, 500], [0, 0], color='saddlebrown', lw=3,
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=10)

    legend_elements = [Line2D([0], [0], color=colors[s], lw=2, label=s) for s in scenarios]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.3),  # Move further down
        ncol=len(scenarios),
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(filename, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# Call functions

plot_single_baseline(
    extreme_mask_dict, model_d2coasts, config,
    scenarios=["present"],
    baseline_type="present",
    title="present",
    filename="/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/d2coast_present_fixed.png"
)

plot_single_baseline(
    extreme_mask_dict, model_d2coasts, config,
    scenarios=["ssp245", "ssp585"],
    baseline_type="present",
    title="future (fixed baseline)",
    filename="/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/d2coast_future_fixed.png"
)

plot_single_baseline(
    extreme_mask_dict, model_d2coasts, config,
    scenarios=["ssp245", "ssp585"],
    baseline_type="present_plus_meandelta",
    title="future (shifted baseline)",
    filename="/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/d2coast_future_shifted.png"
)




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

import pprint

print("\n>>> DUMPING extreme_values:")
pprint.pprint(extreme_values)

print("\n>>> scenarios =", scenarios)
print(">>> regions   =", list(extreme_values.keys()))
print(">>> baseline types =", list(next(iter(extreme_values.values())).keys()))



# === Run plots ===
plot_extreme_histograms("coastal")
plot_extreme_histograms("offshore")


#%%

#######################################################################
# Grouped barcharts for extreme H+ distributions in different regions #
#######################################################################
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pprint

# =============================================
# 1) PARAMETERS & SETUP
# =============================================

scenarios      = ["present", "ssp245", "ssp585"]
colors         = {"present":"#ffab67","ssp245":"#65a342","ssp585":"#9553e0"}
baseline_keys  = ["present", "present_plus_meandelta"]
regions        = ["coastal", "offshore"]

# histogram bins for grouped‐bar
bin_min, bin_max = 0.3e-8, 3e-8
bins = np.linspace(bin_min, bin_max, 31)

# =============================================
# 2) BUILD COASTAL / OFFSHORE MASKS (0–50 km / 50–600 km)
# =============================================
# model_d2coasts["roms_only"] is a DataArray of distance‐to‐coast in km
d2c = model_d2coasts["roms_only"]
coastal_mask  = (d2c >=   0) & (d2c <  50)
offshore_mask = (d2c >=  50) & (d2c <=600)
region_masks  = {"coastal": coastal_mask, "offshore": offshore_mask}

# =============================================
# 3) EXTRACT EXTREME‐[H+] VALUES
# =============================================
# extreme_mask_dict[config][scenario][baseline] → boolean DataArray
# hplus[config][scenario]                 → DataArray of [H+] values

config = "romsoc_fully_coupled"
extreme_values = {
    region: {bl: {} for bl in baseline_keys}
    for region in regions
}

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

# debug: show counts per combo
print("\n>>> extreme_values counts:")
pprint.pprint({
    r: {
        bl: {s: len(extreme_values[r][bl][s]) for s in scenarios}
        for bl in baseline_keys
    }
    for r in regions
})

# =============================================
# 4) GROUPED‐BAR PLOTTING FUNCTION
# =============================================
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

# =============================================
# 5) BUILD 2×2 FIGURE WITH GROUPED BARS
# =============================================
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


# %%

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
# Grouped bar charts

# Define consistent bin edges
bins_clim = np.linspace(-0.01e-8, 1.5e-8, 41)
bins_thresh_fixed = np.linspace(-0.01e-8, 1.3e-8, 41)
bins_thresh_shifted = np.linspace(-0.01e-8, 0.3e-8, 41)

# Function to plot grouped bar charts
def plot_grouped(ax, data_dict, bin_edges, title, xlim):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_width = 0.75 * (bin_centers[1] - bin_centers[0]) / 3

    offsets = [-bar_width, 0, bar_width]

    for scenario, offset in zip(scenarios, offsets):
        data = data_dict[scenario]
        hist, _ = np.histogram(data, bins=bin_edges, density=True)
        ax.bar(bin_centers + offset, hist, width=bar_width,
               label=scenario, color=colors[scenario], alpha=0.7)

    ax.set_title(title, fontsize=14)
    ax.set_xlim(xlim)
    ax.set_xlabel('[H$^+$] anomaly', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)

# === Plot anomalies relative to climatology ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

plot_grouped(ax1, anomalies_clim_fixed, bins_clim, 'Fixed baseline', (-0.01e-8, 1.5e-8))
plot_grouped(ax2, anomalies_clim_moving, bins_clim, 'Shifted baseline', (-0.01e-8, 1.5e-8))

ax1.set_ylabel('Density', fontsize=12)
fig.legend(
    [Line2D([0], [0], color=colors[s], lw=6) for s in scenarios], scenarios,
    loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12, frameon=False
)
plt.suptitle('[H$^+$] anomalies ([H$^+$] minus climatology)', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("/nfs/sea/work/fpfaeffli/plots/statistics/hplus_grouped_bars_anomaly_climatology_fixed_vs_shifted.png", dpi=200, transparent=True)
plt.show()

#%%
# === Plot anomalies relative to climatology ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

plot_grouped(ax1, anomalies_thresh_fixed, bins_thresh_fixed, 'Fixed baseline', (-0.01e-8, 1.3e-8))
plot_grouped(ax2, anomalies_thresh_moving, bins_thresh_shifted, 'Shifted baseline', (-0.01e-8, 0.3e-8))

ax1.set_ylabel('Density', fontsize=12)
fig.legend(
    [Line2D([0], [0], color=colors[s], lw=6) for s in scenarios], scenarios,
    loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12, frameon=False
)
plt.suptitle('[H$^+$] anomalies ([H$^+$] minus threshold)', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("/nfs/sea/work/fpfaeffli/plots/statistics/hplus_grouped_bars_anomaly_threshold_fixed_vs_shifted.png", dpi=200, transparent=True)
plt.show()


#%%

#####################################################
# Grouped bar charts for anomalies for both regions #
#####################################################

# %%

# ——— PARAMETERS ———
scenarios = ['present','ssp245','ssp585']
colors = {
    'present': "#ffab67",
    'ssp245':  "#65a342",
    'ssp585':  "#9553e0"
}

# Bin edges for each plot
bins_clim_fixed   = np.linspace(-0.01e-8, 1.5e-8, 41)
bins_clim_mov     = np.linspace(-0.01e-8, 0.5e-8, 41)
bins_thresh_fixed = np.linspace(-0.01e-8, 1.3e-8, 41)
bins_thresh_mov   = np.linspace(-0.01e-8, 0.3e-8, 41)

def plot_grouped(ax, data_dict, bin_edges, title, xlim):
    """
    ax        : one Axes
    data_dict : dict scenario → 1D array of anomaly values
    bin_edges : array of edges
    title     : string
    xlim      : tuple (xmin, xmax)
    """
    centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    bar_w   = 0.75*(centers[1]-centers[0])/len(scenarios)
    offsets = np.linspace(-bar_w, +bar_w, len(scenarios))

    for scen, off in zip(scenarios, offsets):
        vals, _ = np.histogram(data_dict[scen], bins=bin_edges, density=True)
        ax.bar(centers + off, vals,
               width=bar_w,
               color=colors[scen],
               alpha=0.7,
               label=scen,
               edgecolor='black')
    ax.set_title(title, fontsize=14)
    ax.set_xlim(xlim)
    ax.set_xlabel('[H$^+$] anomaly', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(labelsize=10)


# ——— 1) Climatology‐based anomalies ———
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

plot_grouped(
    ax1,
    anomalies_clim_fixed,
    bins_clim_fixed,
    'Fixed baseline',
    (bins_clim[0], bins_clim[-1])
)
plot_grouped(
    ax2,
    anomalies_clim_moving,
    bins_clim_mov,
    'Shifted baseline',
    (bins_clim[0], bins_clim[-1])
)

ax1.set_ylabel('Density', fontsize=12)
fig.legend(
    [Line2D([0], [0], color=colors[s], lw=6) for s in scenarios],
    scenarios,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=12,
    frameon=False
)
plt.suptitle(
    '[H$^+$] anomalies (minus climatology)',
    fontsize=16,
    y=1.02
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(
    "/nfs/sea/work/fpfaeffli/plots/statistics/"
    "hplus_grouped_bars_anomaly_climatology_fixed_vs_shifted.png",
    dpi=200,
    transparent=True,
    bbox_inches="tight"
)
plt.show()


# ——— 2) Threshold‐based anomalies ———
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

plot_grouped(
    ax3,
    anomalies_thresh_fixed,
    bins_thresh_fixed,
    'Fixed baseline',
    (bins_thresh_fixed[0], bins_thresh_fixed[-1])
)
plot_grouped(
    ax4,
    anomalies_thresh_moving,
    bins_thresh_mov,
    'Shifted baseline',
    (bins_thresh_mov[0], bins_thresh_mov[-1])
)

ax3.set_ylabel('Density', fontsize=12)
fig.legend(
    [Line2D([0], [0], color=colors[s], lw=6) for s in scenarios],
    scenarios,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=12,
    frameon=False
)
plt.suptitle(
    '[H$^+$] anomalies (minus threshold)',
    fontsize=16,
    y=1.02
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(
    "/nfs/sea/work/fpfaeffli/plots/statistics/"
    "hplus_grouped_bars_anomaly_threshold_fixed_vs_shifted.png",
    dpi=200,
    transparent=True,
    bbox_inches="tight"
)
plt.show()



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
coastal_mask = (d2c >= 0) & (d2c < 100)
offshore_mask = (d2c >= 100) & (d2c <= 500)

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
# ssp585 timeseries with present and future clim / thresh

# 1) Define your point
target_lat = 37.0
target_lon = 235.0
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
    "ssp585_clim":  "firebrick",
    "ssp585_thr":   "firebrick",
    "fill_pres":    "cornflowerblue",
    "fill_ssp585":  "firebrick"
}

def plot_ssp585_compare_with_shading(i0, j0):
    scenario = "ssp585"

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

    # 1) H+ line
    ax.plot(t, h, color=colors["hplus"], label="ssp585 [H⁺]", linewidth=1)

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
                    facecolor=colors["fill_ssp585"],
                    edgecolor="none",
                    alpha=0.4,
                    label="shifted baseline extremes")

    # 3) climatologies & thresholds
    ax.plot(t, clim_pres.values.squeeze(),
            color=colors["present_clim"],
            linewidth=1,
            label="present climatology")
    ax.plot(t, clim_ssp.values.squeeze(),
            color=colors["ssp585_clim"],
            linewidth=1,
            label="ssp585 climatology")
    ax.plot(t, p,
            color=colors["present_thr"],
            linestyle="--",
            linewidth=1,
            label="present threshold")
    ax.plot(t, f,
            color=colors["ssp585_thr"],
            linestyle="--",
            linewidth=1,
            label="ssp585 threshold")

    # formatting
    ax.set_xlim(datetime.datetime(2011,1,1), datetime.datetime(2022,1,1))
    ax.set_xlabel("Year")
    ax.set_ylabel("[H⁺] (mol L⁻¹)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.02,0.5), loc="center left")

    plt.tight_layout()
    plt.savefig(
        "/nfs/sea/work/fpfaeffli/plots/statistics/timseries_ssp585_hplus_comparison.png",
        dpi=200, transparent=True, bbox_inches="tight"
    )
    plt.show()

# call it
plot_ssp585_compare_with_shading(i0, j0)

# %%
# Zoomed in for negative anomalies

def plot_ssp585_zoom_sep2011_to_sep2012(i0, j0):
    scenario = "ssp585"

    # extract DataArrays
    hda       = hplus[config][scenario].isel(eta_rho=i0, xi_rho=j0)
    clim_pres = hplus_clim[config]["present"].isel(eta_rho=i0, xi_rho=j0)
    clim_ssp  = hplus_clim[config][scenario].isel(eta_rho=i0, xi_rho=j0)
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
    ax.plot(t, h, color=colors["hplus"], label="SSP585 [H⁺]", lw=1)

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
    ax.plot(t, clim_ssp.values.squeeze(),
            color=colors["ssp585_clim"], lw=1, label="ssp585 climatology")
    ax.plot(t, f, color=colors["ssp585_thr"], ls="--", lw=1, label="ssp585 threshold")

    # zoom window: Oct 1, 2011 to Aug 30, 2012
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
        "/nfs/sea/work/fpfaeffli/plots/statistics/timeseries_ssp585_hplus_zoomed.png",
        dpi=200, transparent=True, bbox_inches="tight"
    )
    plt.show()

# call it
plot_ssp585_zoom_sep2011_to_sep2012(i0, j0)

# %%
