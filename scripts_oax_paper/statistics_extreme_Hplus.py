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

# Plots mean surface pH comparison of the three scenarios:


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

########################################
#         timeseries for methods       #   
########################################


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
# Plot from paper (Figure 2)
# Make a zoomed in version of ssp245 timeseries 
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
    "hplus":        "black",
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
    ax.plot(t, h, color=colors["hplus"], label=f'{scenario} [H⁺]', lw=1.5)

    # exceedance shading
    ax.fill_between(t, h, p, where=(h>p),
                    facecolor=colors["fill_pres"], alpha=0.4,
                    label="fixed baseline extremes")
    ax.fill_between(t, h, f, where=(h>f),
                    facecolor=colors["fill_future"], alpha=0.4,
                    label="shifted baseline extremes")

    # climatologies & thresholds
    ax.plot(t, clim_pres.values.squeeze(),
            color=colors["present_clim"], lw=1, ls="--", label="present climatology")
    ax.plot(t, p, color=colors["present_thr"], lw=1.5, label="present threshold")
    ax.plot(t, f, color=colors["future_thr"], lw=1.5, label=f'{scenario} threshold')

    # zoom window: Jan 1, 2013 to Dec 31, 2015
    ax.set_xlim(
        datetime.datetime(2013, 8, 1),
        datetime.datetime(2015, 8, 1, 23, 59)
    )
    # (optional) tighten y‑limits around the data
    ylim_min = np.nanmin([h, p, f]) * 0.98
    ylim_max = np.nanmax([h, p, f]) * 1.07
    ax.set_ylim(ylim_min, ylim_max)

    # formatting: monthly ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("[H⁺] (mol L⁻¹)", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(False)



    plt.tight_layout()
    plt.savefig(
        f"/nfs/sea/work/fpfaeffli/plots/statistics/aa_finished_zoomed_timeseries_{scenario}_2013_2015_point_{i0}_{j0}.png",
        dpi=200, transparent=True, bbox_inches="tight"
    )
    plt.show()

# call it
plot_timeseries_zoom_2013_to_2015(i0, j0)




#%%
# Plots only the legend for the timeseries plots

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Define your color mapping
colors = {
    "hplus":        "black",
    "present_clim": "cornflowerblue",
    "present_thr":  "cornflowerblue",
    "future_clim":  "firebrick",
    "future_thr":   "firebrick",
    "fill_pres":    "cornflowerblue",
    "fill_future":  "firebrick"
}

# Create legend handles (one per entry)
handles = [
    Line2D([0], [0], color=colors["present_clim"], linestyle="--", lw=1, label="present climatology"),
    Line2D([0], [0], color=colors["present_thr"],  lw=1.5, label="present threshold"),
    Line2D([0], [0], color=colors["future_thr"], lw=1.5, label="ssp245 threshold"),
    Line2D([0], [0], color=colors["hplus"], lw=1.5, label="ssp245 [H⁺]"),
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
    ncol=3,
    frameon=False,
    fancybox=True,
    shadow=False,
    fontsize=14        
)
plt.axis('off')
plt.tight_layout()
plt.savefig(
    "/nfs/sea/work/fpfaeffli/plots/statistics/legend_hplus_timeseries.png",
    dpi=200, transparent=True, bbox_inches="tight"
)
plt.show()




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
