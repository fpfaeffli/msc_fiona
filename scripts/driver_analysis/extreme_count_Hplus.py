"""
author: Fiona Pfäffli
description: Driver analysis of ocean acidification extremes for the SSP585 scenario.
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

from importlib import reload  # Python 3.4+

import get_obs_datasets 
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter

from func_for_clim_thresh import ThreshClimFuncs
import func_for_clim_thresh
reload(func_for_clim_thresh)
from func_for_clim_thresh import ThreshClimFuncs

from plotting_functions_general import PlotFuncs as PlotFuncs

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
landmask_etopo = PlotFuncs.get_etopo_data()


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


# %% COMPUTE THE THRESHOLD EXCEEDANCES (EXTREMES MASK) WITH REGIONAL FILTERING

print("Calculating the extreme masks for each region...")

def calc_extreme_masks_by_region(inst_dict, thresh_dict, ocean_ds):
    extreme_mask_dict_region = dict()

    # Define regions (latitude bounds)
    regions = {
        "northern": (40.5, 50),
        "central": (34.7, 40.5),
        "southern": (30, 34.7)
    }

    for config in configs:
        extreme_mask_dict_region[config] = dict()
        for scenario in scenarios:
            extreme_mask_dict_region[config][scenario] = dict()
            for threshold_type in ["present", "present_plus_meandelta"]:

                print(f"Calculating extreme mask for {config}, {scenario}, {threshold_type}")

                # Get the full dataset and latitudes
                dataset = ocean_ds[config][scenario]
                lat_rho_da = dataset.lat_rho

                # Convert IndexVariable to DataArray
                if isinstance(lat_rho_da, xr.IndexVariable):
                    lat_rho_da = lat_rho_da.to_dataset(name="lat_rho").lat_rho  # Convert to DataArray

                # Expand lat_rho_da if needed
                if lat_rho_da.ndim == 1:
                    lat_rho_da, _ = xr.broadcast(lat_rho_da, inst_dict[config][scenario])

                # Initialize dictionary for region-specific masks
                extreme_mask_dict_region[config][scenario][threshold_type] = {}

                # Apply filtering for each region
                for region, (lat_min, lat_max) in regions.items():
                    
                    # Create latitude mask
                    lat_mask = (lat_rho_da >= lat_min) & (lat_rho_da < lat_max)

                    # Compute extreme mask **with regional filtering**
                    extreme_mask = (inst_dict[config][scenario] > thresh_dict[config][scenario][threshold_type]).where(lat_mask)

                    # Store mask for this region
                    extreme_mask_dict_region[config][scenario][threshold_type][region] = extreme_mask

    return extreme_mask_dict_region

# Compute the region-specific extreme masks
extreme_mask_dict_region = calc_extreme_masks_by_region(hplus, thresholds_mult, ocean_ds)




#%%
###############################
#     COUNTING EXTREMES       #
###############################

# %%
# Count the number of extremes
extreme_counts = dict()
for config in extreme_mask_dict.keys():
    extreme_counts[config] = dict()
    for scenario in extreme_mask_dict[config].keys():
        extreme_counts[config][scenario] = dict()
        for threshold_type in extreme_mask_dict[config][scenario].keys():
            extreme_counts[config][scenario][threshold_type] = np.sum(extreme_mask_dict[config][scenario][threshold_type].values)

print(extreme_counts)

#%%
# Count the number of extremes per region

extreme_counts_region = dict()
for config in extreme_mask_dict_region.keys():
    extreme_counts_region[config] = {}
    for scenario in extreme_mask_dict_region[config].keys():
        extreme_counts_region[config][scenario] = {}
        for region in extreme_mask_dict_region[config][scenario].keys():  # Loop through regions
            extreme_counts_region[config][scenario][region] = {}
            for threshold_type in extreme_mask_dict_region[config][scenario][region].keys():
                # Compute sum only for the respective region
                extreme_counts_region[config][scenario][region][threshold_type] = np.sum(
                    extreme_mask_dict_region[config][scenario][region][threshold_type].values
                )

print(extreme_counts_region)



# %%
# Calculate percentage of extremes
extreme_percentages = dict()
for config in extreme_mask_dict.keys():
    extreme_percentages[config] = dict()
    for scenario in extreme_mask_dict[config].keys():
        extreme_percentages[config][scenario] = dict()
        for threshold_type in extreme_mask_dict[config][scenario].keys():
            total_events = np.prod(extreme_mask_dict[config][scenario][threshold_type].shape)  # Total time * lat * lon
            extreme_count = np.sum(extreme_mask_dict[config][scenario][threshold_type].values)
            extreme_percentages[config][scenario][threshold_type] = (extreme_count / total_events) * 100  # Convert to %


print(extreme_percentages)


# %%
###############################
#     PLOTTING BARCHARTS      #
###############################

# %%
# Plots the percentage of extreme events over the whole timeseries and all grid cells for the relative and absolute threshold combined

# Define color and label mappings
colors = {
    "present": "black",
    "ssp245_fixed_baseline": "indigo",
    "ssp245_moving_baseline": "darkgreen",
    "ssp585_fixed_baseline": "indigo",
    "ssp585_moving_baseline": "darkgreen",
}

# Prepare data for plotting
plot_data_present = extreme_percentages[configs[0]]["present"]["present"]  # Single present value
plot_data_fixed = []
plot_data_moving = []
plot_colors_fixed = []
plot_colors_moving = []

scenarios_filtered = ["ssp245", "ssp585"]  # Only future scenarios

# Add SSP scenarios with fixed & moving baselines
for scenario in scenarios_filtered:
    plot_data_fixed.append(extreme_percentages[configs[0]][scenario]["present"])  # Fixed Baseline
    plot_colors_fixed.append(colors[f"{scenario}_fixed_baseline"])

    plot_data_moving.append(extreme_percentages[configs[0]][scenario]["present_plus_meandelta"])  # Moving Baseline
    plot_colors_moving.append(colors[f"{scenario}_moving_baseline"])

# Define x-positions: Separate Present from the grouped scenarios
bar_width = 0.3
x_positions_present = np.array([0])  # Present at x = 0
x_positions = np.arange(1, len(scenarios_filtered) + 1)  # Start future scenarios at x = 1

# Restore original figure size
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the present scenario bar
bars_present = ax.bar(x_positions_present, [plot_data_present], width=bar_width, color=colors["present"], label="Present")

# Plot the SSP245 and SSP585 bars side by side
bars_fixed = ax.bar(x_positions - bar_width / 2, plot_data_fixed, width=bar_width, color=plot_colors_fixed, label="Fixed baseline")
bars_moving = ax.bar(x_positions + bar_width / 2, plot_data_moving, width=bar_width, color=plot_colors_moving, label="Moving baseline")

# Dynamically adjust label placement to keep them just above the bars
def add_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01 * height, f"{value:.2f}%", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")

# Apply the label placement function
add_labels(bars_present, [plot_data_present])
add_labels(bars_fixed, plot_data_fixed)
add_labels(bars_moving, plot_data_moving)

# Correct x-tick positions and labels
x_positions_all = np.concatenate((x_positions_present, x_positions))  
x_labels = ["Present", "ssp245", "ssp585"]  

ax.set_xticks(x_positions_all)
ax.set_xticklabels(x_labels, fontsize=12)

# Formatting
ax.set_ylabel("Percentage of extreme events (%)")
ax.set_xlabel("Scenario")
ax.set_title("Percentage of extreme H+ events by scenario (fixed vs. moving baseline)")

# Move legend slightly lower WITHOUT shifting the entire figure
legend = ax.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color="black", label="Present"),
    plt.Rectangle((0, 0), 1, 1, color="indigo", label="Fixed baseline"),
    plt.Rectangle((0, 0), 1, 1, color="darkgreen", label="Moving baseline"),
], loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12, frameon=False)

# Ensure the legend does NOT affect the plot height
fig.subplots_adjust(bottom=0.2)  # Reserves space for legend without shifting everything up

# Save plot without cropping issues
output_path = "/nfs/sea/work/fpfaeffli/plots/driver_analysis/extreme_counts/extreme_Hplus_events_percentage_barchart.png"
plt.savefig(output_path, dpi=200, transparent=True, bbox_inches='tight')

# Show plot
plt.show()

#%%
# Plot mean extreme days per year per grid cell for fixed and moving baselines

# Define color and label mappings
colors = {
    "present": "black",
    "ssp245_fixed_baseline": "indigo",
    "ssp245_moving_baseline": "darkgreen",
    "ssp585_fixed_baseline": "indigo",
    "ssp585_moving_baseline": "darkgreen",
}

# Compute total number of time steps
total_time_steps = extreme_mask_dict[configs[0]]["present"]["present"].shape[0]

# Compute mean extreme days per year per grid cell
extreme_days_per_year_present = ((extreme_mask_dict[configs[0]]["present"]["present"].sum(dim="time") / total_time_steps) * 365).mean().values
extreme_days_per_year_fixed = []
extreme_days_per_year_moving = []
plot_colors_fixed = []
plot_colors_moving = []

scenarios_filtered = ["ssp245", "ssp585"]

# Compute mean extreme days per year per grid cell for each scenario
for scenario in scenarios_filtered:
    mean_fixed = ((extreme_mask_dict[configs[0]][scenario]["present"].sum(dim="time") / total_time_steps) * 365).mean().values
    mean_moving = ((extreme_mask_dict[configs[0]][scenario]["present_plus_meandelta"].sum(dim="time") / total_time_steps) * 365).mean().values
    
    extreme_days_per_year_fixed.append(mean_fixed)
    extreme_days_per_year_moving.append(mean_moving)
    
    plot_colors_fixed.append(colors[f"{scenario}_fixed_baseline"])
    plot_colors_moving.append(colors[f"{scenario}_moving_baseline"])

# Define x-positions
bar_width = 0.3
x_positions_present = np.array([0])
x_positions = np.arange(1, len(scenarios_filtered) + 1)

# Restore original figure size
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the present scenario bar
bars_present = ax.bar(x_positions_present, [extreme_days_per_year_present], width=bar_width, color=colors["present"], label="Present")

# Plot the SSP245 and SSP585 bars side by side
bars_fixed = ax.bar(x_positions - bar_width / 2, extreme_days_per_year_fixed, width=bar_width, color=plot_colors_fixed, label="Fixed Baseline")
bars_moving = ax.bar(x_positions + bar_width / 2, extreme_days_per_year_moving, width=bar_width, color=plot_colors_moving, label="Moving Baseline")

# Dynamically adjust label placement to keep them just above the bars
def add_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01 * height, f"{value:.1f}", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")

# Apply the label placement function
add_labels(bars_present, [extreme_days_per_year_present])
add_labels(bars_fixed, extreme_days_per_year_fixed)
add_labels(bars_moving, extreme_days_per_year_moving)

# Correct x-tick positions and labels
x_positions_all = np.concatenate((x_positions_present, x_positions))  
x_labels = ["Present", "SSP245", "SSP585"]  

ax.set_xticks(x_positions_all)
ax.set_xticklabels(x_labels, fontsize=12)

# Formatting
ax.set_ylabel("Mean extreme days per year")
ax.set_xlabel("Scenario")
ax.set_title("Mean extreme H+ days per year (fixed vs. moving baseline)")

# Move legend slightly lower WITHOUT shifting the entire figure
legend = ax.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color="black", label="Present"),
    plt.Rectangle((0, 0), 1, 1, color="indigo", label="Fixed baseline"),
    plt.Rectangle((0, 0), 1, 1, color="darkgreen", label="Moving baseline"),
], loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12, frameon=False)

# Ensure the legend does NOT affect the plot height
fig.subplots_adjust(bottom=0.2)  # Reserves space for legend without shifting everything up

# Save plot without cropping issues
output_path = "/nfs/sea/work/fpfaeffli/plots/driver_analysis/extreme_counts/mean_extreme_Hplus_days_per_year_barchart.png"
plt.savefig(output_path, dpi=200, transparent=True, bbox_inches='tight')

# Show plot
plt.show()



#%%
###############################
#         PLOTTING MAPS       #
###############################

# %%
# Plots mean extreme days per year per grid cell for each scenario on a map

def plot_extreme_days_map(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict, config="romsoc_fully_coupled", scenario="ssp585"):
   
    # Set bounds
    lat_bounds = [40.5, 50]
    lon_bounds = [225, 245]

    dataset = ocean_ds[config][scenario]

    # Extract lat/lon coordinates
    lat_rho = dataset.lat_rho.isel(time=0)
    lon_rho = dataset.lon_rho.isel(time=0)

    # Compute mean extreme days per year per grid cell 
    total_time_steps = extreme_mask_dict[config][scenario]["present"].shape[0]
    extreme_days_per_year = ((extreme_mask_dict[config][scenario]["present"].sum(dim="time") / total_time_steps) * 365).squeeze()

    # Create figure and axis before plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # Add solid black landmask
    ax.contourf(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        levels=[0, 1], colors=['black'], zorder=1
    )
    
   

    # Create filled contour plot
    contour_levels = np.linspace(0, 350, 8) 
    cf = ax.contourf(
        lon_rho, lat_rho, extreme_days_per_year, levels=contour_levels, cmap="inferno", zorder=4, extend="both"
    )

    # Add offshore vs. coastal region contour lines (no shading)
    region_contour_colors = {'offshore_all_lats': 'black', 'coastal_all_lats': 'black'}
    for region, color in region_contour_colors.items():
        if region in model_regions_dict:
            region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']), 0, 1)
            ax.contour(
                lon_rho, lat_rho, region_mask_dummy,
                levels=[0.5], colors=color, zorder=5, linewidths=1.2, linestyles="solid"
            )

    # Set labels and titles
    ax.set_title(f"Mean extreme H+ days per year ({scenario})", fontsize=16)
    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)  # Always include ylabel

    # Format latitude & longitude labels
    yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
    xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels([f"{lat}°N" for lat in yticks])
    ax.set_xticklabels([f"{xtick}°E" for xtick in xticks])

    # Add colorbar below the plot with adjusted placement
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", extend="both", pad=0.1)

    # Set predefined tick values for the colorbar
    cbar.set_ticks([0, 50, 100, 150, 200, 250, 300, 350])
    cbar.set_ticklabels([str(tick) for tick in [0, 50, 100, 150, 200, 250, 300, 350]])

    cbar.set_label("Mean extreme H+ days per year", fontsize=14)

    # Save
    output_path = f'/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/extreme_Hplus_days_per_year_map_{scenario}.png'
    #plt.savefig(output_path, dpi=150, transparent=True, bbox_inches='tight')
    plt.show()



# %%
plot_extreme_days_map(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict, config="romsoc_fully_coupled", scenario="present")
plot_extreme_days_map(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict, config="romsoc_fully_coupled", scenario="ssp245")
plot_extreme_days_map(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict, config="romsoc_fully_coupled", scenario="ssp585")





# %%


def plot_extreme_days_comparison(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict, config="romsoc_fully_coupled"):
    """
    Plots a side-by-side comparison of mean extreme H+ days per year 
    for the scenarios: present, SSP245, and SSP585.
    """
    
    # Define the scenarios to plot
    scenarios = ["present", "ssp245", "ssp585"]

    # Set bounds
    lat_bounds = [25, 50]
    lon_bounds = [225, 245]

    # Create a single figure with 3 subplots (side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Loop through scenarios and plot each one
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        dataset = ocean_ds[config][scenario]

        # Extract lat/lon coordinates
        lat_rho = dataset.lat_rho.isel(time=0)
        lon_rho = dataset.lon_rho.isel(time=0)

        # Compute mean extreme days per year per grid cell
        total_time_steps = extreme_mask_dict[config][scenario]["present"].shape[0]
        extreme_days_per_year = ((extreme_mask_dict[config][scenario]["present"].sum(dim="time") / total_time_steps) * 365).squeeze()

        # Add solid black landmask
        ax.contourf(
            landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
            levels=[0, 1], colors=['black'], zorder=1
        )
        
        # Add contour levels
        contour_levels = np.linspace(0, extreme_days_per_year.max(), 20)

        # Create filled contour plot
        cf = ax.contourf(
            lon_rho, lat_rho, extreme_days_per_year, levels=contour_levels, cmap="inferno", zorder=4, extend="both"
        )

        # Add offshore vs. coastal region contour lines (no shading)
        region_contour_colors = {'offshore_all_lats': 'black', 'coastal_all_lats': 'black'}
        for region, color in region_contour_colors.items():
            if region in model_regions_dict:
                region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']), 0, 1)
                ax.contour(
                    lon_rho, lat_rho, region_mask_dummy,
                    levels=[0.5], colors=color, zorder=5, linewidths=1.2, linestyles="solid"
                )

        # Set labels and titles
        ax.set_title(f"{scenario}", fontsize=16)
        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_xlabel("Longitude", fontsize=12)
        if i == 0:  # Only set ylabel for first subplot
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

    #Set predefined tick values for the colorbar
    cbar.set_ticks([0, 50, 100, 150, 200, 250, 300, 350])
    cbar.set_ticklabels([str(tick) for tick in [0, 50, 100, 150, 200, 250, 300, 350]])

    cbar.set_label("Mean extreme H+ days per year", fontsize=14)

    # Add suptitle
    fig.suptitle("Comparison of mean extreme H+ days per year for different scenarios", fontsize=16)

    # Save 
    output_path = '/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/extreme_Hplus_days_per_year_comparison.png'
    plt.savefig(output_path, dpi=150, transparent=True, bbox_inches='tight')
    plt.show()


# %%
plot_extreme_days_comparison(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict, config="romsoc_fully_coupled")


#%%

#Maps but with the bin approach



# %%

# Distance to coast plot

def plot_extreme_Hplus_days_vs_distance(ocean_ds, model_d2coasts, extreme_mask_dict, config="romsoc_fully_coupled"):
    

    # Define scenarios and colors
    scenarios = ["ssp245", "ssp585"]
    colors = {"ssp245": "green", "ssp585": "purple"}

    # Define distance bins (every 10 km up to 380 km)
    bins_d2coast = np.arange(0, 380, 10)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Loop over scenarios
    for scenario in scenarios:
        # Compute mean extreme Hplus days per year per grid cell
        total_time_steps = extreme_mask_dict[config][scenario]["present"].shape[0]
        extreme_days_per_year = ((extreme_mask_dict[config][scenario]["present"].sum(dim="time") / total_time_steps) * 365).squeeze()

        # Flatten the arrays
        extreme_days_flat = extreme_days_per_year.values.flatten()
        d2coast_flat = model_d2coasts["roms_only"].values.flatten()


        # Remove NaNs
        valid_mask = ~np.isnan(extreme_days_flat) & ~np.isnan(d2coast_flat)
        extreme_days_flat = extreme_days_flat[valid_mask]
        d2coast_flat = d2coast_flat[valid_mask]

        # Compute mean and percentiles for each distance bin
        mean_values, p25_values, p75_values = [], [], []

        for bdx in range(len(bins_d2coast) - 1):
            bin_mask = (d2coast_flat >= bins_d2coast[bdx]) & (d2coast_flat < bins_d2coast[bdx + 1])
            if np.any(bin_mask):
                mean_values.append(np.mean(extreme_days_flat[bin_mask]))
                p25_values.append(np.percentile(extreme_days_flat[bin_mask], 25))
                p75_values.append(np.percentile(extreme_days_flat[bin_mask], 75))
            else:
                mean_values.append(np.nan)
                p25_values.append(np.nan)
                p75_values.append(np.nan)

        # Plot with shading for IQR
        ax.plot(bins_d2coast[:-1], mean_values, color=colors[scenario], alpha=1, linewidth=2, label=scenario.upper())
        ax.fill_between(bins_d2coast[:-1], p75_values, p25_values, color=colors[scenario], alpha=0.35)

    # Formatting
    ax.set_xlim([0, 360])
    ax.grid(linestyle="--", alpha=0.25)
    ax.axvline(100, color="black", linestyle="--", linewidth=2)  # Vertical line at 100 km
    ax.set_xlabel("Distance to coast (km)", fontsize=12)
    ax.set_ylabel("Mean extreme H+ days per year ", fontsize=12)
    ax.set_title("Mean extreme H+ days per year vs. distance to coast", fontsize=16)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="ssp245"),
        Line2D([0], [0], color="purple", lw=2, label="ssp585"),
    ]
    ax.legend(handles=legend_elements, loc="center right", fontsize=12)

    plt.ylim(220, 366)

    # Save & Show
    output_path = "/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/extreme_Hplus_days_vs_coast_future.png"
    plt.savefig(output_path, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# Call the function for Hplus
plot_extreme_Hplus_days_vs_distance(ocean_ds, model_d2coasts, extreme_mask_dict, config="romsoc_fully_coupled")








#%%
# Distance to coast plot for whole area

def plot_extreme_Hplus_days_vs_distance(ocean_ds, model_d2coasts, extreme_mask_dict, config="romsoc_fully_coupled"):
    
    # Define scenarios and colors (including present)
    scenarios = ["present", "ssp245", "ssp585"]
    colors = {"present": "black", "ssp245": "green", "ssp585": "purple"}

    # Define distance bins (every 10 km up to 380 km)
    bins_d2coast = np.arange(0, 380, 10)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Loop over scenarios
    for scenario in scenarios:
        # Compute mean extreme Hplus days per year per grid cell
        total_time_steps = extreme_mask_dict[config][scenario]["present"].shape[0]
        extreme_days_per_year = ((extreme_mask_dict[config][scenario]["present"].sum(dim="time") / total_time_steps) * 365).squeeze()

        # Flatten the arrays
        extreme_days_flat = extreme_days_per_year.values.flatten()
        d2coast_flat = model_d2coasts["roms_only"].values.flatten()

        # Remove NaNs
        valid_mask = ~np.isnan(extreme_days_flat) & ~np.isnan(d2coast_flat)
        extreme_days_flat = extreme_days_flat[valid_mask]
        d2coast_flat = d2coast_flat[valid_mask]

        # Compute mean and percentiles for each distance bin
        mean_values, p25_values, p75_values = [], [], []

        for bdx in range(len(bins_d2coast) - 1):
            bin_mask = (d2coast_flat >= bins_d2coast[bdx]) & (d2coast_flat < bins_d2coast[bdx + 1])
            if np.any(bin_mask):
                mean_values.append(np.mean(extreme_days_flat[bin_mask]))
                p25_values.append(np.percentile(extreme_days_flat[bin_mask], 25))
                p75_values.append(np.percentile(extreme_days_flat[bin_mask], 75))
            else:
                mean_values.append(np.nan)
                p25_values.append(np.nan)
                p75_values.append(np.nan)

        # Plot with shading for IQR (exclude shading for present run)
        ax.plot(bins_d2coast[:-1], mean_values, color=colors[scenario], alpha=1, linewidth=2, label=scenario.upper())
        
        if scenario != "present":  # No IQR shading for present
            ax.fill_between(bins_d2coast[:-1], p75_values, p25_values, color=colors[scenario], alpha=0.2)

    # Formatting
    ax.set_xlim([0, 360])
    ax.grid(linestyle="--", alpha=0.25)
    ax.axvline(100, color="black", linestyle="--", linewidth=2)  # Vertical line at 100 km
    ax.set_xlabel("Distance to coast (km)", fontsize=12)
    ax.set_ylabel("Mean extreme H+ days per year", fontsize=12)
    ax.set_title("Mean extreme H+ days per year vs. distance to coast", fontsize=16)

    # Legend (adding present)
    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="present"),
        Line2D([0], [0], color="green", lw=2, label="ssp245"),
        Line2D([0], [0], color="purple", lw=2, label="ssp585"),
    ]
    ax.legend(handles=legend_elements, loc="center right", fontsize=12)

    plt.ylim(0, 366)  # Keep Y-axis limit

    # Save & Show
    output_path = "/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/extreme_Hplus_days_vs_coast_with_present.png"
    plt.savefig(output_path, dpi=200, transparent=True, bbox_inches="tight")
    plt.show()

# Call the function for Hplus
plot_extreme_Hplus_days_vs_distance(ocean_ds, model_d2coasts, extreme_mask_dict, config="romsoc_fully_coupled")

# %%

#