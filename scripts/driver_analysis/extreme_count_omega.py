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
from get_model_datasets_omega_surf_only import ModelGetter as ModelGetter

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
params = ThresholdParameters.omega_arag_instance() #5th percentile threshold

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
            if varia == 'omega_arag_offl':
                vdict[config][scenario] = ocean_ds[config][scenario].temp.load()
            
    return vdict


omega = load_data_in('omega_arag_offl')


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


omega_clim = load_clim_in('omega_arag_offl')


#%% LOAD in the threshold for Omega
print("Loading the threshold for Omega...")
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
        thresholds_mult[config][scenario]['present_plus_meandelta'] = concatenated_threshold + (omega_clim[config][scenario].mean(dim='time') - omega_clim[config]['present'].mean(dim='time'))
        
    return thresholds_mult

thresholds_mult = load_thresh_in('omega_arag_offl', 5)

# %% COMPUTE THE THRESHOLD EXCEEDANCES (EXTREMES MASK) for Omega with only relative threshold

print("Calculating the extreme masks for Omega...")
def calc_extreme_masks_rel(inst_dict, thresh_dict):
    extreme_mask_dict_rel = dict()
    for config in configs:
        extreme_mask_dict_rel[config] = dict()
        for scenario in scenarios:
            extreme_mask_dict_rel[config][scenario] = dict()
            for threshold_type in ['present', 'present_plus_meandelta']:
                extreme_mask_dict_rel[config][scenario][threshold_type] = inst_dict[config][scenario] > thresh_dict[config][scenario][threshold_type]
    return extreme_mask_dict_rel

extreme_mask_dict_rel = calc_extreme_masks_rel(omega, thresholds_mult)

#%% COMPUTE THE THRESHOLD EXCEEDANCES (EXTREMES MASK) for Omega with both relative and absolute thresholds


print("Calculating the extreme masks for Omega with updated absolute threshold filtering...")

#Define upper threshold
upper_thresh = 1.0


def calc_extreme_masks_rel_abs(inst_dict, thresh_dict,  upper_thresh):
    extreme_mask_dict_rel_abs = dict()
    for config in configs:
        extreme_mask_dict_rel_abs[config] = dict()
        for scenario in scenarios:
            extreme_mask_dict_rel_abs[config][scenario] = dict()
            for threshold_type in ['present', 'present_plus_meandelta']:
                # Original mask for exceeding the 5th percentile threshold
                relative_extreme_mask = inst_dict[config][scenario] > thresh_dict[config][scenario][threshold_type]
                
                # Additional absolute threshold filter: Keep only values BELOW 2
                absolute_extreme_mask = inst_dict[config][scenario] < upper_thresh
                
                # Combined mask: Only keep events that satisfy both conditions
                extreme_mask_dict_rel_abs[config][scenario][threshold_type] = relative_extreme_mask & absolute_extreme_mask
                
    return extreme_mask_dict_rel_abs

# Apply the function with the new absolute threshold (removing values above 2)
extreme_mask_dict_rel_abs = calc_extreme_masks_rel_abs(omega, thresholds_mult, upper_thresh)

#%%
###############################
#     COUNTING EXTREMES       #
###############################

# %%
# Count the number of extremes

# Initialize dictionaries for counting extreme events
extreme_counts = dict()
extreme_counts_rel_abs = dict()  # For relative + absolute threshold filtering

# Count the number of extreme events for both methods
for config in configs:
    extreme_counts[config] = dict()
    extreme_counts_rel_abs[config] = dict()
    
    for scenario in scenarios:
        extreme_counts[config][scenario] = dict()
        extreme_counts_rel_abs[config][scenario] = dict()
        
        for threshold_type in ['present', 'present_plus_meandelta']:
            # Relative threshold count
            extreme_counts[config][scenario][threshold_type] = np.sum(extreme_mask_dict_rel[config][scenario][threshold_type].values)
            
            # Relative + Absolute threshold count
            extreme_counts_rel_abs[config][scenario][threshold_type] = np.sum(extreme_mask_dict_rel_abs[config][scenario][threshold_type].values)

# Print counts
print("Extreme counts (relative threshold only):")
print(extreme_counts)
print("\nExtreme counts (relative + absolute threshold):")
print(extreme_counts_rel_abs)




# %%
# Calculate percentage of extremes

# Initialize dictionaries for percentage calculations
extreme_percentages = dict()
extreme_percentages_rel_abs = dict()  # For relative + absolute threshold filtering

# Calculate percentage of extremes
for config in configs:
    extreme_percentages[config] = dict()
    extreme_percentages_rel_abs[config] = dict()
    
    for scenario in scenarios:
        extreme_percentages[config][scenario] = dict()
        extreme_percentages_rel_abs[config][scenario] = dict()
        
        for threshold_type in ['present', 'present_plus_meandelta']:
            # Total grid cells (time * lat * lon)
            total_events = np.prod(extreme_mask_dict_rel[config][scenario][threshold_type].shape)
            
            # Relative threshold percentage
            extreme_count_rel = np.sum(extreme_mask_dict_rel[config][scenario][threshold_type].values)
            extreme_percentages[config][scenario][threshold_type] = (extreme_count_rel / total_events) * 100  # Convert to %
            
            # Relative + Absolute threshold percentage
            extreme_count_rel_abs = np.sum(extreme_mask_dict_rel_abs[config][scenario][threshold_type].values)
            extreme_percentages_rel_abs[config][scenario][threshold_type] = (extreme_count_rel_abs / total_events) * 100  # Convert to %

# Print percentages
print("Extreme percentages (relative threshold only):")
print(extreme_percentages)
print("\nExtreme percentages (relative + absolute threshold):")
print(extreme_percentages_rel_abs)


print(extreme_percentages)


# %%
###############################
#     PLOTTING BARCHARTS      #
###############################

# %%
# Barplots which compare relative approach vs relative + absolute approach


# Define color and label mappings
colors = {
    "present": "black",
    "ssp245_fixed_baseline": "indigo",
    "ssp245_moving_baseline": "darkgreen",
    "ssp585_fixed_baseline": "indigo",
    "ssp585_moving_baseline": "darkgreen",
}

# Prepare data for plotting (Relative Approach)
plot_data_present_rel = extreme_percentages[configs[0]]["present"]["present"]  # Single present value
plot_data_fixed_rel = []
plot_data_moving_rel = []

# Prepare data for plotting (Relative + Absolute Approach)
plot_data_present_rel_abs = extreme_percentages_rel_abs[configs[0]]["present"]["present"]  # Single present value
plot_data_fixed_rel_abs = []
plot_data_moving_rel_abs = []

plot_colors_fixed = []
plot_colors_moving = []

scenarios_filtered = ["ssp245", "ssp585"]  # Only future scenarios

# Add SSP scenarios with fixed & moving baselines
for scenario in scenarios_filtered:
    # Relative approach
    plot_data_fixed_rel.append(extreme_percentages[configs[0]][scenario]["present"])  # Fixed Baseline
    plot_data_moving_rel.append(extreme_percentages[configs[0]][scenario]["present_plus_meandelta"])  # Moving Baseline
    
    # Relative + Absolute approach
    plot_data_fixed_rel_abs.append(extreme_percentages_rel_abs[configs[0]][scenario]["present"])  # Fixed Baseline
    plot_data_moving_rel_abs.append(extreme_percentages_rel_abs[configs[0]][scenario]["present_plus_meandelta"])  # Moving Baseline

    # Colors
    plot_colors_fixed.append(colors[f"{scenario}_fixed_baseline"])
    plot_colors_moving.append(colors[f"{scenario}_moving_baseline"])

# Define x-positions: Separate Present from the grouped scenarios
bar_width = 0.3
x_positions_present = np.array([0])  # Present at x = 0
x_positions = np.arange(1, len(scenarios_filtered) + 1)  # Start future scenarios at x = 1

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Titles for each subplot
subplot_titles = [
    "Relative threshold only",
    "Relative + absolute threshold"
]

# Data for both plots
plot_data = [
    (plot_data_present_rel, plot_data_fixed_rel, plot_data_moving_rel),
    (plot_data_present_rel_abs, plot_data_fixed_rel_abs, plot_data_moving_rel_abs)
]

# Loop through both subplots
for i, ax in enumerate(axes):
    # Unpack data
    plot_data_present, plot_data_fixed, plot_data_moving = plot_data[i]

    # Plot the present scenario bar
    ax.bar(x_positions_present, [plot_data_present], width=bar_width, color=colors["present"], label="present")

    # Plot the SSP245 and SSP585 bars side by side
    ax.bar(x_positions - bar_width / 2, plot_data_fixed, width=bar_width, color=plot_colors_fixed, label="Fixed Baseline")
    ax.bar(x_positions + bar_width / 2, plot_data_moving, width=bar_width, color=plot_colors_moving, label="Moving Baseline")

    # Add percentage labels on bars
    ax.text(x_positions_present[0], plot_data_present + 0.5, f"{plot_data_present:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for j, (x, fixed, moving) in enumerate(zip(x_positions, plot_data_fixed, plot_data_moving)):
        ax.text(x - bar_width / 2, fixed + 0.5, f"{fixed:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(x + bar_width / 2, moving + 0.5, f"{moving:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Correct x-tick positions and labels
    x_positions_all = np.concatenate((x_positions_present, x_positions))  
    x_labels = ["present", "ssp245", "ssp585"]  

    ax.set_xticks(x_positions_all)
    ax.set_xticklabels(x_labels, fontsize=12)

    # Formatting
    ax.set_ylabel("Percentage of extreme events (%)")
    ax.set_xlabel("Scenario")
    ax.set_title(subplot_titles[i])

# Create a single shared legend **BELOW** the plots
handles = [
    plt.Rectangle((0, 0), 1, 1, color="black", label="Present"),
    plt.Rectangle((0, 0), 1, 1, color="indigo", label="Fixed baseline"),
    plt.Rectangle((0, 0), 1, 1, color="darkgreen", label="Moving baseline"),
]

fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=False)

# Add overall title
fig.suptitle("Comparison of threshold approaches for extreme Ω events", fontsize=16)

# Add a second smaller suptitle to the right upper corner
fig.text(0.97, 0.95, f'Ω < {upper_thresh}', ha='right', fontsize=12)

# Adjust layout to leave more space at the bottom for the legend
fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leaves more space at the bottom for the legend and overall title

# Save plot
output_path = f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/extreme_counts/extreme_events_percentage_barchart_dual_upper_thresh_{upper_thresh}.png'
plt.savefig(output_path, dpi=200, transparent=True, bbox_inches='tight')

# Show plot
plt.show()


#%%
# Plots the percentage of extreme events over the whole timeseries and all grid cells for the relative and absolute threshold combined

# Define color and label mappings
colors = {
    "present": "black",
    "ssp245_fixed_baseline": "indigo",
    "ssp245_moving_baseline": "darkgreen",
    "ssp585_fixed_baseline": "indigo",
    "ssp585_moving_baseline": "darkgreen",
}

# Prepare data for plotting (Relative + Absolute Threshold Approach)
plot_data_present_rel_abs = extreme_percentages_rel_abs[configs[0]]["present"]["present"]  # Single present value
plot_data_fixed_rel_abs = []
plot_data_moving_rel_abs = []
plot_colors_fixed = []
plot_colors_moving = []

scenarios_filtered = ["ssp245", "ssp585"]  # Only future scenarios

# Add SSP scenarios with fixed & moving baselines
for scenario in scenarios_filtered:
    plot_data_fixed_rel_abs.append(extreme_percentages_rel_abs[configs[0]][scenario]["present"])  # Fixed Baseline
    plot_colors_fixed.append(colors[f"{scenario}_fixed_baseline"])

    plot_data_moving_rel_abs.append(extreme_percentages_rel_abs[configs[0]][scenario]["present_plus_meandelta"])  # Moving Baseline
    plot_colors_moving.append(colors[f"{scenario}_moving_baseline"])

# Define x-positions: Separate Present from the grouped scenarios
bar_width = 0.3
x_positions_present = np.array([0])  # Present at x = 0
x_positions = np.arange(1, len(scenarios_filtered) + 1)  # Start future scenarios at x = 1

# Restore original figure size
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the present scenario bar
bars_present = ax.bar(x_positions_present, [plot_data_present_rel_abs], width=bar_width, color=colors["present"], label="Present")

# Plot the SSP245 and SSP585 bars side by side
bars_fixed = ax.bar(x_positions - bar_width / 2, plot_data_fixed_rel_abs, width=bar_width, color=plot_colors_fixed, label="Fixed baseline")
bars_moving = ax.bar(x_positions + bar_width / 2, plot_data_moving_rel_abs, width=bar_width, color=plot_colors_moving, label="Moving baseline")

# **Dynamically adjust label placement to keep them just above the bars**
def add_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01 * height, f"{value:.2f}%", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")

# Apply the label placement function
add_labels(bars_present, [plot_data_present_rel_abs])
add_labels(bars_fixed, plot_data_fixed_rel_abs)
add_labels(bars_moving, plot_data_moving_rel_abs)

# Correct x-tick positions and labels
x_positions_all = np.concatenate((x_positions_present, x_positions))  
x_labels = ["Present", "ssp245", "ssp585"]  

ax.set_xticks(x_positions_all)
ax.set_xticklabels(x_labels, fontsize=12)

# Formatting
ax.set_ylabel("Percentage of extreme events (%)")
ax.set_xlabel("Scenario")
ax.set_title("Percentage of extreme Ω events by scenario (fixed vs. moving baseline)")

# Add a second smaller suptitle to the right upper corner
fig.text(0.97, 0.95, f'Ω < {upper_thresh}', ha='right', fontsize=12)


# **Move legend slightly lower WITHOUT shifting the entire figure**
legend = ax.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color="black", label="Present"),
    plt.Rectangle((0, 0), 1, 1, color="indigo", label="Fixed baseline"),
    plt.Rectangle((0, 0), 1, 1, color="darkgreen", label="Moving baseline"),
], loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12, frameon=False)

# **Ensure the legend does NOT affect the plot height**
fig.subplots_adjust(bottom=0.2)  # Reserves space for legend without shifting everything up

# Save plot without cropping issues
output_path = f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/extreme_counts/extreme_omega_events_percentage_barchart_rel_abs_upper_thresh_{upper_thresh}.png'
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
total_time_steps = extreme_mask_dict_rel_abs[configs[0]]["present"]["present"].shape[0]

# Compute mean extreme days per year per grid cell
extreme_days_per_year_present = ((extreme_mask_dict_rel_abs[configs[0]]["present"]["present"].sum(dim="time") / total_time_steps) * 365).mean().values
extreme_days_per_year_fixed = []
extreme_days_per_year_moving = []
plot_colors_fixed = []
plot_colors_moving = []

scenarios_filtered = ["ssp245", "ssp585"]

# Compute mean extreme days per year per grid cell for each scenario
for scenario in scenarios_filtered:
    mean_fixed = ((extreme_mask_dict_rel_abs[configs[0]][scenario]["present"].sum(dim="time") / total_time_steps) * 365).mean().values
    mean_moving = ((extreme_mask_dict_rel_abs[configs[0]][scenario]["present_plus_meandelta"].sum(dim="time") / total_time_steps) * 365).mean().values
    
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
ax.set_ylabel("Mean extreme days per yearl")
ax.set_xlabel("Scenario")
ax.set_title("Mean extreme Ω days per year (fixed vs. moving baseline)")


# Add a second smaller suptitle to the right upper corner
fig.text(0.97, 0.95, f'Ω < {upper_thresh}', ha='right', fontsize=12)


# Move legend slightly lower WITHOUT shifting the entire figure
legend = ax.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color="black", label="Present"),
    plt.Rectangle((0, 0), 1, 1, color="indigo", label="Fixed baseline"),
    plt.Rectangle((0, 0), 1, 1, color="darkgreen", label="Moving baseline"),
], loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12, frameon=False)

# Ensure the legend does NOT affect the plot height
fig.subplots_adjust(bottom=0.2)  # Reserves space for legend without shifting everything up

# Save plot without cropping issues
output_path = f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/extreme_counts/mean_extreme_omega_days_per_year_barchart_upper_thresh_{upper_thresh}.png'
plt.savefig(output_path, dpi=200, transparent=True, bbox_inches='tight')

# Show plot
plt.show()



###############################
#         PLOTTING MAPS       #
###############################

# %%
# Plots mean extreme days per year per grid cell for each scenario on a map
def plot_extreme_days_map(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict_rel, config="romsoc_fully_coupled", scenario="ssp585"):
   
    # Set bounds
    lat_bounds = [25, 50]
    lon_bounds = [225, 245]

    dataset = ocean_ds[config][scenario]

    # Extract lat/lon coordinates
    lat_rho = dataset.lat_rho.isel(time=0)
    lon_rho = dataset.lon_rho.isel(time=0)

    # Compute mean extreme days per year per grid cell 
    total_time_steps = extreme_mask_dict_rel[config][scenario]["present"].shape[0]
    extreme_days_per_year = ((extreme_mask_dict_rel[config][scenario]["present"].sum(dim="time") / total_time_steps) * 365).squeeze()
    

    # Create figure and axis before plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    
   # Add solid black landmask
    ax.contourf(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        levels=[0, 1], colors=['black'], zorder=1
    )

    # Create filled contour plot
    contour_levels = np.linspace(0, extreme_days_per_year.max(), 20)

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
    ax.set_title(f"Mean extreme Ω days per year ({scenario})", fontsize=16)
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

    #Set predefined tick values for the colorbar
    cbar.set_ticks([0, 50, 100, 150, 200, 250, 300, 350])
    cbar.set_ticklabels([str(tick) for tick in [0, 50, 100, 150, 200, 250, 300, 350]])

    cbar.set_label("Mean extreme Ω days per year", fontsize=14)

    # Add a second smaller suptitle to the right upper corner for omega threshold
    fig.text(0.97, 0.95, f'Ω < {upper_thresh}', ha='right', fontsize=12)



    # Save
    output_path = f'/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/extreme_omega_days_per_year_map_{scenario}_pper_thresh_{upper_thresh}.png'
    plt.savefig(output_path, dpi=150, transparent=True, bbox_inches='tight')
    plt.show()



# %%
plot_extreme_days_map(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict_rel, config="romsoc_fully_coupled", scenario="present")
plot_extreme_days_map(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict_rel, config="romsoc_fully_coupled", scenario="ssp245")
plot_extreme_days_map(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict_rel, config="romsoc_fully_coupled", scenario="ssp585")





# %%


def plot_extreme_days_comparison(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict_rel, config="romsoc_fully_coupled"):
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
        total_time_steps = extreme_mask_dict_rel[config][scenario]["present"].shape[0]
        extreme_days_per_year = ((extreme_mask_dict_rel[config][scenario]["present"].sum(dim="time") / total_time_steps) * 365).squeeze()

        # Add solid black landmask
        ax.contourf(
            landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
            levels=[0, 1], colors=['black'], zorder=1
        )
        
        # Add contour levels
        contour_levels = np.linspace(0, extreme_days_per_year.max(), 20)

        # Create filled contour plot
        cf = ax.contourf(
            lon_rho, lat_rho, extreme_days_per_year, levels=contour_levels, cmap="inferno", zorder=4, extend="max"
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

    # Add a second smaller suptitle to the right upper corner for omega threshold
    fig.text(0.97, 0.95, f'Ω < {upper_thresh}', ha='right', fontsize=12)

    # Save 
    output_path = f'/nfs/sea/work/fpfaeffli/plots/extreme_days_per_year/extreme_days_per_year_comparison_pper_thresh_{upper_thresh}.png'
    plt.savefig(output_path, dpi=150, transparent=True, bbox_inches='tight')
    plt.show()


# %%
plot_extreme_days_comparison(ocean_ds, landmask_etopo, model_regions_dict, extreme_mask_dict_rel, config="romsoc_fully_coupled")

# %%
