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
scenarios = ['present'] # ,'present'
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


#%% Plotting

# Function to find the closest grid point to a given latitude and longitude

def find_closest_grid_point(lat_rho, lon_rho, target_lat, target_lon):
    """
    Finds the closest (eta_rho, xi_rho) indices to a given latitude and longitude.
    """
    # Compute distance (great-circle distance approximation)
    dist = np.sqrt((lat_rho - target_lat) ** 2 + (lon_rho - target_lon) ** 2)
    
    print(f"Distance array shape: {dist.shape}")  # Debugging

    # Get the index of the minimum distance
    min_idx = np.argmin(dist)  # Flattened index
    indices = np.unravel_index(min_idx, dist.shape)  # Convert to 2D indices

    print(f"Closest grid indices: {indices}")  # Debugging

    # Ensure only two values are returned
    if len(indices) == 2:
        return int(indices[0]), int(indices[1])
    else:
        raise ValueError(f"Unexpected number of indices returned: {indices}")

#%% 

#Map with costal offshore bins

def plot_neutral_map(ocean_ds, landmask_etopo, bounds, model_regions_dict, config="romsoc_fully_coupled", scenario="ssp585"):
   
    lat_bounds, lon_bounds = bounds
    dataset = ocean_ds[config][scenario]

    # Extract lat_rho and lon_rho (use first time step efficiently)
    lat_rho = dataset.lat_rho.isel(time=0)
    lon_rho = dataset.lon_rho.isel(time=0)

    # Debugging: Print dataset shape
    print(f"Dataset lat_rho shape: {lat_rho.shape}, lon_rho shape: {lon_rho.shape}")

    # Plot map
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("white")  # Ensure the ocean inside the panel is white

    # Add solid black landmask first
    ax.contourf(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        levels=[0, 1], colors=['black'], zorder=1
    )

    # Add offshore vs. coastal boundaries with shading
    downsample_factor = 5  # Adjust as needed
    region_colors = {'offshore_all_lats': 'lightgreen', 'coastal_all_lats': 'plum'}  # Valid light purple
    region_labels = {'offshore_all_lats': 'offshore region', 'coastal_all_lats': 'coastal region'}
    legend_patches = []

    for region, color in region_colors.items():
        if region in model_regions_dict:
            region_mask_dummy = xr.where(np.isnan(model_regions_dict[region]['mask']), 0, 1)
            region_mask_dummy = region_mask_dummy[::downsample_factor, ::downsample_factor]  # Downsample
            ax.contourf(
                lon_rho[::downsample_factor, ::downsample_factor], 
                lat_rho[::downsample_factor, ::downsample_factor], 
                region_mask_dummy,
                levels=[0.5, 1], colors=[color], alpha=0.4, zorder=2
            )
            ax.contour(
                lon_rho[::downsample_factor, ::downsample_factor], 
                lat_rho[::downsample_factor, ::downsample_factor], 
                region_mask_dummy,
                levels=[0.5], colors='k', zorder=3, linewidths=0.8
            )
            # Create legend entries
            legend_patches.append(mpatches.Patch(color=color, label=region_labels[region]))
    
    # Add legend with colors
    ax.legend(handles=legend_patches, loc='upper right', frameon=True, fontsize=12)
    
    # Formatting: Axis limits
    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)

    # Add labels and ticks with ° symbols
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Format latitude & longitude labels
    yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
    xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels([f"{lat}°N" for lat in yticks])
    ax.set_xticklabels([f"{xtick}°E" for xtick in xticks])

    # Title
    ax.set_title("Study region with coastal and offshore regions", fontsize=14)

    # Save & Show
    #plt.savefig(f'/nfs/sea/work/fpfaeffli/plots/neutral_maps/neutral_map_offshore_coastal_region.png', dpi=150, transparent=True, bbox_inches='tight')
    plt.show()

# Example execution
bounds = ([25, 50], [225, 245])  # Example latitude and longitude bounds
plot_neutral_map(ocean_ds, landmask_etopo, bounds, model_regions_dict)

#%%
# Map which plots the whole model output region

def plot_model_domain_with_grid(ocean_ds, landmask_etopo, config="romsoc_fully_coupled", scenario="ssp585"):
    
    dataset = ocean_ds[config][scenario]
    lat_rho = dataset.lat_rho.isel(time=0)
    lon_rho = dataset.lon_rho.isel(time=0)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot landmask UNDERNEATH the domain
    ax.contourf(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        levels=[0, 1], colors=['black'], zorder=2
    )

    # Fill the model domain with blue
    ocean_mask = np.ones_like(lat_rho)
    ax.contourf(
        lon_rho, lat_rho, ocean_mask,
        levels=[0.5, 1.5], colors=['skyblue'], alpha=0.7, zorder=1
    )

    # Plot ROMS gridlines (no outer border)
    stride = 10  # adjust to 1 for full grid
    for i in range(0, lat_rho.shape[0], stride):  # horizontal grid lines
        ax.plot(lon_rho[i, :], lat_rho[i, :], color='black', linewidth=0.4, alpha=0.5, zorder=3)
    for j in range(0, lat_rho.shape[1], stride):  # vertical grid lines
        ax.plot(lon_rho[:, j], lat_rho[:, j], color='black', linewidth=0.4, alpha=0.5, zorder=3)

    # Axis formatting
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_xlim(lon_rho.min(), lon_rho.max())
    ax.set_ylim(lat_rho.min(), lat_rho.max())

    # Set ticks at every 10°
    yticks = np.arange(np.floor(lat_rho.min()/10)*10, np.ceil(lat_rho.max()/10)*10 + 1, 10)
    xticks = np.arange(np.floor(lon_rho.min()/10)*10, np.ceil(lon_rho.max()/10)*10 + 1, 10)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels([f"{lat:.0f}°N" for lat in yticks])
    ax.set_xticklabels([f"{lon:.0f}°E" for lon in xticks])

    # General formatting
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.show()


plot_model_domain_with_grid(ocean_ds, landmask_etopo)


#%%
# Function to plot entire model domain with gridlines and mean pH


def plot_model_pH_with_grid(ocean_ds, landmask_etopo, config="romsoc_fully_coupled", scenario="present"):
    """
    Plots surface pH (pH_offl) over the full ROMS model domain.
    Overlays model gridlines and landmask. Limits color range to pH 1–15.
    """
    dataset = ocean_ds[config][scenario]
    lat_rho = dataset.lat_rho.isel(time=0)
    lon_rho = dataset.lon_rho.isel(time=0)
    
    # Extract pH (select first time step and surface depth)
    pH = dataset['pH_offl'].isel(depth=0).mean(dim='time', skipna=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot pH field with fixed color range
    p = ax.pcolormesh(lon_rho, lat_rho, pH, shading='auto', cmap='cmo.ice', vmin=1, vmax=15, zorder=1)
    cbar = plt.colorbar(p, ax=ax, orientation='vertical', label='pH')

    # Overlay gridlines
    stride = 5
    for i in range(0, lat_rho.shape[0], stride):
        ax.plot(lon_rho[i, :], lat_rho[i, :], color='gray', linewidth=0.4, alpha=0.5, zorder=2)
    for j in range(0, lat_rho.shape[1], stride):
        ax.plot(lon_rho[:, j], lat_rho[:, j], color='gray', linewidth=0.4, alpha=0.5, zorder=2)

    # Overlay landmask
    ax.contourf(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        levels=[0, 1], colors=['black'], zorder=3
    )

    # Axis formatting
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_xlim(lon_rho.min(), lon_rho.max())
    ax.set_ylim(lat_rho.min(), lat_rho.max())

    # Ticks at 10° intervals
    yticks = np.arange(np.floor(lat_rho.min()/10)*10, np.ceil(lat_rho.max()/10)*10 + 1, 10)
    xticks = np.arange(np.floor(lon_rho.min()/10)*10, np.ceil(lon_rho.max()/10)*10 + 1, 10)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels([f"{lat:.0f}°N" for lat in yticks])
    ax.set_xticklabels([f"{lon:.0f}°E" for lon in xticks])

    ax.set_aspect('auto')
    ax.set_title("Surface Ocean pH (pH_offl)", fontsize=14)
    plt.tight_layout()
    plt.show()

plot_model_pH_with_grid(ocean_ds, landmask_etopo)


#%%
# Function to plot a neutral map with a single marked point and a solid black landmask
target_lat = 38.5
target_lon = 236.5

def plot_neutral_map_with_point(ocean_ds, landmask_etopo, bounds, target_lat, target_lon, config="romsoc_fully_coupled", scenario="present"):
    """
    Plots a neutral map with a single marked point and a solid black landmask.
    Allows selection of point using latitude & longitude instead of grid indices.
    """
    lat_bounds, lon_bounds = bounds
    dataset = ocean_ds[config][scenario]

    # Convert West longitudes (e.g., 125°W → 235°E)
    if target_lon < 0:
        target_lon = 360 + target_lon

    # Extract lat_rho and lon_rho (use first time step)
    lat_rho = dataset.lat_rho.values[0, :, :]
    lon_rho = dataset.lon_rho.values[0, :, :]

    # Debugging: Print shapes
    print(f"Processed lat_rho shape: {lat_rho.shape}")
    print(f"Processed lon_rho shape: {lon_rho.shape}")

    # Find closest grid point
    point_eta_rho, point_xi_rho = find_closest_grid_point(lat_rho, lon_rho, target_lat, target_lon)
    
    # Extract lat/lon of the closest grid point
    point_lat = lat_rho[point_eta_rho, point_xi_rho]
    point_lon = lon_rho[point_eta_rho, point_xi_rho]

    # Convert longitude to 0-360° if necessary
    if point_lon < 0:
        point_lon += 360

    # Debugging: Print found indices and lat/lon
    print(f"Closest grid point (eta_rho, xi_rho): ({point_eta_rho}, {point_xi_rho})")
    print(f"Grid Lat/Lon: ({point_lat}, {point_lon})")

    # Plot map
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor("lightblue")

    # Add solid black landmask using pcolormesh
    ax.pcolormesh(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        cmap="Greys", vmin=0, vmax=1, zorder=1
    )

    # Plot selected point in **red**
    ax.scatter(point_lon, point_lat, color='red', marker='o', s=100, label="Selected Point", zorder=3)

    # Formatting: Axis limits
    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)

    # Add labels and ticks with ° symbols
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Format latitude & longitude labels
    yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
    xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels([f"{lat}°N" for lat in yticks])
    ax.set_xticklabels([f"{xtick}°E" if xtick <= 180 else f"{360 - xtick}°W" for xtick in xticks])

    # Add legend
    ax.legend(loc="upper right")

    # Title
    ax.set_title(f"Neutral Map with Selected Grid Point ({target_lat}°N, {target_lon}°E)", fontsize=14)

    # Save & Show
    plt.savefig(f'/nfs/sea/work/fpfaeffli/plots/neutral_maps/neutral_map_lat{target_lat}_lon{target_lon}.png', dpi=200, transparent=True, bbox_inches='tight')
    plt.show()

#%%

lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

# Example 1: Input in WEST (Convert 125°W → 235°E)
target_lat = 38.5
target_lon = 236.5
# Example 2: Input in EAST (No conversion needed)
# target_lon = 125  # Example: Longitude 125°E

# Call function
plot_neutral_map_with_point(ocean_ds, landmask_etopo, bounds, target_lat, target_lon, config="romsoc_fully_coupled", scenario="present")


#%%
def plot_neutral_map_with_two_points(ocean_ds, landmask_etopo, bounds, points_latlon, config="romsoc_fully_coupled", scenario="present"):
    """
    Plots a neutral map with two marked points and a solid black landmask.
    Allows selection of points using latitude & longitude instead of grid indices.
    """
    lat_bounds, lon_bounds = bounds
    dataset = ocean_ds[config][scenario]

    # Extract lat_rho and lon_rho (first time step only)
    lat_rho = dataset.lat_rho.values[0, :, :]
    lon_rho = dataset.lon_rho.values[0, :, :]

    # Convert longitude bounds (if using west longitudes)
    lon_bounds = tuple(lon + 360 if lon < 0 else lon for lon in lon_bounds)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot a neutral background (light gray ocean)
    ax.set_facecolor("lightblue")

    # Add solid black landmask using pcolormesh
    ax.pcolormesh(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        cmap="Greys", vmin=0, vmax=1, zorder=1
    )

    # Define colors and labels for each point
    point_colors = ["red", "blue"]
    point_labels = ["Point 1", "Point 2"]

    # Process each point
    for i, (target_lat, target_lon) in enumerate(points_latlon):
        # Convert West longitudes (e.g., -125°W → 235°E)
        if target_lon < 0:
            target_lon = 360 + target_lon

        # Find closest grid point
        eta_rho, xi_rho = find_closest_grid_point(lat_rho, lon_rho, target_lat, target_lon)

        # Extract lat/lon of the closest grid point
        point_lat = lat_rho[eta_rho, xi_rho]
        point_lon = lon_rho[eta_rho, xi_rho]

        # Convert longitude to 0-360° if necessary
        if point_lon < 0:
            point_lon += 360

        # Scatter plot for each point
        ax.scatter(point_lon, point_lat, color=point_colors[i], marker='o', s=100, label=point_labels[i], zorder=3)

        # Debugging: Print point information
        print(f"{point_labels[i]} -> Target: ({target_lat}, {target_lon}) | Closest Grid: ({point_lat}, {point_lon})")

    # Formatting: Axis limits
    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)

    # Add labels and ticks with ° symbols
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Format latitude & longitude labels
    yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
    xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels([f"{lat}°N" for lat in yticks])
    ax.set_xticklabels([f"{xtick}°E" if xtick <= 180 else f"{360 - xtick}°W" for xtick in xticks])

    # Add legend
    ax.legend(loc="upper right")

    # Title
    ax.set_title("Neutral Map with Selected Grid Points", fontsize=14)

    # Save & Show
    plt.savefig("/nfs/sea/work/fpfaeffli/plots/neutral_map_two_points.png", dpi=200, transparent=True, bbox_inches='tight')
    plt.show()

#%%
lat_bounds = (30, 47)
lon_bounds = (225, 245)  # Ensure longitudes are in 0-360° format
bounds = (lat_bounds, lon_bounds)

# Define the 2 selected points (lat, lon)
points_latlon = [
    (40, -125),  # Point 1: 40°N, 125°W  (Converted to 235°E)
    (40, -132)   # Point 2: 41°N, 127°W  (Converted to 233°E)
]

# Call the function with ocean_ds
plot_neutral_map_with_two_points(ocean_ds, landmask_etopo, bounds, points_latlon, config="romsoc_fully_coupled", scenario="present")


# %%


def plot_neutral_map_with_four_points(ocean_ds, landmask_etopo, bounds, points_latlon, config="romsoc_fully_coupled", scenario="present"):
    """
    Plots a neutral map with four marked points and a solid black landmask.
    Allows selection of points using latitude & longitude instead of grid indices.
    """
    lat_bounds, lon_bounds = bounds
    dataset = ocean_ds[config][scenario]

    # Extract lat_rho and lon_rho (first time step only)
    lat_rho = dataset.lat_rho.values[0, :, :]
    lon_rho = dataset.lon_rho.values[0, :, :]

    # Convert longitude bounds (if using west longitudes)
    lon_bounds = tuple(lon + 360 if lon < 0 else lon for lon in lon_bounds)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot a neutral background (light gray ocean)
    ax.set_facecolor("lightblue")

    # Add solid black landmask using pcolormesh
    ax.pcolormesh(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        cmap="Greys", vmin=0, vmax=1, zorder=1
    )

    # Define colors and labels for each point
    point_colors = ["red", "blue", "green", "purple"]
    point_labels = ["Point 1", "Point 2", "Point 3", "Point 4"]

    # Process each point
    for i, (target_lat, target_lon) in enumerate(points_latlon):
        # Convert West longitudes (e.g., -125°W → 235°E)
        if target_lon < 0:
            target_lon = 360 + target_lon

        # Find closest grid point
        eta_rho, xi_rho = find_closest_grid_point(lat_rho, lon_rho, target_lat, target_lon)

        # Extract lat/lon of the closest grid point
        point_lat = lat_rho[eta_rho, xi_rho]
        point_lon = lon_rho[eta_rho, xi_rho]

        # Convert longitude to 0-360° if necessary
        if point_lon < 0:
            point_lon += 360

        # Scatter plot for each point
        ax.scatter(point_lon, point_lat, color=point_colors[i], marker='o', s=100, label=point_labels[i], zorder=3)

        # Debugging: Print point information
        print(f"{point_labels[i]} -> Target: ({target_lat}, {target_lon}) | Closest Grid: ({point_lat}, {point_lon})")

    # Formatting: Axis limits
    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)

    # Add labels and ticks with ° symbols
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Format latitude & longitude labels
    yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
    xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels([f"{lat}°N" for lat in yticks])
    ax.set_xticklabels([f"{xtick}°E" if xtick <= 180 else f"{360 - xtick}°W" for xtick in xticks])

    # Add legend
    ax.legend(loc="upper right")

    # Title
    ax.set_title("Neutral Map with Selected Grid Points", fontsize=14)

    # Save & Show
    # plt.savefig("/nfs/sea/work/fpfaeffli/plots/neutral_map_four_points.png", dpi=200, transparent=True, bbox_inches='tight')
    plt.show()

#%%
lat_bounds = (30, 47)
lon_bounds = (225, 245)  # Ensure longitudes are in 0-360° format
bounds = (lat_bounds, lon_bounds)

# Define the 4 selected points (lat, lon)
points_latlon = [
    (40, -125),  # Point 1: 40°N, 125°W  (Converted to 235°E)
    (40, -127),  # Point 2: 41°N, 120°W  (Converted to 240°E)
    (40, -129),   # Point 3: 42°N, 125°E  (Stays the same)
    (40, -131)    # Point 4: 43°N, 130°E  (Stays the same)
]

# Call the function with ocean_ds
plot_neutral_map_with_four_points(ocean_ds, landmask_etopo, bounds, points_latlon, config="romsoc_fully_coupled", scenario="present")


# %%
