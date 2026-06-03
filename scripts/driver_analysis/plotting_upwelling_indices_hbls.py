"""
author: Fiona Pfäffli
description: This file plots present day vs future extremes for the different thresholds
"""

#%% enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

# load the package
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter
import glob
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

#%% 
# Define the threshold 
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.Hplus_instance() #95th percentile threshold

#%%
###############################
#         LOADING DATA        #
###############################

#%%
#for hbls

# Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present','ssp585', 'ssp245'] # ,'ssp245'
configs = ['romsoc_fully_coupled'] # [ms_only'] 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
depth = 0
vert_struct = 'avg'


ocean_ds_raw = dict()
for config in configs:
     ocean_ds_raw[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds_raw[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)

#%%
#for NO3, DIC, T, salt

# Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present','ssp585', 'ssp245'] # ,'ssp245'
configs = ['romsoc_fully_coupled'] # [ms_only'] 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
depth = 0
vert_struct = 'zavg'

ocean_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)

#%%
#for w

# Define the base path for w data
base_path = "/nfs/sea/work/fpfaeffli/"

# Define subdirectories for each scenario
w_scenario_dirs = {
    "present": f"{base_path}/fionas_avg_present/",
    "ssp245": f"{base_path}/fionas_avg_ssp245/",
    "ssp585": f"{base_path}/fionas_avg_ssp585/",
}

# Function to process dataset before merging
def preprocess(ds):
    """Ensure 'time' is set as a coordinate before merging."""
    if "time" in ds.dims and "time" not in ds.coords:
        ds = ds.assign_coords(time=ds["time"])  # Promote time from dimension to coordinate
    
    return ds


#%%
# Initialize w_datasets
w_datasets = {}  

# Loop through each config and scenario
configs = ["romsoc_fully_coupled"]  

for config in configs:
    w_datasets[config] = {}  # Initialize for each config

    for scenario, path in w_scenario_dirs.items():
        # Get all files
        w_files = sorted(glob.glob(f"{path}/z_avg_*.nc"))

        # Filter only 2011-2021 files
        w_files = [f for f in w_files if any(f"z_avg_{year}" in f for year in range(2011, 2022))]

        print(f"Filtered number of files for {scenario}: {len(w_files)}")
        print(f"First 5 filtered files:\n{w_files[:5]}")

        
        # Load all w files as a single dataset
        print(f"Loading w for {scenario}...")
        w_ds = xr.open_mfdataset(
            w_files, 
            combine="nested", 
            concat_dim="time", 
            preprocess=preprocess, 
            parallel=True
        )

        # Store the dataset in the dictionary
        w_datasets[config][scenario] = w_ds["w"].isel(depth=10).load()  # Extract w at depth=10

        print(f"Successfully loaded w for {scenario}.")


#%% 
# Initialize the variables dictionary
variables = dict()

# List of variables from ocean_ds (50m depth)
var_list = ["DIC", "Alk", "NO3", "temp", "salt"] #, "DIC", "Alk", "emp", "alt"

# Loop through each config and scenario
for config in configs:
    variables[config] = dict()
    
    for scenario in scenarios:
        print(f'Getting variables for {config} - {scenario}...')
        
        # Initialize scenario dictionary
        variables[config][scenario] = dict()

        # Load hbls from ocean_ds_raw
        # if scenario in ocean_ds_raw[config]:
        #     if "hbls" in ocean_ds_raw[config][scenario]:
        #         print(f"Loading hbls for {scenario}...")
        #         variables[config][scenario]["hbls"] = ocean_ds_raw[config][scenario]["hbls"].load()

        # Load NO3, DIC, Alk, Temp, Salt from ocean_ds at depth=10 (-50m)
        if scenario in ocean_ds[config]:
            for var in var_list:
                if var in ocean_ds[config][scenario]:
                    print(f"Loading {var} at 50m for {scenario}...")
                    variables[config][scenario][var] = ocean_ds[config][scenario][var].isel(depth=10).load()

        # Load w from w_datasets
        if scenario in w_datasets[config]:
            print(f"Loading w for {scenario}...")
            variables[config][scenario]["w"] = w_datasets[config][scenario]  # Already at depth=10

print("All variables loaded!")

#%%
# Fixing  time alignment issues:

for config in configs:
    for scenario in scenarios:
        if "w" in variables[config][scenario]:
            num_timesteps = variables[config][scenario]["w"].shape[0]  # Get time length

            # Generate correct time values from 2011-01-01 onward
            correct_time = pd.date_range("2011-01-01", periods=num_timesteps, freq="D")

            # Assign proper time coordinate
            variables[config][scenario]["w"] = variables[config][scenario]["w"].assign_coords(time=("time", correct_time))

print("Time alignment applied to all scenarios!")



#%%
###############################
#      UPWELLING INDICES      #
###############################

# %%
# Convert w from meters per second to meters per day

def convert_w_to_m_per_day(w_data):
    """
    Convert w (upwelling velocity) from m/s to m/day.
    """
    # Extract w
    w_data = variables[config][scenario]["w"]

    # Check initial min and max values
    print(f"Before conversion: Min = {w_data.min().values}, Max = {w_data.max().values}")

    # Perform the conversion from m/s to m/day
    w_data = w_data * 86400  # Convert from m/s to m/day
    
    # Check converted min and max values
    print(f"After conversion: Min = {w_data.min().values}, Max = {w_data.max().values}")
    
    return w_data


w_data = variables[config][scenario]["w"]

# Convert w_data and print the min/max values
w_data_converted = convert_w_to_m_per_day(w_data)



# %%
# Compute all upwelling indices for all scenarios
for config in configs:
    for scenario in scenarios:
        print(f"Computing indices for {config} - {scenario}...")

        # Extract NO3
        NO3_data = variables[config][scenario]["NO3"]
        DIC_data = variables[config][scenario]["DIC"]
        Alk_data = variables[config][scenario]["Alk"]
        Temp_data = variables[config][scenario]["temp"]
        Salt_data = variables[config][scenario]["salt"]

        # Compute CUTI (upwelling velocity)
        CUTI = w_data_converted  # w is directly the upwelling velocity

        # Compute BEUTI (biological upwelling transport)
        BEUTI = NO3_data * w_data_converted  # Element-wise multiplication

        # Compute rest of upwelling
        DIC_upwelling = DIC_data * w_data_converted
        Alk_upwelling = Alk_data * w_data_converted
        Temp_upwelling = Temp_data * w_data_converted
        Salt_upwelling = Salt_data * w_data_converted

        # Store in variables dictionary
        variables[config][scenario]["CUTI"] = CUTI
        variables[config][scenario]["BEUTI"] = BEUTI
        variables[config][scenario]["DIC_upwelling"] = DIC_upwelling
        variables[config][scenario]["Alk_upwelling"] = Alk_upwelling
        variables[config][scenario]["Temp_upwelling"] = Temp_upwelling
        variables[config][scenario]["Salt_upwelling"] = Salt_upwelling

        print(f"Upwelling indices computed for {config} - {scenario}!")

print("All indices calculations completed!")


#%%
# Assign lon and lat to indices and w

print("Assign lon/lat to indices...")
for config in configs:
    for scenario in scenarios:
        # Extract lat/lon from NO3
        lat_data = variables[config][scenario]["NO3"].coords["lat"]
        lon_data = variables[config][scenario]["NO3"].coords["lon"]

        # Assign them to CUTI & BEUTI
        variables[config][scenario]["CUTI"] = variables[config][scenario]["CUTI"].assign_coords(lat=lat_data, lon=lon_data)
        variables[config][scenario]["BEUTI"] = variables[config][scenario]["BEUTI"].assign_coords(lat=lat_data, lon=lon_data)
        variables[config][scenario]["w"] = variables[config][scenario]["w"].assign_coords(lat=lat_data, lon=lon_data)
        #variables[config][scenario]["hbls"] = variables[config][scenario]["hbls"].assign_coords(lat=lat_data, lon=lon_data)
print("Assigned lat/lon coordinates to hbls, w, CUTI and BEUTI!")

#%%
###############################
#        CALC CLIM HBLS       #
###############################


# %%
varias = ['hbls']

# Calculate climatology for hbls to determine climatological depth of mixed layer
print('Get the climatology for hbls')
clims = dict()
for config in configs:
     clims[config] = dict()
     for scenario in scenarios:
          clims[config][scenario] = dict()
          for varia in varias:
               print(f'Calculate the climatology for: {config}, {scenario}, {varia}.')
               dummy_clim = ThreshClimFuncs.calc_clim(params,variables[config][scenario][varia])
               smoothed_clim = ThreshClimFuncs.smooth_array(params,dummy_clim)
               smoothed_clim = smoothed_clim.rename({'day_of_year_adjusted': 'time'})
               smoothed_clim = smoothed_clim.assign_coords({'time': pd.date_range('2015-01-01','2015-12-31')})
               clims[config][scenario][varia] = ThreshClimFuncs.repeat_array_multiple_years(smoothed_clim)



#%%
###############################
#         PLOTTING HBLS       #
###############################


# %%
# Define the function to plot the mean climatological mixed layer depth (hbls) for all scenarios over the whole year

def plot_mean_hbls_map(hbls_dict, landmask_etopo, bounds, config="romsoc_fully_coupled", scenarios=["present", "ssp585", "ssp245"], filename=None):
    """
    Plots the mean climatological mixed layer depth (hbls) for multiple scenarios.
    """
    lat_bounds, lon_bounds = bounds

    # Extract hbls dataset correctly
    dataset = hbls_dict[config]["present"]["hbls"]  # FIXED!
    
    # Detect correct latitude and longitude names
    lat_name = next((coord for coord in dataset.coords if "lat" in coord.lower()), None)
    lon_name = next((coord for coord in dataset.coords if "lon" in coord.lower()), None)

    lat_rho = dataset[lat_name].values
    lon_rho = dataset[lon_name].values

    # Create figure with subplots
    fig, axs = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6), constrained_layout=True)
    levels = np.linspace(0, 60, 40)  # Depth scale

    for idx, (scenario, ax) in enumerate(zip(scenarios, axs if len(scenarios) > 1 else [axs])):
        print(f"Plotting scenario: {scenario}")

        # Extract scenario-specific hbls data
        hbls_data = hbls_dict[config][scenario]["hbls"]  
        hbls_mean = hbls_data.mean(dim="time")

        # landmask
        ax.contourf(landmask_etopo.coords['lon'], landmask_etopo.coords['lat'], landmask_etopo, 
                    levels=1, colors='black', zorder=1)

        # Plot data
        contour = ax.contourf(lon_rho, lat_rho, hbls_mean, levels=levels, cmap=cmocean.cm.deep_r,
                              vmin=0, vmax=60, extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_title(scenario, fontsize=14)
        ax.set_xlabel("Longitude")
        if idx == 0:
            ax.set_ylabel("Latitude")

        # Add degree symbols and N/W indicators
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

    # Add colorbar
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label="Mean mixed layer depth (m)")

    # Add title
    plt.suptitle("Mean climatological mixed layer depth (hbls)", fontsize=20, y=1.05)

    # Save & Show
    if filename is not None:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    plt.show()

# Define bounds 
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

# Load landmask
landmask_etopo = PlotFuncs.get_etopo_data()

# Define output filename
filename_hbls_map = "/nfs/sea/work/fpfaeffli/plots/driver_analysis/hbls_lots/mean_clim_hbls_map.png"

# Call function to plot
plot_mean_hbls_map(
    clims, landmask_etopo, bounds,
    scenarios=["present", "ssp245", "ssp585"], 
    filename=filename_hbls_map
)



#%%
# Plots the monthly means of the climatological mixed layer depth (hbls) for all scenarios


def plot_monthly_hbls_maps(hbls_dict, landmask_etopo, bounds, config="romsoc_fully_coupled", scenario="present", filename=None):
    """
    Plots the monthly mean climatological mixed layer depth (hbls) for a given scenario.
    """
    lat_bounds, lon_bounds = bounds

    # Extract hbls dataset for the scenario
    hbls_data = hbls_dict[config][scenario]["hbls"]
    
    # Detect latitude and longitude names
    lat_name = next((coord for coord in hbls_data.coords if "lat" in coord.lower()), None)
    lon_name = next((coord for coord in hbls_data.coords if "lon" in coord.lower()), None)

    lat_rho = hbls_data[lat_name].values
    lon_rho = hbls_data[lon_name].values

    # **Group by month and take the mean**
    hbls_monthly = hbls_data.groupby("time.month").mean(dim="time")

    # Create a figure with 12 subplots (3 rows, 4 columns)
    fig, axs = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    levels = np.linspace(0, 60, 40)  # Depth scale

    for month in range(1, 13):  # Loop over 12 months
        ax = axs[(month - 1) // 4, (month - 1) % 4]  # Correct subplot index
        
        # Select the monthly mean
        hbls_mean = hbls_monthly.sel(month=month)

        # Add landmask
        ax.contourf(landmask_etopo.coords['lon'], landmask_etopo.coords['lat'], landmask_etopo, 
                    levels=1, colors='black', zorder=1)

        # Plot data
        contour = ax.contourf(lon_rho, lat_rho, hbls_mean, levels=levels, cmap=cmocean.cm.deep_r,
                              vmin=0, vmax=60, extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        month_names = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
        ax.set_title(month_names[month - 1], fontsize=14)
        ax.set_xlabel("Longitude")
        if month in [1, 5, 9]:  # Label y-axis only on the leftmost subplots
            ax.set_ylabel("Latitude")

        # Add degree symbols
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

    # Add a colorbar
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label="Mean mixed layer depth (m)", shrink=0.7)

    # Add a title
    plt.suptitle(f"Monthly mean climatological mixed layer depth (hbls) - {scenario}", fontsize=20, y=1.05)

    # Save & Show
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    plt.show()


# **Define bounds and load landmask**
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

landmask_etopo = PlotFuncs.get_etopo_data()



# **Call function to plot for a specific scenario**
plot_monthly_hbls_maps(
    clims, landmask_etopo, bounds,
    scenario="present",  # Change scenario if needed
    filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/hbls_lots/monthly_clim_hbls_present.png'
)

plot_monthly_hbls_maps(
    clims, landmask_etopo, bounds,
    scenario="ssp245",  # Change scenario if needed
    filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/hbls_lots/monthly_clim_hbls_ssp245.png'
)

plot_monthly_hbls_maps(
    clims, landmask_etopo, bounds,
    scenario="ssp585",  # Change scenario if needed
    filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/hbls_lots/monthly_clim_hbls_ssp585.png'
)

# %%

# Plots the anomalies of the monthly mean climatological mixed layer depth (hbls) between present and future scenarios

def plot_monthly_hbls_anomalies(hbls_dict, landmask_etopo, bounds, config="romsoc_fully_coupled", future_scenario="ssp245", filename=None):
    """
    Plots the monthly mean climatological mixed layer depth (hbls) anomalies for a given future scenario 
    relative to the present-day climate.
    """
    lat_bounds, lon_bounds = bounds

    # Extract present-day and future hbls datasets
    hbls_present = hbls_dict[config]["present"]["hbls"]
    hbls_future = hbls_dict[config][future_scenario]["hbls"]

    # Detect latitude and longitude names
    lat_name = next((coord for coord in hbls_present.coords if "lat" in coord.lower()), None)
    lon_name = next((coord for coord in hbls_present.coords if "lon" in coord.lower()), None)

    lat_rho = hbls_present[lat_name].values
    lon_rho = hbls_present[lon_name].values

    # **Compute Monthly Mean**
    hbls_present_monthly = hbls_present.groupby("time.month").mean(dim="time")
    hbls_future_monthly = hbls_future.groupby("time.month").mean(dim="time")

    # **Calculate Anomalies (Future - Present)**
    hbls_anomaly = hbls_future_monthly - hbls_present_monthly

    # Create a figure with 12 subplots (3 rows, 4 columns)
    fig, axs = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    levels = np.linspace(-10, 10, 20)  # Set anomaly color range (adjust if needed)

    for month in range(1, 13):  # Loop over 12 months
        ax = axs[(month - 1) // 4, (month - 1) % 4]  # Correct subplot index
        
        # Select the monthly anomaly
        hbls_diff = hbls_anomaly.sel(month=month)

        # Add landmask
        ax.contourf(landmask_etopo.coords['lon'], landmask_etopo.coords['lat'], landmask_etopo, 
                    levels=1, colors='black', zorder=1)

        # Plot anomaly (future - present)
        contour = ax.contourf(lon_rho, lat_rho, hbls_diff, levels=levels, cmap=cmocean.cm.balance,
                              vmin=-10, vmax=10, extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        month_names = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
        ax.set_title(month_names[month - 1], fontsize=14)
        ax.set_xlabel("Longitude")
        if month in [1, 5, 9]:  # Label y-axis only on the leftmost subplots
            ax.set_ylabel("Latitude")

        # Add degree symbols
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

    # Add a colorbar
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label="Mixed layer depth anomaly (m)", shrink=0.7)

    # Add a title
    plt.suptitle(f"Monthly anomalies in climatological mixed layer depth ({future_scenario} - present)", fontsize=20, y=1.05)

    # Save & Show
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    plt.show()


# **Define bounds and load landmask**
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

landmask_etopo = PlotFuncs.get_etopo_data()

# **Call function to plot anomalies for each future scenario**
plot_monthly_hbls_anomalies(
    clims, landmask_etopo, bounds,
    future_scenario="ssp245",  
    filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/hbls_lots/monthly_hbls_anomaly_ssp245.png'
)

plot_monthly_hbls_anomalies(
    clims, landmask_etopo, bounds,
    future_scenario="ssp585",  
    filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/hbls_lots/monthly_hbls_anomaly_ssp585.png'
)



# %%

###############################
#     PLOTTING INDICES        #
###############################
# %%

# Plots maps of the monthly means of all the upwelling indices (CUTI, BEUTI, DIC upwelling, Alk upwelling) for all scenarios


def plot_monthly_upwelling_indices_maps(variables, landmask_etopo, bounds, index_name="CUTI", config="romsoc_fully_coupled", scenario="present", filename=None):
    """
    Plots the monthly mean maps of the selected upwelling index (CUTI, BEUTI, DIC_upwelling, Alk_upwelling) for a given scenario.
    """
    lat_bounds, lon_bounds = bounds

    # Extract dataset for the selected index (CUTI, BEUTI, DIC_upwelling, Alk_upwelling)
    index_data = variables[config][scenario][index_name]

    # Detect latitude and longitude names
    lat_name = "lat"
    lon_name = "lon"

    lat_rho = index_data[lat_name].values
    lon_rho = index_data[lon_name].values

    # **Group by month and take the mean**
    index_monthly = index_data.groupby("time.month").mean(dim="time")

    # Create a figure with 12 subplots (3 rows, 4 columns)
    fig, axs = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)

    # Define color scale based on the index
    if index_name == "CUTI":
        levels = np.linspace(0, 5, 20)  # CUTI scale (upwelling velocity)
        cmap = cmocean.cm.thermal
        color_label = "CUTI (m/day)"
    elif index_name == "BEUTI":
        levels = np.linspace(0, 200, 20)  # BEUTI scale (biological upwelling transport)
        cmap = cmocean.cm.thermal
        color_label = "BEUTI (mMol NO3 m⁻² d⁻¹)"
    elif index_name == "DIC_upwelling":
        levels = np.linspace(0, 20000, 20)  # DIC upwelling scale
        cmap = cmocean.cm.thermal
        color_label = "DIC Upwelling (mMol C m⁻² d⁻¹)"
    elif index_name == "Alk_upwelling":
        levels = np.linspace(0, 20000, 20)  # Alk upwelling scale
        cmap = cmocean.cm.thermal
        color_label = "Alk Upwelling (mMol m⁻² d⁻¹)"
    elif index_name == "Temp_upwelling":
        levels = np.linspace(0, 10, 20)
        cmap = cmocean.cm.thermal
        color_label = "Temp Upwelling (°C m d⁻¹)"
    elif index_name == "Salt_upwelling":
        levels = np.linspace(0, 10, 20)
        cmap = cmocean.cm.thermal
        color_label = "Salt Upwelling (PSU m d⁻¹)"


    for month in range(1, 13):  # Loop over 12 months
        ax = axs[(month - 1) // 4, (month - 1) % 4]  # Correct subplot index
        
        # Select the monthly mean
        index_mean = index_monthly.sel(month=month)

        # Add landmask
        ax.contourf(landmask_etopo["lon"], landmask_etopo["lat"], landmask_etopo, 
                    levels=1, colors='black', zorder=3)

        # Plot data
        contour = ax.contourf(lon_rho, lat_rho, index_mean, levels=levels, cmap=cmap,
                              extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        month_names = ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"]
        ax.set_title(month_names[month - 1], fontsize=14)
        ax.set_xlabel("Longitude")
        if month in [1, 5, 9]:  # Label y-axis only on the leftmost subplots
            ax.set_ylabel("Latitude")

        # Add degree symbols
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

    # Add a colorbar
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label=color_label, shrink=0.7)

    # Add a title
    if index_name == "CUTI":
        plt.suptitle(f"Monthly mean Coastal Upwelling Transport Index (CUTI) at 50m depth - {scenario}", fontsize=20, y=1.05)
    elif index_name == "BEUTI":
        plt.suptitle(f"Monthly mean Biological Upwelling Transport Index (BEUTI) at 50m depth - {scenario}", fontsize=20, y=1.05)
    elif index_name == "DIC_upwelling":
        plt.suptitle(f"Monthly mean DIC upwelling at 50m depth - {scenario}", fontsize=20, y=1.05)
    elif index_name == "Alk_upwelling":
        plt.suptitle(f"Monthly mean Alk upwelling at 50m depth - {scenario}", fontsize=20, y=1.05)
    elif index_name == "Temp_upwelling":
        plt.suptitle(f"Monthly mean Temperature upwelling at 50m depth - {scenario}", fontsize=20, y=1.05)
    elif index_name == "Salt_upwelling":
        plt.suptitle(f"Monthly mean Salinity upwelling at 50m depth - {scenario}", fontsize=20, y=1.05)


    # Save & Show
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    plt.show()


# **Define bounds and load landmask**
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

landmask_etopo = PlotFuncs.get_etopo_data()

# **Call function to plot for each scenario**
for scenario in ["present", "ssp245", "ssp585"]:
    for index_name in [ "Temp_upwelling", "Salt_upwelling"]: #"CUTI", "BEUTI", "DIC_upwelling", "Alk_upwelling",
        plot_monthly_upwelling_indices_maps(
            variables, landmask_etopo, bounds,
            index_name=index_name, scenario=scenario,
            filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/upwelling/monthly_{index_name}_{scenario}_whole_region.png'
        )


# %%

# Plots a comparison of yearly means of all the upwelling indices for all scenarios

def plot_mean_upwelling_index_scenarios(variables, landmask_etopo, bounds, index_name="CUTI", config="romsoc_fully_coupled", filename=None):
    """
    Plots the mean upwelling index (CUTI, BEUTI, DIC_upwelling, Alk_upwelling) over the entire time period
    for all three scenarios ("present", "ssp245", "ssp585") in one figure.
    """
    lat_bounds, lon_bounds = bounds
    scenarios = ["present", "ssp245", "ssp585"]
    
    # Define color scale based on the index
    if index_name == "CUTI":
        levels = np.linspace(0, 5, 20)
        cmap = cmocean.cm.thermal
        color_label = "CUTI (m/day)"
    elif index_name == "BEUTI":
        levels = np.linspace(0, 200, 20)
        cmap = cmocean.cm.thermal
        color_label = "BEUTI (mMol NO3 m⁻² d⁻¹)"
    elif index_name == "DIC_upwelling":
        levels = np.linspace(0, 20000, 20)
        cmap = cmocean.cm.thermal
        color_label = "DIC Upwelling (mMol C m⁻² d⁻¹)"
    elif index_name == "Alk_upwelling":
        levels = np.linspace(0, 20000, 20)
        cmap = cmocean.cm.thermal
        color_label = "Alk Upwelling (mMol m⁻² d⁻¹)"
    elif index_name == "Temp_upwelling":
        levels = np.linspace(0, 10, 20)
        cmap = cmocean.cm.thermal
        color_label = "Temp Upwelling (°C m d⁻¹)"
    elif index_name == "Salt_upwelling":
        levels = np.linspace(0, 10, 20)
        cmap = cmocean.cm.thermal
        color_label = "Salt Upwelling (PSU m d⁻¹)"

    # Create figure with 3 subplots (one for each scenario)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        
        # Extract dataset for the selected index
        index_data = variables[config][scenario][index_name]

        # Detect latitude and longitude names
        lat_name = "lat"
        lon_name = "lon"

        lat_rho = index_data[lat_name].values
        lon_rho = index_data[lon_name].values

        # Compute mean over entire time period
        index_mean = index_data.mean(dim="time")

        # **Debugging: Print mean and range for each scenario**
        print(f"\n{index_name} - {scenario}:")
        print(f"Mean value: {np.nanmean(index_mean.values):.4f}")
        print(f"Min value: {np.nanmin(index_mean.values):.4f}, Max value: {np.nanmax(index_mean.values):.4f}")

        # Add landmask
        ax.contourf(landmask_etopo["lon"], landmask_etopo["lat"], landmask_etopo, levels=1, colors='black', zorder=3)

        # Plot data
        contour = ax.contourf(lon_rho, lat_rho, index_mean, levels=levels, cmap=cmap, extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_xlabel("Longitude")
        if i == 0:
            ax.set_ylabel("Latitude")

        # Add degree symbols
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

        # Title for each subplot
        ax.set_title(f"{scenario}", fontsize=16)

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.25, -0.04, 0.5, 0.02])  # Adjusted placement
    cbar = plt.colorbar(contour, cax=cbar_ax, orientation="horizontal", label=color_label)
    
    # Add overall title
    if index_name == "CUTI":
        plt.suptitle(f"Comparison of mean CUTI for all scenarios", fontsize=18)
    elif index_name == "BEUTI":
        plt.suptitle(f"Comparison of mean BEUTI for all scenarios", fontsize=18)
    elif index_name == "DIC_upwelling":
        plt.suptitle(f"Comparison of mean DIC upwelling for all scenarios", fontsize=18)
    elif index_name == "Alk_upwelling":
        plt.suptitle(f"Comparison of mean alkalinity upwelling for all scenarios", fontsize=18)
    elif index_name == "Temp_upwelling":
        plt.suptitle(f"Comparison of mean temperature upwelling for all scenarios", fontsize=18)
    elif index_name == "Salt_upwelling":
        plt.suptitle(f"Comparison of mean salinity upwelling for all scenarios", fontsize=18)


    # Save & Show
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    plt.show()


# **Define bounds and load landmask**
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

landmask_etopo = PlotFuncs.get_etopo_data()

# **Call function to plot for each upwelling index**
for index_name in ["BEUTI", "CUTI"]: #"CUTI", "BEUTI", "DIC_upwelling", "Alk_upwelling", "Temp_upwelling", "Salt_upwelling"
    plot_mean_upwelling_index_scenarios(
        variables, landmask_etopo, bounds,
        index_name=index_name,
        filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/upwelling/mean_{index_name}_all_scenarios.png'
    )



#%%

# Plots the anomalies of the yearly mean upwelling indices between present and future scenarios


def plot_upwelling_anomalies(variables, landmask_etopo, bounds, index_name="CUTI", config="romsoc_fully_coupled", filename=None):
    """
    Plots the anomaly of the selected upwelling index (CUTI, BEUTI, DIC_upwelling, Alk_upwelling) 
    for SSP245 and SSP585 compared to the present scenario.
    """
    
    lat_bounds, lon_bounds = bounds

    # Extract dataset for each scenario
    index_present = variables[config]["present"][index_name].mean(dim="time")
    index_ssp245 = variables[config]["ssp245"][index_name].mean(dim="time")
    index_ssp585 = variables[config]["ssp585"][index_name].mean(dim="time")

    # Compute **anomalies** (SSP - Present)
    anomaly_ssp245 = index_ssp245 - index_present
    anomaly_ssp585 = index_ssp585 - index_present

    # Detect latitude and longitude names
    lat_name = "lat"
    lon_name = "lon"

    lat_rho = index_present[lat_name].values
    lon_rho = index_present[lon_name].values

    # **Manually specify vmin and vmax for each index**
    vmin_vmax_dict = {
        "CUTI": (-2, 2),
        "BEUTI": (-50, 50),
        #"DIC_upwelling": (-5000, 5000),
        #"Alk_upwelling": (-5000, 5000),
        "Temp_upwelling": (-1, 1),
        "Salt_upwelling": (-0.1, 0.1),
    }

    vmin, vmax = vmin_vmax_dict.get(index_name, (-1, 1))  # Default range if index is missing

    # Define contour levels
    levels = np.linspace(vmin, vmax, 20)
    cmap = cmocean.cm.balance  # Diverging colormap for positive/negative changes

    # Define colorbar label based on index
    color_labels = {
        "CUTI": "CUTI anomaly (future - present)",
        "BEUTI": "BEUTI anomaly (future - present)",
        #"DIC_upwelling": "DIC upwelling anomaly (future - present)",
        #"Alk_upwelling": "Alk upwelling anomaly (future - present)",
        "Temp_upwelling": "Temperature upwelling anomaly (future - present)",
        "Salt_upwelling": "Salinity upwelling anomaly (future - present)",
    }
    color_label = color_labels.get(index_name, "Anomaly (future - present)")

    # Create figure with 2 subplots (SSP245, SSP585 anomalies)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    scenarios = ["ssp245", "ssp585"]
    anomalies = [anomaly_ssp245, anomaly_ssp585]

    for i, (scenario, anomaly) in enumerate(zip(scenarios, anomalies)):
        ax = axs[i]

        # Add landmask
        ax.contourf(landmask_etopo["lon"], landmask_etopo["lat"], landmask_etopo, 
                    levels=[0, 1], colors='black', zorder=3)

        # Plot data
        contour = ax.contourf(lon_rho, lat_rho, anomaly, levels=levels, cmap=cmap, 
                              extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        ax.set_title(f"{scenario} - present", fontsize=14)
        ax.set_xlabel("Longitude")
        if i == 0:
            ax.set_ylabel("Latitude")

        # Add degree symbols
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

    # Add a colorbar
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label=color_label, shrink=0.8)

    # Add a title
    plt.suptitle(color_label, fontsize=18)

    # Save & Show
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    plt.show()


# **Define bounds and load landmask**
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

landmask_etopo = PlotFuncs.get_etopo_data()

# **Call function to plot anomalies for each upwelling index**
for index_name in ["CUTI", "BEUTI"]: #"CUTI", "BEUTI", "DIC_upwelling", "Alk_upwelling",
    plot_upwelling_anomalies(
        variables, landmask_etopo, bounds,
        index_name=index_name,
        filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/upwelling/anomaly_{index_name}.png'
    )



#%%
###############################
#     PLOTTING VARIABLES     #
###############################

# %%
# Plot monthly means of w and NO3 for all scenarios

def plot_monthly_maps(variables, landmask_etopo, bounds, var_name="w", config="romsoc_fully_coupled", scenario="present", filename=None):
    """
    Plots the monthly mean maps of the selected variable (w or NO3) for a given scenario.
    """
    lat_bounds, lon_bounds = bounds

    # Extract dataset for the selected variable
    var_data = variables[config][scenario][var_name]
    
    # Detect latitude and longitude names
    lat_name = "lat"
    lon_name = "lon"

    lat_rho = var_data[lat_name].values
    lon_rho = var_data[lon_name].values

    # **Group by month and take the mean**
    var_monthly = var_data.groupby("time.month").mean(dim="time")

    # Create a figure with 12 subplots (3 rows, 4 columns)
    fig, axs = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)

    # Define color scale based on the variable
    if var_name == "w":
        levels = np.linspace(0, 0.0001, 20)  # w scale (vertical velocity)
        cmap = cmocean.cm.dense
        color_label = "Vertical velocity (m/s)"
    else:
        levels = np.linspace(0, np.nanmax(var_monthly), 20)  # NO3 scale (concentration)
        cmap = cmocean.cm.thermal
        color_label = "Nitrate Concentration (mmol/m³)"

    for month in range(1, 13):  # Loop over 12 months
        ax = axs[(month - 1) // 4, (month - 1) % 4]  # Correct subplot index
        
        # Select the monthly mean
        var_mean = var_monthly.sel(month=month)

        # Add landmask
        ax.contourf(landmask_etopo["lon"], landmask_etopo["lat"], landmask_etopo, 
                    levels=1, colors='black', zorder=3)

        # Plot data
        contour = ax.contourf(lon_rho, lat_rho, var_mean, levels=levels, cmap=cmap,
                              extend="both", zorder=2)

        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)
        month_names = ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"]
        ax.set_title(month_names[month - 1], fontsize=14)
        ax.set_xlabel("Longitude")
        if month in [1, 5, 9]:  # Label y-axis only on the leftmost subplots
            ax.set_ylabel("Latitude")

        # Add degree symbols
        yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 5)
        xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_yticklabels([f"{lat}°N" for lat in yticks])
        ax.set_xticklabels([f"{360 - lon}°W" for lon in xticks])

    # Add a colorbar
    cbar = fig.colorbar(contour, ax=axs, orientation="vertical", label=color_label, shrink=0.7)

    # Add a title
    if var_name == "w":
        plt.suptitle(f"Monthly mean vertical velocity (w) at 50m depth ({scenario})", fontsize=20, y=1.05)
    elif var_name == "NO3":
        plt.suptitle(f"Monthly mean nitrate concentration at 50m depth ({scenario})", fontsize=20, y=1.05)

    # Save & Show
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    plt.show()


# **Define bounds and load landmask**
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

landmask_etopo = PlotFuncs.get_etopo_data()

# **Call function to plot for each scenario**
for scenario in ["present", "ssp245", "ssp585"]:
    plot_monthly_maps(
        variables, landmask_etopo, bounds,
        var_name="w", scenario=scenario,
        filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/upwelling/monthly_w_{scenario}.png'
    )

    plot_monthly_maps(
        variables, landmask_etopo, bounds,
        var_name="NO3", scenario=scenario,
        filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/upwelling/monthly_NO3_{scenario}.png'
    )