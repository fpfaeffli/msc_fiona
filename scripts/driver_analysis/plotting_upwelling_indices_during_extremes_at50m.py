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

from get_auxiliary_data import AuxGetter as AuxGetter

from matplotlib.lines import Line2D

#%% 
# Define the threshold 
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.Hplus_instance() #95th percentile threshold



#%%
###############################
# LOADING MODEL,THRESH,CLIM   #
###############################

#%%
#for NO3, DIC, T, salt, Alk, Hplus

# Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present','ssp585', 'ssp245'] # ,'ssp245'
configs = ['romsoc_fully_coupled'] # [ms_only'] 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'zavg'

ocean_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)


#%%

# Loading in climatologies and thresholds for Hplus for extreme mask calculations

print("Loading in climatologies for Hplus...")
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
print("Climatologies loaded!")

#%%
#Loading in the thresholds for Hplus
print("Loading the threshold for Hplus...")
def load_thresh_in(varia,thresh_value):
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

thresholds_mult = load_thresh_in('Hplus',95)


#%%
###############################
#      LOADING VARIABLES      #
###############################

#%%
# Loading in Hplus data to make the extreme mask calculations:

# load the data at the respective location
def load_data_in(varia):
    vdict = dict()
    for config in configs:
        vdict[config] = dict()
        for scenario in scenarios:
            print(f'Getting Hplus for config {config} and scenario {scenario}.')
            print(varia)
            if varia == 'Hplus':
                pH = ocean_ds[config][scenario].pH_offl.isel(depth=0).load()
                vdict[config][scenario] = np.power(10, -pH)
    return vdict

hplus = load_data_in('Hplus')

#%%


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

# Initialize w_datasets
w_datasets = {}


# Loop through each config and scenario
configs = ["romsoc_fully_coupled"]

for config in configs:
    w_datasets[config] = {}  # Initialize for each config

    for scenario, path in w_scenario_dirs.items():
        w_datasets[config][scenario] = {}  # Initialize scenario dictionary

        # Get all files
        w_files = sorted(glob.glob(f"{path}/z_avg_*.nc"))

        # Filter only 2011-2021 files
        w_files = [f for f in w_files if any(f"z_avg_{year}" in f for year in range(2011, 2022))]

        # Print which file gets loaded currently
        for f in w_files:
            print(f"Loading file: {f}")

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

        # Store the dataset in the dictionary at the given depth level
        w_datasets[config][scenario] = w_ds["w"].isel(depth=10).load()  

        print(f"Successfully loaded w for {scenario}.")


# %%
# Storing all the variables for the upwelling indices computations in a dictionary

# Initialize the variables dictionary
variables = {}

# List of variables from ocean_ds (50m depth)
var_list = ["DIC"]

# Loop through each config and scenario
for config in configs:
    variables[config] = {}  # Initialize config dictionary

    for scenario in scenarios:
        print(f'Getting variables for {config} - {scenario}...')

        # Initialize scenario dictionary
        variables[config][scenario] = {}

        # Ensure depth key exists in the dictionary before assignment
        variables[config][scenario]= {}

        # Load NO3, DIC, Alk, Temp, Salt from ocean_ds at depth=10 (-50m)
        if scenario in ocean_ds.get(config, {}):  # Safely check if scenario exists
            for var in var_list:
                if var in ocean_ds[config][scenario]:
                    print(f"Loading {var} at 50m for {scenario}...")
                    variables[config][scenario][var] = (
                        ocean_ds[config][scenario][var].isel(depth=10).load()
                    )

        # Load w from w_datasets
        if scenario in w_datasets[config]:
            print(f"Loading w for {scenario}...")
            variables[config][scenario]["w"] = w_datasets[config][scenario]  # Already at depth=10

print("All variables loaded!")

#%%
# Fix w_ds time by assigning proper datetime coordinates
num_days = w_ds.sizes["time"]  # Get the number of time steps
correct_time = pd.date_range("2011-01-01", periods=num_days, freq="D")  # Generate full dates

# Assign correct time values
w_ds = w_ds.assign_coords(time=("time", correct_time))

print("Fixed `w_ds` time format. Checking corrected values:")
print(w_ds["time"].values[:50])  # Verify first 50 time values

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
#      CALC EXTREME MASK      #
###############################

#%%
# Calculating the extreme masks

print("Calculating the extreme masks...")
def calc_extreme_masks(inst_dict,thresh_dict):
    extreme_mask_dict = dict()
    for config in configs:
        extreme_mask_dict[config] = dict()
        for scenario in scenarios:
            extreme_mask_dict[config][scenario] = dict()
            for threshold_type in ['present','present_plus_meandelta']:
                print(f'Calculating extreme mask for {inst_dict}: {config} and {scenario} and {threshold_type}')
                extreme_mask_dict[config][scenario][threshold_type] = inst_dict[config][scenario] > thresh_dict[config][scenario][threshold_type]
    return extreme_mask_dict

extreme_mask_dict = calc_extreme_masks(hplus,thresholds_mult)


#%%
###############################
#   CALC UPWELLING INDICES    #
###############################

# %%
# Convert w from meters per second to meters per day

def convert_w_to_m_per_day(w_data):
    """
    Convert w (upwelling velocity) from m/s to m/day.
    """
    # Extract w at 50m depth
    w_data = variables[config][scenario].get("w")

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
        #NO3_data = variables[config][scenario].get("NO3")
        #DIC_data = variables[config][scenario].get("DIC")
        #Alk_data = variables[config][scenario].get("Alk")
        #Temp_data = variables[config][scenario].get("temp")
        #Salt_data = variables[config][scenario].get("salt")

        # Compute CUTI (upwelling velocity)
        CUTI = w_data_converted  # w is directly the upwelling velocity

        # Compute BEUTI (biological upwelling transport)
        #BEUTI = NO3_data * w_data_converted  # Element-wise multiplication

        # Compute rest of upwelling
        #DIC_upwelling = DIC_data * w_data_converted
        #Alk_upwelling = Alk_data * w_data_converted
        #Temp_upwelling = Temp_data * w_data_converted
        #Salt_upwelling = Salt_data * w_data_converted

        # Store in variables dictionary
        variables[config][scenario]["CUTI"] = CUTI
        #variables[config][scenario]["BEUTI"] = BEUTI
        #variables[config][scenario]["DIC_upwelling"] = DIC_upwelling
        #variables[config][scenario]["Alk_upwelling"] = Alk_upwelling
        #variables[config][scenario]["Temp_upwelling"] = Temp_upwelling
        #variables[config][scenario]["Salt_upwelling"] = Salt_upwelling

        print(f"Upwelling indices computed for {config} - {scenario}!")

print("All indices calculations completed!")


#%%
# Assign lon and lat to indices and w

print("Assign lon/lat to indices...")
for config in configs:
    for scenario in scenarios:
        # Extract lat/lon from NO3
        lat_data = variables[config][scenario]["DIC"].coords["lat"]
        lon_data = variables[config][scenario]["DIC"].coords["lon"]

        # Assign them to CUTI & BEUTI
        variables[config][scenario]["CUTI"] = variables[config][scenario]["CUTI"].assign_coords(lat=lat_data, lon=lon_data)
        #variables[config][scenario]["BEUTI"] = variables[config][scenario]["BEUTI"].assign_coords(lat=lat_data, lon=lon_data)
        variables[config][scenario]["w"] = variables[config][scenario]["w"].assign_coords(lat=lat_data, lon=lon_data)
        
print("Assigned lat/lon coordinates")

#%%

###############################
#   APPLY EXTREME MASKS       #
###############################



#%%
# Apply extreme masks to upwelling indices

def apply_extreme_masks_to_upwelling(variables, extreme_mask_dict):
    """
    Apply extreme masks to upwelling indices (CUTI, BEUTI, DIC_upwelling, etc.)
    to filter out timesteps and locations where Hplus extremes occur.
    
    Parameters:
    variables (dict): Dictionary containing upwelling indices for different scenarios.
    extreme_mask_dict (dict): Dictionary of extreme masks for different scenarios and thresholds.
    
    Returns:
    dict: A new dictionary with masked upwelling indices.
    """
    masked_variables = {}

    
    for config in configs:
        masked_variables[config] = {}
        for scenario in scenarios:
            masked_variables[config][scenario] = {}
            for threshold_type in ['present', 'present_plus_meandelta']:
                print(f"Applying mask for {config} - {scenario} using {threshold_type}")
                
                # Initialize dictionary for masked variables
                masked_variables[config][scenario][threshold_type] = {}
                
                for var_name in ["Temp_upwelling", "Salt_upwelling"]: #, "BEUTI", "DIC_upwelling", "Alk_upwelling", "Temp_upwelling", "Salt_upwelling"
                    if var_name in variables[config][scenario]:
                        masked_variables[config][scenario][threshold_type][var_name] = (
                            variables[config][scenario][var_name]
                            .where(extreme_mask_dict[config][scenario][threshold_type], np.NaN)
                        )
                
    return masked_variables

# Apply the function to filter upwelling indices
masked_upwelling_indices = apply_extreme_masks_to_upwelling(variables, extreme_mask_dict)

print("Extreme masks applied to upwelling indices!")


#%%













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
        levels = np.linspace(0, 5000, 20)  # DIC upwelling scale
        cmap = cmocean.cm.thermal
        color_label = "DIC Upwelling (mMol C m⁻² d⁻¹)"
    elif index_name == "Alk_upwelling":
        levels = np.linspace(0, 5000, 20)  # Alk upwelling scale
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
        plt.suptitle(f"Monthly mean CUTI at 50m depth during surface [H+] extremes - {scenario}", fontsize=20, y=1.05)
    elif index_name == "BEUTI":
        plt.suptitle(f"Monthly mean BEUTI at 50m depth during surface [H+] extremes - {scenario}", fontsize=20, y=1.05)
    elif index_name == "DIC_upwelling":
        plt.suptitle(f"Monthly mean DIC upwelling at 50m depth during surface [H+] extremes - {scenario}", fontsize=20, y=1.05)
    elif index_name == "Alk_upwelling":
        plt.suptitle(f"Monthly mean Alk upwelling at 50m depth during surface [H+] extremes - {scenario}", fontsize=20, y=1.05)
    elif index_name == "Temp_upwelling":
        plt.suptitle(f"Monthly mean Temperature upwelling at 50m depth during surface [H+] extremes - {scenario}", fontsize=20, y=1.05)
    elif index_name == "Salt_upwelling":
        plt.suptitle(f"Monthly mean Salinity upwelling at 50m depth during surface [H+] extremes - {scenario}", fontsize=20, y=1.05)


    # Save & Show
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    plt.show()


# **Define bounds and load landmask**
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

landmask_etopo = AuxGetter.get_etopo_data()

# **Call function to plot for each scenario**
for scenario in ["present", "ssp245", "ssp585"]:
    for index_name in [ "Alk_upwelling", "DIC_upwelling"]: #"CUTI", "BEUTI", "DIC_upwelling", "Alk_upwelling",
        plot_monthly_upwelling_indices_maps(
            variables, landmask_etopo, bounds,
            index_name=index_name, scenario=scenario,
            filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/upwelling/monthly_{index_name}_{scenario}_whole_region.png'
        )



# %%
# Plots mean upwelling index on the native grid


def plot_mean_upwelling_index_scenarios(masked_variables, landmask_etopo, bounds, index_name="CUTI", config="romsoc_fully_coupled", filename=None):
    """
    Plots the mean upwelling index (CUTI, BEUTI, DIC_upwelling, Alk_upwelling) over the entire time period
    for all three scenarios ("present", "ssp245", "ssp585") in one figure.
    The plot includes two threshold types: 'present' and 'present_plus_meandelta'.
    """
    lat_bounds, lon_bounds = bounds
    scenarios = ["present", "ssp245", "ssp585"]
    
    # Define color scale based on the index
    if index_name == "CUTI":
        levels = np.linspace(0, 5, 20)
        cmap = cmocean.cm.thermal
        color_label = "CUTI (m/day)"
    elif index_name == "BEUTI":
        levels = np.linspace(0, 50, 20)
        cmap = cmocean.cm.thermal
        color_label = "BEUTI (mMol NO3 m⁻² d⁻¹)"
    elif index_name == "DIC_upwelling":
        levels = np.linspace(0, 5000, 20)
        cmap = cmocean.cm.thermal
        color_label = "DIC Upwelling (mMol C m⁻² d⁻¹)"
    elif index_name == "Alk_upwelling":
        levels = np.linspace(0, 5000, 20)
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)  # Two rows: one for present and one for future scenarios
    
    threshold_types = ["present", "present_plus_meandelta"]

    for j, threshold_type in enumerate(threshold_types):  # Iterate over the two threshold types
        for i, scenario in enumerate(scenarios):
            ax = axes[j][i]  # Select subplot position for threshold_type and scenario
            
            # Extract dataset for the selected index
            index_data = masked_variables[config][scenario][threshold_type].get(index_name, None)
            if index_data is None:
                continue  # Skip if the index does not exist for the threshold_type and scenario
            
            # Detect latitude and longitude names
            lat_name = "lat"
            lon_name = "lon"

            lat_rho = index_data[lat_name].values
            lon_rho = index_data[lon_name].values

            # Compute mean over entire time period
            index_mean = index_data.mean(dim="time")

            # **Debugging: Print mean and range for each scenario**
            print(f"\n{index_name} - {scenario} ({threshold_type}):")
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
            ax.set_title(f"{scenario} ({threshold_type})", fontsize=14)

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.25, -0.04, 0.5, 0.02])  # Adjusted placement
    cbar = plt.colorbar(contour, cax=cbar_ax, orientation="horizontal", label=color_label)
    
    # Add overall title
    if index_name == "CUTI":
        plt.suptitle(f"Mean CUTI during surface [H+] extremes", fontsize=18)
    elif index_name == "BEUTI":
        plt.suptitle(f"Mean BEUTI during surface [H+] extremes", fontsize=18)
    elif index_name == "DIC_upwelling":
        plt.suptitle(f"Mean DIC upwelling during surface [H+] extremes", fontsize=18)
    elif index_name == "Alk_upwelling":
        plt.suptitle(f"Mean alkalinity upwelling during surface [H+] extremes", fontsize=18)
    elif index_name == "Temp_upwelling":
        plt.suptitle(f"Mean temperature during surface [H+] extremes", fontsize=18)
    elif index_name == "Salt_upwelling":
        plt.suptitle(f"Mean salinity upwelling during surface [H+] extremes", fontsize=18)


    # Save & Show
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    plt.show()


# **Define bounds and load landmask**
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

landmask_etopo = AuxGetter.get_etopo_data()

# **Call function to plot for each upwelling index**
for index_name in [ "CUTI"]: #"CUTI", "BEUTI", "DIC_upwelling", "Alk_upwelling", "Temp_upwelling", "Salt_upwelling"
    plot_mean_upwelling_index_scenarios(
        masked_upwelling_indices, landmask_etopo, bounds,
        index_name=index_name,
        filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/upwelling/mean_{index_name}_during_extremes.png'
    )

#%%
# Plots mean upwelling indeces during extremes on 5x5 gridcells average

import xarray as xr
import numpy as np

def plot_mean_upwelling_index_scenarios(masked_variables, landmask_etopo, bounds, index_name="CUTI", config="romsoc_fully_coupled", filename=None):
    """
    Plots the mean upwelling index (CUTI, BEUTI, DIC_upwelling, Alk_upwelling) over the entire time period
    for all three scenarios ("present", "ssp245", "ssp585") in one figure.
    The plot includes two threshold types: 'present' and 'present_plus_meandelta'.
    """
    lat_bounds, lon_bounds = bounds
    scenarios = ["present", "ssp245", "ssp585"]
    
    # Define color scale based on the index
    if index_name == "CUTI":
        levels = np.linspace(0, 1, 20)
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)  # Two rows: one for present and one for future scenarios
    
    threshold_types = ["present", "present_plus_meandelta"]

    for j, threshold_type in enumerate(threshold_types):  # Iterate over the two threshold types
        for i, scenario in enumerate(scenarios):
            ax = axes[j][i]  # Select subplot position for threshold_type and scenario
            
            # Extract dataset for the selected index
            index_data = masked_variables[config][scenario][threshold_type].get(index_name, None)
            if index_data is None:
                continue  # Skip if the index does not exist for the threshold_type and scenario
            

            # Select specific slices for eta_rho and xi_rho to avoid conflicts with dimensions
            index_data_rolled = index_data.isel(eta_rho=slice(None), xi_rho=slice(None))  # No conflicting dimensions

            # Apply rolling mean for a 5x5 window over the eta_rho and xi_rho coordinates
            index_data_rolled = index_data_rolled.rolling(eta_rho=5, xi_rho=5, center=True).mean()

            # Compute mean over entire time period
            index_mean = index_data_rolled.mean(dim="time")

            # Proceed with plotting



            # Detect latitude and longitude names
            lat_name = "lat"
            lon_name = "lon"

            lat_rho = index_data[lat_name].values
            lon_rho = index_data[lon_name].values

    

            # **Debugging: Print mean and range for each scenario**
            print(f"\n{index_name} - {scenario} ({threshold_type}):")
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
            ax.set_title(f"{scenario} ({threshold_type})", fontsize=14)

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.25, -0.04, 0.5, 0.02])  # Adjusted placement
    cbar = plt.colorbar(contour, cax=cbar_ax, orientation="horizontal", label=color_label)
    
    # Add overall title
    if index_name == "CUTI":
        plt.suptitle(f"Mean CUTI during surface [H+] extremes", fontsize=18)
    elif index_name == "BEUTI":
        plt.suptitle(f"Mean BEUTI during surface [H+] extremes", fontsize=18)
    elif index_name == "DIC_upwelling":
        plt.suptitle(f"Mean DIC upwelling during surface [H+] extremes", fontsize=18)
    elif index_name == "Alk_upwelling":
        plt.suptitle(f"Mean alkalinity upwelling during surface [H+] extremes", fontsize=18)
    elif index_name == "Temp_upwelling":
        plt.suptitle(f"Mean temperature during surface [H+] extremes", fontsize=18)
    elif index_name == "Salt_upwelling":
        plt.suptitle(f"Mean salinity upwelling during surface [H+] extremes", fontsize=18)


    # Save & Show
    if filename:
        plt.savefig(filename, dpi=200, transparent=True, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    plt.show()


# **Define bounds and load landmask**
lat_bounds = (30, 47)
lon_bounds = (225, 245)
bounds = (lat_bounds, lon_bounds)

landmask_etopo = AuxGetter.get_etopo_data()

# **Call function to plot for each upwelling index**
for index_name in ["CUTI", "BEUTI"]: #"CUTI", "BEUTI", "DIC_upwelling", "Alk_upwelling", "Temp_upwelling", "Salt_upwelling"
    plot_mean_upwelling_index_scenarios(
        masked_upwelling_indices, landmask_etopo, bounds,
        index_name=index_name,
        filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/upwelling/mean_{index_name}_during_extremes_5x5_gridcells.png'
    )


#%%

# Plots the anomalies of the yearly mean upwelling indices between present and future scenarios


def plot_upwelling_anomalies(masked_variables, landmask_etopo, bounds, index_name="CUTI", config="romsoc_fully_coupled", filename=None):
    """
    Plots the anomaly of the selected upwelling index (CUTI, BEUTI, DIC_upwelling, Alk_upwelling) 
    for SSP245 and SSP585 compared to the present scenario.
    """
    
    lat_bounds, lon_bounds = bounds

    # Extract dataset for each scenario
    index_present = masked_variables[config]["present"][index_name].mean(dim="time")
    index_ssp245 = masked_variables[config]["ssp245"][index_name].mean(dim="time")
    index_ssp585 = masked_variables[config]["ssp585"][index_name].mean(dim="time")

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
        #"Temp_upwelling": (-1, 1),
        #"Salt_upwelling": (-0.1, 0.1),
    }

    vmin, vmax = vmin_vmax_dict.get(index_name, (-1, 1))  # Default range if index is missing

    # Define contour levels
    levels = np.linspace(vmin, vmax, 20)
    cmap = cmocean.cm.balance  # Diverging colormap for positive/negative changes

    # Define colorbar label based on index
    color_labels = {
        "CUTI": "CUTI anomaly during extremes (future - present)",
        "BEUTI": "BEUTI anomaly during extremes(future - present)",
        #"DIC_upwelling": "DIC upwelling anomaly during extremes (future - present)",
        #"Alk_upwelling": "Alk upwelling anomaly during extremes (future - present)",
        #"Temp_upwelling": "Temperature upwelling anomaly during extremes (future - present)",
        #"Salt_upwelling": "Salinity upwelling anomaly  during extremes (future - present)",
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

landmask_etopo = AuxGetter.get_etopo_data()

# **Call function to plot anomalies for each upwelling index**
for index_name in ["CUTI", "BEUTI"]: #"CUTI", "BEUTI", "DIC_upwelling", "Alk_upwelling",
    plot_upwelling_anomalies(
        masked_upwelling_indices, landmask_etopo, bounds,
        index_name=index_name,
        filename=f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/upwelling/anomaly_{index_name}_during_extremes.png'
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

landmask_etopo = AuxGetter.get_etopo_data()

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