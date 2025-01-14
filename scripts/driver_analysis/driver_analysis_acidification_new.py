"""
author: Fiona Pfäffli
description: Driver analysis of ocean acidification extremes 
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
import pandas as pd
import cmocean
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter

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

#%% 
#########################
#           1           #
#########################

# Define the threshold 
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.Hplus_instance() #95th percentile threshold

# Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present','ssp585'] # ,'ssp245'
configs = ['romsoc_fully_coupled'] # [ms_only'] 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '001'
vert_struct = 'zavg'    # 'avg'
depth = 0

#%% 
# Get the model datasets for the oceanic and atmospheric variables

ocean_ds = dict()
for config in configs:
     ocean_ds[config] = dict()
     for scenario in scenarios:
          print(f'--{config}, {scenario}--')
          print('ocean...')
          ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config,scenario,simulation_type,ensemble_run,model_temp_resolution,vert_struct,vtype='oceanic',parent_model=parent_model)

#%% 
# load the data at the respective location

variables = dict()
for config in configs:
     variables[config] = dict()
     for scenario in scenarios:
          print(f'Getting the variables for config {config} and scenario {scenario}.')
          #
          # oceanic variables
          variables[config][scenario] = dict()
          print('Hplus')
          # Convert pH to [H+] concentration
          pH = ocean_ds[config][scenario].pH_offl.isel(depth=0).load()
          variables[config][scenario]['Hplus'] = np.power(10, -pH)
          

#%%
varias = ['Hplus']

#%% 
# Get the climatology for each variable
print('Get the climatology')
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
# Get the threshold for each variable
print('Get the present day threshold')
varia = 'Hplus'
thresholds = dict()
for config in configs:
    thresholds[config] = dict()
    for scenario in ['present']:  # ,'ssp245','ssp585'
        thresholds[config][scenario] = dict()
        print(f'{config}, {scenario}')
        threshold, threshold_366 = ModelGetter.get_threshold('Hplus', 0, 'relative', 95, config, scenario)  # variable, depth_level, threshold_type, threshold_value, config, scenario
        concatenated_threshold = ModelGetter.concatenate_yearly_arrays(threshold, threshold_366, start_year=2011, end_year=2021)
        #rename 'day_of_year_adjusted' to 'time' in the concatenated threshold
        if 'day_of_year_adjusted' in concatenated_threshold.dims:
            concatenated_threshold = concatenated_threshold.rename({'day_of_year_adjusted': 'time'})
        elif 'day_of_year_adjusted' in concatenated_threshold.coords:
            concatenated_threshold = concatenated_threshold.rename_coords({'day_of_year_adjusted': 'time'})
        #assign the renamed threshold to the dictionary
        thresholds[config][scenario][varia] = concatenated_threshold

#%%
# Adjust the thresholds
print('Adjust the thresholds')
thresholds_mult = dict()
for config in configs:
     thresholds_mult[config] = dict()
     for scenario in ['present','ssp585']:
          thresholds_mult[config][scenario] = dict()
          print(f'{config}, {scenario}')
          thresholds_mult[config][scenario]['present'] = thresholds[config]['present'][varia]
          thresholds_mult[config][scenario]['present_plus_meandelta'] = thresholds[config]['present'][varia] + (clims[config][scenario][varia].mean(dim='time') - clims[config]['present'][varia].mean(dim='time'))



#%%
#########################
#           2           #
#########################

# Variables to load the sensitivity fields and the extreme masks
scenario = 'ssp585'
depthlevel = 0
eta_rho_cho = 500
xi_rho_cho = 200

#%%
# Load the variables, thresholds, and sensitivity fields
present = variables['romsoc_fully_coupled']['present']['Hplus']
future = variables['romsoc_fully_coupled'][scenario]['Hplus']
thresholds_present = thresholds_mult['romsoc_fully_coupled'][scenario]['present']
thresholds_present_meandelta = thresholds_mult['romsoc_fully_coupled'][scenario]['present_plus_meandelta']

# Compute baselines
future_fixed_baseline = future - thresholds_present
future_moving_baseline = future - thresholds_present_meandelta
print("Future baseline computed")

# Define masks for the two threshold types
masks_fixed = {
    'non_extremes': (future_fixed_baseline <= 0) & (present <= 0),
    'new_extremes': (future_fixed_baseline > 0) & (present <= 0),
    'disappearing_extremes': (future_fixed_baseline <= 0) & (present > 0),
    'intensifying_extremes': (future_fixed_baseline > 0) & (present > 0) & (future_fixed_baseline >= present),
    'weakening_extremes': (future_fixed_baseline > 0) & (present > 0) & (future_fixed_baseline < present),
}

masks_moving = {
    'non_extremes': (future_moving_baseline <= 0) & (present <= 0),
    'new_extremes': (future_moving_baseline > 0) & (present <= 0),
    'disappearing_extremes': (future_moving_baseline <= 0) & (present > 0),
    'intensifying_extremes': (future_moving_baseline > 0) & (present > 0) & (future_moving_baseline >= present),
    'weakening_extremes': (future_moving_baseline > 0) & (present > 0) & (future_moving_baseline < present),
}

print("Masks defined")


#%%
# Save the present scenario, future baselines, and masks to NetCDF files

# Define the output directory
output_dir = "/nfs/sea/work/fpfaeffli/baselines_and_masks/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Save the present scenario
present_ds = xr.Dataset({'present': present})
present_ds.to_netcdf(os.path.join(output_dir, "present_scenario.nc"))
print("Present data saved with coordinates")

# Save the future baselines
future_fixed_baseline_ds = xr.Dataset({'future_fixed_baseline': future_fixed_baseline})
future_fixed_baseline_ds.to_netcdf(os.path.join(output_dir, "future_fixed_baseline.nc"))
print("Future fixed baseline saved with coordinates")

future_moving_baseline_ds = xr.Dataset({'future_moving_baseline': future_moving_baseline})
future_moving_baseline_ds.to_netcdf(os.path.join(output_dir, "future_moving_baseline.nc"))
print("Future moving baseline saved with coordinates")

# Save the fixed masks
masks_fixed_ds = xr.Dataset({f'mask_fixed_{key}': value for key, value in masks_fixed.items()})
masks_fixed_ds.to_netcdf(os.path.join(output_dir, "extremes_masks_fixed_baseline.nc"))
print("Fixed baseline masks saved with coordinates")

# Save the moving masks
masks_moving_ds = xr.Dataset({f'mask_moving_{key}': value for key, value in masks_moving.items()})
masks_moving_ds.to_netcdf(os.path.join(output_dir, "extremes_masks_moving_baseline.nc"))
print("Moving baseline masks saved with coordinates")



#%%
# Load hte present scenario, future baselines, and masks from netCDFfiles

output_dir = "/nfs/sea/work/fpfaeffli/baselines_and_masks/"

# Load present 
present_ds = xr.open_dataset(os.path.join(output_dir, "present_scenario.nc"))
present = present_ds['present']
print("Present loaded:", present)

# Load future baselines
future_fixed_baseline_ds = xr.open_dataset(os.path.join(output_dir, "future_fixed_baseline.nc"))
future_fixed_baseline = future_fixed_baseline_ds['future_fixed_baseline']
print("Future fixed baseline loaded:", future_fixed_baseline)

future_moving_baseline_ds = xr.open_dataset(os.path.join(output_dir, "future_moving_baseline.nc"))
future_moving_baseline = future_moving_baseline_ds['future_moving_baseline']
print("Future moving baseline loaded:", future_moving_baseline)

# Load masks
masks_fixed_ds = xr.open_dataset(os.path.join(output_dir, "extremes_masks_fixed_baseline.nc"))
masks_fixed = {var: masks_fixed_ds[var] for var in masks_fixed_ds.data_vars}
print("Fixed baseline masks loaded:", list(masks_fixed.keys()))

masks_moving_ds = xr.open_dataset(os.path.join(output_dir, "extremes_masks_moving_baselie.nc"))
masks_moving = {var: masks_moving_ds[var] for var in masks_moving_ds.data_vars}
print("Moving baseline masks loaded:", list(masks_moving.keys()))


#%%
#########################
#           3           #
#########################

# Aggregates all the monthly chunks of the sensitivity fields into data arrays for the whole hindcast period
# This was necessary to have the same dimension and shape for the sensitivity fields and the extreme masks

# Define the range of years and other configurations
years = range(2011, 2022)  # Hindcast period (2011-2021 inclusive)
scenario = 'ssp585'

# Paths to ensemble data
base_path = '/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled'

# File pattern 
file_pattern = 'pactcs30_romsoc_fully_coupled_{scenario}_daily_{year}_*_mocsy_sensitivities_depthlevel_0.nc'

# Output directory
output_dir = "/nfs/sea/work/fpfaeffli/aggregated_sensitivities/"
os.makedirs(output_dir, exist_ok=True)

# Function to aggregate and save for the specific variable (Alk, T, S, DIC)
def aggregate_sensitivity_field(variable_name, base_path, file_pattern, years, output_dir):
    all_files = []
    for year in years:
        path = os.path.join(base_path, f"{scenario}/daily/z_avg/mocsy_co2_chemistry/{year}/")
        files = sorted(glob.glob(os.path.join(path, file_pattern.format(scenario=scenario, year=year))))
        all_files.extend(files)

    print(f"Aggregating files for {variable_name}...")
    
    # Load and concatenate all files along the time dimension
    ds = xr.open_mfdataset(all_files, concat_dim="time", combine="nested", parallel=True)
    
    # Extract the specific variable as a DataArray
    if variable_name not in ds.data_vars:
        raise ValueError(f"Variable {variable_name} not found in the dataset.")
    
    data_array = ds[variable_name]

    # Save the aggregated DataArray to a NetCDF file
    output_file = os.path.join(output_dir, f"aggregated_{variable_name}_all_years.nc")
    data_array.to_netcdf(output_file)
    print(f"Saved aggregated {variable_name} to {output_file}")
    return data_array

# List of sensitivity variables to process
sensitivity_variables = ['dh_dtem', 'dh_dsal', 'dh_ddic', 'dh_dalk']

# Process each variable and save as a separate file
for var_name in sensitivity_variables:
    aggregate_sensitivity_field(var_name, base_path, file_pattern, years, output_dir)

#%%

# Loading the aggregated sensitivity fields from netCDF files

# Define the directory where the files are stored
aggregated_dir = "/nfs/sea/work/fpfaeffli/aggregated_sensitivities/"

# Define the sensitivity variables
sensitivity_variables = ['dh_dtem', 'dh_dsal', 'dh_ddic', 'dh_dalk']

# Load each variable as its own 
sensitivities_dh_dtem = xr.open_dataarray(os.path.join(aggregated_dir, "aggregated_dh_dtem_all_years.nc"))
sensitivities_dh_dsal = xr.open_dataarray(os.path.join(aggregated_dir, "aggregated_dh_dsal_all_years.nc"))
sensitivities_dh_ddic = xr.open_dataarray(os.path.join(aggregated_dir, "aggregated_dh_ddic_all_years.nc"))
sensitivities_dh_dalk = xr.open_dataarray(os.path.join(aggregated_dir, "aggregated_dh_dalk_all_years.nc"))

# Verify each variable
print(f"sensitivities_dh_dtem: dims = {sensitivities_dh_dtem.dims}, shape = {sensitivities_dh_dtem.shape}")
print(f"sensitivities_dh_dsal: dims = {sensitivities_dh_dsal.dims}, shape = {sensitivities_dh_dsal.shape}")
print(f"sensitivities_dh_ddic: dims = {sensitivities_dh_ddic.dims}, shape = {sensitivities_dh_ddic.shape}")
print(f"sensitivities_dh_dalk: dims = {sensitivities_dh_dalk.dims}, shape = {sensitivities_dh_dalk.shape}")



#%%
#########################
#           4          #
#########################

# Function to apply extreme masks to sensitivity fields
def apply_extreme_mask(sensitivity_field, masks):
    masked_fields = {}
    for mask_name, mask in masks.items():
        # Apply the mask and store the result
        masked_field = sensitivity_field.where(mask, other=np.nan)
        masked_fields[mask_name] = masked_field
    return masked_fields

# Apply extreme masks to sensitivities for fixed baseline
print("Applying fixed baseline masks...")
masked_sensitivities_fixed = {
    'dh_dtem': apply_extreme_mask(sensitivities_dh_dtem, masks_fixed),
    'dh_dsal': apply_extreme_mask(sensitivities_dh_dsal, masks_fixed),
    'dh_ddic': apply_extreme_mask(sensitivities_dh_ddic, masks_fixed),
    'dh_dalk': apply_extreme_mask(sensitivities_dh_dalk, masks_fixed),
}
print("Fixed baseline masks applied.")

# Apply extreme masks to sensitivities for moving baseline
print("Applying moving baseline masks...")
masked_sensitivities_moving = {
    'dh_dtem': apply_extreme_mask(sensitivities_dh_dtem, masks_moving),
    'dh_dsal': apply_extreme_mask(sensitivities_dh_dsal, masks_moving),
    'dh_ddic': apply_extreme_mask(sensitivities_dh_ddic, masks_moving),
    'dh_dalk': apply_extreme_mask(sensitivities_dh_dalk, masks_moving),
}
print("Moving baseline masks applied.")


# %%
# Save the masked sensitivities to NetCDF files (all drivers seperately)

# Define the output directory
output_dir = "/nfs/sea/work/fpfaeffli/masked_sensitivities/"
os.makedirs(output_dir, exist_ok=True)

# Save the masked sensitivities for fixed baseline
print("Saving masked sensitivities for fixed baseline...")
for var_name, masked_fields in masked_sensitivities_fixed.items():
    for mask_name, masked_field in masked_fields.items():
        # Create a file name based on variable and mask name
        output_file = os.path.join(output_dir, f"masked_{var_name}_{mask_name}_fixed.nc")
        masked_field.to_netcdf(output_file)
        print(f"Saved {var_name} with {mask_name} mask (fixed baseline) to {output_file}")

# Save the masked sensitivities for moving baseline
print("Saving masked sensitivities for moving baseline...")
for var_name, masked_fields in masked_sensitivities_moving.items():
    for mask_name, masked_field in masked_fields.items():
        # Create a file name based on variable and mask name
        output_file = os.path.join(output_dir, f"masked_{var_name}_{mask_name}_moving.nc")
        masked_field.to_netcdf(output_file)
        print(f"Saved {var_name} with {mask_name} mask (moving baseline) to {output_file}")

print("All masked sensitivities for each driver seperately have been saved.")



# %%
# Save the masked sensitivities to NetCDF files (all drivers combined, for each extreme type)

import os
import glob
import xarray as xr

# Define the directory containing the masked files
masked_dir = "/nfs/sea/work/fpfaeffli/masked_sensitivities/"

# List of extreme types
extreme_types = ["non_extremes", "new_extremes", "disappearing_extremes", "intensifying_extremes", "weakening_extremes"]

# Loop through each extreme type and combine its variables
for extreme_type in extreme_types:
    # Find all files for the specific extreme type
    files_to_combine = glob.glob(os.path.join(masked_dir, f"masked_*_{extreme_type}_*.nc"))
    
    if not files_to_combine:
        print(f"No files found for {extreme_type}. Skipping...")
        continue
    
    # Create a dictionary to hold the loaded data arrays
    data_arrays = {}
    
    # Load each file and store it in the dictionary
    for file in files_to_combine:
        var_name = os.path.basename(file).split("_")[1]  # Extract variable name (e.g., dh_dtem)
        print(f"Loading {file} for variable {var_name}...")
        # Open the dataset with `xr.open_dataset` to handle the conflicting dimension issue
        dataset = xr.open_dataset(file)
        # Remove the coordinate variable if it conflicts with a dimension
        if "xi_rho" in dataset.coords and "xi_rho" in dataset.dims:
            dataset = dataset.reset_coords("xi_rho", drop=True)
        if "eta_rho" in dataset.coords and "eta_rho" in dataset.dims:
            dataset = dataset.reset_coords("eta_rho", drop=True)
        # Add the data variable to the dictionary
        data_arrays[var_name] = dataset[var_name]
    
    # Combine all variables into an xarray.Dataset
    combined_dataset = xr.Dataset(data_arrays)
    
    # Save the combined dataset to a new NetCDF file
    output_file = os.path.join(masked_dir, f"combined_{extreme_type}_sensitivities.nc")
    combined_dataset.to_netcdf(output_file)
    print(f"Combined dataset for {extreme_type} saved to {output_file}")

print("All extreme types processed and saved.")

#%% 
#########################
#           5           #
#########################

# Function to rename dimensions and variables eta_rho and xi_rho to avoid conflicts and save it into a new file

def rename_colliding_dims_in_netcdf(infile, outfile=None):
   
    if outfile is None:
        # Default behavior: append '_renamed' to the original filename
        base, ext = os.path.splitext(infile)
        outfile = base + "_renamed.nc"

    # Open original file in read mode
    with nc.Dataset(infile, mode='r') as src:
        # Create a new netCDF for output, using the same file format
        with nc.Dataset(outfile, mode='w', format=src.file_format) as dst:

            # 1) Copy global attributes
            for attr_name in src.ncattrs():
                dst.setncattr(attr_name, getattr(src, attr_name))

            # 2) Rename dimensions if needed
            dim_rename = {}
            for dim_name, dim_obj in src.dimensions.items():
                new_dim_name = dim_name
                if dim_name == 'xi_rho':
                    new_dim_name = 'xi_rho_dim'
                elif dim_name == 'eta_rho':
                    new_dim_name = 'eta_rho_dim'

                dim_rename[dim_name] = new_dim_name
                # Create this dimension in the destination file
                dst.createDimension(
                    new_dim_name,
                    (len(dim_obj) if not dim_obj.isunlimited() else None)
                )

            # 3) Copy and rename variables
            for var_name, var_obj in src.variables.items():
                # Decide if this variable name must be changed
                if var_name == 'xi_rho':
                    new_var_name = 'xi_rho_var'
                elif var_name == 'eta_rho':
                    new_var_name = 'eta_rho_var'
                else:
                    new_var_name = var_name

                # Map old dimensions to new dimension names
                var_dims = [dim_rename[d] for d in var_obj.dimensions]

                # Handle _FillValue properly:
                if '_FillValue' in var_obj.ncattrs():
                    fill_value = var_obj.getncattr('_FillValue')
                else:
                    fill_value = None

                # Create the variable in the destination file
                dst_var = dst.createVariable(
                    new_var_name,
                    var_obj.dtype,
                    dimensions=var_dims,
                    fill_value=fill_value
                )

                # Copy over all other variable attributes EXCEPT _FillValue
                for attr_name in var_obj.ncattrs():
                    if attr_name == '_FillValue':
                        continue
                    attr_val = var_obj.getncattr(attr_name)
                    dst_var.setncattr(attr_name, attr_val)

                # Copy the data
                dst_var[:] = var_obj[:]

    print(f"Created '{outfile}' with renamed dims/vars.\n")


def batch_rename_masked_files(input_dir, output_dir=None):
    """
    Looks in 'input_dir' for netCDF files that:
      - Start with 'masked_'
      - End with '.nc'
      - Do NOT contain 'ens00'
    For each, call rename_colliding_dims_in_netcdf to fix dimension collisions,
    writing the new file into 'output_dir' (or, if None, into 'input_dir').

    Adjust as needed for your specific naming/filtering preferences.
    """
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    # Iterate over files in input_dir
    for filename in os.listdir(input_dir):
        # Must start with 'masked_' and end with '.nc'
        if filename.startswith('masked_') and filename.endswith('.nc'):
            # Skip any containing 'ens00'
            if 'ens00' in filename:
                print(f"Skipping {filename} (contains 'ens00').")
                continue

            infile_path = os.path.join(input_dir, filename)
            base, ext = os.path.splitext(filename)
            out_name = base + "_renamed.nc"
            outfile_path = os.path.join(output_dir, out_name)

            print(f"Processing {filename} => {out_name} ...")
            rename_colliding_dims_in_netcdf(infile_path, outfile=outfile_path)


if __name__ == "__main__":
    input_directory = "/nfs/sea/work/fpfaeffli/masked_sensitivities"
    output_directory = "/nfs/sea/work/fpfaeffli/masked_sensitivities" 

    batch_rename_masked_files(input_directory, output_directory)





# %%

# Load the renamed files for masked sensitivities

def load_renamed_nc_files(directory):
    # Pattern to match: anything ending in '_renamed.nc'
    pattern = os.path.join(directory, "*_renamed.nc")
    file_list = sorted(glob.glob(pattern))

    datasets = {}
    for filepath in file_list:
        filename = os.path.basename(filepath)
        # Skip if 'ens00' in filename
        if "ens00" in filename:
            print(f"Skipping {filename} (contains 'ens00').")
            continue
        
        # Open the renamed file in xarray
        ds = xr.open_dataset(filepath)
        
        # Create a short key for the dict, e.g. remove '.nc'
        key = filename.replace(".nc", "")
        datasets[key] = ds
        print(f"Loaded '{filename}' as '{key}'")

    return datasets

# Load netCDF files
if __name__ == "__main__":
    renamed_dir = "/nfs/sea/work/fpfaeffli/masked_sensitivities"  # or your chosen dir
    loaded_datasets = load_renamed_nc_files(renamed_dir)



# %%
#########################
#           6           #
#########################

# Crop the netCDFs to remove the NaNs in the x/y coordinates for plotting afterwards 

def crop_and_save_netcdf(file_path, lat_bounds, lon_bounds, output_dir):
    """
    Crops a netCDF file to the lat/lon of the CaCS  and saves the result.
    """
    print(f"Cropping file: {file_path}")
    fname = os.path.basename(file_path)
    output_file = os.path.join(output_dir, fname.replace(".nc", "_cropped.nc"))

    # Open the dataset
    ds = xr.open_dataset(file_path)

    # Identify lat/lon
    possible_lat_names = ["lat", "lat_rho", "eta_rho", "eta_rho_dim"]
    possible_lon_names = ["lon", "lon_rho", "xi_rho", "xi_rho_dim"]

    lat_name = None
    lon_name = None

    for name in possible_lat_names:
        if name in ds.variables:
            lat_name = name
            break
    for name in possible_lon_names:
        if name in ds.variables:
            lon_name = name
            break

    if not lat_name or not lon_name:
        print(f"  ERROR: Lat/Lon not found in {file_path}. Skipping.")
        ds.close()
        return

    # Extract lat/lon
    lat = ds[lat_name]
    lon = ds[lon_name]

    # Check if lat/lon are 2D
    if lat.ndim != 2 or lon.ndim != 2:
        print(f"  ERROR: Lat/Lon are not 2D in {file_path}. Skipping.")
        ds.close()
        return

    # Crop based on the specified bounds
    crop_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1]) & \
                (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    # Mask the data with NaNs outside the crop area
    cropped_ds = ds.where(crop_mask, drop=True)

    # Save the cropped dataset
    cropped_ds.to_netcdf(output_file)
    print(f"  Saved cropped file to: {output_file}")

    # Close the datasets
    ds.close()
    cropped_ds.close()


def main():
    # Directory containing *renamed* netCDF files
    input_dir = "/nfs/sea/work/fpfaeffli/masked_sensitivities"
    output_dir = "/nfs/sea/work/fpfaeffli/cropped_sensitivities"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the bounding box for cropping
    lat_bounds = (30, 47)
    lon_bounds = (225, 245)

    # Find all *_renamed.nc files
    file_pattern = os.path.join(input_dir, "*_renamed.nc")
    file_list = sorted(glob.glob(file_pattern))

    if not file_list:
        print("No *_renamed.nc files found.")
        return

    # Loop through files and crop
    for file_path in file_list:
        crop_and_save_netcdf(file_path, lat_bounds, lon_bounds, output_dir)


if __name__ == "__main__":
    main()



#%%
# PLOTTING 

################################
#              6               #
################################


# At the moment only for Alkalinity
# Shows the mean sensitivity over time for a specific extreme type, but only there were extreme events got detected.


# Define the directory containing cropped NetCDF files
directory = "/nfs/sea/work/fpfaeffli/cropped_sensitivities/"

# Define the bounding box for cropping
lat_bounds = (30, 47)
lon_bounds = (225, 245)

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith("_renamed_cropped.nc"):  # Filter for cropped sensitivity files
        filepath = os.path.join(directory, filename)
        print(f"Processing file: {filename}")
        
        try:
            # Open and process the file
            with Dataset(filepath, mode="r") as nc:
                # Check and process data (as in your code)
                variables = list(nc.variables.keys())
                if 'dh_dalk' not in variables:
                    print(f"'dh_dalk' not found in {filename}. Available variables: {variables}")
                    continue
                
                data = nc.variables['dh_dalk'][:]
                lon = nc.variables['lon'][:]
                lat = nc.variables['lat'][:]
                Lon2D, Lat2D = np.meshgrid(lon, lat) if len(lon.shape) == 1 else (lon, lat)
                data_mean = np.mean(data, axis=0)

                # Ensure correct shapes before plotting
                if data_mean.shape == Lon2D.shape and data_mean.shape == Lat2D.shape:
                    # Only one plot per file
                    plt.figure(figsize=(12, 6))
                    contour = plt.contourf(Lon2D, Lat2D, data_mean, levels=20, cmap='viridis')
                    
                    # Add landmask
                    landmask_etopo = PlotFuncs.get_etopo_data()
                    plt.contourf(landmask_etopo.lon, landmask_etopo.lat, landmask_etopo, colors='#000000')

                    # Set plot limits to zoom into the CaCS
                    plt.xlim(lon_bounds)
                    plt.ylim(lat_bounds)

                    # Finalize plot
                    plt.colorbar(contour, label='Mean of dh_dalk')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.title(f'Mean of dh_dalk for {filename}')
                    plt.show()
                    plt.close()  # Prevents residual plots
                else:
                    print(f"Shape mismatch in {filename}: data_mean {data_mean.shape}, "
                          f"Lon2D {Lon2D.shape}, Lat2D {Lat2D.shape}. Skipping plot.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


# %%
