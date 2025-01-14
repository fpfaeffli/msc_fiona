#%% Enable the visibility of the modules for the import functions
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

#%% Load the necessary packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter


#%% Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present','ssp245','ssp585']
configs = ['romsoc_fully_coupled']
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '000'
vert_struct = 'zavg'    # 'avg'

#%% Get the model datasets for the oceanic variables
ocean_ds = dict()
for config in configs:
    ocean_ds[config] = dict()
    for scenario in scenarios:
        print(f'--{config}, {scenario}--')
        print('Loading ocean data...')
        ocean_ds[config][scenario] = ModelGetter.get_model_dataset(config, scenario, simulation_type, ensemble_run, model_temp_resolution, vert_struct, vtype='oceanic', parent_model=parent_model)

#%%
# Get the variables for each config and scenario
variables = dict()
for config in configs:
    variables[config] = dict()
    for scenario in scenarios:
        print(f'Getting the variables for config {config} and scenario {scenario}.')

        # Load the variables
        # Compute Hplus from pH_offl
        variables[config][scenario] = dict()
        if "pH_offl" in ocean_ds[config][scenario]:
            print('Calculating Hplus...')
            pH = ocean_ds[config][scenario]["pH_offl"].isel(depth=0).load()
            variables[config][scenario]['Hplus'] = np.power(10, -pH)

        #Load the rest of the variables
        for var in ['salt', 'DIC', 'Alk', 'omega_arag_offl', 'temp']:
            if var in ocean_ds[config][scenario]:
                print(f"Loading {var}...")
                variables[config][scenario][var] = ocean_ds[config][scenario][var].isel(depth=0).load()
   
      
#%% Define variables
varias = ['Hplus', 'salt', 'DIC', 'Alk', 'omega_arag_offl', 'temp'] 

#%% Get model regions and distance to coast

# Get the distance to coast file from ROMS and the regions over which to calculate the statistics
model_d2coasts = dict()
model_d2coasts['roms_only'] = ModelGetter.get_distance_to_coast(vtype='oceanic')

# Get the model regions
model_regions = dict()
model_regions['roms_only'] = GetRegions.define_CalCS_regions(model_d2coasts['roms_only'].lon, model_d2coasts['roms_only'].lat, model_d2coasts['roms_only'])

# Get the model area
model_area = ModelGetter.get_model_area()

#%%
# Calculate the regional means for each variable

regions = {
    "northern": (40.5, 50),
    "central": (34.7, 40.5),
    "southern": (30, 34.7)
}

for config in configs:
    for scenario in scenarios:
        for var in varias:
            print(f"\nRegional means for {var} in {config}, {scenario}:")

            data = variables[config][scenario][var]

            # Filter out all values over 15 to avoid the land points getting included in the mean (for pH and omega)
            if var in varias == ['Hplus', 'omega_arag_offl']:
                data = data.where(data <= 15)


            for region_name, (lat_min, lat_max) in regions.items():
                # Create a mask for the latitude range
                lat = data.coords["lat"]
                mask = (lat >= lat_min) & (lat <= lat_max)

                region_data = data.where(mask, drop=True)

                # Resample the data to daily means
                daily_data = region_data.resample(time='1D').mean(dim="time", skipna=True)

                for day in daily_data['time']:
                    daily_mean = daily_data.sel(time=str(day.values)).mean(dim=["eta_rho", "xi_rho"], skipna=True).values.item()
                    print(f"  {region_name.capitalize()} Region ({lat_min}° to {lat_max}°), Day {day.values}: {daily_mean}")


#%% Save daily means to a NetCDF file
output_netcdf_file = f"/nfs/sea/work/fpfaeffli/calculated_means/daily_means/daily_means_{varias}_for_distinct_regions_NCS.nc"

dataset_dict = {}
for region_name in regions.keys():
    for var in varias:
        times = np.array([np.datetime64(date) for date in daily_mean[region_name][var].keys()])
        values = np.array(list(daily_mean[region_name][var].values()))

        # Create DataArray for each variable and region
        dataset_dict[f"{region_name}_{var}"] = xr.DataArray(
            data=values,
            dims=["time"],
            coords={"time": times},
            name=f"{region_name}_{var}"
        )

# Combine all DataArrays into a Dataset and save as NetCDF
xr.Dataset(dataset_dict).to_netcdf(output_netcdf_file)
print(f"Daily means saved to {output_netcdf_file}.")

#%% Function to plot all variables for each region
def plot_timeseries_all_variables(daily_means_file, region_name, variables, start_date, end_date):
    """
    Plot timeseries for all variables in a single plot for a specific region.

    Parameters:
        daily_means_file (str): Path to the NetCDF file containing daily means.
        region_name (str): The name of the region to plot (e.g., "northern").
        variables (list): List of variable names to include in the plot.
        start_date (str): Start date for the plot (e.g., "2011-01-01").
        end_date (str): End date for the plot (e.g., "2021-12-31").
    """
    dataset = xr.open_dataset(daily_means_file)

    plt.figure(figsize=(12, 8))
    for var in variables:
        data_key = f"{region_name}_{var}"
        if data_key in dataset:
            data = dataset[data_key].sel(time=slice(start_date, end_date))
            plt.plot(data.time, data, label=f"{var.capitalize()}")

    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.title(f"Daily Timeseries for All Variables in {region_name.capitalize()} Region")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%% Main plotting loop
variables = ['Hplus', 'salt', 'DIC', 'Alk', 'omega_arag_offl', 'temp']
for region_name in regions.keys():
    plot_timeseries_all_variables(output_netcdf_file, region_name, variables, "2011-01-01", "2021-12-31")



# %%
 #Calculated means
 