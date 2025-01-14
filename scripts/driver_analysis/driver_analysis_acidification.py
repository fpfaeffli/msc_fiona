"""
author: Fiona Pfäffli
description: Driver analysis of ocean acidification extremes 
"""

#%%
import sys
import os
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/climatologies_and_thresholds/')

import numpy as np
import xarray as xr
import glob
import pandas as pd
import matplotlib.pyplot as plt

from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter
from func_for_clim_thresh import ThreshClimFuncs

#%% Define parameters for loading sensitivity fields
eta_rho_cho = 500
xi_rho_cho = 200
depthlevel = 0
year = 2015
month = 'all' # Options: specific month (e.g., 6) or 'all'
scenario = 'present'

#%% Load sensitivity fields
if month == 'all':
    ds = xr.open_mfdataset(sorted(glob.glob(f'/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled/{scenario}/daily/z_avg/z_avg_2015_*_37zlevs_full_1x1meanpool_downsampling.nc')),concat_dim='time',combine='nested')
    ds['time'] = pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D')

    ds_sens = xr.open_mfdataset(sorted(glob.glob(f'/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled/{scenario}/daily/z_avg/mocsy_co2_chemistry/{year}/pactcs30_romsoc_fully_coupled_{scenario}_daily_{year}_*_mocsy_sensitivities_depthlevel_0.nc')))
else:
    ds = xr.open_dataset(f'/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled/{scenario}/daily/z_avg/z_avg_2015_{month:03d}_37zlevs_full_1x1meanpool_downsampling.nc')
    ds_sens = xr.open_dataset(f'/nfs/sea/work/koehne/roms/analysis/pactcs30/future_sim/romsoc_fully_coupled/{scenario}/daily/z_avg/mocsy_co2_chemistry/{year}/pactcs30_romsoc_fully_coupled_{scenario}_daily_{year}_{month:03d}_mocsy_sensitivities_depthlevel_0.nc')

#%% Extract variables and sensitivities
omegaa = ds.omega_arag_offl.isel(depth=depthlevel,eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
pH = ds.pH_offl.isel(depth=depthlevel,eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
hplus = 10**(-pH)
temp = ds.temp.isel(depth=depthlevel,eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
dic = ds.DIC.isel(depth=depthlevel,eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
alk = ds.Alk.isel(depth=depthlevel,eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
salt = ds.salt.isel(depth=depthlevel,eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)

dh_dtemp = ds_sens.dh_dtem.isel(eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
dh_dsalt = ds_sens.dh_dsal.isel(eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
dh_ddic = ds_sens.dh_ddic.isel(eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
dh_dalk = ds_sens.dh_dalk.isel(eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)

domegaa_dtemp = ds_sens.domegaa_dtem.isel(eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
domegaa_dsalt = ds_sens.domegaa_dsal.isel(eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
domegaa_ddic = ds_sens.domegaa_ddic.isel(eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)
domegaa_dalk = ds_sens.domegaa_dalk.isel(eta_rho=eta_rho_cho, xi_rho=xi_rho_cho)

#%% 
# Plot an individual timeseries throughout a full year with the anomalies relative to the annual mean
# from Eike's script "test_sensitivities.py"

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(hplus - np.mean(hplus), label=r'Model output $H^{+}_{anom}$', color='k', linewidth=3)
ax[0].plot(dh_dtemp * (temp - np.mean(temp)), label=r'$\frac{\partial H^{+}}{\partial T} T_{anom}$', color='C1', linewidth=2)
ax[0].plot(dh_dsalt * (salt - np.mean(salt)), label=r'$\frac{\partial H^{+}}{\partial S} S_{anom}$', color='C3', linewidth=2)
ax[0].plot(dh_ddic * (dic - np.mean(dic)), label=r'$\frac{\partial H^{+}}{\partial DIC} DIC_{anom}$', color='C0', linewidth=2)
ax[0].plot(dh_dalk * (alk - np.mean(alk)), label=r'$\frac{\partial H^{+}}{\partial Alk} Alk_{anom}$', color='C2', linewidth=2)
ax[0].plot(
    dh_dtemp * (temp - np.mean(temp)) +
    dh_dsalt * (salt - np.mean(salt)) +
    dh_ddic * (dic - np.mean(dic)) +
    dh_dalk * (alk - np.mean(alk)),
    label='sum of linearized terms', linewidth=3, color='C4')
ax[0].legend(loc='lower left', bbox_to_anchor=(1.02, 0.1))

ax[1].plot(omegaa - np.mean(omegaa), label=r'Model output $\Omega_{anom}$', color='k', linewidth=3)
ax[1].plot(domegaa_dtemp * (temp - np.mean(temp)), label=r'$\frac{\partial \Omega_{A}}{\partial T} T_{anom}$', color='C1', linewidth=2)
ax[1].plot(domegaa_dsalt * (salt - np.mean(salt)), label=r'$\frac{\partial \Omega_{A}}{\partial S} S_{anom}$', color='C3', linewidth=2)
ax[1].plot(domegaa_ddic * (dic - np.mean(dic)), label=r'$\frac{\partial \Omega_{A}}{\partial DIC} DIC_{anom}$', color='C0', linewidth=2)
ax[1].plot(domegaa_dalk * (alk - np.mean(alk)), label=r'$\frac{\partial \Omega_{A}}{\partial Alk} Alk_{anom}$', color='C2', linewidth=2)
ax[1].plot(
    domegaa_dtemp * (temp - np.mean(temp)) +
    domegaa_dsalt * (salt - np.mean(salt)) +
    domegaa_ddic * (dic - np.mean(dic)) +
    domegaa_dalk * (alk - np.mean(alk)),
    label='sum of linearized terms', linewidth=3, color='C4')
ax[1].legend(loc='lower left', bbox_to_anchor=(1.02, 0.1))
plt.tight_layout()
plt.savefig('decomposition_hplus_omega_anom_mean.png', dpi=200)
plt.show()

#%% 
######## Extreme detection #########

# Define the threshold 
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')
from set_thresh_and_clim_params import ThresholdParameters as ThresholdParameters
params = ThresholdParameters.Hplus_instance() #95th percentile threshold

# Defining variables
model_temp_resolution = 'daily' # 'monthly'
scenarios = ['present', 'ssp585'] # ,'ssp245'
configs = ['romsoc_fully_coupled'] # [ms_only'] 
simulation_type = 'hindcast'
parent_model = 'mpi-esm1-2-hr'
ensemble_run = '001'
vert_struct = 'zavg'    # 'avg'
depth = 0

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


# %%
#% Get the distance to coast file from ROMS and the regions over which to calculate the statistics
model_d2coasts = dict()
model_d2coasts['roms_only'] = ModelGetter.get_distance_to_coast(vtype='oceanic')

# Get the model regions
model_regions = dict()
model_regions['roms_only'] = GetRegions.define_CalCS_regions(model_d2coasts['roms_only'].lon, model_d2coasts['roms_only'].lat, model_d2coasts['roms_only'])

# Get the model area
model_area = ModelGetter.get_model_area()



#%% 

# Define the threshold type
scenario = 'ssp585'
threshold_type = 'present'

# Calculate differences relative to the threshold
present = variables['romsoc_fully_coupled']['present']['Hplus'] - thresholds['romsoc_fully_coupled']['present']['Hplus']
print("Present calculated")
future = variables['romsoc_fully_coupled'][scenario]['Hplus'] - thresholds_mult['romsoc_fully_coupled'][scenario]['present']
print("Future calculated")

# Define masks for different extreme types
new_extremes_mask = (future > 0) * (present <= 0)
disappearing_extremes_mask = (future <= 0) * (present > 0)
intensifying_extremes_mask = (future > 0) * (present > 0) * (future >= present)
weakening_extremes_mask = (future > 0) * (present > 0) * (future < present)


#%%

# Define the directory to save 
output_directory = "/nfs/sea/work/fpfaeffli/long_lats_extremes"

# Extract indices from mask
def extract_indices_from_mask(mask):
    indices = np.where(mask)  
    eta_indices = indices[1]  
    xi_indices = indices[2] 
    return eta_indices, xi_indices
print("Indices extracted")

# Create a dictionary to store 
extreme_indices = {
    "new_extremes": extract_indices_from_mask(new_extremes_mask.values),
    "disappearing_extremes": extract_indices_from_mask(disappearing_extremes_mask.values),
    "intensifying_extremes": extract_indices_from_mask(intensifying_extremes_mask.values),
    "weakening_extremes": extract_indices_from_mask(weakening_extremes_mask.values),
}
print("Dictionary created")

# Save 
for extreme_type, indices in extreme_indices.items():
    eta_indices, xi_indices = indices
    np.save(os.path.join(output_directory, f'{extreme_type}_eta_rho.npy'), eta_indices)
    np.save(os.path.join(output_directory, f'{extreme_type}_xi_rho.npy'), xi_indices)

print(f"Saved indices for all extreme types in {output_directory}.")


#%% Define bins for distance to coast
bins_d2coast = [0, 100, np.max(model_d2coasts['roms_only'].values)]  # coastal: 0–100 km, offshore: >100 km
bin_labels = ['coastal', 'offshore']

#Load extreme indices
new_extreme_eta_rho = np.load('new_extremes_eta_rho.npy')
new_extreme_xi_rho = np.load('new_extremes_xi_rho.npy')

# Load distance to coast data
distance_to_coast = model_d2coasts['roms_only']


#%% Classify extremes into bins

# Load the distance-to-coast array
distance_to_coast_flat = distance_to_coast.values.flatten()

# Flatten extreme indices into 1D
flat_indices = np.ravel_multi_index((new_extreme_eta_rho, new_extreme_xi_rho), distance_to_coast.shape)
print("Flatting done")

# Extract distances for all extreme points in a single operation
extreme_distances = distance_to_coast_flat[flat_indices]
print("Distances extracted")

# Create masks for onshore and offshore
onshore_mask = (extreme_distances <= bins_d2coast[1])
offshore_mask = (extreme_distances > bins_d2coast[1])
print("Masks created")

# Extract onshore and offshore indices
onshore_indices = [(new_extreme_eta_rho[i], new_extreme_xi_rho[i]) for i in np.where(onshore_mask)[0]]
offshore_indices = [(new_extreme_eta_rho[i], new_extreme_xi_rho[i]) for i in np.where(offshore_mask)[0]]

print(f"Number of onshore extremes: {len(onshore_indices)}")
print(f"Number of offshore extremes: {len(offshore_indices)}")


#%% Aggregate anomalies and sensitivities for onshore and offshore
def aggregate_extremes(indices, ds, ds_sens):

    aggregated = {
        'temp_anom': [],
        'sal_anom': [],
        'dic_anom': [],
        'alk_anom': [],
        'dh_dtemp': [],
        'dh_dsal': [],
        'dh_ddic': [],
        'dh_dalk': []
    }
    
    for eta, xi in indices:
        # Calculate anomalies
        aggregated['temp_anom'].append(ds.temp.isel(eta_rho=eta, xi_rho=xi) - ds.temp.mean(dim='time'))
        aggregated['sal_anom'].append(ds.salt.isel(eta_rho=eta, xi_rho=xi) - ds.salt.mean(dim='time'))
        aggregated['dic_anom'].append(ds.DIC.isel(eta_rho=eta, xi_rho=xi) - ds.DIC.mean(dim='time'))
        aggregated['alk_anom'].append(ds.Alk.isel(eta_rho=eta, xi_rho=xi) - ds.Alk.mean(dim='time'))

        # Extract sensitivities
        aggregated['dh_dtemp'].append(ds_sens.dh_dtem.isel(eta_rho=eta, xi_rho=xi))
        aggregated['dh_dsal'].append(ds_sens.dh_dsal.isel(eta_rho=eta, xi_rho=xi))
        aggregated['dh_ddic'].append(ds_sens.dh_ddic.isel(eta_rho=eta, xi_rho=xi))
        aggregated['dh_dalk'].append(ds_sens.dh_dalk.isel(eta_rho=eta, xi_rho=xi))
    
    # Aggregate by averaging over all indices
    # Combines the time series across all extreme points and calculates the mean over all extreme locations
    for key in aggregated:
        aggregated[key] = xr.concat(aggregated[key], dim='index').mean(dim='index')
    
    return aggregated

# Aggregate for onshore and offshore
onshore_aggregated = aggregate_extremes(onshore_indices, ds, ds_sens)
offshore_aggregated = aggregate_extremes(offshore_indices, ds, ds_sens)

#%% Plot H⁺ decomposition for onshore and offshore
def plot_hplus_decomposition(aggregated, region_label, save_filename):
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    # H⁺ decomposition
    ax[0].plot(10**9 * aggregated['dh_dtemp'] * aggregated['temp_anom'], label=r'$\frac{\partial H^{+}}{\partial T} T_{anom}$', color='C1', linewidth=2)
    ax[0].plot(10**9 * aggregated['dh_dsal'] * aggregated['sal_anom'], label=r'$\frac{\partial H^{+}}{\partial S} S_{anom}$', color='C3', linewidth=2)
    ax[0].plot(10**9 * aggregated['dh_ddic'] * aggregated['dic_anom'], label=r'$\frac{\partial H^{+}}{\partial DIC} DIC_{anom}$', color='C0', linewidth=2)
    ax[0].plot(10**9 * aggregated['dh_dalk'] * aggregated['alk_anom'], label=r'$\frac{\partial H^{+}}{\partial Alk} Alk_{anom}$', color='C2', linewidth=2)
    ax[0].plot(
        10**9 * (
            aggregated['dh_dtemp'] * aggregated['temp_anom'] +
            aggregated['dh_dsal'] * aggregated['sal_anom'] +
            aggregated['dh_ddic'] * aggregated['dic_anom'] +
            aggregated['dh_dalk'] * aggregated['alk_anom']
        ),
        label='Sum of linearized terms', color='C4', linewidth=3)
    ax[0].set_title(f'{region_label} H⁺ Decomposition')
    ax[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax[0].set_ylabel('H⁺ Anomalies (nmol L⁻¹)')
    ax[0].grid()

    # Ωₐ decomposition
    ax[1].plot(aggregated['dh_dtemp'] * aggregated['temp_anom'], label=r'$\frac{\partial \Omega_{A}}{\partial T} T_{anom}$', color='C1', linewidth=2)
    ax[1].plot(aggregated['dh_dsal'] * aggregated['sal_anom'], label=r'$\frac{\partial \Omega_{A}}{\partial S} S_{anom}$', color='C3', linewidth=2)
    ax[1].plot(aggregated['dh_ddic'] * aggregated['dic_anom'], label=r'$\frac{\partial \Omega_{A}}{\partial DIC} DIC_{anom}$', color='C0', linewidth=2)
    ax[1].plot(aggregated['dh_dalk'] * aggregated['alk_anom'], label=r'$\frac{\partial \Omega_{A}}{\partial Alk} Alk_{anom}$', color='C2', linewidth=2)
    ax[1].plot(
        aggregated['dh_dtemp'] * aggregated['temp_anom'] +
        aggregated['dh_dsal'] * aggregated['sal_anom'] +
        aggregated['dh_ddic'] * aggregated['dic_anom'] +
        aggregated['dh_dalk'] * aggregated['alk_anom'],
        label='Sum of linearized terms', color='C4', linewidth=3)
    ax[1].set_title(f'{region_label} Ωₐ Decomposition')
    ax[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax[1].set_ylabel('Ωₐ Anomalies')
    ax[1].set_xlabel('Time (days)')
    ax[1].grid()

    plt.tight_layout()
    plt.savefig(save_filename, dpi=200)
    plt.show()

#%% Plot onshore and offshore
plot_hplus_decomposition(onshore_aggregated, 'Onshore', 'onshore_hplus_decomposition.png')
plot_hplus_decomposition(offshore_aggregated, 'Offshore', 'offshore_hplus_decomposition.png')
