"""
author: Fiona Pfäffli
description: PDF of all H+ values for the different scenarios.
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
from scipy.stats import gaussian_kde


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

#%%

#############################
#       PLOTTING          #
#############################

#%%
# Plot pdf of all H+ values at a given point for the different scenarios

# --- 1) define your point ---
target_lat, target_lon = 38.5, 236.5 # 233 or 236.5
config     = "romsoc_fully_coupled"
ds0        = ocean_ds[config]["present"]
lat2d, lon2d = ds0.lat.values, ds0.lon.values
idx_flat   = ((lat2d - target_lat)**2 + (lon2d - target_lon)**2).argmin()
i0, j0     = np.unravel_index(idx_flat, lat2d.shape)

# --- 2) scenarios and colors ---
scenarios = ['present','ssp245','ssp585']
colors = {
    'present':      "#ffab67",
    'ssp245':       "#65a342",
    'ssp585':       "#9553e0",
    # light/dark for shading/lines in absolute subplot
    'ssp245_light': "#8ad764",
    'ssp245_dark':  "#3c7d1f",
    'ssp585_light': "#b98bf0",
    'ssp585_dark':  "#6c2eb3",
}

# --- 3) extract series at the point ---
series = {}
for scen in scenarios:
    da = hplus[config][scen].isel(eta_rho=i0, xi_rho=j0)
    arr = da.values.squeeze()
    arr = arr[~np.isnan(arr)]
    series[scen] = arr

# --- 4) thresholds (absolute) ---
fixed_thr = thresholds_mult[config]['present']['present'] \
                .isel(eta_rho=i0, xi_rho=j0).mean().item()
shifted_thr = {
    scen: thresholds_mult[config][scen]['present_plus_meandelta'] \
                .isel(eta_rho=i0, xi_rho=j0).mean().item()
    for scen in ['ssp245','ssp585']
}

# --- shared figure/axes ---
fig, axes = plt.subplots(1, 2, figsize=(13,5), sharey=True)
# Reduce horizontal space between subplots
fig.subplots_adjust(wspace=0.1, bottom=0.18)  # wspace controls horizontal gap

ax_abs, ax_ctr = axes

# =========================
# LEFT: ABSOLUTE PDFs
# =========================

# Use a reasonable grid built from data (robust percentiles)
all_abs_vals = np.concatenate([series[s] for s in scenarios])
lo, hi = np.percentile(all_abs_vals, [0.5, 99.5])
pad = 0.1 * (hi - lo) if hi > lo else 0.0
grid_abs = np.linspace(max(1e-12, lo - pad), hi + pad, 600)

# Shared bandwidth (fair smoothing)
shared_bw_abs = gaussian_kde(all_abs_vals).covariance_factor()

kdes_abs = {}
for scen in scenarios:
    kde = gaussian_kde(series[scen], bw_method=shared_bw_abs)
    kdes_abs[scen] = kde
    dens = kde(grid_abs)
    ax_abs.plot(grid_abs, dens, color=colors[scen], lw=2, label=f"{scen}")

    # shading beyond its threshold
    thr = fixed_thr if scen == 'present' else shifted_thr[scen]
    mask = grid_abs >= thr
    fill_col = colors['present'] if scen=='present' else colors[f"{scen}_light"]
    ax_abs.fill_between(grid_abs[mask], 0, dens[mask], color=fill_col, alpha=0.30)

    # mean marker (diamond 'D')
    mu = series[scen].mean()
    ax_abs.plot(mu, kdes_abs[scen](mu), marker='D', color=colors[scen], markersize=8, label=None)

    # threshold line (dashed)
    y_thr = kde(thr)[0]
    clr = colors['present'] if scen=='present' else colors[f"{scen}_dark"]
    ax_abs.plot([thr, thr], [0, y_thr], linestyle='--', color=clr, lw=1.5)

# composite shading: any > present-day threshold
mask_any = grid_abs >= fixed_thr
max_dens = np.maximum.reduce([kdes_abs[s](grid_abs) for s in scenarios])
ax_abs.fill_between(grid_abs[mask_any], 0, max_dens[mask_any], color='grey', alpha=0.15)
ax_abs.set_xlabel("[H⁺] (mol L⁻¹)", fontsize=12)
ax_abs.grid(alpha=0.3)
ax_abs.tick_params(labelsize=12)

# Add legend for the mean marker (diamond 'D')
mean_marker_legend = Line2D([0], [0], marker='D', color='k', linestyle='None', markersize=8, label='scenario mean')
ax_abs.legend(handles=[mean_marker_legend], loc='upper right', frameon=False)

# =========================
# RIGHT: CENTERED PDFs
# =========================

# Center each series by its own mean
centered = {s: (series[s] - series[s].mean()) for s in scenarios}
means = {s: series[s].mean() for s in scenarios}

# thresholds in deviation space
fixed_thr_abs = fixed_thr
shifted_thr_abs = shifted_thr  # already defined
thr_dev = {
    'present': fixed_thr_abs - means['present'],
    'ssp245':  shifted_thr_abs['ssp245'] - means['ssp245'],
    'ssp585':  shifted_thr_abs['ssp585'] - means['ssp585'],
}

# Grid for centered plot (symmetric, robust)
all_ctr_vals = np.concatenate([centered[s] for s in scenarios])
lo_c, hi_c = np.percentile(all_ctr_vals, [0.5, 99.5])
pad_c = 0.1 * (hi_c - lo_c) if hi_c > lo_c else 0.0
left = min(lo_c - pad_c, -abs(hi_c - lo_c))  # ensure some symmetry if ranges are weird
right = max(hi_c + pad_c, abs(hi_c - lo_c))
grid_ctr = np.linspace(left, right, 600)

# Shared bandwidth for centered data
shared_bw_ctr = gaussian_kde(all_ctr_vals).covariance_factor()

dens_ctr = {}
for scen in scenarios:
    kde = gaussian_kde(centered[scen], bw_method=shared_bw_ctr)
    dens = kde(grid_ctr)
    dens_ctr[scen] = dens
    ax_ctr.plot(grid_ctr, dens, color=colors[scen], lw=2, label=f"{scen} (centered)")

    # threshold deviation line (dashed)
    y_thr = kde(thr_dev[scen])[0]
    ax_ctr.plot([thr_dev[scen], thr_dev[scen]], [0, y_thr],
                linestyle='--', color=colors[scen], lw=1.5)

    # shading (deviation >= threshold deviation)
    mask = grid_ctr >= thr_dev[scen]
    ax_ctr.fill_between(grid_ctr[mask], 0, dens[mask], color=colors[scen], alpha=0.25)

# mean lines (all at 0 after centering) – draw one neutral and annotate in legend
#ax_ctr.axvline(0, color='k', linestyle=':', lw=1.2, label="mean (0 after centering)")

ax_ctr.set_xlabel("Deviation from mean [H⁺] (mol L⁻¹)", fontsize=12)
ax_ctr.grid(alpha=0.3)
ax_ctr.tick_params(labelsize=12)

# --- shared y-label + unified legend ---
axes[0].set_ylabel("Density", fontsize=12)

# single combined legend (bottom center)
lines_labels = [ax.get_legend_handles_labels() for ax in axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# Remove duplicate labels while preserving order
seen = set()
uniq_lines, uniq_labels = [], []
for ln, lb in zip(lines, labels):
    if lb not in seen:
        uniq_lines.append(ln)
        uniq_labels.append(lb)
        seen.add(lb)

from matplotlib.ticker import ScalarFormatter

class OneDecimalSciFormatter(ScalarFormatter):
    def _set_format(self, vmin=None, vmax=None):
        self.format = "%.1f"   # always one decimal

# apply to both subplots
for ax in (ax_abs, ax_ctr):
    fmt = OneDecimalSciFormatter(useMathText=False)
    fmt.set_scientific(True)     # force scientific notation
    fmt.set_powerlimits((0, 0))  # always show offset (e.g. 1e8)
    ax.yaxis.set_major_formatter(fmt)


# --- save ---
outpath = f'/nfs/sea/work/fpfaeffli/plots/H+_PDFs/Hplus_PDF_offshore_dual_{target_lat}_{target_lon}.png'
plt.savefig(outpath, dpi=220, transparent=True, bbox_inches='tight')
plt.show()
print("Saved:", outpath)


#%%
# only plot the legend


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase

# colour definitions
colors = {
    'present':      "#ffab67",
    'ssp245_light': "#8ad764",
    'ssp245':       "#65a342",
    'ssp245_dark':  "#3c7d1f",
    'ssp585_light': "#b98bf0",
    'ssp585':       "#9553e0",
    'ssp585_dark':  "#6c2eb3",
}

# custom handler for the two‑colour square
class HandlerMultiColorPatch(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        n = len(orig_handle.colors)
        sw = width / n
        patches = []
        for i, c in enumerate(orig_handle.colors):
            r = Rectangle((xdescent + i*sw, ydescent),
                          sw, height,
                          facecolor=c, edgecolor='none',
                          transform=trans)
            patches.append(r)
        return patches

# build the multi‑colour proxy
multi = Patch()
multi.colors = [colors['ssp245_light'], colors['ssp585_light']]
multi.set_label("adjusted baseline future extremes")

# legend entries for top block (3 cols)
top_handles = [
    Line2D([0],[0], color=colors['present'], lw=2, label='present [H⁺]'),
    Line2D([0],[0], color=colors['ssp245'],  lw=2, label='ssp245 [H⁺]'),
    Line2D([0],[0], color=colors['ssp585'],  lw=2, label='ssp585 [H⁺]'),
    
    # thresholds
    Line2D([0],[0], color=colors['present'],     linestyle='--', lw=1.5, label='present-day threshold'),
    Line2D([0],[0], color=colors['ssp245_dark'], linestyle='--', lw=1.5, label='ssp245 adjusted threshold'),
    Line2D([0],[0], color=colors['ssp585_dark'], linestyle='--', lw=1.5, label='ssp585 adjusted threshold'),
  
    # patches
    Patch(facecolor=colors['present'],      alpha=0.3, label='present extremes'),
    multi,
    Patch(facecolor='grey', alpha=0.2,     label='fixed baseline future extremes'),
    Line2D([],[], color='none',             label=''),  # spacer
    Line2D([],[], color='none',             label=''),  # spacer
    Line2D([],[], color='none',             label=''),  # spacer
]

fig, ax = plt.subplots(figsize=(8,2))
ax.axis('off')

# 1) top legend
lg1 = ax.legend(
    handles=top_handles,
    handler_map={ multi: HandlerMultiColorPatch() },
    ncol=4,
    loc='upper center',
    frameon=True,
)

plt.tight_layout()
plt.savefig(
    '/nfs/sea/work/fpfaeffli/plots/H+_PDFs/Hplus_PDF_legend.png',
    dpi=200, transparent=True, bbox_inches='tight'
)
plt.show()



# %%
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

    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot a neutral background (light gray ocean)
    ax.set_facecolor("white")

    # Add solid black landmask using pcolormesh
    ax.pcolormesh(
        landmask_etopo.lon, landmask_etopo.lat, landmask_etopo,
        cmap="Greys", vmin=0, vmax=1, zorder=1
    )

    # Define colors and labels for each point
    point_colors = ["coral", "saddlebrown"]
    point_labels = ["coastal", "offshore"]

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
        ax.scatter(point_lon, point_lat, color=point_colors[i], marker='o', s=25, label=point_labels[i], zorder=3)

        # Debugging: Print point information
        print(f"{point_labels[i]} -> Target: ({target_lat}, {target_lon}) | Closest Grid: ({point_lat}, {point_lon})")

    # Offshore vs coastal contours
    region_contour_colors = {
        'offshore_central': 'black',
        'coastal_central': 'black'
    }
    for region, color in region_contour_colors.items():
        if region in model_regions_dict:
            mask = model_regions_dict[region]['mask']
            mask_dummy = xr.where(np.isnan(mask), 0, 1)
            ax.contour(
                lon_rho, lat_rho, mask_dummy,
                levels=[0.5],
                colors=[color],
                linewidths=1,
                zorder=4
            )

    # Formatting: Axis limits
    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)

    # Add labels and ticks with ° symbols
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Format latitude & longitude labels
    yticks = np.arange(lat_bounds[0], lat_bounds[1] + 1, 3)
    xticks = np.arange(lon_bounds[0], lon_bounds[1] + 1, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels([f"{lat}°N" for lat in yticks])
    ax.set_xticklabels([f"{xtick}°E" if xtick <= 180 else f"{360 - xtick}°W" for xtick in xticks])

    # Add legend
    ax.legend(loc="upper right")

    
    # Save & Show
    plt.savefig("/nfs/sea/work/fpfaeffli/plots/H+_PDFs/neutral_map_two_points.png", dpi=200, transparent=True, bbox_inches='tight')
    plt.show()


lat_bounds = (32, 46)
lon_bounds = (225, 245)  # Ensure longitudes are in 0-360° format
bounds = (lat_bounds, lon_bounds)

# Define the 2 selected points (lat, lon)
points_latlon = [
    (38.5, 236.5),  
    (38.5, 233)   
]

# Call the function with ocean_ds
plot_neutral_map_with_two_points(ocean_ds, landmask_etopo, bounds, points_latlon, config="romsoc_fully_coupled", scenario="present")


# %%
