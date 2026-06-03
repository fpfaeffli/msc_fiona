"""
Author: Fiona Pfäffli
Script: Plot global ppm vs. pH, Ω, or H+ concentrations for different scenarios and regions.
"""
#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

SAVEDIR = '/nfs/sea/work/fpfaeffli/plots/pH_vs_CO2/'
# CO2 ppm values
ppm_values = [414, 600, 1100] # 414ppm = present, 600ppm = ssp245 scenario, 1100ppm = ssp585 scenario

# Calculate the pH values for different regions and scenarios

# !!!! use iacpy3_2022 environment for this script !!!!

 
# enable the visibility of the modules for the import functions
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '/home/koehne/Documents/publications/paper_future_simulations/scripts_clean/modules/'))
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/evaluation/')
sys.path.append('/home/fpfaeffli/msc_fiona/scripts/modules/')


# load the package
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from get_obs_datasets import ObsGetter as ObsGetter
from get_study_regions import GetRegions as GetRegions
from get_model_datasets import ModelGetter as ModelGetter
from plotting_functions_evaluation import Plotter as Plotter

from importlib import reload  # Python 3.4+
import plotting_functions_evaluation
reload(plotting_functions_evaluation)
from plotting_functions_evaluation import Plotter as Plotter

import get_model_datasets
reload(get_model_datasets)
from get_model_datasets import ModelGetter as ModelGetter

import get_obs_datasets
reload(get_obs_datasets)
from get_obs_datasets import ObsGetter as ObsGetter

from regridding_tools import Regridder as Regridder

from get_auxiliary_data import AuxGetter as AuxGetter



#%% Setting the variable to compare and across which models to compare

var = 'pH_offl'  

dep = 0  # m
vert_struct = 'zavg'

plot_resolution = 'monthly'
model_temp_resolution = 'monthly'
obs_temp_resolution = 'monthly'

scenarios = ['present', 'ssp245', 'ssp585']  # ✅ now a list
simulation_type = 'hindcast'
ensemble_run = '000'
vtype = 'oceanic'
configs = ['romsoc_fully_coupled']
parent_model = 'mpi-esm1-2-hr'  

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


#%% Getting model datasets
model_ds = dict()
model_da = dict()


for config in configs:
    model_ds[config] = dict()
    model_da[config] = dict()

    for scenario in scenarios:
        print(f'Loading: {config}, {scenario}, {simulation_type}, {ensemble_run}, {model_temp_resolution}, {vert_struct}, {vtype}')

        # Pass parent_model only for future scenarios
        if scenario == 'present':
            model_ds[config][scenario] = ModelGetter.get_model_dataset(
                config=config,
                scenario=scenario,
                simulation_type=simulation_type,
                ensemble_run=ensemble_run,
                temp_resolution=model_temp_resolution,
                vert_struct=vert_struct,
                vtype=vtype
            )
        else:
            model_ds[config][scenario] = ModelGetter.get_model_dataset(
                config=config,
                scenario=scenario,
                simulation_type=simulation_type,
                ensemble_run=ensemble_run,
                temp_resolution=model_temp_resolution,
                vert_struct=vert_struct,
                vtype=vtype,
                parent_model=parent_model  # required for ssp245, ssp585
            )

        # Variable selection
        var_data = model_ds[config][scenario][var]
        if vert_struct == 'zavg':
            if var_data.ndim == 4:
                model_da[config][scenario] = var_data.sel(depth=dep)
            else:
                model_da[config][scenario] = var_data
        elif vert_struct == 'avg' and var_data.ndim == 3:
            model_da[config][scenario] = var_data
        else:
            raise Exception('Not yet implemented for this vertical structure.')

#%% 

# Load model data for all configs and scenarios
for config in configs:
    for scenario in scenarios:
        print(f'Loading model data for config: {config}, scenario: {scenario}')
        model_da[config][scenario].load()
        print('Done')

#%%
# Calculate regional means for all regions
target_regions = ['all_dists_central', 'coastal_central', 'offshore_central']
regional_data_model = dict()

for config in configs:
    regional_data_model[config] = dict()
    for scenario in scenarios:
        print(f'Calculating regional means for variable {var} and scenario: {scenario}')
        regional_data_model[config][scenario] = dict()
        for region in target_regions:
            if region in model_regions_dict:
                regional_mean = (
                    (model_da[config][scenario] * model_regions_dict[region]['mask'])
                    .weighted(model_area.fillna(0))
                    .mean(("eta_rho", "xi_rho"))
                    .groupby("time.month")
                    .mean("time")
                )
                regional_data_model[config][scenario][region] = regional_mean

                # Print mean value across all months
                print(f"{scenario} - {region}: {regional_mean.mean().values:.3e}")


#%%


ppm_values = np.array([414, 600, 1100])
bar_width = 130


mean_pH = {
    'all_dists_central':  [8.047, 7.896, 7.740],
    'coastal_central':    [7.982, 7.861, 7.738],
    'offshore_central':   [8.052, 7.899, 7.740]
}

# Calculated in the script detect_count_H+_extreme.py (/home/fpfaeffli/msc_fiona/scripts_oax_paper/)
mean_extreme_days_fixed = {
    'all_dists_central':  [19.0, 334.3, 358.6],
    'coastal_central':    [19.7, 221.3, 316.8],
    'offshore_central':   [18.9, 345.5, 362.8]
}

mean_extreme_days_moving = {
    'all_dists_central':  [19.0,  48.8, 113.8],
    'coastal_central':    [19.7,  38.3,  80.0],
    'offshore_central':   [18.9,  49.8, 117.1]
}



region_labels = ['whole region', 'coastal', 'offshore']
regions = ['all_dists_central', 'coastal_central', 'offshore_central']
scenario_labels = ['present', 'ssp245', 'ssp585']

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

######################
#     Plotting       #
######################


# %%


fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

for idx, region in enumerate(regions):
    ax1 = axs[idx]

    # Set spine color
    if "coastal" in region.lower():
        spine_color = "coral"
    elif "offshore" in region.lower():
        spine_color = "saddlebrown"
    else:
        spine_color = "black"

    for side in ['top', 'bottom', 'left', 'right']:
        ax1.spines[side].set_edgecolor(spine_color)
        ax1.spines[side].set_linewidth(1)
    ax1.tick_params(axis='x', colors=spine_color)
    ax1.tick_params(axis='y', colors=spine_color)
    ax1.yaxis.label.set_color(spine_color)

    # pH curve
    ax1.plot(ppm_values, mean_pH[region],
             marker='o', color='black', linewidth=2)
    ax1.set_ylabel('pH', fontsize=14, color='black')
    ax1.set_ylim(7.7, 8.2)
    ax1.set_xlabel('$CO_2$ [ppm]', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(ppm_values)

    # bars on second y-axis
    ax2 = ax1.twinx()
    # present
    ax2.bar(ppm_values[0], mean_extreme_days_fixed[region][0],
            width=bar_width, color=colors['present'], alpha=0.8)
    ax2.text(ppm_values[0], mean_extreme_days_fixed[region][0] + 5,
             f'{mean_extreme_days_fixed[region][0]:.1f}',
             ha='center', fontsize=12)

    # future scenarios
    for i, scen in enumerate(['ssp245', 'ssp585']):
        x = ppm_values[i + 1]
        ax2.bar(x - bar_width/2.8, mean_extreme_days_fixed[region][i + 1],
                width=bar_width, color=colors[f"{scen}_light"], alpha=0.7)
        ax2.bar(x + bar_width/2.8, mean_extreme_days_moving[region][i + 1],
                width=bar_width, color=colors[f"{scen}_dark"], alpha=0.7)

        ax2.text(x - bar_width/2.8, mean_extreme_days_fixed[region][i + 1] + 12,
                 f'{mean_extreme_days_fixed[region][i + 1]:.0f}',
                 ha='center', fontsize=12)
        ax2.text(x + bar_width/2.8, mean_extreme_days_moving[region][i + 1] + 12,
                 f'{mean_extreme_days_moving[region][i + 1]:.0f}',
                 ha='center', fontsize=12)

    ax2.set_ylim(0, 400)
    ax2.set_ylabel('Extreme [H$^+$] days/year', fontsize=14)

    # styling & title
    if "coastal" in region.lower():
        title_color = "coral"
    elif "offshore" in region.lower():
        title_color = "saddlebrown"
    else:
        title_color = "black"

    ax1.set_title(region_labels[idx], fontsize=14, color=title_color)
    ax1.set_zorder(10)
    ax1.patch.set_visible(False)

    # hide ax2 spines
    for side in ['top', 'bottom', 'left', 'right']:
        ax2.spines[side].set_visible(False)

    ax1.grid(False)
    ax2.grid(False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(
    '/nfs/sea/work/fpfaeffli/plots/pH_vs_CO2/ppm_pH_extreme_coastal_vs_offshore.png',
    dpi=150, transparent=True, bbox_inches='tight'
)
plt.show()


# %%

# Just the legend for the plot
legend_elements = [
    Line2D([0], [0], color='black', marker='o', linestyle='-', linewidth=2, label='pH'),
    Line2D([0], [0], color='darkgrey', marker='^', linestyle='-', linewidth=2, label='Ω'),
    
    Patch(facecolor=colors['present'], label='present', alpha=0.8),
    Patch(facecolor=colors['ssp245_light'], label='ssp245 (fixed baseline)', alpha=0.7),
    Patch(facecolor=colors['ssp245_dark'], label='ssp245 (shifted baseline)', alpha=0.7),
    Patch(facecolor=colors['ssp585_light'], label='ssp585 (fixed baseline)', alpha=0.7),
    Patch(facecolor=colors['ssp585_dark'], label='ssp585 (shifted baseline)', alpha=0.7),
]

fig.legend(handles=legend_elements,
           loc='lower center', bbox_to_anchor=(0.5, -0.1),
           ncol=4, fontsize=10, frameon=False)

fig.legend(handles=legend_elements,
           loc='lower center', bbox_to_anchor=(0.5, -0.1),
           ncol=4, fontsize=10, frameon=False)


# Create figure just for legend
fig, ax = plt.subplots(figsize=(2.5, 3))
ax.axis('off')  # Hide axes

# Add legend
legend = ax.legend(handles=legend_elements,
                   title='Legend',
                   loc='center',
                   frameon=True,
                   fontsize=10,
                   title_fontsize=11,
                   borderpad=1.0,
                   labelspacing=1.2)

# Style legend box
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')

# Save or show
plt.tight_layout()
plt.savefig('/nfs/sea/work/fpfaeffli/plots/pH_vs_CO2/legend_scenarios_boxed.png', dpi=150, bbox_inches='tight', transparent=False)
plt.show()

# %%
# %%
# Plotting extreme days and pH for coastal vs. offshore in one figure

#Define data


# --- your scenario colour dictionary ---
colors = {
    'present': "#ffab67",  # warm orange
    'ssp245':  "#65a342",  # original green
    'ssp585':  "#9553e0",  # original purple
}

ppm_values = np.array([414, 600, 1100])
scenarios  = ['present', 'ssp245', 'ssp585']
col_list   = [colors[s] for s in scenarios]

# Shifted‑baseline extreme days (mean_extreme_days_moving)
coastal_days  = np.array([19.7, 38.3, 80.0])
offshore_days = np.array([18.9, 49.8, 117.1])

# Mean pH values
coastal_pH  = np.array([7.982, 7.861, 7.738])
offshore_pH = np.array([8.052, 7.899, 7.740])

bar_width = 80
fig, ax1 = plt.subplots(figsize=(8, 7))

# --- Bars for extremes (old bar colours) ---
ax1.bar(ppm_values - bar_width/2, coastal_days,
        width=bar_width, label='coastal extremes',
        color='coral', alpha=0.7)
ax1.bar(ppm_values + bar_width/2, offshore_days,
        width=bar_width, label='offshore extremes',
        color='saddlebrown', alpha=0.7)

# Annotate bar values
for x, cd, od in zip(ppm_values, coastal_days, offshore_days):
    ax1.text(x - bar_width/2, cd + 1, f'{cd:.1f}',
             ha='center', va='bottom', fontsize=11)
    ax1.text(x + bar_width/2, od + 1, f'{od:.1f}',
             ha='center', va='bottom', fontsize=11)

ax1.set_xlabel('$CO_2$ [ppm]', fontsize=12)
ax1.set_ylabel('Extreme [H$^+$] days / year', fontsize=12)

# --- Colour the x‑ticks by scenario ---
ax1.set_xticks(ppm_values)
# Add scenario labels below ppm values
xtick_labels = [f"{x}\n{scen}" for x, scen in zip(ppm_values, scenarios)]
ax1.set_xticklabels(xtick_labels, fontsize=11)
for lbl, c in zip(ax1.get_xticklabels(), col_list):
    lbl.set_color(c)

ax1.set_ylim(0, max(offshore_days)*1.1)

# --- pH curves on secondary y-axis, matching bar colours ---
ax2 = ax1.twinx()

# dashed lines in coral/saddlebrown
ax2.plot(ppm_values, coastal_pH,  linestyle='--',
         color='coral',      linewidth=2,
         label='coastal pH')
ax2.plot(ppm_values, offshore_pH, linestyle='--',
         color='saddlebrown', linewidth=2,
         label='offshore pH')

# scenario‐coloured markers on top
ax2.scatter(ppm_values, coastal_pH, marker='o', s=30,
            color='coral', edgecolor='coral')
ax2.scatter(ppm_values, offshore_pH, marker='s', s=30,
            color='saddlebrown', edgecolor='saddlebrown')

ax2.set_ylabel('pH', fontsize=12)
ax2.set_ylim(7.6, 8.2)

# --- Combined legend ---
bars, bar_labels   = ax1.get_legend_handles_labels()
lines, line_labels = ax2.get_legend_handles_labels()
ax1.legend(bars + lines, bar_labels + line_labels,
           loc='center left', bbox_to_anchor=(1.25, 0.5),
           frameon=False)

plt.tight_layout()
plt.savefig(
    '/nfs/sea/work/fpfaeffli/plots/pH_vs_CO2/coastal_vs_offshore_extreme_days_pH.png',
    dpi=150, transparent=True, bbox_inches='tight'
)
plt.show()

# %%
