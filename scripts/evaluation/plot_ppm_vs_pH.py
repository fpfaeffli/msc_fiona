"""
Author: Fiona Pfäffli
Script: Plot global ppm vs. pH, Ω, or H+ concentrations for different scenarios and regions.
"""
#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
# Function to calculate H+ concentrations from pH
def calculate_Hplus(pH_list):
    return [10 ** -pH for pH in pH_list]


# 1. Plot for all regions combined (coastal + offshore averaged)
def plot_all_regions_combined(ppm_values, data, ylabel, title, save_path, ylim=None):
    plt.figure(figsize=(10, 6))
    colors = {"Northern": 'blue', "Central": 'green', "Southern": 'red'}

    for region, areas in data.items():
        combined_values = np.mean([areas['coastal'], areas['offshore']], axis=0)
        plt.plot(ppm_values, combined_values, marker='o', linestyle='-', color=colors[region], label=f"{region} (All)")

    plt.xlabel('$CO_2$ [ppm]', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=16)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()



# 2. Plot split into coastal and offshore regions
def plot_split_coastal_offshore(ppm_values, data, ylabel, title, save_path, ylim=None):
    plt.figure(figsize=(12, 8))
    colors = {"Northern": 'blue', "Central": 'green', "Southern": 'red'}

    for region, areas in data.items():
        for area, values in areas.items():
            linestyle = '-' if area == "coastal" else '--'
            plt.plot(ppm_values, values, marker='o', linestyle=linestyle, color=colors[region],
                     label=f"{region} - {area.capitalize()}")

    plt.xlabel('$CO_2$ [ppm]', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=16)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()


# 3. Plot general mean values for coastal and offshore regions
def plot_general_means(ppm_values, coastal_means, offshore_means, ylabel, title, save_path, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.plot(ppm_values, coastal_means, marker='o', linestyle='-', color='blue', label='Coastal Mean')
    plt.plot(ppm_values, offshore_means, marker='^', linestyle='--', color='green', label='Offshore Mean')

    plt.xlabel('$CO_2$ [ppm]', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=16)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=200, transparent=True, bbox_inches='tight')
    plt.show()

#%%

SAVEDIR = '/nfs/sea/work/fpfaeffli/plots/pH_vs_CO2/'
# CO2 ppm values
ppm_values = [414, 600, 1100] # 414ppm = present, 600ppm = ssp245 scenario, 1100ppm = ssp585 scenario

# pH values for coastal and offshore regions
# means calculated in this script: /home/fpfaeffli/msc_fiona/scripts/evaluation/plot_mean_stdev_present_vs_future.py

pH_values = {
    "Northern": {"coastal": [8.03, 7.90, 7.76], "offshore": [8.05, 7.89, 7.72]},
    "Central": {"coastal": [8.00, 7.87, 7.74], "offshore": [8.05, 7.89, 7.73]},
    "Southern": {"coastal": [8.04, 7.90, 7.74], "offshore": [8.06, 7.90, 7.73]},
}

# General mean values for coastal and offshore regions
pH_coastal_means = np.mean([pH_values[region]["coastal"] for region in pH_values], axis=0)
pH_offshore_means = np.mean([pH_values[region]["offshore"] for region in pH_values], axis=0)

# omega values for coastal and offshore regions

omega_values = {
    "Northern": {"coastal": [2.49, 1.98, 1.53], "offshore": [2.82, 2.21, 1.68]},
    "Central": {"coastal": [2.13, 1.71, 1.35], "offshore": [2.35, 1.80, 1.34]},
    "Southern": {"coastal": [2.75, 2.15, 1.63], "offshore": [2.72, 2.07, 1.54]},
}
# General mean values for coastal and offshore regions
omega_coastal_means = np.mean([omega_values[region]["coastal"] for region in omega_values], axis=0)
omega_offshore_means = np.mean([omega_values[region]["offshore"] for region in omega_values], axis=0)


# Calculate H+ concentrations
Hplus_values = {
    region: {area: calculate_Hplus(values) for area, values in areas.items()}
    for region, areas in pH_values.items()
}
# General mean values for coastal and offshore regions
Hplus_coastal_means = np.mean([Hplus_values[region]["coastal"] for region in Hplus_values], axis=0)
Hplus_offshore_means = np.mean([Hplus_values[region]["offshore"] for region in Hplus_values], axis=0)



#%%
# Plot pH vs CO2

# 1. Plot pH for all regions combined
plot_all_regions_combined(
    ppm_values=ppm_values,
    data=pH_values,
    ylabel="pH",
    title="pH vs CO2 for All Regions Combined (Coastal + Offshore)",
    save_path=os.path.join(SAVEDIR, "pH_vs_CO2_combined_regions.png"),
    ylim=(7.7, 8.2)
)

#%%
# 2. Plot pH split into coastal and offshore regions
plot_split_coastal_offshore(
    ppm_values=ppm_values,
    data=pH_values,
    ylabel="pH",
    title="pH vs CO2 Split into Coastal and Offshore Regions",
    save_path=os.path.join(SAVEDIR, "pH_vs_CO2_split_coastal_offshore.png"),
    ylim=(7.7, 8.2)
)

#%%
# 3. Plot general mean pH values
plot_general_means(
    ppm_values=ppm_values,
    coastal_means=pH_coastal_means,
    offshore_means=pH_offshore_means,
    ylabel="pH",
    title="Mean pH vs CO2 for Coastal and Offshore Regions",
    save_path=os.path.join(SAVEDIR, "pH_vs_CO2_general_means.png"),
    ylim=(7.7, 8.2)
)

#%%
# Plot Ω vs CO2

# 1. Plot Ω for all regions combined
plot_all_regions_combined(
    ppm_values=ppm_values,
    data=omega_values,
    ylabel="Ω",
    title="Ω vs CO2 for All Regions Combined (Coastal + Offshore)",
    save_path=os.path.join(SAVEDIR, "omega_vs_CO2_combined_regions.png"),
    ylim=(1.0, 3.0)
)

# 2. Plot Ω split into coastal and offshore regions
plot_split_coastal_offshore(
    ppm_values=ppm_values,
    data=omega_values,
    ylabel="Ω",
    title="Ω vs CO2 Split into Coastal and Offshore Regions",
    save_path=os.path.join(SAVEDIR, "omega_vs_CO2_split_coastal_offshore.png"),
    ylim=(1.0, 3.0)
)

# 3. Plot general mean Ω values
plot_general_means(
    ppm_values=ppm_values,
    coastal_means=omega_coastal_means,
    offshore_means=omega_offshore_means,
    ylabel="Ω",
    title="Mean Ω vs CO2 for Coastal and Offshore Regions",
    save_path=os.path.join(SAVEDIR, "omega_vs_CO2_general_means.png"),
    ylim=(1.0, 3.0)
)

# %%
# Plot H+ vs CO2

# 1. Plot H+ for all regions combined
plot_all_regions_combined(
    ppm_values=ppm_values,
    data=Hplus_values,
    ylabel="H+ [mol/L]",
    title="H+ vs CO2 for All Regions Combined (Coastal + Offshore)",
    save_path=os.path.join(SAVEDIR, "Hplus_vs_CO2_combined_regions.png")
)

# 2. Plot H+ split into coastal and offshore regions
plot_split_coastal_offshore(
    ppm_values=ppm_values,
    data=Hplus_values,
    ylabel="H+ [mol/L]",
    title="H+ vs CO2 Split into Coastal and Offshore Regions",
    save_path=os.path.join(SAVEDIR, "Hplus_vs_CO2_split_coastal_offshore.png")
)

# 3. Plot general mean H+ values
plot_general_means(
    ppm_values=ppm_values,
    coastal_means=Hplus_coastal_means,
    offshore_means=Hplus_offshore_means,
    ylabel="H+ [mol/L]",
    title="Mean H+ vs CO2 for Coastal and Offshore Regions",
    save_path=os.path.join(SAVEDIR, "Hplus_vs_CO2_general_means.png")
)



#%%
# CO2 ppm values
ppm_values = [414, 600, 1100] # 414ppm = present, 600ppm = ssp245 scenario, 1100ppm = ssp585 scenario

# pH values for coastal and offshore regions
pH_values = {
    "Northern": [8.04, 7.90, 7.74],
    "Central": [8.03, 7.88, 7.73],
    "Southern": [8.05, 7.90, 7.74],
}

# General mean values for all regions
pH_means = np.mean([pH_values[region] for region in pH_values], axis=0)

# Plot pH vs CO2 for the three scenarios
plt.figure(figsize=(10, 6))
plt.plot(ppm_values, pH_means, marker='o', linestyle='-', color='blue')

plt.xlabel('$CO_2$ [ppm]', fontsize=12)
plt.ylabel('pH', fontsize=12)
plt.title('pH vs CO2 ', fontsize=16)
plt.ylim(7.7, 8.2)
plt.legend()
plt.grid(True)
plt.show()
# %%
