
### Colorbar for driver analysis maps:

# %%
# %%
# %%
# %%
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import matplotlib.colors as mcolors

# Define colormap and limits
cmap = cmo.balance  # Replace with your desired colormap
vmin, vmax = -3, 3  # Fixed color scale limits
levels = np.linspace(vmin, vmax, 40)  # 40 discrete levels

# Create figure for the colorbar
fig, ax = plt.subplots(figsize=(1.5, 8))  # Adjust size as needed
fig.subplots_adjust(left=0.5, right=0.9)

# Use BoundaryNorm to ensure discrete color blocks
norm = mcolors.BoundaryNorm(levels, cmap.N, extend="both")

# Create the colorbar with discrete blocks
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
sm.set_clim(vmin, vmax)

# Add colorbar with extensions
cbar = fig.colorbar(sm, cax=ax, orientation="vertical",
                     label=r"Change in H$^+$ concentration [$10^{-9}$ mol/L]", extend="both",
                     ticks=np.linspace(vmin, vmax, 11))  # 11 tick marks for readability

# Save and show
plt.savefig("/nfs/sea/work/fpfaeffli/plots/plots_for_overleaf/colorbar_driver_plots.png", dpi=200, transparent=True, bbox_inches='tight')
plt.show()


#%%

# Standalone titles

# %%
import matplotlib.pyplot as plt

# Create a blank figure
fig, ax = plt.subplots(figsize=(6, 1))  # Adjust size as needed
ax.set_axis_off()  # Hide axes

# Add text
ax.text(0.5, 0.5, "Extreme Ω days per year under a moving baseline",
        fontsize=14, ha="center", va="center")

# Save and show
plt.savefig("/nfs/sea/work/fpfaeffli/plots/plots_for_overleaf/title_extreme_days_per_year_omega_moving.png", dpi=200, transparent=True, bbox_inches='tight')
plt.show()

# %%

import matplotlib.pyplot as plt

# Create a blank figure
fig, ax = plt.subplots(figsize=(6, 1))  # Adjust size as needed
ax.set_axis_off()  # Hide axes

# Add text
ax.text(0.5, 0.5, "ensemble 1",
        fontsize=14, ha="center", va="center")

# Save and show
plt.savefig("/nfs/sea/work/fpfaeffli/plots/plots_for_overleaf/ensemble1.png", dpi=200, transparent=True, bbox_inches='tight')
plt.show()

# %%
