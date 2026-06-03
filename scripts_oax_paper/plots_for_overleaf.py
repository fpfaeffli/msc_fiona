
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

# Add tex
ax.text(0.5, 0.5, "Probability density function",
        fontsize=14, ha="center", va="center", color="black")

# Save and show
plt.savefig("/nfs/sea/work/fpfaeffli/plots/plots_for_overleaf/coastal_point_description.png", dpi=200, transparent=True, bbox_inches='tight')
plt.show()

# %%

import matplotlib.pyplot as plt

# Create a blank figure
fig, ax = plt.subplots(figsize=(6, 1))  # Adjust size as needed
ax.set_axis_off()  # Hide axes

# Add text
ax.text(0.5, 0.5, "Model setup",
        fontsize=14, ha="center", va="center", color="black")

# Save and show
plt.savefig("/nfs/sea/work/fpfaeffli/plots/plots_for_overleaf/Model_setup.png", dpi=200, transparent=True, bbox_inches='tight')
plt.show()

# %%
# Plot the labels

# %%
# %%
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = "/nfs/sea/work/fpfaeffli/plots/plots_for_overleaf/"
os.makedirs(output_dir, exist_ok=True)

# Panel labels
panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)']

# Font and figure settings
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': False
})

# Generate labeled boxes
for label in panel_labels:
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.set_axis_off()
    
    ax.text(
        0.5, 0.5, label,
        ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
        fontsize=18, transform=ax.transAxes
    )
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"panel_label_{label.strip(')')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

print(f"Labeled boxes saved to: {output_dir}")

# %%
# Horizontal colorbar for present vs future maps:

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import cmocean

# Dummy data for colorbar visualization
dummy_large = np.linspace(0, 300, 100).reshape(10, 10)
dummy_small = np.linspace(0, 25, 100).reshape(10, 10)

# Colormaps
cmap_small_range = cm.get_cmap('cmo.tempo', 15)
cmap_big_range = cm.get_cmap('cmo.amp', 15)

# Create a new figure only for colorbars
fig = plt.figure(figsize=(6, 2.5))

# First colorbar (big range, top)
cbax1 = fig.add_axes([0.15, 0.8, 0.7, 0.15])  # [left, bottom, width, height]
img1 = plt.imshow(dummy_large, cmap=cmap_big_range, vmin=0, vmax=300)
cb1 = plt.colorbar(img1, cax=cbax1, orientation='horizontal')
cb1.set_label('Extreme days per year')
cb1.set_ticks([0, 50, 100, 150, 200, 250, 300])

# Second colorbar (small range, bottom)
cbax2 = fig.add_axes([0.15, 0.2, 0.7, 0.15])
img2 = plt.imshow(dummy_small, cmap=cmap_small_range, vmin=0, vmax=25)
cb2 = plt.colorbar(img2, cax=cbax2, orientation='horizontal')
cb2.set_label('Extreme days per year')
cb2.set_ticks([0, 5, 10, 15, 20, 25])

plt.savefig("horizontal_colorbars.png", dpi=200, bbox_inches='tight')
plt.show()

# %%
