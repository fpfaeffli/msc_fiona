"""
Author: Fiona Pfäffli
Description: This code plots the results of the salinity normalized driver analysis in a piechart.

"""
#%%
# Import libraries

import numpy as np
import matplotlib.pyplot as plt

#%%
##############################
#            DATA            #
##############################

# Order: [Temperature, DIC, Alkalinity, Salinity]
scenarios = {
    "present": [0.7, 88.4, 10.7, 0.2],
    "ssp245":  [7.7, 83.0,  9.1, 0.1],
    "ssp585":  [13.5, 84.8, 1.7, 0.0]
}

labels = ["Temperature contribution", "DIC contribution",
          "Alkalinity contribution", "Salinity contribution"]

# Palette 
colors = {
    "Temperature contribution": "#d97706",  # warm orange
    "DIC contribution":         "#80b6ff",  # soft blue
    "Alkalinity contribution":  "#3ba370",  # teal/green
    "Salinity contribution":    "#f2b600",  # amber/yellow
}

##############################
#        PLOTTING CODE        #
##############################

#%% Pie chart function 
def make_pie(ax, data, title):
    wedges, _ = ax.pie(
        data,
        startangle=90,
        labels=None,
        colors=[colors[l] for l in labels],
        wedgeprops=dict(linewidth=1.2, edgecolor="white"),
    )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=16, pad=60, color="black")  # higher up, not bold

    # compute percentages
    fracs = np.asarray(data, dtype=float)
    fracs = fracs / fracs.sum() * 100.0

    # outside labels with overlap-avoidance
    used_y = []
    for w, pct, lab in zip(wedges, fracs, labels):
        ang_deg = (w.theta2 + w.theta1) / 2.0
        ang = np.deg2rad(ang_deg)

        # place labels a bit outside
        r_text = 1.35
        x = r_text * np.cos(ang)
        y = r_text * np.sin(ang)

        # nudge vertically to avoid overlaps
        for _ in range(24):
            if any(abs(y - yy) < 0.09 for yy in used_y):
                y += 0.09 * (1 if y >= 0 else -1)
            else:
                break
        used_y.append(y)

        ha = "left" if x >= 0 else "right"
        ax.annotate(
            f"{pct:.1f}%",   # always show (even 0.0%)
            xy=(np.cos(ang)*1.0, np.sin(ang)*1.0),   # arrow tail on wedge rim
            xytext=(x, y),
            ha=ha, va="center",
            fontsize=12, color=colors[lab],
            arrowprops=dict(arrowstyle="-", lw=1.4, color=colors[lab],
                            shrinkA=0, shrinkB=0)
        )

# ---- figure styling ----
plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none"})
fig, axs = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True, facecolor="none")

for ax, (title, data) in zip(axs, scenarios.items()):
    make_pie(ax, data, title)

# shared legend (further down)
handles = [plt.Line2D([0], [0], marker='o', linestyle='',
                      markerfacecolor=colors[l], markeredgecolor='black',
                      markersize=10, label=l) for l in labels]
leg = fig.legend(handles=handles, loc="lower center", ncol=4,
                 frameon=False, bbox_to_anchor=(0.5, -0.14), fontsize=14)
for txt in leg.get_texts():
    txt.set_color("black")

fig.savefig(f'/nfs/sea/work/fpfaeffli/plots/driver_analysis/sal_norm_driver_analysis_pies.png', dpi=300, transparent=True, bbox_inches="tight")
plt.show()
# %%
