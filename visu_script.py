# -------------------------- imports and settings ---------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker

# -------------------------- imports and settings -----------
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams.update({
    'font.family': 'serif',   # default font for all text
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 12,
    'mathtext.fontset': 'dejavuserif', 
})


# -------------------------- load file and data ------------
path = "/Volumes/HardDevice/Fall25/CHEME6130/assignment6/mini"
ene_filename = "avg_energy.txt"
rdf_filename = "avg_rdf.txt"
ene_file = os.path.join(path, ene_filename)
#rdf_file = os.path.join(path, rdf_filename)
ene_data = np.loadtxt(ene_file)
#rdf_data = np.loadtxt(rdf_file)


# -------------------------- extract parameters and visualize --
# total energy v.s. steps
subset = ene_data
ene_frames = subset[:, 0]
total_e = subset[:, 1]
ene_steps = [(frame_id+1)*50 for frame_id in ene_frames]
delta = ["0.02", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5"]
accept = ["260", "256", "237", "233", "224", "191", "186"]
delta = np.array(delta, dtype=float)
accept = np.array(accept, dtype=float)
accept = accept / 500 * 100
total = 0
for e in total_e:
    total += e
avg = total/len(total_e)
#r = rdf_data[:, 0]
#gr = rdf_data[:, 1]

# plot delta-accept
fig, ax = plt.subplots(figsize=(6.4, 4.8))
formatter = ticker.ScalarFormatter(useMathText=True)
plt.tick_params(axis="both", direction="in")
ax.xaxis.set_major_formatter(formatter)
ax.ticklabel_format(axis="x", style="sci")

ax.set_xlabel(r"$\delta x_{max}$")
ax.set_ylabel(f"Acceptance rate / %")
plt.scatter(delta, accept, color="orange", marker="s", zorder=3)
plt.plot([delta[0], delta[-1]], [accept[0], accept[-1]], linestyle='--', color="grey", alpha=0.5)
xticks = [d for d in delta if d != 0.05]
plt.xticks(xticks)
xmin = min(delta)
xmax = max(delta)
ymin = min(accept)
ymax = max(accept)
padx = (xmax - xmin) * 0.0025
pady =  (ymax - ymin) * 0.05
ax.set_xlim(xmin - padx, xmax + padx)
ax.set_ylim(min(ymin-pady, ymax-pady))
plt.tight_layout()
plt.savefig(f"{path}/accept.jpg", dpi=1000)
plt.show()
"""
# smoothed rdf data
fig, ax = plt.subplots(figsize=(6.4, 4.8))
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((2, 3))
plt.tick_params(axis="both", direction="in")
ax.xaxis.set_major_formatter(formatter)
ax.ticklabel_format(axis="x", style="sci")

ax.set_xlim(r[136], 500)
ax.set_ylim(min(gr), 0.8)
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$g(r)$")
plt.plot(r, gr, linewidth=2, color="#8b3a2b")
plt.tight_layout()
plt.savefig(f"{path}/rdf.jpg", dpi=1000)
plt.show()



# plot energy-step
fig, ax = plt.subplots(figsize=(6.4, 4.8))
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((2, 3))
plt.tick_params(axis="both", direction="in")
ax.xaxis.set_major_formatter(formatter)
ax.ticklabel_format(axis="x", style="sci")

ax.set_xlim(0, ene_steps[-1])
#ax.set_ylim(-1000, 200)
ax.set_xlabel("Number of Steps")
ax.set_ylabel(r"$E_{total}$")
plt.plot(ene_steps, total_e, color="green", alpha=0.3)
plt.hlines(y=avg, xmin=ene_steps[0], xmax=ene_steps[-1], colors='grey', linestyles='--', linewidth=1.5, label=f"Average Total Energy = {avg:.3f}")
plt.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(f"{path}/energy.jpg", dpi=1000)
plt.show()
"""

