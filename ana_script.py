# -------------------------- imports and settings ---------------------
import os 
import log_ana
from log_ana import Log
import numpy as np
import matplotlib.pyplot as plt


# -------------------------- load file and extract data ---------------
# attention: frame_id 0 - 440 (441 frames in total)
path = "/Volumes/HardDevice/Fall25/CHEME6130/assignment6/mini"
n_parallel = 10
n_last_frames = 290
n_bins = 500
log_files = []
para_energies = []
rdf_parallels = []

for i in range(1, n_parallel+1):
    frame_energies = []
    filename = f"d3.5_s80000_{i}.txt"
    log_file = Log(
        path=path,
        filename=filename
    )
    n_frames = log_file.count_frames()

    
    # extract frame energies from 10 parallel simulations
    for j in range(n_frames):
        frame_energy = log_file.compute_etotal(frame_id=j)
        frame_energies.append(frame_energy)
    para_energies.append(frame_energies)
para_energies = np.array(para_energies)
average_energies = np.mean(para_energies, axis=0)
average_energies = average_energies[1:]

"""
    # compte RDF from last 10 frames of 10 parallel simulations
    rdf_samples = []
    for j in range(n_last_frames):
        gr = log_file.compute_rdf(density=0.9, n_bins=n_bins, frame_id=n_frames-j-1)
        rdf_samples.append(gr)
    rdf_parallels.append(rdf_samples)
rdf_parallels = np.array(rdf_parallels)
print("Shape:", rdf_parallels.shape)
avg_rdfs = np.mean(rdf_parallels, axis=(0,1))
"""

# -------------------------- data post-processing and save ---------------
"""
# compute RDF of equilibrium configuration (from frame 151 to 440, 290 frames totally)
avg_filename = "avg_rdf.txt"
avg_rdf_file = os.path.join(path, avg_filename)
with open(avg_rdf_file, "w") as rdf:
    for i, average in enumerate(avg_rdfs):
        rdf.write(f"{i} {average}\n")
"""


# write out data of average energy
avg_filename = f"avg_energy.txt"
avg_ene_file = os.path.join(path, avg_filename)
with open(avg_ene_file, "w") as avg:
    for i, average in enumerate(average_energies):
        avg.write(f"{i+1} {average}\n")
# test size of average energy data sets
print(len(para_energies))
print(len(para_energies[0]))
print(len(average_energies))

    