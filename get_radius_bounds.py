# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (VSCode)
#     language: python
#     name: vs_env
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
import torch
import yaml
from helpers.data_transforms import  load_in_data


plt.style.use("../science.mplstyle")


# %%
from helpers.plotting import plot_hists_1d, plot_corner_hist_2d

# %%


BIN_BOUND = 5
NUM_BINS = 100
NUM_FEATURES = 5



working_dir = "/pscratch/sd/r/rmastand/muon_collider"


# %%

feature_labels = ["log($E$) [Gev]", "$x$", "$y$", "$z$", "$t$ [s]", "context"]
log_vars = []



# %%
data_dict = {}

for col_name in ["OuterTrackerBarrelCollection", "InnerTrackerBarrelCollection", "VertexBarrelCollection"]:
    data_dict[col_name] = load_in_data([col_name], "xy", working_dir, 0.1)[0]
        

bins_dict = {
    0: np.linspace(-20, 0, NUM_BINS),
    1: np.linspace(0, 1500, NUM_BINS),
    2: np.linspace(-4, 4, NUM_BINS),
    3: np.linspace(-1500, 1500, NUM_BINS),
    4: np.linspace(-5, 25, NUM_BINS),
}

# %%

plot_hists_1d(data_dict, bins_dict, log_dims=log_vars, labels=feature_labels)




# %%
radii = {
    "OuterTrackerBarrelCollection": [819, 1153, 1486],
    "InnerTrackerBarrelCollection": [127, 340, 554],
    "VertexBarrelCollection": [30, 51, 74, 102],
}


mask_definitions = {
    col_name: {
        
    "lower_bounds":[],
    "upper_bounds":[],
}              
   for col_name in data_dict.keys() 
                   }


for col_name in data_dict.keys():

    r = np.sqrt(data_dict[col_name][:,1]**2 + data_dict[col_name][:,2]**2)
    
    for radius in radii[col_name]:

        bins = np.linspace(radius - 10, radius + 10, 1000)
        
        hist, bin_edges = np.histogram(r, bins)

        for i in range(len(hist)-1):
            if hist[i] == 0 and hist[i+1] > 0:
                mask_definitions[col_name]["lower_bounds"].append(bin_edges[i])
            if hist[i] > 0 and hist[i+1] == 0:
                mask_definitions[col_name]["upper_bounds"].append(bin_edges[i+1])

    
        plt.figure()
        for j in range(len(mask_definitions[col_name]["lower_bounds"])):
            plt.axvspan(mask_definitions[col_name]["lower_bounds"][j], mask_definitions[col_name]["upper_bounds"][j], alpha = 0.5, color = "pink")
            
        plt.hist(r, bins = bins)
        plt.xlim([radius - 10, radius + 10])
        plt.axvline(radius, color = "r")

        plt.title(f"{col_name}")
        plt.yscale("log")
        plt.show()

# %%
with open("mask_definitions.pkl", "wb") as ofile:
    pickle.dump(mask_definitions, ofile)


# %%


# %%

# %%

# %%
import numpy as np
from numba import njit, prange
from tqdm import tqdm


def get_delta_R_neighbors(data_array, R, NN):
    # Extract coordinates
    x = data_array[:NN, 1]
    y = data_array[:NN, 2]
    z = data_array[:NN, 3]

    # Convert to eta, phi
    phi = np.arctan2(y, x)
    rT = np.sqrt(x**2 + y**2)
    eta = np.arcsinh(z / rT)

    N = len(eta)
    print(col_name)
    print("N =", N)

    # Bin size ~ ΔR
    deta = 2 * R
    dphi = 2 * R
    eta_min = eta.min()
    phi_min = -np.pi

    eta_bin = np.floor((eta - eta_min) / deta).astype(int)
    phi_bin = np.floor((phi - phi_min) / dphi).astype(int)

    # Build grid
    grid = {}
    print(len(grid.keys()))
    for i in range(N):
        key = (eta_bin[i], phi_bin[i])
        if key not in grid:
            grid[key] = []
        grid[key].append(i)

    # Initialize neighbor counts
    neighbor_counts = np.zeros(N, dtype=int)

    # Search neighboring bins
    for i in tqdm(range(N)):
        eb = eta_bin[i]
        pb = phi_bin[i]

        for de in [-1, 0, 1]:
            for dp in [-1, 0, 1]:
                key = (eb + de, pb + dp)
                if key not in grid:
                    continue
                for j in grid[key]:
                    if j == i:
                        continue
                    d_eta = eta[j] - eta[i]
                    d_phi = phi[j] - phi[i]
                    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
                    dR = np.sqrt(d_eta**2 + d_phi**2)
                    if dR <= R:
                        neighbor_counts[i] += 1

    return neighbor_counts

    

import numpy as np
from numba import njit

@njit
def get_delta_R_neighbors_numba_exact(data_array, R, NN):
    # Extract coordinates
    x = data_array[:NN, 1]
    y = data_array[:NN, 2]
    z = data_array[:NN, 3]

    # Convert to eta, phi
    phi = np.arctan2(y, x)
    rT = np.sqrt(x**2 + y**2)
    eta = np.arcsinh(z / rT)

    N = len(eta)

    # Bin size ~ ΔR
    deta = 2 * R
    dphi = 2 * R
    eta_min = eta.min()
    phi_min = -np.pi

    eta_bin = np.floor((eta - eta_min) / deta).astype(np.int64)
    phi_bin = np.floor((phi - phi_min) / dphi).astype(np.int64)

    # Number of bins
    n_eta_bins = eta_bin.max() + 1
    n_phi_bins = phi_bin.max() + 1
    n_bins = n_eta_bins * n_phi_bins

    # Count points per bin
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(N):
        b = eta_bin[i] * n_phi_bins + phi_bin[i]
        bin_counts[b] += 1

    # Compute start indices (cumulative sum)
    bin_start = np.zeros(n_bins + 1, dtype=np.int64)
    total = 0
    for b in range(n_bins):
        bin_start[b] = total
        total += bin_counts[b]
    bin_start[n_bins] = total

    # Flattened array storing all points
    bin_points = np.zeros(N, dtype=np.int64)
    temp_count = np.zeros(n_bins, dtype=np.int64)
    for i in range(N):
        b = eta_bin[i] * n_phi_bins + phi_bin[i]
        idx = bin_start[b] + temp_count[b]
        bin_points[idx] = i
        temp_count[b] += 1

    # Neighbor counting
    neighbor_counts = np.zeros(N, dtype=np.int64)
    for i in range(N):
        eb = eta_bin[i]
        pb = phi_bin[i]

        # exactly 3x3 neighbor bins
        for de in [-1, 0, 1]:
            for dp in [-1, 0, 1]:
                ebi = eb + de
                pbi = pb + dp
                if ebi < 0 or ebi >= n_eta_bins or pbi < 0 or pbi >= n_phi_bins:
                    continue

                # indices of points in this neighbor bin
                start = bin_start[ebi * n_phi_bins + pbi]
                end = bin_start[ebi * n_phi_bins + pbi + 1]
                for k in range(start, end):
                    j = bin_points[k]
                    if j == i:
                        continue
                    d_eta = eta[j] - eta[i]
                    d_phi = phi[j] - phi[i]
                    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
                    dR = np.sqrt(d_eta ** 2 + d_phi ** 2)
                    if dR <= R:
                        neighbor_counts[i] += 1

    return neighbor_counts


# %%
paths = {
"OuterTrackerBarrelCollection":"/pscratch/sd/r/rmastand/muon_collider/npys/flow_samples/OuterTrackerBarrelCollection_NCSF_OTBC_S_2048.npy",
    "InnerTrackerBarrelCollection":"/pscratch/sd/r/rmastand/muon_collider/npys/flow_samples/InnerTrackerBarrelCollection_NCSF_ITBC_S_2048.npy",
    "VertexBarrelCollection":"/pscratch/sd/r/rmastand/muon_collider/npys/flow_samples/VertexBarrelCollection_NCSF_VBC_S_2048.npy",
}
for col_name in data_dict.keys():

    neighbor_counts_data = get_delta_R_neighbors_numba_exact(data_dict[col_name], 0.1, 10000)


    a = np.load(paths[col_name])
    neighbor_counts_flow = get_delta_R_neighbors_numba_exact(a, 0.1, 10000)

    max_val0 = np.max(neighbor_counts_data)
    max_val1 = np.max(neighbor_counts_flow)
    max_val = np.max([max_val0, max_val1])


    plt.figure()
    plt.hist(neighbor_counts_data, bins=np.arange(0, max_val, 1), histtype = "step", label = "data")
    plt.hist(neighbor_counts_flow, bins=np.arange(0, max_val, 1), histtype = "step", label = "flow")
    plt.legend()
    plt.xlabel("Number of neighbors within ΔR <= 0.1")
    plt.ylabel("Count")
    plt.title(col_name)
    plt.show()

# %%

paths = {
"OuterTrackerBarrelCollection":"/pscratch/sd/r/rmastand/muon_collider/npys/flow_samples/OuterTrackerBarrelCollection_NCSF_OTBC_S_2048.npy",
    "InnerTrackerBarrelCollection":"/pscratch/sd/r/rmastand/muon_collider/npys/flow_samples/InnerTrackerBarrelCollection_NCSF_ITBC_S_2048.npy",
    "VertexBarrelCollection":"/pscratch/sd/r/rmastand/muon_collider/npys/flow_samples/VertexBarrelCollection_NCSF_VBC_S_2048.npy",
}
for col_name in data_dict.keys():

    neighbor_counts_data = get_delta_R_neighbors(data_dict[col_name], 0.1, 10000)


    a = np.load(paths[col_name])
    neighbor_counts_flow = get_delta_R_neighbors(a, 0.1, 10000)

    max_val0 = np.max(neighbor_counts_data)
    max_val1 = np.max(neighbor_counts_flow)
    max_val = np.max([max_val0, max_val1])


    plt.figure()
    plt.hist(neighbor_counts_data, bins=np.arange(0, max_val, 1), histtype = "step", label = "data")
    plt.hist(neighbor_counts_flow, bins=np.arange(0, max_val, 1), histtype = "step", label = "flow")
    plt.legend()
    plt.xlabel("Number of neighbors within ΔR <= 0.1")
    plt.ylabel("Count")
    plt.title(col_name)
    plt.show()  

# %%

# %%

# %%
