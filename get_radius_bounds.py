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
data_dict = {}

for col_name in ["OuterTrackerBarrelCollection", "InnerTrackerBarrelCollection", "VertexBarrelCollection"]:
    data_dict[col_name], _, feature_labels = load_in_data([col_name], "xy", working_dir, 1)
        

bins_dict = {
    0: np.linspace(-20, 0, NUM_BINS),
    1: np.linspace(0, 1500, NUM_BINS),
    2: np.linspace(-4, 4, NUM_BINS),
    3: np.linspace(-1500, 1500, NUM_BINS),
    4: np.linspace(-5, 25, NUM_BINS),
}

# %%

plot_hists_1d(data_dict, bins_dict, log_dims=[], labels=feature_labels)


# %%
radii = {
    "OuterTrackerBarrelCollection": [819, 1153, 1486],
    "InnerTrackerBarrelCollection": [127, 340, 554],
    "VertexBarrelCollection": [30, 51, 74, 102],
}


mask_definitions = {
    col_name: { 
        r: { 
            "lower_bounds":[],
            "upper_bounds":[],
            
           } for r in radii[col_name]
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
                mask_definitions[col_name][radius]["lower_bounds"].append(bin_edges[i])
            if hist[i] > 0 and hist[i+1] == 0:
                mask_definitions[col_name][radius]["upper_bounds"].append(bin_edges[i+1])

    
        plt.figure()
        for j in range(len(mask_definitions[col_name][radius]["lower_bounds"])):
            plt.axvspan(mask_definitions[col_name][radius]["lower_bounds"][j], mask_definitions[col_name][radius]["upper_bounds"][j], alpha = 0.5, color = "pink")
            
        plt.hist(r, bins = bins)
        plt.xlim([radius - 10, radius + 10])
        plt.axvline(radius, color = "r")

        plt.title(f"{col_name}\nradius={radius}mm")
        plt.yscale("log")
        plt.xlabel("$r$ [mm]")
        plt.ylabel("Counts")
        plt.show()

# %%
with open("mask_definitions.pkl", "wb") as ofile:
    pickle.dump(mask_definitions, ofile)


# %%


# %%

# %%

# %%

# %%

# %%

# %%

# %%
