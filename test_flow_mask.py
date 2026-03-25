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
#     display_name: Python (muon_collider_env)
#     language: python
#     name: muon_collider_env
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
import yaml
import zuko
from helpers.data_transforms import preprocess_data, inverse_preprocess_data, load_in_data

from helpers.models.DNN import count_parameters
from helpers.evaluation import get_kl_dist, discriminate_data_from_samples, get_wasserstein_dist, get_delta_R_neighbors_numba_exact, get_delta_R_neighbors_kdtree
from helpers.plotting import plot_hists_1d, plot_corner_hist_2d


plt.style.use("../science.mplstyle")

# %%
BIN_BOUND = 5
NUM_BINS = 60
NUM_FEATURES = 5
NUM_BDTS = 10


ZUKO_ID = "NCSF"
NAME = "VBC_M_4096_rphi"
NUM_COND_INPUTS = 0
COLLECTION_NAME = "VertexBarrelCollection"
SEED = 8

EVALUATE_SAMPLES = False

FEATURES = "rphi"
working_dir = "/pscratch/sd/r/rmastand/muon_collider"
log_vars = [1]

save_dir = f"{working_dir}/zuko_outputs/{ZUKO_ID}/{NAME}"



# %%
data, _, feature_labels = load_in_data([COLLECTION_NAME], FEATURES, working_dir, 0.5)
flow_samples = np.load(f"{save_dir}/flow_samples.npy")

# if FEATURES == "rphi":
#     r = np.sqrt(flow_samples[:, 1]**2 + flow_samples[:, 2]**2)
#     theta = np.arctan2(flow_samples[:, 2], flow_samples[:, 1])
#     flow_samples[:,1] = r
#     flow_samples[:,2] = theta

bins_dict = {}
bins_dict_preproc = {i:np.linspace(-BIN_BOUND, BIN_BOUND, NUM_BINS) for i in range(data.shape[1])}

for i in range(data.shape[1]):
    if i in log_vars:
        bins_dict[i] = np.logspace(np.log10(0.9*np.min(data[:,i])), np.log10(1.1*np.max(data[:,i])), NUM_BINS) 
    else:
        bins_dict[i] = np.linspace(np.min(data[:,i] - 3), np.max(data[:,i] + 3), NUM_BINS) 


# %%
plot_hists_1d({"data":data, "samples": flow_samples}, bins_dict, log_dims=log_vars, labels=feature_labels)


try:
    with open(f"{save_dir}/results.txt") as ifile:
        a = ifile.readlines()[-1]
        print(a)
except:
    pass


# %%
plt.figure(figsize = (15, 15))
plt.scatter(data[:,1]*np.cos(data[:,2]), data[:,1]*np.sin(data[:,2]), s = 0.001)
plt.xlim(-130, 130)
plt.ylim(-130, 130)
plt.show()


plt.figure(figsize = (15, 15))
plt.scatter(flow_samples[:,1]*np.cos(flow_samples[:,2]), flow_samples[:,1]*np.sin(flow_samples[:,2]), s = 0.001)
plt.xlim(-130, 130)
plt.ylim(-130, 130)
plt.show()

# %%
fig_samp, axes_samp = plot_corner_hist_2d(
   flow_samples,
    feature_labels=feature_labels,
    bins_dict=bins_dict,
    log_dims=log_vars,
    title= "flow_samples",
)


# %% [markdown]
# # Restrict layers
#

# %%
import pickle
# https://github.com/key4hep/k4geo/blob/main/MuColl/MAIA/compact/MAIA_v0/OuterTracker_o2_v06_01.xml
with open("mask_definitions.pkl", "rb") as ifile:
    mask_definitions = pickle.load(ifile)



def make_mask(r):

    with open("mask_definitions.pkl", "rb") as ifile:
        mask_definitions = pickle.load(ifile)

    mask = []
    for radius in mask_definitions[COLLECTION_NAME].keys():
        for i in range(len(mask_definitions[COLLECTION_NAME][radius]["lower_bounds"])):
           
    
            tmp = (
                (r >= mask_definitions[COLLECTION_NAME][radius]["lower_bounds"][i]) &
                (r <= mask_definitions[COLLECTION_NAME][radius]["upper_bounds"][i])
            )
    
            mask.append(tmp.reshape(-1,1))

    mask = np.any(np.hstack(mask), axis=1)

    return mask


for col_name in mask_definitions.keys():
    print(col_name)
    for r in mask_definitions[col_name].keys():
        print("r:", r)
        print("   lower bounds:", mask_definitions[col_name][r]["lower_bounds"])
        print("   upper bounds:", mask_definitions[col_name][r]["upper_bounds"])
    print()


# %%
if FEATURES == "rphi":
    r_data = data[:,1]
    r_flow = flow_samples[:,1]
else: 
    r_data = np.sqrt(data[:,1]**2+data[:,2]**2)
    r_flow = np.sqrt(flow_samples[:,1]**2+flow_samples[:,2]**2)

mask_flow = make_mask(r_flow)

print("Num. data samples:", len(r_data))
print("Num. flow samples:",len(mask_flow), "\nNum. flow samples pass mask:", sum(mask_flow), "\n% flow samples pass mask:", sum(mask_flow)/len(mask_flow))


plt.figure()
plt.hist(r_data, bins = np.linspace(0, 1500, 1000), histtype = "step", label = "data")
plt.hist(r_flow, bins = np.linspace(0, 1500, 1000), histtype = "step", label = "flow")
plt.legend()
plt.yscale("log")
plt.xlabel("$r$ [mm]")
plt.ylabel("Counts")
plt.show()


# %%

for radius in mask_definitions[COLLECTION_NAME].keys():

    bins = np.linspace(radius - 10, radius + 10, 1000)
   

    plt.figure(figsize = (10, 5))
    for j in range(len(mask_definitions[COLLECTION_NAME][radius]["lower_bounds"])):
        plt.axvspan(mask_definitions[COLLECTION_NAME][radius]["lower_bounds"][j], mask_definitions[COLLECTION_NAME][radius]["upper_bounds"][j], alpha = 0.5, color = "pink")
        
    plt.hist(r_flow, bins = bins, histtype = "step", label = "flow", density = True)
    plt.hist(r_flow[mask_flow], bins = bins, histtype = "step", label = "masked flow", density = True)
    plt.hist(r_data, bins = bins, histtype = "step", label = "data", density = True)
    plt.xlim([radius - 10, radius + 10])
    plt.axvline(radius, color = "r")
    plt.legend()

    plt.title(f"{COLLECTION_NAME}\nr={radius}mm")
    plt.yscale("log")
    plt.xlabel("$r$ [mm]")
    plt.ylabel("Density")
    plt.show()


# %%
plot_hists_1d({"samples":flow_samples, "masked flow samples":flow_samples[mask_flow]}, bins_dict, log_dims=log_vars, labels=feature_labels)


fig_samp, axes_samp = plot_corner_hist_2d(
        flow_samples,
        feature_labels=feature_labels,
        bins_dict=bins_dict,
        log_dims=log_vars,
        title= "flow samples",
    )



fig_samp, axes_samp = plot_corner_hist_2d(
        flow_samples[mask_flow],
        feature_labels=feature_labels,
        bins_dict=bins_dict,
        log_dims=log_vars,
        title= "masked flow samples",
    )


# %%

# %%

# %%

# %%
import torch

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)

def run_eval_suite(data, samples, R_values):

    print(f"Len data: {len(data)}, len samples: {len(samples)}")


    # Wasserstein dist
    # wass_dists_samples = get_wasserstein_dist(data, samples)
    # wass_dists_gaussians = get_wasserstein_dist(np.random.normal(size = data.shape), np.random.normal(size = samples.shape))                                            
    # for i, wass_dist in enumerate(wass_dists_samples):
    #     print("Feature {i} Wasserstein distance: {wass_dist} (for gaussian: {wass_dist_gauss})".format(i=i, wass_dist=wass_dist, wass_dist_gauss=wass_dists_gaussians[i]))

    # BDT
    auc_mean, auc_std, best_epoch_list, max_epochs, bdt_list, samples_test, loc_scores = discriminate_data_from_samples(
                                            data,
                                            samples,
                                            3,
                                            "configs/bdt.yml",
                                            model_type="bdt",
                                            plot_losses=True,
                                            device="cuda",
                                            val_size = 0.25, 
        plot_dir = save_dir
                                        )

    print(f"auc {auc_mean} \pm {auc_std}. best epoch {best_epoch_list} of {max_epochs}.\n")

    
    plt.figure()
    plt.hist(loc_scores, bins = np.linspace(0, 1, 100), histtype = "step", density = True)
    plt.yscale("log")
    plt.xlabel("scores")
    plt.ylabel("Density")
    plt.show()
    
    
    X_plot = {"samples":  samples_test}
    percentiles = [90, 95, 99, 99.9]
    for p in percentiles:
        X_plot[f"samples, top {p}%"] =  samples_test[loc_scores >= np.percentile(loc_scores, p)]

    plot_hists_1d({"data":data, **X_plot}, bins_dict, log_dims=log_vars, labels=feature_labels)

    # Clustering
    for R in R_values:
        neighbor_counts_data = get_delta_R_neighbors_kdtree(data, R, FEATURES)
        neighbor_counts_flow = get_delta_R_neighbors_kdtree(samples, R, FEATURES)
    
        max_val0 = np.max(neighbor_counts_data)
        max_val1 = np.max(neighbor_counts_flow)
        max_val = np.max([max_val0, max_val1])
    
        bin_spacing = int(max_val/NUM_BINS)
        plt.figure()
        plt.hist(neighbor_counts_data, bins=np.arange(0, max_val, bin_spacing), histtype = "step", label = "data")
        plt.hist(neighbor_counts_flow, bins=np.arange(0, max_val, bin_spacing), histtype = "step", label = "flow")
        plt.legend()
        plt.xlabel(f"# neighbors within $\Delta$R $\leq$ {R}")
        plt.ylabel("Count")
        plt.show()



# %%
N = 100_000

indices = np.random.choice(data.shape[0], size=N, replace=False)
indices_mask = np.random.choice(sum(mask_flow), size=N, replace=False)



run_eval_suite(data[indices], flow_samples[indices], [0.1, 0.2, 0.4])


# %%

           


# %%

# %%



run_eval_suite(data[indices], flow_samples[mask_flow][indices_mask], [0.1, 0.2, 0.4])


# %%



# %%

# %%

# %%

# %%

# %%


# %%

# %%
