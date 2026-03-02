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
import argparse
import yaml
from numba import cuda
import zuko
from helpers.data_transforms import preprocess_data, inverse_preprocess_data

from helpers.models.DNN import count_parameters
from helpers.evaluation import get_kl_dist, discriminate_data_from_samples

plt.style.use("../science.mplstyle")


# %%
from helpers.plotting import plot_hists_1d, plot_corner_hist_2d

# %%


BIN_BOUND = 5
NUM_BINS = 100
NUM_FEATURES = 5



ZUKO_ID = "NCSF"
NAME = "XL_2048"
BATCH_SIZE = 256
NUM_COND_INPUTS = 0

SEED = 8




# %%
save_dir = f"/pscratch/sd/r/rmastand/muon/zuko_outputs/{ZUKO_ID}/{NAME}"


with open(f"{save_dir}/wandb/latest-run/files/config.yaml") as ifile:
    configs = yaml.safe_load(ifile)


# %%

# computing
#device = cuda.get_current_device()
#device.reset()
torch.set_num_threads(2)
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)
seed = int(SEED)
torch.manual_seed(seed)
np.random.seed(seed)

# %%
collections_TrackerHitPlane = [
    "IBTrackerHits",
    "IBTrackerHitsConed",
    "IETrackerHits", 
    "IETrackerHitsConed",
    "OBTrackerHits",
    "OBTrackerHitsConed", 
    "OETrackerHits",        
    "OETrackerHitsConed",          
    "VBTrackerHits",               
    "VBTrackerHitsConed",   
    "VETrackerHits",  
    "VETrackerHitsConed",        
]

collections_SimTrackerHit = [
  #  "InnerTrackerBarrelCollection",
   # "InnerTrackerBarrelCollectionConed",
 #   "InnerTrackerEndcapCollection",
    #"InnerTrackerEndcapCollectionConed", 
    "OuterTrackerBarrelCollection",     
   # "OuterTrackerBarrelCollectionConed",
 #   "OuterTrackerEndcapCollection",  
   # "OuterTrackerEndcapCollectionConed",
 #   "VertexBarrelCollection",   
   # "VertexBarrelCollectionConed",   
  #  "VertexEndcapCollection",
   # "VertexEndcapCollectionConed",
]



feature_labels = ["log($E$) [Gev]", "$x$", "$y$", "$z$", "$t$ [s]", "context"]
log_vars = []

# %%

data = []
context = []

for i, collection in enumerate(collections_SimTrackerHit): # TODO coned too?

    tmp_data = np.load(f"/pscratch/sd/r/rmastand/muon/npys/{collection}_SimTrackerHit.npy")
    tmp_context = int(i)*np.ones((tmp_data.shape[0],1))
    context.append(tmp_context)
    data.append(tmp_data[:int(len(tmp_data)*.1)])


data = np.vstack(data)[:,[0,2,3,4,5]]
#data = data[data[:,0] > 2e-6]
data[:,0] = np.log(data[:,0])
context = np.vstack(context)

bins_dict = {}
bins_dict_preproc = {i:np.linspace(-BIN_BOUND, BIN_BOUND, NUM_BINS) for i in range(data.shape[1])}

for i in range(data.shape[1]):
    if i in log_vars:
        bins_dict[i] = np.logspace(np.log10(0.9*np.min(data[:,i])), np.log10(1.1*np.max(data[:,i])), NUM_BINS) 
    else:
        bins_dict[i] = np.linspace(np.min(data[:,i] - 3), np.max(data[:,i] + 3), NUM_BINS) 

# %%


X_preproc = preprocess_data(data, ".", ZUKO_ID)
if NUM_COND_INPUTS == 1:
    X_preproc = np.hstack([X_preproc,  context])
elif NUM_COND_INPUTS == 0:
    pass
else:
    print("ERROR")
    exit()



# %%


hidden_features = [int(x) for x in configs["HIDDEN_FEATURES"]["value"].split(",")]


if ZUKO_ID == "NSF":
    flow = zuko.flows.NSF(5, 0, transforms=configs["TRANSFORMS"]["value"], hidden_features=hidden_features).to(device)
elif ZUKO_ID == "NCSF":
    flow = zuko.flows.NCSF(5, 0, transforms=configs["TRANSFORMS"]["value"], hidden_features=hidden_features, bins=16).to(device)
elif ZUKO_ID == "SOSPF":
    flow = zuko.flows.SOSPF(5, 0, transforms=configs["TRANSFORMS"]["value"], hidden_features=hidden_features, degree=4, polynomials=3).to(device)
elif ZUKO_ID == "UNAF":
    flow = zuko.flows.UNAF(5, 0, transforms=configs["TRANSFORMS"]["value"], hidden_features=hidden_features).to(device)
elif ZUKO_ID == "CNF":
    flow = zuko.flows.CNF(5, 0, hidden_features=hidden_features, freqs=configs["FREQS"]["value"]).to(device)
else:
    print("ERROR: Unknown ZUKO_ID")
    exit()


num_params = count_parameters(flow)
print(f"Number of trainable parameters: {num_params}")



# %%


print("Making flow samples...")

eval_flow = flow

eval_flow.load_state_dict(torch.load(f"{save_dir}/test.pt"))

num_samples_total = data.shape[0] 
sample_batch_size = 8192

samples_flow = []
for i in tqdm(range(0, num_samples_total, sample_batch_size)):
    if i + sample_batch_size > num_samples_total:
        nn = num_samples_total - i
    else:
        nn = sample_batch_size
    loc_samples = eval_flow().sample((nn,)).detach().cpu().numpy()
    samples_flow.append(loc_samples)
samples_flow = np.concatenate(samples_flow)

X_samples = {
        "flow_samples":samples_flow
        }



# %%
for key in X_samples.keys():
    
    X_samples[key] = inverse_preprocess_data( X_samples[key] , ".", ZUKO_ID)


# %%

plot_hists_1d({"data":data, **X_samples}, bins_dict, log_dims=log_vars, labels=feature_labels)


try:
    with open(f"{save_dir}/results.txt") as ifile:
        a = ifile.readlines()[-1]
        print(a)
except:
    pass


# %%
for key in X_samples.keys():
    fig_samp, axes_samp = plot_corner_hist_2d(
        X_samples[key],
        feature_labels=feature_labels,
        bins_dict=bins_dict,
        log_dims=log_vars,
        title= key,
    )



# %% [markdown]
# # Restrict layers

# %%

# https://github.com/key4hep/k4geo/blob/main/MuColl/MAIA/compact/MAIA_v0/OuterTracker_o2_v06_01.xml


def get_x_y_mask(X, layer_radii=(819, 1153, 1486), thickness=15):
    layer_radii = np.asarray(layer_radii)   # convert list → numpy array
    half_t = thickness / 2

    r = np.sqrt(X[:,1]**2 + X[:,2]**2)

    mask = np.any(
        np.abs(r[:, None] - layer_radii[None, :]) <= half_t,
        axis=1
    )
    return mask


m = get_x_y_mask(data)
print("Num. hits before mask:", len(m))
print("Num. hits after mask:", sum(m))


plt.figure(figsize = (5,5))
plt.hist2d(data[:,1], data[:,2], bins = [np.linspace(-1500,1500,500), np.linspace(-1500,1500,500)])
plt.show()


plt.figure(figsize = (5,5))
plt.hist2d(data[m][:,1], data[m][:,2], bins = [np.linspace(-1500,1500,500), np.linspace(-1500,1500,500)])
plt.show()


# %%
m = get_x_y_mask(X_samples["flow_samples"])
print("Num. hits before mask:", len(m))
print("Num. hits after mask:", sum(m))



plt.figure(figsize = (5,5))
plt.hist2d(X_samples["flow_samples"][:,1], X_samples["flow_samples"][:,2], bins = [np.linspace(-1500,1500,500), np.linspace(-1500,1500,500)])
plt.show()


plt.figure(figsize = (5,5))
plt.hist2d(X_samples["flow_samples"][m][:,1], X_samples["flow_samples"][m][:,2], bins = [np.linspace(-1500,1500,500), np.linspace(-1500,1500,500)])
plt.show()



# %%
print(ZUKO_ID, NAME)
plot_hists_1d({"data":data, **X_samples, "truncated":X_samples["flow_samples"][m]}, bins_dict, log_dims=log_vars, labels=feature_labels)
plt.savefig(f"{save_dir}/hists_final")


fig_samp, axes_samp = plot_corner_hist_2d(
        X_samples[key][m],
        feature_labels=feature_labels,
        bins_dict=bins_dict,
        log_dims=log_vars,
        title= key,
    )

# %%
