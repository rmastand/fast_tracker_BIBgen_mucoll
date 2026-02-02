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
#     display_name: Python (torchcfm)
#     language: python
#     name: torchcfm
# ---

# %%
import math
import os
import time
from tqdm import tqdm
from numba import cuda
import argparse
import pickle
import copy


import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE

from helpers.data_transforms import preprocess_data, inverse_preprocess_data
from helpers.models.DNN import NeuralNet, count_parameters
from helpers.sampling import get_cfm_samples
from helpers.plotting import plot_hists_1d, plot_corner_hist_2d


from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *

from torchcfm.utils import *
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("-n_epochs", "--N_EPOCHS", type=int, default=10)
parser.add_argument("-plot", "--PLOT_EPOCH_INTERVAL", type=int, default=1)
parser.add_argument("-bs", "--BATCH_SIZE", type=int, default=1024)
parser.add_argument("-n_samp", "--N_SAMPLE", type=int, default=50000)
parser.add_argument("-lr", "--LEARNING_RATE", type=float, default=1e-3)
parser.add_argument("-m", "--MODEL", type=str, default="CFM")
parser.add_argument("-train", "--TRAIN_MODEL", action="store_true")
parser.add_argument("-eval", "--EVAL_MODEL", action="store_true")
parser.add_argument("-p", "--TRAINED_MODEL_PATH", type=str)
args = parser.parse_args()


savedir = "models"
os.makedirs(savedir, exist_ok=True)


# %%
device = cuda.get_current_device()
device.reset()
torch.set_num_threads(2)
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

print( "Using device: " + str( device ), flush=True)

# %%
N_EPOCHS = args.N_EPOCHS
PLOT_EPOCH_INTERVAL = args.PLOT_EPOCH_INTERVAL
BATCH_SIZE = args.BATCH_SIZE
N_BINS = 200
BIN_BOUND = 4
N_SAMPLE = args.N_SAMPLE
LEARNING_RATE = args.LEARNING_RATE
log_vars = [0]

# %%
# load in data
collections_SimTrackerHit = [
 #   "InnerTrackerBarrelCollection",
   # "InnerTrackerBarrelCollectionConed",
   # "InnerTrackerEndcapCollection",
    #"InnerTrackerEndcapCollectionConed", 
    "OuterTrackerBarrelCollection",     
   # "OuterTrackerBarrelCollectionConed",
   # "OuterTrackerEndcapCollection",  
   # "OuterTrackerEndcapCollectionConed",
  #  "VertexBarrelCollection",   
   # "VertexBarrelCollectionConed",   
  #  "VertexEndcapCollection",
   # "VertexEndcapCollectionConed",
]


data = []
context = []


for i, collection in enumerate(collections_SimTrackerHit): # TODO coned too?

    tmp_data = np.load(f"/pscratch/sd/r/rmastand/muon/npys/{collection}_SimTrackerHit.npy")
    tmp_context = int(i)*np.ones((tmp_data.shape[0],1))
    context.append(tmp_context)
    data.append(tmp_data[:int(len(tmp_data)*1)])


data = np.vstack(data)[:,[0,2,3,4,5]]

bins_dict = {}

for i in range(data.shape[1]):
    if i in log_vars:
        bins_dict[i] = np.logspace(np.log10(0.9*np.min(data[:,i])), np.log10(1.1*np.max(data[:,i])), 100) 
    else:
        bins_dict[i] = np.linspace(np.min(data[:,i] - 10), np.max(data[:,i] + 10), 100) 

N_FEATURES = data.shape[1]
print("Data shape:", data.shape)


data = preprocess_data(data, ".")

from sklearn.model_selection import train_test_split
data_train, data_val = train_test_split(data, test_size=0.2, random_state=42)

#plot_data({"data": data})

# %% [markdown]
# ### Conditional Flow Matching
#
# First we implement the basic conditional flow matching. As in the paper, we have
# $$
# \begin{align}
# z &= (x_0, x_1) \\
# q(z) &= q(x_0)q(x_1) \\
# p_t(x | z) &= \mathcal{N}(x | t * x_1 + (1 - t) * x_0, \sigma^2) \\
# u_t(x | z) &= x_1 - x_0
# \end{align}
# $$
# When $\sigma = 0$ this is equivalent to zero-steps of rectified flow. We find that small $\sigma$ helps to regularize the problem ymmv.

# %%

# %% [markdown]
# ### Action Matching (Neklyudov et al. 2022)
#
# Next we try a variant called action matching. Here we parametrize the velocity field $v_\theta(t, x)$ as $\nabla s_\theta(t, x)$ where $s_\theta(t, x): \mathbb{R} \times \mathbb{R}^d \to \mathbb{R}$ is interpreted as the **action**. This is an interesting parameterization because of its link with optimal transport. Namely this velocity performs instantaneous optimal transport flow over $p_t(x)$. This is slightly different than optimal transport between marginals, but is also quite interesting. Action matching can be summarized in the following way:
# $$
# \begin{align}
# z &= (x_0, x_1) \\
# q(z) &= q(x_0)q(x_1) \\
# p_t(x | z) &= \mathcal{N}(x | (1-t) x_0 + t x_1, \sigma^2)\\
# L_{AM}(\theta) &= s_\theta(0, x_0) - s_\theta(1, x_1) + \frac{1}{2} \| \nabla_x s_\theta(t, x_t)\|^2 + \frac{\partial}{\partial t} s_\theta(t, x_t)
# \end{align}
# $$
# Note that the authors again consider $\sigma = 0$ (i.e. a Dirac around $\mu_t$) but we keep the general form for consistency assuming that $\mathcal{N}(x | \mu_t, 0)$ is a degenerate Dirac centered at $\mu_t$. Our standard parameterization seems to be more difficult to fit with this loss (3-layer MLP with width 64 and SELU activations). It's unclear to me why this is the case, but as suggested in their repo using ReLU, Swish, Swish activations works much better.


# %%


def run_cfm_model(MLP_base_model, FM, name, model_id, train_model, path_to_trained_model):

    assert model_id in ["cfm", "action"]

    if model_id == "cfm":
        model = MLP_base_model # MLP base model      
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        

    elif model_id == "action":
        action = MLP_base_model
        model = GradModel(action) 
        optimizer = torch.optim.Adam(action.parameters(), lr=LEARNING_RATE)

    model = model.to(device)
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    if train_model:
        print("Training model")
    
        losses_train, losses_val = [], []
        best_val_loss = 1e10
                
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers = 8, pin_memory = True, drop_last = True)
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, num_workers = 8, pin_memory = True, drop_last = True)
            
        for k in range(N_EPOCHS):
    
            epoch_losses_train, epoch_losses_val = [], []

            # TRAINING LOOP
            pbar = tqdm(train_loader, desc="Train batches", leave=False, )
            for x1 in pbar:
                optimizer.zero_grad()
        
                x0 = torch.normal(size=(BATCH_SIZE, N_FEATURES), mean=0.0, std=1.0).to(device)
                x1 = x1.to(device).float()
    
                if model_id == "cfm":
                    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                    vt = model(torch.cat([xt, t[:, None]], dim=-1))
                    loss = torch.mean((vt - ut) ** 2)
                elif model_id == "action":
                    t = torch.rand(BATCH_SIZE, 1).requires_grad_(True).to(device)
                    xt = (t * x1 + (1 - t) * x0).detach().requires_grad_(True)
                    st = torch.sum(action(torch.cat([xt, t], dim=-1)))
                    dsdx, dsdt = torch.autograd.grad(st, (xt, t), create_graph=True, retain_graph=True)
                    #xt.requires_grad, t.requires_grad = False, False
                    xt = xt.detach()
                    t = t.detach()
                    a0 = action(torch.cat([x0, torch.zeros(BATCH_SIZE, 1).to(device)], dim=-1))
                    a1 = action(torch.cat([x1, torch.ones(BATCH_SIZE, 1).to(device)], dim=-1))
                    loss = a0 - a1 + 0.5 * (dsdx**2).sum(1, keepdims=True) + dsdt
                    loss = loss.mean()
    
                epoch_losses_train.append(loss.item())
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.3e}, epoch {k}")
    
            losses_train.append(np.mean(epoch_losses_train))

            # VALIDATION LOOP
            pbar = tqdm(val_loader, desc="Val batches", leave=False, )
            for x1 in pbar:
        
                x0 = torch.normal(size=(BATCH_SIZE, N_FEATURES), mean=0.0, std=1.0).to(device)
                x1 = x1.to(device).float()
    
                if model_id == "cfm":
                    with torch.no_grad():
                        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                        vt = model(torch.cat([xt, t[:, None]], dim=-1))
                        loss = torch.mean((vt - ut) ** 2)
                elif model_id == "action":
                    t = torch.rand(BATCH_SIZE, 1).requires_grad_(True).to(device)
                    xt = (t * x1 + (1 - t) * x0).detach().requires_grad_(True)
                    st = torch.sum(action(torch.cat([xt, t], dim=-1)))
                    dsdx, dsdt = torch.autograd.grad(st, (xt, t), create_graph=True, retain_graph=True)
                    #xt.requires_grad, t.requires_grad = False, False
                    xt = xt.detach()
                    t = t.detach()
                    a0 = action(torch.cat([x0, torch.zeros(BATCH_SIZE, 1).to(device)], dim=-1))
                    a1 = action(torch.cat([x1, torch.ones(BATCH_SIZE, 1).to(device)], dim=-1))
                    loss = a0 - a1 + 0.5 * (dsdx**2).sum(1, keepdims=True) + dsdt
                    loss = loss.mean()
    
                epoch_losses_val.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.3e}, epoch {k}")
        
                losses_val.append(np.mean(epoch_losses_val))
    
                if losses_val[-1] < best_val_loss:
                    best_val_loss = losses_val[-1]
                    print("new best val loss", losses_val[-1])
                    
                    torch.save(model.state_dict(), f"{savedir}/{name}.pt")
    
            if (k + 1) % PLOT_EPOCH_INTERVAL == 0:
               
                plt.figure()
                plt.plot(losses_train, label="train")
                plt.plot(losses_val, label="val")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(f"plots/{name}_losses")
    
                if model_id == "cfm":
                    node = NeuralODE(
                        torch_wrapper(model),
                        solver="dopri5", 
                        sensitivity="adjoint", 
                        atol=1e-4, 
                        rtol=1e-4
                            )
                    samples = get_cfm_samples(node, N_FEATURES, N_SAMPLE, BATCH_SIZE, device=device)
                    
                elif model_id == "action":
                    action_cpu = copy.deepcopy(action).to("cpu")  # make sure parameters are on CPU
                    model_cpu = GradModel(action_cpu)
                    node = NeuralODE(
                        torch_wrapper(model_cpu),
                        solver="euler",
                        sensitivity="adjoint",  
                        atol=1e-4,
                        rtol=1e-4
                    )

                    samples = get_cfm_samples(node, N_FEATURES, N_SAMPLE, BATCH_SIZE, device="cpu")
                
    
                loc_data_dict = {"data": inverse_preprocess_data( data_val , "."),
                        "generated": inverse_preprocess_data( samples , ".")}
                plot_hists_1d(loc_data_dict, bins_dict)
                plt.savefig(f"plots/{name}_hists")
    
                for key in loc_data_dict.keys():
                    fig_samp, axes_samp = plot_corner_hist_2d(
                        loc_data_dict[key],
                        feature_labels=[f"feature {i}" for i in range(N_FEATURES)],
                        bins_dict=bins_dict,
                        log_dims=log_vars,
                        title= key,
                    )
                    plt.savefig(f"plots/{name}_corner_{key}")
    
        

    else:
        print("Evaluating model")
        
        if model_id == "cfm":
            model.load_state_dict(torch.load(path_to_trained_model, weights_only=True))
            model.eval()
            
            node = NeuralODE(
                torch_wrapper(model),
                solver="dopri5", 
                sensitivity="adjoint", 
                atol=1e-4, 
                rtol=1e-4
                    )
            
        elif model_id == "action":
            action.load_state_dict(torch.load(path_to_trained_model, weights_only=True))
            action.eval()
            action_cpu = copy.deepcopy(action).to("cpu")  # make sure parameters are on CPU
            model_cpu = GradModel(action_cpu)
            node_cpu = NeuralODE(
                torch_wrapper(model_cpu),
                solver="euler",
                sensitivity="adjoint",  
                atol=1e-4,
                rtol=1e-4
            )
            
        samples = get_cfm_samples(node, N_FEATURES, 10*N_SAMPLE, BATCH_SIZE, device=device)

        loc_data_dict = {"data": inverse_preprocess_data( data , "."),
                "generated": inverse_preprocess_data( samples , ".")}
        plot_hists_1d(loc_data_dict, bins_dict)
        plt.savefig(f"plots/{name}_hists_final")

        for key in loc_data_dict.keys():
            fig_samp, axes_samp = plot_corner_hist_2d(
                loc_data_dict[key],
                feature_labels=[f"feature {i}" for i in range(N_FEATURES)],
                bins_dict=bins_dict,
                log_dims=log_vars,
                title= key,
            )
            plt.savefig(f"plots/{name}_corner_{key}_final")



# %%
flow_model_dicts = {
    "ConditionalFlowMatcher": {
        "model": NeuralNet([256, 256, 1], N_FEATURES+1, activation=torch.nn.SELU()),
        "FM": ConditionalFlowMatcher(sigma=0.1),
        "name": "CFM_sigma01"
   },
    "ExactOptimalTransportConditionalFlowMatcher": {
        "model":  NeuralNet([256, 256, 1], N_FEATURES+1, activation=torch.nn.SELU()),
        "FM": ExactOptimalTransportConditionalFlowMatcher(sigma=0.1),
        "name": "EOTCFM_sigma01"
  },
    "SchrodingerBridgeConditionalFlowMatcher":{
        "model":  NeuralNet([256, 256, 1], N_FEATURES+1, activation=torch.nn.SELU()),
        "FM": SchrodingerBridgeConditionalFlowMatcher(sigma=0.5, ot_method="exact"),
        "name": "SBCFM_sigma05_exact"
    },
    "VariancePreservingConditionalFlowMatcher":{
       "model": NeuralNet([256, 256, 1], N_FEATURES+1, activation=torch.nn.SELU()),
          "FM": VariancePreservingConditionalFlowMatcher(sigma=0.1),
        "name": "VPCFM_sigma01"
    }
}


flow_model_action_dicts= {
    "MLP": {
        "action":   NeuralNet([256, 256, 1], N_FEATURES+1, activation=torch.nn.SELU()),
        "name": "action"
    }
}


if args.MODEL == "CFM":
    run_cfm_model(
        flow_model_dicts["ConditionalFlowMatcher"]["model"],
        flow_model_dicts["ConditionalFlowMatcher"]["FM"],
        flow_model_dicts["ConditionalFlowMatcher"]["name"],
        "cfm", 
        args.TRAIN_MODEL, 
        args.TRAINED_MODEL_PATH
    )
elif args.MODEL == "OTCFM":
    run_cfm_model(
        flow_model_dicts["ExactOptimalTransportConditionalFlowMatcher"]["model"],
        flow_model_dicts["ExactOptimalTransportConditionalFlowMatcher"]["FM"],
        flow_model_dicts["ExactOptimalTransportConditionalFlowMatcher"]["name"],
        "cfm", 
        args.TRAIN_MODEL, 
        args.TRAINED_MODEL_PATH
    )
elif args.MODEL == "SBCFM":
    run_cfm_model(
        flow_model_dicts["SchrodingerBridgeConditionalFlowMatcher"]["model"],
        flow_model_dicts["SchrodingerBridgeConditionalFlowMatcher"]["FM"],
        flow_model_dicts["SchrodingerBridgeConditionalFlowMatcher"]["name"],
        "cfm", 
        args.TRAIN_MODEL, 
        args.TRAINED_MODEL_PATH
    )
elif args.MODEL == "VPCFM":
    run_cfm_model(
        flow_model_dicts["VariancePreservingConditionalFlowMatcher"]["model"],
        flow_model_dicts["VariancePreservingConditionalFlowMatcher"]["FM"],
        flow_model_dicts["VariancePreservingConditionalFlowMatcher"]["name"],
        "cfm", 
        args.TRAIN_MODEL, 
        args.TRAINED_MODEL_PATH
    )
elif args.MODEL == "action":
    run_cfm_model(
        flow_model_action_dicts["MLP"]["action"],
        None,
        flow_model_action_dicts["MLP"]["name"],
        "action", 
        args.TRAIN_MODEL, 
        args.TRAINED_MODEL_PATH
    )
else:
    print("ERROR: unspecified flow model")
        
        
        
