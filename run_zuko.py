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
#from numba import cuda
import wandb
import zuko
from helpers.models.DNN import count_parameters
from helpers.data_transforms import (
    preprocess_data,
    inverse_preprocess_data,
    inverse_preprocess_data_torch,
    load_in_data,
)
from helpers.evaluation import get_kl_dist, discriminate_data_from_samples
from helpers.flow import sample_from_flow
plt.style.use("../science.mplstyle")

# %%
from helpers.plotting import plot_hists_1d, plot_corner_hist_2d

# %%


BIN_BOUND = 5
NUM_BINS = 100
NUM_FEATURES = 5

feature_indices_dict = {
    "InnerTrackerBarrelCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
    "InnerTrackerEndcapCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
    "OuterTrackerBarrelCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
    "OuterTrackerEndcapCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
    "VertexBarrelCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
    "VertexEndcapCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
}



parser = argparse.ArgumentParser()
parser.add_argument("--ZUKO_ID", type=str, default="NSF", help="Zuko model ID")
parser.add_argument("--NAME", type=str, default="", help="Name")
parser.add_argument("--COLLECTION_LIST", type=str, default="OuterTrackerBarrelCollection")
parser.add_argument("--WORKING_DIR", default="/pscratch/sd/r/rmastand/muon_collider", type=str, help="Where to store model outputs and plots")
parser.add_argument("--FEATURES", default="xy")
parser.add_argument("--FEATURE_ORDER", default=None, help="Comma-separated list of feature indices to specify order. If None, uses default order.")
parser.add_argument("--OVERSAMPLE", default=1, type=int)


parser.add_argument("--SEED", type=int, default=8, help="Random seed")
parser.add_argument("--NUM_EPOCHS", type=int, default=5, help="Number of training epochs")
parser.add_argument("--LEARNING_RATE", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--TRAINING_FRAC", type=float, default=1, help="How much training data to use")
parser.add_argument("--BATCH_SIZE", type=int, default=512, help="Batch size")
parser.add_argument("--NUM_COND_INPUTS", type=int, default=0, help="Number of conditional inputs")
parser.add_argument("--TRANSFORMS", type=int, default=3, help="Number of transforms ")
parser.add_argument("--HIDDEN_FEATURES", type=str, default="32,32,32", help="Number of hidden features")
parser.add_argument("--FREQS", type=int, default=3, help="Freqs for CNF")
parser.add_argument("--BINS", type=int, default=16, help="Freqs for CNF")
parser.add_argument("--DEGREE", type=int, default=3, help="Freqs for CNF")
parser.add_argument("--POLYNOMIALS", type=int, default=4, help="Freqs for CNF")


parser.add_argument("--PLOT_EPOCH_INTERVAL", type=int, default=100, help="Interval for plotting during training")
parser.add_argument("--TRAIN_FLOW", action="store_true", help="Whether to train the flow")
parser.add_argument("--EVAL_FLOW", action="store_true", help="Whether to evaluate the flow after training")
parser.add_argument("--PHI_LOCAL", action="store_true", help="Whether to evaluate the flow after training")

parser.add_argument("--NUM_BDTS", type=int, default=5, help="For sample evaluation")


args = parser.parse_args()


# %%
save_dir = f"{args.WORKING_DIR}/zuko_outputs/{args.ZUKO_ID}/{args.NAME}"
os.makedirs(save_dir, exist_ok=True)
wandb.init(
    project="zuko-flows",          # change if you want
    name=f"{args.ZUKO_ID}_{args.NAME}",
    config=vars(args),             # logs all argparse params
    dir=save_dir
)


# %%

# computing
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)
seed = int(args.SEED)
torch.manual_seed(seed)
np.random.seed(seed)

collection_list = [x for x in args.COLLECTION_LIST.split(",")]
print(collection_list)

log_vars = []

# %%
FEATURE_ORDER = None if args.FEATURE_ORDER is None else [int(x) for x in args.FEATURE_ORDER.split(",")]
X, feature_labels = load_in_data(collection_list, args.FEATURES, args.WORKING_DIR, args.TRAINING_FRAC, args.NUM_COND_INPUTS, feature_order=FEATURE_ORDER, use_local_phi=args.PHI_LOCAL)
print(f"Data has shape {X.shape}")
print("Feature labels:", feature_labels)
NUM_FEATURES = X.shape[1] - args.NUM_COND_INPUTS


bins_dict = {}
bins_dict_preproc = {i:np.linspace(-BIN_BOUND, BIN_BOUND, NUM_BINS) for i in range(X.shape[1])}

for i in range(X.shape[1]):
    if i in log_vars:
        bins_dict[i] = np.logspace(np.log10(0.9*np.min(X[:,i])), np.log10(1.1*np.max(X[:,i])), NUM_BINS) 
    else:
        bins_dict[i] = np.linspace(np.min(X[:,i] - 1), np.max(X[:,i] + 1), NUM_BINS) 


fig_samp, axes_samp = plot_corner_hist_2d(
        X,
        feature_labels=feature_labels,
        bins_dict=bins_dict,
        log_dims=log_vars,
        title= "data",
    )
plt.savefig(f"{save_dir}/data_final")
plt.close()

# %%



# plot_hists_1d({"data":data}, bins_dict, log_dims=log_vars, labels=feature_labels)
# plt.show()

# plot_hists_1d({"data":X_preproc}, bins_dict_preproc, log_dims=[], labels = feature_labels)
# plt.show()





# %%
# train val split
from sklearn.model_selection import train_test_split


X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

X_train = preprocess_data(X_train, save_dir, args.ZUKO_ID, args.NUM_COND_INPUTS, scaler_exists=False)
X_val = preprocess_data(X_val, save_dir, args.ZUKO_ID, args.NUM_COND_INPUTS, scaler_exists=True)



print(f"Train data has shape {X_train.shape}.")
print(f"Val data has shape {X_val.shape}.")

train_loader = torch.utils.data.DataLoader(X_train, batch_size=args.BATCH_SIZE, shuffle=True, num_workers = 0, pin_memory = True)
val_loader = torch.utils.data.DataLoader(X_val, batch_size=args.BATCH_SIZE, shuffle=False, num_workers = 0, pin_memory = True)

# %%
import zuko


hidden_features = [int(x) for x in args.HIDDEN_FEATURES.split(",")]

if args.ZUKO_ID == "NSF":
    flow = zuko.flows.NSF(NUM_FEATURES, args.NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features).to(device)
#elif args.ZUKO_ID == "GMM":
#    flow = zuko.flows.GMM(NUM_FEATURES, args.NUM_COND_INPUTS, components=30, hidden_features=[256] * 5).to(device)
#elif args.ZUKO_ID == "NICE":
#    flow = zuko.flows.NICE(NUM_FEATURES, args.NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features).to(device)
elif args.ZUKO_ID == "MAF":
   flow = zuko.flows.MAF(NUM_FEATURES, args.NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features).to(device)
elif args.ZUKO_ID == "NCSF":
    flow = zuko.flows.NCSF(NUM_FEATURES, args.NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features, bins=args.BINS).to(device)
elif args.ZUKO_ID == "SOSPF":
    flow = zuko.flows.SOSPF(NUM_FEATURES, args.NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features, degree=args.DEGREE, polynomials=args.POLYNOMIALS).to(device)
#elif args.ZUKO_ID == "NAF":
#    flow = zuko.flows.NAF(NUM_FEATURES, args.NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features).to(device)
elif args.ZUKO_ID == "UNAF":
    flow = zuko.flows.UNAF(NUM_FEATURES, args.NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features).to(device)
elif args.ZUKO_ID == "CNF":
    flow = zuko.flows.CNF(NUM_FEATURES, args.NUM_COND_INPUTS, hidden_features=hidden_features, freqs=args.FREQS).to(device)
#elif args.ZUKO_ID == "GF":
#    flow = zuko.flows.GF(NUM_FEATURES, args.NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features, components=8).to(device)
#elif args.ZUKO_ID == "BPF":
#    flow = zuko.flows.BPF(NUM_FEATURES, args.NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features, degree=16).to(device)
else:
    print("ERROR: Unknown ZUKO_ID")
    exit()


num_params = count_parameters(flow)
print(f"Number of trainable parameters: {num_params}")


wandb.log({"num_trainable_params": num_params})
wandb.run.summary["num_trainable_params"] = num_params


def run_training_step(data_loader, epoch, global_step, is_val_step=False):
    flow.eval() if is_val_step else flow.train()

    text_desc = "val" if is_val_step else "train"
    losses_ll, losses_mmap, losses_total = [], [], []
    bad_fracs = []

    pbar = tqdm(data_loader, desc=f"{text_desc} batches", leave=False)

    for x in pbar:
        if not is_val_step:
            optimizer.zero_grad()

        x = x.to(device).float()

        if args.NUM_COND_INPUTS > 0:
            x_data = x[:, :-args.NUM_COND_INPUTS]
            x_context = x[:, -args.NUM_COND_INPUTS:]
            dist = flow(x_context)
            #loss_ll = -dist.log_prob(x_data).mean()
        else:
            x_data = x
            x_context = None
            dist = flow()
            #loss_ll = -dist.log_prob(x_data).mean()


        # -----------------------------
        # Log likelihood loss
        # -----------------------------
        log_prob = dist.log_prob(x_data)
        loss_ll = -log_prob.mean()



        total_loss = loss_ll

        if not is_val_step:
  
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()

            global_step += 1


        losses_ll.append(loss_ll.item())
        losses_total.append(total_loss.item())

        pbar.set_postfix(loss=f"{total_loss.item():.3e}")

    metrics = {
        f"{text_desc}/ll": np.mean(losses_ll),
        f"{text_desc}/total": np.mean(losses_total),
    }

    return metrics, global_step



if args.TRAIN_FLOW:
    print("Training flow...")

    # Train to maximize the log-likelihood
    optimizer = torch.optim.Adam(flow.parameters(), lr=args.LEARNING_RATE)

    # %%
    losses_train, losses_val = [], []
    best_val_loss = 1e10

    global_step = 0

    for k in range(args.NUM_EPOCHS):

        epoch_losses_train_ll, epoch_losses_val_ll = [], []
        epoch_losses_train_mask, epoch_losses_val_mask = [], []
        epoch_losses_train_total, epoch_losses_val_total = [], []

        train_losses, global_step = run_training_step(train_loader, k, global_step, is_val_step=False)

        with torch.no_grad():
            val_losses, _ = run_training_step(val_loader, k, global_step, is_val_step=True)

        wandb.log(
            {
                
                "epoch": k,
                **train_losses,
                **val_losses,
            },
)
        
        losses_train.append(train_losses["train/total"])
        losses_val.append(val_losses["val/total"])
        
        if val_losses["val/total"] < best_val_loss:
            best_val_loss = val_losses["val/total"]
            torch.save(flow.state_dict(), f"{save_dir}/test.pt")

        if (k + 1) % args.PLOT_EPOCH_INTERVAL == 0:
            flow.eval()
        
            x_plot = next(iter(val_loader)).to(device).float()
        
            if args.NUM_COND_INPUTS > 0:
                x_plot_data = x_plot[:, :-args.NUM_COND_INPUTS]
                x_plot_context = x_plot[:, -args.NUM_COND_INPUTS:]
                factor = 1
                context_to_sample = x_plot_context.repeat_interleave(factor, dim=0)
                samples = sample_from_flow(flow, N=factor * len(x_plot_data), x_context=context_to_sample)
            else:
                factor = 1
                samples = sample_from_flow(flow, N=factor * len(x_plot))
        
            loc_data_dict = {
                "data": inverse_preprocess_data(
                    x_plot.detach().cpu().numpy(),
                    save_dir,
                    args.ZUKO_ID,
                    args.NUM_COND_INPUTS,
                ),
               "generated": inverse_preprocess_data(
                    samples,
                    save_dir,
                    args.ZUKO_ID,
                    args.NUM_COND_INPUTS,
                ),
            }
            plot_hists_1d(loc_data_dict, bins_dict, log_dims=log_vars, labels=feature_labels)
            plt.savefig(f"{save_dir}/hists")
            plt.close()

            # for key in loc_data_dict.keys():
            #     fig_samp, axes_samp = plot_corner_hist_2d(
            #         loc_data_dict[key],
            #         feature_labels=feature_labels,
            #         bins_dict=bins_dict,
            #         log_dims=log_vars,
            #         title= key,
            #     )
            #     plt.savefig(f"{save_dir}/corner_{key}")
            #     plt.close()



# %%
if args.EVAL_FLOW:

    print("Making flow samples...")

    eval_flow = flow

    eval_flow.load_state_dict(torch.load(f"{save_dir}/test.pt"))

    num_samples_total = X_train.shape[0] 
    sample_batch_size = 8192

    samples_flow = []
    for i in tqdm(range(0, num_samples_total, sample_batch_size)):
        if i + sample_batch_size > num_samples_total:
            nn = num_samples_total - i
        else:
            nn = sample_batch_size


        if args.NUM_COND_INPUTS > 0:
            context_to_sample = torch.tensor(
                X_train[i:i+nn, -args.NUM_COND_INPUTS:], dtype=torch.float32
            ).to(device)


        loc_samples = sample_from_flow(eval_flow, N=args.OVERSAMPLE, x_context=context_to_sample if args.NUM_COND_INPUTS > 0 else None)


        samples_flow.append(loc_samples)
    samples_flow = np.concatenate(samples_flow)
  

    
    samples_flow = inverse_preprocess_data( samples_flow , save_dir, args.ZUKO_ID, args.NUM_COND_INPUTS)
    np.save(f"{save_dir}/flow_samples.npy", samples_flow)


    # %%

    plot_hists_1d({"data":X, "generated":samples_flow}, bins_dict, log_dims=log_vars, labels=feature_labels)
    plt.savefig(f"{save_dir}/hists_final")
    plt.close()


    # %%
    fig_samp, axes_samp = plot_corner_hist_2d(
        samples_flow,
        feature_labels=feature_labels,
        bins_dict=bins_dict,
        log_dims=log_vars,
        title= "generated",
    )
    plt.savefig(f"{save_dir}/corner_generated_final")
    plt.close()



    # %% [markdown]
    # # Train a BDT to discriminate flow from samples

    # %%

    with open(f"{save_dir}/results.txt", "w") as ofile:
        ks_dists_samples = get_kl_dist(X, samples_flow)
        ks_dists_gaussians = get_kl_dist(np.random.normal(size = X.shape), np.random.normal(size =  samples_flow.shape))
        
                                                
        for i, ks_dist in enumerate(ks_dists_samples):
            ofile.write("Feature {i} KL div: {ks_dist} (for gaussian: {ks_gauss})".format(i=i, ks_dist=ks_dist, ks_gauss=ks_dists_gaussians[i]))
            ofile.write("\n")
        
        auc_mean, auc_std, best_epoch_list, max_epochs, _, _, _ = discriminate_data_from_samples(
            X,
            samples_flow,
            args.NUM_BDTS,
            "configs/bdt.yml",
            model_type="bdt",
            plot_losses=True,
            device=device,
            val_size=0.2,
            subsample_frac=0.25,  # optional subsample for large datasets,
            plot_dir=save_dir
        )
        
        ofile.write(f"auc {auc_mean} pm {auc_std}. best epoch {best_epoch_list} of {max_epochs}.\n")

    wandb.log({
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "bdt_best_epoch": np.mean(best_epoch_list)
            })
        
    wandb.run.summary["auc_mean"] = auc_mean
    wandb.run.summary["auc_std"] = auc_std


wandb.finish()
    # %%





