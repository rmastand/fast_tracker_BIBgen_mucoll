import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import argparse
import wandb
import zuko
from helpers.models.DNN import count_parameters
from helpers.data_transforms import (
    preprocess_data,
    inverse_preprocess_data,
    load_in_data,
    inverse_geometry_transform,
    export_dataset
)
from helpers.evaluation import evaluate_samples
from helpers.flow import sample_from_flow
#plt.style.use("../science.mplstyle")

# %%
from helpers.plotting import plot_hists_1d, plot_corner_hist_2d

SCRIPT_DIR = Path(__file__).resolve().parent
TABDDPM_ROOT = SCRIPT_DIR / "diffusion" / "tabddpm_official"
TABDDPM_SCRIPTS = TABDDPM_ROOT / "scripts"

for path in (str(TABDDPM_ROOT), str(TABDDPM_SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

# shiyu I had to override your last commit


# TabDDPM official training and sampling entry points

# %%


BIN_BOUND = 5
NUM_BINS = 100

feature_indices_dict = {
    "InnerTrackerBarrelCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
    "InnerTrackerEndcapCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
    "OuterTrackerBarrelCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
    "OuterTrackerEndcapCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
    "VertexBarrelCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
    "VertexEndcapCollection": {"r": 2, "phi": 3, "z": 4, "side": 5, "layer": 6},
}

# setup
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL", choices=["flow", "tabddpm"],default="flow", help="Generative_model")
parser.add_argument("--ZUKO_ID", type=str, default="NSF", help="Zuko model ID")
parser.add_argument("--NAME", type=str, default="", help="Name")
parser.add_argument("--WORKING_DIR", default="/scratch/midway3/rmastand/muon_collider", type=str, help="Where to store model outputs and plots")
parser.add_argument("--WANDB_DIR", default="/scratch/midway3/rmastand/wandb", type=str, help="Where to store model outputs and plots")
parser.add_argument("--DATA_DIR", default="/scratch/midway3/rmastand/nuGun_pT_0_50", type=str, help="Where to store model outputs and plots")
parser.add_argument("--OVERSAMPLE", default=1, type=int)

# data + evaluation
parser.add_argument("--COLLECTION_LIST", type=str, default="OuterTrackerBarrelCollection")
parser.add_argument("--BASIS", choices=["xy", "rphi", "local_phi", "local_rphi"], default="rphi", help="Coordinate basis used for training")
parser.add_argument("--FEATURE_ORDER", default=None, help="Comma-separated list of feature indices to specify order. If None, uses default order.")
parser.add_argument("--TRAIN", action="store_true", help="Whether to train the flow")
parser.add_argument("--EVAL", action="store_true", help="Whether to evaluate the flow after training")
parser.add_argument("--NUM_BDTS", type=int, default=5, help="For sample evaluation")
parser.add_argument("--BDT_SUBSAMPLE_FRAC", type=float, default=0.25, help="Evaluation subsample fraction")
parser.add_argument("--SEED", type=int, default=8, help="Random seed")  # shiyu: do you have a random seed?

# flow-specific arguments
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


# TabDDPM-specific arguments
parser.add_argument("--STEPS", type=int, default=5000, help="Number of TabDDPM training steps")
parser.add_argument("--WEIGHT_DECAY", type=float, default=0.0, help="TabDDPM optimizer weight decay")
parser.add_argument("--NUM_TIMESTEPS", type=int, default=100, help="Number of diffusion timesteps")
parser.add_argument("--SAMPLE_BATCH_SIZE", type=int, default=4096, help="TabDDPM sampling batch size")
parser.add_argument("--SCHEDULER", type=str, default="cosine", help="Diffusion noise scheduler")
parser.add_argument("--D_LAYERS", type=str, default="1024,512,512,1024", help="TabDDPM MLP hidden layers")
parser.add_argument("--DIM_T", type=int, default=256, help="Diffusion timestep embedding dimension")
parser.add_argument("--NORMALIZATION", type=str, default="quantile", help="TabDDPM numerical normalization")
parser.add_argument("--Y_MODE", choices=["cond", "joint", "none"], default="cond", help="TabDDPM conditioning mode")

args = parser.parse_args()









# %%
if args.MODEL == "flow":
    save_dir = f"{args.WORKING_DIR}/zuko_outputs/{args.NAME}"
    import zuko
elif args.MODEL == "tabddpm":
    save_dir = f"{args.WORKING_DIR}/ddpm_outputs/{args.NAME}"
    from sample import sample as tabddpm_sample
    from train import train as tabddpm_train

os.makedirs(save_dir, exist_ok=True)
wandb_dir = f"{args.WANDB_DIR}"
os.makedirs(wandb_dir, exist_ok=True)

# Keep W&B and saved configs limited to parameters used by the selected model
flow_only_args = "ZUKO_ID OVERSAMPLE NUM_EPOCHS TRANSFORMS HIDDEN_FEATURES FREQS BINS DEGREE POLYNOMIALS PLOT_EPOCH_INTERVAL CHECKPOINT_EPOCH_INTERVAL TRAIN_FLOW EVAL_FLOW".split()
tabddpm_only_args = "STEPS WEIGHT_DECAY NUM_TIMESTEPS SAMPLE_BATCH_SIZE SCHEDULER D_LAYERS DIM_T NORMALIZATION Y_MODE TRAIN_TABDDPM EVAL_TABDDPM".split()
unused_args = tabddpm_only_args if args.MODEL == "flow" else flow_only_args
run_config = {key:value for key, value in vars(args).items() if key not in unused_args}

wandb.init(
    project="muon_collider",
    name=args.NAME,
    config=run_config,
    dir=wandb_dir
)

# computing
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)
seed = int(args.SEED)
torch.manual_seed(seed)
np.random.seed(seed)

collection_list = [x for x in args.COLLECTION_LIST.split(",")]
print(f"Running on collection {collection_list}")

log_vars = []

# %%
FEATURE_ORDER = None if args.FEATURE_ORDER is None else [int(x) for x in args.FEATURE_ORDER.split(",")]

# shiyu whose local phu transformation did you use?
X, feature_labels = load_in_data(collection_list, args.BASIS, args.DATA_DIR, args.TRAINING_FRAC, args.NUM_COND_INPUTS, feature_order=FEATURE_ORDER)

# Keep a global reference for evaluating the final global samples
# shiyu walk my though this and all of data_transforms
X_global = inverse_geometry_transform(X, args.BASIS, collection_list[0], FEATURE_ORDER)
global_feature_labels = [label.replace(" (local)", "") for label in feature_labels]

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


# make a common train-test split
X_train, X_val, train_indices, val_indices = train_test_split(X, np.arange(len(X)), test_size=0.2, random_state=42)

fig_samp, axes_samp = plot_corner_hist_2d(
        X_train,
        feature_labels=feature_labels,
        bins_dict=bins_dict,
        log_dims=log_vars,
        title= "data",
    )
plt.savefig(f"{save_dir}/data_final")
plt.close()


if args.MODEL == "tabddpm":

    # shiyu I don't understand this function
    def pack_condition_rows(condition, model):
        """Map each unique condition row to a TabDDPM class index."""
        condition = np.asarray(condition)

        if condition.ndim == 1:
            condition = condition[:, None]

        unique_rows, condition_ids = np.unique(
            condition,
            axis=0,
            return_inverse=True,
        )

        return condition_ids.astype(np.int64), unique_rows.astype(np.float32)

    if args.Y_MODE == "none":
        X_num = X.astype(np.float32)
        y_values = np.zeros(len(X_num), dtype=np.int64)
        y_lookup = None
    else:
        X_num = X[:, :NUM_FEATURES].astype(np.float32)
        y_values, y_lookup = pack_condition_rows(X[:, NUM_FEATURES:])
        np.save(Path(save_dir) / "y_lookup.npy", y_lookup)

    is_y_cond = args.Y_MODE == "cond"

    # Configure the official TabDDPM model and data transformations
    model_params = {
        "num_classes": int(n_classes),
        "is_y_cond": bool(is_y_cond),
        "rtdl_params": {
            "d_layers": [
                int(value)
                for value in args.D_LAYERS.split(",")
                if value.strip()
            ],
            "dropout": 0.0,
        },
        "dim_t": int(args.DIM_T),
    }

    T_dict = {
        "seed": int(seed),
        "normalization": args.NORMALIZATION,
        "num_nan_policy": None,
        "cat_nan_policy": None,
        "cat_min_frequency": None,
        "cat_encoding": None,
        "y_policy": "default",


    }

elif args.MODEL == "flow":
    X_num = X[:, :NUM_FEATURES].astype(np.float32)
    y_values = X[:, NUM_FEATURES:]

n_classes = export_dataset(
    dataset_dir,
    X_num,
    y_values,
    train_indices,
    val_indices,
)

# Save the resolved run configuration before training
run_config["feature_labels"] = feature_labels
run_config["num_features"] = int(NUM_FEATURES)

if args.MODEL == "tabddpm":
    run_config["n_classes"] = int(n_classes)
    run_config["is_y_cond"] = bool(is_y_cond)

with open(
    Path(save_dir) / "run_config.json",
    "w",
    encoding="utf-8",
) as output_file:
    json.dump(run_config, output_file, indent=2)


if args.TRAIN:
    # Train TabDDPM with the official implementation
    if args.MODEL == "tabddpm":
        tabddpm_train(
            parent_dir=str(save_dir),
            real_data_path=str(dataset_dir),
            steps=args.STEPS,
            lr=args.LEARNING_RATE,
            weight_decay=args.WEIGHT_DECAY,
            batch_size=args.BATCH_SIZE,
            model_type="mlp",
            model_params=model_params,
            num_timesteps=args.NUM_TIMESTEPS,
            gaussian_loss_type="mse",
            scheduler=args.SCHEDULER,
            T_dict=T_dict,
            num_numerical_features=X_num.shape[1],
            device=device,
            seed=args.SEED,
            change_val=False,
        )

        loss_history = pd.read_csv(Path(save_dir) / "loss.csv")

        for row in loss_history.itertuples(index=False):
            wandb.log({
                "training_step": int(row.step),
                "train/mloss": float(row.mloss),
                "train/gloss": float(row.gloss),
                "train/total": float(row.loss),
            })

    elif args.MODEL == "flow":

        # preprocess the flow data

        X_train = preprocess_data(X_train, save_dir, args.ZUKO_ID, args.NUM_COND_INPUTS, scaler_exists=False)
        X_val = preprocess_data(X_val, save_dir, args.ZUKO_ID, args.NUM_COND_INPUTS, scaler_exists=True)

        print(f"Train data has shape {X_train.shape}.")
        print(f"Val data has shape {X_val.shape}.")

        train_loader = torch.utils.data.DataLoader(X_train, batch_size=args.BATCH_SIZE, shuffle=True, num_workers = 0, pin_memory = True)
        val_loader = torch.utils.data.DataLoader(X_val, batch_size=args.BATCH_SIZE, shuffle=False, num_workers = 0, pin_memory = True)

        hidden_features = [int(x) for x in args.HIDDEN_FEATURES.split(",")]

        # choose flow model

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


        # Train to maximize the log-likelihood
        optimizer = torch.optim.Adam(flow.parameters(), lr=args.LEARNING_RATE)

        # %%
        losses_train, losses_val = [], []
        best_val_loss = 1e10
        start_epoch = 0
      
        global_step = 0

        for k in range(start_epoch, args.NUM_EPOCHS):

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


if args.EVAL:

    if args.MODEL == "tabddpm":

        tabddpm_sample(
            parent_dir=str(save_dir),
            real_data_path=str(dataset_dir),
            batch_size=args.SAMPLE_BATCH_SIZE,
            num_samples=len(X_num),
            model_type="mlp",
            model_params=model_params,
            model_path=str(Path(save_dir) / "model.pt"),
            num_timesteps=args.NUM_TIMESTEPS,
            gaussian_loss_type="mse",
            scheduler=args.SCHEDULER,
            T_dict=T_dict,
            num_numerical_features=X_num.shape[1],
            disbalance=None,
            device=device,
            seed=args.SEED,
            change_val=False,
        )

        X_generated = np.load(Path(save_dir) / "X_num_train.npy").astype(np.float32)
        y_generated = np.load(Path(save_dir) / "y_train.npy").astype(np.int64)

        if args.Y_MODE == "none":
            samples = X_generated
        else:
            condition_generated = y_lookup[y_generated]
            samples = np.concatenate([X_generated, condition_generated], axis=1)

        samples_global = inverse_geometry_transform(samples, args.BASIS, collection_list[0], FEATURE_ORDER)
        np.save(Path(save_dir) / "tabddpm_samples.npy", samples_global)

        evaluate_samples(samples, samples_global)

    elif args.MODEL == "flow":


        eval_flow = flow

        eval_flow.load_state_dict(torch.load(f"{save_dir}/test.pt"))

        num_samples_total = X_train.shape[0] 
        sample_batch_size = 8192

        samples = []
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
            
            
            samples.append(loc_samples)
        samples = np.concatenate(samples)
    

        
        samples = inverse_preprocess_data( samples , save_dir, args.ZUKO_ID, args.NUM_COND_INPUTS)
        samples_global = inverse_geometry_transform(samples, args.BASIS, collection_list[0], FEATURE_ORDER)
        np.save(f"{save_dir}/flow_samples.npy", samples_global)

        evaluate_samples(samples, samples_global)


        # # %%
    #
    # plot_hists_1d({"data":X, "generated":samples}, bins_dict, log_dims=log_vars, labels=feature_labels)
    # plt.savefig(f"{save_dir}/hists_final")
    # plt.close()
    #
    #
    # # %%
    # fig_samp, axes_samp = plot_corner_hist_2d(
    #     samples,
    #     feature_labels=feature_labels,
    #     bins_dict=bins_dict,
    #     log_dims=log_vars,
    #     title= "generated",
    # )
    # plt.savefig(f"{save_dir}/corner_generated_final")
    # plt.close()
    #
    #
    #
    # # %% [markdown]
    # # # Train a BDT to discriminate flow from samples
    #
    # # %%
    #
    # with open(f"{save_dir}/results.txt", "w") as ofile:
    #     ks_dists_samples = get_kl_dist(X, samples)
    #     ks_dists_gaussians = get_kl_dist(np.random.normal(size = X.shape), np.random.normal(size =  samples.shape))
    #
    #
    #     for i, ks_dist in enumerate(ks_dists_samples):
    #         ofile.write("Feature {i} KL div: {ks_dist} (for gaussian: {ks_gauss})".format(i=i, ks_dist=ks_dist, ks_gauss=ks_dists_gaussians[i]))
    #         ofile.write("\n")
    #
    #     auc_mean, auc_std, best_epoch_list, max_epochs, _, _, _ = discriminate_data_from_samples(
    #         X,
    #         samples,
    #         args.NUM_BDTS,
    #         "configs/bdt.yml",
    #         model_type="bdt",
    #         plot_losses=True,
    #         device=device,
    #         val_size=0.2,
    #         subsample_frac=0.25,  # optional subsample for large datasets,
    #         plot_dir=save_dir
    #     )
    #
    #     ofile.write(f"auc {auc_mean} pm {auc_std}. best epoch {best_epoch_list} of {max_epochs}.\n")
    #
    # wandb.log({
    #     "auc_mean": auc_mean,
    #     "auc_std": auc_std,
    #     "bdt_best_epoch": np.mean(best_epoch_list)
    #         })
    #
    # wandb.run.summary["auc_mean"] = auc_mean
    # wandb.run.summary["auc_std"] = auc_std

   


wandb.finish()
