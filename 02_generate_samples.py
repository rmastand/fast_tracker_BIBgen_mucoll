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
import yaml
from helpers.models.DNN import count_parameters
from helpers.data_transforms import (
    preprocess_data,
    inverse_preprocess_data,
    load_in_data,
    inverse_geometry_transform,
    export_dataset,
    pack_condition_rows
)
from helpers.evaluation import evaluate_samples
from helpers.flow import run_training_step, sample_from_flow, build_xy_z_lookup, snap_z_to_detector_xy
from helpers.material_map import apply_material_map_hybrid
plt.style.use("../science.mplstyle")

# %%
from helpers.plotting import plot_hists_1d, plot_corner_hist_2d

SCRIPT_DIR = Path(__file__).resolve().parent
TABDDPM_ROOT = SCRIPT_DIR / "diffusion" / "tabddpm_official"
TABDDPM_SCRIPTS = TABDDPM_ROOT / "scripts"

for path in (str(TABDDPM_ROOT), str(TABDDPM_SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

# setup
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL", choices=["flow", "tabddpm"],default="flow", help="Generative_model")
parser.add_argument("--ZUKO_ID", type=str, default="NCSF", help="Zuko model ID")
parser.add_argument("--NAME", type=str, default="", help="Name")
parser.add_argument("--OVERSAMPLE", default=1, type=int, help="Number of generated samples per reference row")
parser.add_argument("--CONFIG_PATH", default="configs", type=str, help="Path to the configuration file")



# data + evaluation
parser.add_argument("--COLLECTION_LIST", type=str, default="OuterTrackerBarrelCollection")
parser.add_argument("--BASIS", choices=["xy", "rphi", "local_phi"], default="rphi", help="Coordinate basis used for training")
parser.add_argument("--TRAIN", action="store_true", help="Whether to train the flow")
parser.add_argument("--EVAL", action="store_true", help="Whether to evaluate the flow after training")
parser.add_argument("--NUM_BDTS", type=int, default=1, help="For sample evaluation")
parser.add_argument("--BDT_SUBSAMPLE_FRAC", type=float, default=1.0, help="Evaluation subsample fraction")
parser.add_argument("--SEED", type=int, default=8, help="Random seed") 
parser.add_argument("--TRAINING_FRAC", type=float, default=1, help="How much training data to use")


# flow-specific arguments
parser.add_argument("--NUM_EPOCHS", type=int, default=5, help="Number of training epochs")
# Keep the Flow defaults; for TabDDPM, use --LEARNING_RATE 5e-4 --BATCH_SIZE 4096.
parser.add_argument("--LEARNING_RATE", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--BATCH_SIZE", type=int, default=512, help="Batch size")
parser.add_argument("--TRANSFORMS", type=int, default=3, help="Number of transforms ")
parser.add_argument("--HIDDEN_FEATURES", type=str, default="32,32,32", help="Number of hidden features")
parser.add_argument("--FREQS", type=int, default=3, help="Freqs for CNF")
parser.add_argument("--BINS", type=int, default=16, help="Freqs for CNF")
parser.add_argument("--DEGREE", type=int, default=3, help="Freqs for CNF")
parser.add_argument("--POLYNOMIALS", type=int, default=4, help="Freqs for CNF")
parser.add_argument("--PLOT_EPOCH_INTERVAL", type=int, default=50, help="Interval for plotting during training")


# TabDDPM-specific arguments
parser.add_argument("--STEPS", type=int, default=300000, help="Number of TabDDPM training steps")
parser.add_argument("--WEIGHT_DECAY", type=float, default=0.0, help="TabDDPM optimizer weight decay")
parser.add_argument("--NUM_TIMESTEPS", type=int, default=1000, help="Number of diffusion timesteps")
parser.add_argument("--SAMPLE_BATCH_SIZE", type=int, default=4096*2, help="TabDDPM sampling batch size")
parser.add_argument("--SCHEDULER", type=str, default="cosine", help="Diffusion noise scheduler")
parser.add_argument("--D_LAYERS", type=str, default="4096,4096,4096,4096,4096,4096", help="TabDDPM MLP hidden layers")
parser.add_argument("--DIM_T", type=int, default=2048, help="Diffusion timestep embedding dimension")
parser.add_argument("--NORMALIZATION", type=str, default="quantile", help="TabDDPM numerical normalization")
parser.add_argument("--Y_MODE", choices=["cond", "joint", "none"], default="cond", help="TabDDPM conditioning mode")

args = parser.parse_args()





with open(f"{args.CONFIG_PATH}.yaml", "r") as f:
    configs = yaml.safe_load(f)

BIN_BOUND = configs["BIN_BOUND"]
NUM_BINS = configs["NUM_BINS"]
FEATURE_ORDER = configs["FEATURE_ORDER"]
SAVE_DIR = configs["PATH_TO_OUTPUT_DIR"]
WANDB_DIR = configs["PATH_TO_WANDB_DIR"]
NUM_COND_INPUTS = configs["NUM_COND_INPUTS"]
FEATURE_INDICES_DICT = configs["FEATURE_INDICES_DICT"]



# %%
if args.MODEL == "flow":
    save_dir = f"{SAVE_DIR}/zuko_outputs/{args.NAME}"
    import zuko
elif args.MODEL == "tabddpm":
    save_dir = f"{SAVE_DIR}/ddpm_outputs/{args.NAME}"
    # TabDDPM official training and sampling entry points
    from sample import sample as tabddpm_sample
    from train import train as tabddpm_train

dataset_dir = Path(save_dir) / "dataset"
os.makedirs(save_dir, exist_ok=True)
wandb_dir = f"{WANDB_DIR}"
os.makedirs(wandb_dir, exist_ok=True)
plots_dir = f"{save_dir}/plots"
os.makedirs(plots_dir, exist_ok=True)

# Keep W&B and saved configs limited to parameters used by the selected model
flow_only_args = "ZUKO_ID NUM_EPOCHS TRANSFORMS HIDDEN_FEATURES FREQS BINS DEGREE POLYNOMIALS PLOT_EPOCH_INTERVAL CHECKPOINT_EPOCH_INTERVAL TRAIN_FLOW EVAL_FLOW".split()
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

X, feature_labels = load_in_data(collection_list, args.BASIS, configs["PATH_TO_DATA_DIR"], args.TRAINING_FRAC, NUM_COND_INPUTS, feature_order=FEATURE_ORDER)


# Keep a global reference for evaluating the final global samples
X_global = inverse_geometry_transform(X, args.BASIS, collection_list[0], FEATURE_ORDER)
global_feature_labels = [label.replace(" (local)", "") for label in feature_labels]

print(f"Data has shape {X.shape}")
print("Feature labels:", feature_labels)
NUM_FEATURES = X.shape[1] - NUM_COND_INPUTS

bins_dict = {}
bins_dict_preproc = {i:np.linspace(-BIN_BOUND, BIN_BOUND, NUM_BINS) for i in range(X.shape[1])}

for i in range(X.shape[1]):
    if i in log_vars:
        bins_dict[i] = np.logspace(np.log10(0.9*np.min(X[:,i])), np.log10(1.1*np.max(X[:,i])), NUM_BINS) 
    else:
        bins_dict[i] = np.linspace(np.min(X[:,i] - 1), np.max(X[:,i] + 1), NUM_BINS) 

# Pack condition rows once for the shared stratified split and TabDDPM class labels.
condition_ids = None
condition_lookup = None
if NUM_COND_INPUTS > 0:
    condition_ids, condition_lookup = pack_condition_rows(X[:, NUM_FEATURES:])

X_train, X_val, train_indices, val_indices = train_test_split(X, np.arange(len(X)), test_size=0.2, random_state=42, stratify=condition_ids)
fig_samp, axes_samp = plot_corner_hist_2d(
        X_train,
        feature_labels=feature_labels,
        bins_dict=bins_dict,
        log_dims=log_vars,
        title= "data",
    )
plt.savefig(f"{plots_dir}/data_final")
plt.close()

if args.MODEL == "tabddpm":
    if args.Y_MODE == "none": # no conditioning, just train on the features
        X_values = X.astype(np.float32)
        y_values = np.zeros(len(X_values), dtype=np.int64)
        y_lookup = None
    else: # conditioning on the features, so pack the conditioning features into a single integer label
        X_values = X[:, :NUM_FEATURES].astype(np.float32)
        y_values = condition_ids
        y_lookup = condition_lookup
        np.save(Path(save_dir) / "y_lookup.npy", y_lookup)
    is_y_cond = args.Y_MODE == "cond"


elif args.MODEL == "flow":
    X_values = X
    y_values = np.zeros((len(X_values), 1)) # conditioning features are stored within X for the flow


n_classes = export_dataset(
    dataset_dir,
    X_values,
    y_values,
    train_indices,
    val_indices,
)

if args.MODEL == "tabddpm":


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
    

    hidden_features = [int(x) for x in args.HIDDEN_FEATURES.split(",")]

    # choose flow model
    if args.ZUKO_ID == "NSF":
        flow = zuko.flows.NSF(NUM_FEATURES, NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features, bins=args.BINS).to(device)
    elif args.ZUKO_ID == "MAF":
        flow = zuko.flows.MAF(NUM_FEATURES, NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features).to(device)
    elif args.ZUKO_ID == "NCSF":
        flow = zuko.flows.NCSF(NUM_FEATURES, NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features, bins=args.BINS).to(device)
    elif args.ZUKO_ID == "SOSPF":
        flow = zuko.flows.SOSPF(NUM_FEATURES, NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features, degree=args.DEGREE, polynomials=args.POLYNOMIALS).to(device)
    elif args.ZUKO_ID == "UNAF":
        flow = zuko.flows.UNAF(NUM_FEATURES, NUM_COND_INPUTS, transforms=args.TRANSFORMS, hidden_features=hidden_features).to(device)
    elif args.ZUKO_ID == "CNF":
        flow = zuko.flows.CNF(NUM_FEATURES, NUM_COND_INPUTS, hidden_features=hidden_features, freqs=args.FREQS).to(device)
    else:
        print("ERROR: Unknown ZUKO_ID")
        exit()


    num_params = count_parameters(flow)
    print(f"Number of trainable parameters: {num_params}")


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

    print(f"\nTraining {args.MODEL} model...")
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
            num_numerical_features=X_values.shape[1],
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

        X_train = preprocess_data(X_train, save_dir, args.ZUKO_ID, NUM_COND_INPUTS, scaler_exists=False)
        X_val = preprocess_data(X_val, save_dir, args.ZUKO_ID, NUM_COND_INPUTS, scaler_exists=True)

        print(f"Train data has shape {X_train.shape}.")
        print(f"Val data has shape {X_val.shape}.")

        train_loader = torch.utils.data.DataLoader(X_train, batch_size=args.BATCH_SIZE, shuffle=True, num_workers = 0, pin_memory = True)
        val_loader = torch.utils.data.DataLoader(X_val, batch_size=args.BATCH_SIZE, shuffle=False, num_workers = 0, pin_memory = True)

        models_dir = f"{save_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        
        

        wandb.log({"num_trainable_params": num_params})
        wandb.run.summary["num_trainable_params"] = num_params


        # Train to maximize the log-likelihood
        optimizer = torch.optim.Adam(flow.parameters(), lr=args.LEARNING_RATE)

        # %%
        losses_train, losses_val = [], []
        best_val_loss = 1e10
        start_epoch = 0
      

        for k in range(start_epoch, args.NUM_EPOCHS):

            train_losses = run_training_step(flow, optimizer, train_loader, device, k, NUM_COND_INPUTS, is_val_step=False)

            with torch.no_grad():
                val_losses = run_training_step(flow, optimizer, val_loader, device, k, NUM_COND_INPUTS, is_val_step=True)

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
                best_ckpt_path = f"{models_dir}/best_model.pt"
                torch.save(
                    {
                        "epoch": k,
                        "model_state_dict": flow.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_losses["val/total"],
                    },
                    best_ckpt_path,
                )

            if (k + 1) % args.PLOT_EPOCH_INTERVAL == 0:
                print(f"     Making eval plots at epoch {k}")

                flow.eval()
                with torch.no_grad():
                
                    x_plot = next(iter(val_loader)).to(device).float()
                
                    if NUM_COND_INPUTS > 0:
                        x_plot_data = x_plot[:, :-NUM_COND_INPUTS]
                        x_plot_context = x_plot[:, -NUM_COND_INPUTS:]
                        samples = sample_from_flow(flow, N=args.OVERSAMPLE, x_context=x_plot_context if NUM_COND_INPUTS > 0 else None)
                    else:
                        samples = sample_from_flow(flow, N=args.OVERSAMPLE * len(x_plot))
                
                    loc_data_dict = {
                        "data": inverse_preprocess_data(
                            x_plot.detach().cpu().numpy(),
                            save_dir,
                            args.ZUKO_ID,
                            NUM_COND_INPUTS,
                        ),

                    "generated": inverse_preprocess_data(
                            samples,
                            save_dir,
                            args.ZUKO_ID,
                            NUM_COND_INPUTS,
                        ),
                    }
                    plot_hists_1d(loc_data_dict, bins_dict, log_dims=log_vars, labels=feature_labels)
                    plt.savefig(f"{plots_dir}/hists_epoch_{k}")
                    plt.close()

                torch.save(
                    {
                        "epoch": k,
                        "model_state_dict": flow.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_losses["val/total"],
                    },
                    f"{models_dir}/checkpoint_epoch_{k}.pt",
                )

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

    print(f"\nEvaluating {args.MODEL} model...")

    # Determine output filename and seed, so repeated runs produce different samples.
    existing_files = sorted(Path(save_dir).glob("generated_samples*.npy"))
    n_existing = len(existing_files)
    if n_existing == 0:
        samples_out_path = Path(save_dir) / "generated_samples.npy"
        sample_seed = seed
    else:
        samples_out_path = Path(save_dir) / f"generated_samples_{n_existing + 1}.npy"
        sample_seed = seed + n_existing
        print(f"     Found existing generated samples file(s): {[f.name for f in existing_files]}")
        print(f"     Saving new samples to {samples_out_path.name} with seed {sample_seed}.")

    torch.manual_seed(sample_seed)
    np.random.seed(sample_seed)

    print("     Making samples...")
    sample_indices = np.tile(np.arange(len(X)), args.OVERSAMPLE)

    if args.MODEL == "tabddpm":

        col_name = collection_list[0]
        is_endcap = "Endcap" in col_name

        # Build z-snapping KDTree lookup from real data (only needed for endcap collections)
        if is_endcap:
            z_lookup = build_xy_z_lookup(
                X_global,
                FEATURE_INDICES_DICT["side"],
                FEATURE_INDICES_DICT["layer"],
                FEATURE_INDICES_DICT["r"],
                FEATURE_INDICES_DICT["phi"],
                FEATURE_INDICES_DICT["z"],
            )

        num_samples_total = len(sample_indices)
        # y condition (class label encoding the conditioning variables) per output slot
        y_per_slot = y_values[sample_indices] if args.Y_MODE == "cond" else None

        # kwargs shared across all tabddpm_sample calls in the rejection loop
        tabddpm_kwargs = dict(
            parent_dir=str(save_dir),
            real_data_path=str(dataset_dir),
            batch_size=args.SAMPLE_BATCH_SIZE,
            model_type="mlp",
            model_params=model_params,
            model_path=str(Path(save_dir) / "model.pt"),
            num_timesteps=args.NUM_TIMESTEPS,
            gaussian_loss_type="mse",
            scheduler=args.SCHEDULER,
            T_dict=T_dict,
            num_numerical_features=X_values.shape[1],
            disbalance=None,
            device=device,
            change_val=False,
        )

        # output_samples[i] holds one mask-passing sample for the context of X[sample_indices[i]]
        output_samples = np.empty((num_samples_total, X.shape[1]), dtype=np.float32)
        # unfilled[i] is True until slot i receives a mask-passing sample
        unfilled = np.ones(num_samples_total, dtype=bool)

        max_rounds = 100
        for rnd in range(max_rounds):
            remaining = np.where(unfilled)[0]
            if len(remaining) == 0:
                break

            n_rem = len(remaining)
            print(f"     Round {rnd+1}: {n_rem}/{num_samples_total} tabddpm slots remaining ({100*len(remaining)/num_samples_total:.1f}%).", flush=True)

            # Generate one sample per remaining slot using the exact y condition for that slot.
            # tabddpm_sample overwrites X_num_train.npy / y_train.npy each call; reload after.
            tabddpm_sample(
                **tabddpm_kwargs,
                num_samples=n_rem,
                seed=sample_seed + rnd,
                y_to_sample=y_per_slot[remaining] if args.Y_MODE == "cond" else None,
            )

            X_gen = np.load(Path(save_dir) / "X_num_train.npy").astype(np.float32)
            y_gen = np.load(Path(save_dir) / "y_train.npy").astype(np.int64)

            if args.Y_MODE == "none":
                loc_samples = X_gen
            else:
                # y_lookup maps class label -> original conditioning values (side, layer, module, sensor)
                cond_gen = y_lookup[y_gen]
                loc_samples = np.concatenate([X_gen, cond_gen], axis=1)

            # Undo geometry transform (no-op for rphi; local->global phi for local_phi)
            loc_samples = inverse_geometry_transform(loc_samples, args.BASIS, col_name, FEATURE_ORDER)

            # Snap z to nearest real detector hit for endcap collections
            if is_endcap:
                loc_samples[:, FEATURE_INDICES_DICT["z"]] = snap_z_to_detector_xy(
                    loc_samples[:, FEATURE_INDICES_DICT["r"]],
                    loc_samples[:, FEATURE_INDICES_DICT["phi"]],
                    loc_samples[:, FEATURE_INDICES_DICT["side"]],
                    loc_samples[:, FEATURE_INDICES_DICT["layer"]],
                    z_lookup,
                )

            # Apply material map: keep only samples that land on detector material
            mask = apply_material_map_hybrid(
                {col_name: loc_samples}, None, col_name, FEATURE_INDICES_DICT
            )

            passing = np.where(mask)[0]
            output_samples[remaining[passing]] = loc_samples[passing]
            unfilled[remaining[passing]] = False
            print(f"       Filled {len(passing)} new samples this round.", flush=True)

        if np.any(unfilled):
            # Save the X context rows that never converged for external inspection / retry
            unfilled_ctx_path = samples_out_path.parent / (samples_out_path.stem + "_unfilled_contexts.npy")
            np.save(unfilled_ctx_path, X[sample_indices[np.where(unfilled)[0]]])
            print(f"WARNING: {unfilled.sum()} tabddpm slots still unfilled after {max_rounds} rounds. "
                  f"Unfilled contexts saved to {unfilled_ctx_path.name}. "
                  "Saving only the filled samples.", flush=True)

        # samples is in global physical space (inverse_geometry applied inside the loop above);
        # trimmed to only the slots that were successfully filled
        samples = output_samples[~unfilled]


   


    elif args.MODEL == "flow":

        ckpt = torch.load(f"{save_dir}/models/best_model.pt", map_location=device, weights_only=False)
        flow.load_state_dict(ckpt["model_state_dict"])
        flow.eval()

        col_name = collection_list[0]
        is_endcap = "Endcap" in col_name

        # Build z-snapping KDTree lookup from real data (only needed for endcap collections)
        if is_endcap:
            z_lookup = build_xy_z_lookup(
                X_global,
                FEATURE_INDICES_DICT["side"],
                FEATURE_INDICES_DICT["layer"],
                FEATURE_INDICES_DICT["r"],
                FEATURE_INDICES_DICT["phi"],
                FEATURE_INDICES_DICT["z"],
            )

        num_samples_total = len(sample_indices)
        sample_batch_size = 4096

        # output_samples[i] holds one mask-passing sample for the context of X[sample_indices[i]]
        output_samples = np.empty((num_samples_total, X.shape[1]), dtype=np.float32)
        # unfilled[i] is True until slot i receives a mask-passing sample
        unfilled = np.ones(num_samples_total, dtype=bool)

        max_rounds = 100
        with torch.no_grad():
            for rnd in range(max_rounds):
                remaining = np.where(unfilled)[0]
                if len(remaining) == 0:
                    break

                print(f"     Round {rnd+1}: {len(remaining)}/{num_samples_total} flow slots remaining ({100*len(remaining)/num_samples_total:.1f}%).", flush=True)
                n_filled_this_round = 0

                for i in tqdm(range(0, len(remaining), sample_batch_size), desc=f"       Sampling round {rnd+1}"):
                    batch_out_idx = remaining[i:i+sample_batch_size]
                    nn = len(batch_out_idx)

                    # Use the exact conditioning variables from X for each output slot
                    if NUM_COND_INPUTS > 0:
                        context_to_sample = torch.tensor(
                            X[sample_indices[batch_out_idx], -NUM_COND_INPUTS:], dtype=torch.float32
                        ).to(device)
                    else:
                        context_to_sample = None

                    n_samples = 1 if context_to_sample is not None else nn
                    loc_samples = sample_from_flow(flow, N=n_samples, x_context=context_to_sample)

                    # Undo preprocessing (scaler applied to feature columns only; context unchanged)
                    loc_samples = inverse_preprocess_data(loc_samples, save_dir, args.ZUKO_ID, NUM_COND_INPUTS)
                    # Undo geometry transform (no-op for rphi; local->global phi for local_phi)
                    loc_samples = inverse_geometry_transform(loc_samples, args.BASIS, col_name, FEATURE_ORDER)

                    # Snap z to nearest real detector hit for endcap collections
                    if is_endcap:
                        loc_samples[:, FEATURE_INDICES_DICT["z"]] = snap_z_to_detector_xy(
                            loc_samples[:, FEATURE_INDICES_DICT["r"]],
                            loc_samples[:, FEATURE_INDICES_DICT["phi"]],
                            loc_samples[:, FEATURE_INDICES_DICT["side"]],
                            loc_samples[:, FEATURE_INDICES_DICT["layer"]],
                            z_lookup,
                        )

                    # Apply material map: keep only samples that land on detector material
                    mask = apply_material_map_hybrid(
                        {col_name: loc_samples}, None, col_name, FEATURE_INDICES_DICT
                    )

                    # Store passing samples into the output array
                    passing = np.where(mask)[0]
                    output_samples[batch_out_idx[passing]] = loc_samples[passing]
                    unfilled[batch_out_idx[passing]] = False
                    n_filled_this_round += len(passing)

                print(f"       Filled {n_filled_this_round} new samples this round.", flush=True)

        if np.any(unfilled):
            # Save the X context rows that never converged for external inspection / retry
            unfilled_ctx_path = samples_out_path.parent / (samples_out_path.stem + "_unfilled_contexts.npy")
            np.save(unfilled_ctx_path, X[sample_indices[np.where(unfilled)[0]]])
            print(f"WARNING: {unfilled.sum()} flow slots still unfilled after {max_rounds} rounds. "
                  f"Unfilled contexts saved to {unfilled_ctx_path.name}. "
                  "Saving only the filled samples.", flush=True)

        # samples is in global physical space (inverse_preprocess + inverse_geometry applied above);
        # trimmed to only the slots that were successfully filled
        samples = output_samples[~unfilled]




    # Both model branches now apply inverse_preprocess + inverse_geometry_transform inline
    # (inside the rejection-sampling loops above), so samples is already in global physical space.
    samples_global = samples
    np.save(samples_out_path, samples_global)

    print("     Done making samples. Made samples with shape:", samples_global.shape)

    print("     Making plots...")

    fig_samp, axes_samp = plot_corner_hist_2d(samples, feature_labels=feature_labels, bins_dict=bins_dict, log_dims=log_vars, title= f"generated_{args.BASIS}",)
    plt.savefig(f"{plots_dir}/corner_generated_{args.BASIS}_final")
    plt.close()

    fig_samp, axes_samp = plot_corner_hist_2d(samples_global, feature_labels=global_feature_labels, bins_dict=bins_dict, log_dims=log_vars, title= f"generated_global",)
    plt.savefig(f"{plots_dir}/corner_generated_global_final")
    plt.close()

    print("     Done making plots.")
            
    # print("     Comparing samples to target...")
    # results_dir = evaluate_samples(samples, samples_global, args.BASIS, X, X_global, feature_labels, global_feature_labels, save_dir, NUM_BINS, args.NUM_BDTS, args.BDT_SUBSAMPLE_FRAC, device, log_vars)
    # wandb.log(results_dir)
    # print(results_dir)



      

print("Done!")
wandb.finish()
