import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
import yaml
from helpers.data_transforms import  load_in_data

from helpers.evaluation import run_eval_suite_BDTs
from helpers.plotting import plot_hists_1d, make_2d_plots
from helpers.material_map import  build_masked_datasets
from helpers.flow import  build_xy_z_lookup, snap_z_to_detector_xy


plt.style.use("../science.mplstyle")

import torch
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "Using device: " + str( device ), flush=True)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_BDTs", type=int, default=5, help="Number of BDTs to train for evaluation")
parser.add_argument("--MODEL", choices=["flow_NCSF", "flow_NSF", "tabddpm"],default="flow", help="Generative_model")
parser.add_argument("--BASIS", choices=["xy", "rphi", "local_phi"], default="rphi", help="Coordinate basis used for training")
parser.add_argument("--PLOTS_DIR", default="plots", help="Directory to save plots")
parser.add_argument("--NUM_SAMPLES_TO_COMPARE", type=int, default=100_000, help="Number of samples to compare for BDT evaluation")
parser.add_argument("--CONFIGS_PATH", type=str, default="configs_train", help="Path to the configs file")
run_single_feature_BDTs = False
args = parser.parse_args()


if not os.path.exists(args.PLOTS_DIR):
    os.makedirs(args.PLOTS_DIR)



with open(f"{args.CONFIGS_PATH}.yaml", "r") as f:
    configs = yaml.safe_load(f)

BIN_BOUND = configs["BIN_BOUND"]
NUM_BINS = configs["NUM_BINS"]
ALL_COLLECTIONS =["InnerTrackerBarrelCollection"]
FEATURE_ORDER = configs["FEATURE_ORDER"]
FEATURE_INDICES_DICT = configs["FEATURE_INDICES_DICT"]
PATH_TO_DATA_DIR = configs["PATH_TO_DATA_DIR"]
NUM_COND_INPUTS = configs["NUM_COND_INPUTS"]
PATH_TO_DATA_BDT = configs["PATH_TO_DATA_BDT"]
log_vars = []
# print("CHECK THAT YOU'RE LOADING IN THE RIGHT FILE")
# exit()


# load in samples
if args.MODEL == "flow_NCSF":

    if args.BASIS == "rphi":

        exit()

                            
    elif args.BASIS == "local_phi":
       PATHS_TO_SAMPLES = {
                "OuterTrackerBarrelCollection": f"{PATH_TO_DATA_BDT}/flow_NCSF_bins8_local",
                "OuterTrackerEndcapCollection": f"{PATH_TO_DATA_BDT}/flow_NCSF_bins8_local",
                "InnerTrackerBarrelCollection": f"{PATH_TO_DATA_BDT}/flow_NCSF_bins8_local",
                "InnerTrackerEndcapCollection": f"{PATH_TO_DATA_BDT}/flow_NCSF_bins8_local",
                "VertexBarrelCollection": f"{PATH_TO_DATA_BDT}/flow_NCSF_bins8_local",
                "VertexEndcapCollection": f"{PATH_TO_DATA_BDT}/flow_NCSF_bins8_local",
                                }


elif args.MODEL == "tabddpm":

    if args.BASIS == "rphi":
        exit()
           
    elif args.BASIS == "local_phi":
        PATHS_TO_SAMPLES = {
            "OuterTrackerBarrelCollection": f"{PATH_TO_DATA_BDT}/diff_local",
            "OuterTrackerEndcapCollection": f"{PATH_TO_DATA_BDT}/diff_local",
            "InnerTrackerBarrelCollection": f"{PATH_TO_DATA_BDT}/diff_local",
            "InnerTrackerEndcapCollection": f"{PATH_TO_DATA_BDT}/diff_local",
            "VertexBarrelCollection": f"{PATH_TO_DATA_BDT}/diff_local",
            "VertexEndcapCollection": f"{PATH_TO_DATA_BDT}/diff_local/",
                            }





for i, col_name in enumerate(ALL_COLLECTIONS):

    print(f"Loading in data and samples for {col_name}...", flush=True)
    all_data_dir, all_samples_dir = {}, {}
    bins_dict = {}
    feature_labels_dict = {}

    X, feature_labels = load_in_data([col_name], "rphi", PATH_TO_DATA_DIR, 1, NUM_COND_INPUTS, FEATURE_ORDER)
    
    all_data_dir[col_name] = X


    try:
        all_samples_dir[col_name] = np.load(f"{PATHS_TO_SAMPLES[col_name]}_{col_name}_post_cellID_filtering.npy")

    except:
        print(f"Could not load generated samples for {col_name} from {PATHS_TO_SAMPLES[col_name]}/generated_samples.npy")
        # move on to next collection
        continue

    # len(X) == len(all_samples_dir[col_name]), f"Number of samples in {col_name} does not match between data and generated samples"
    feature_labels_dict[col_name] = feature_labels
    
    
    bins_dict[col_name] = {}

    for i in range(all_data_dir[col_name].shape[1]):
        if i in log_vars:
            bins_dict[col_name][i] = np.logspace(np.log10(0.9*np.min(all_data_dir[col_name][:,i])), np.log10(1.1*np.max(all_data_dir[col_name][:,i])), NUM_BINS) 
        else:
            bins_dict[col_name][i] = np.linspace(np.min(all_data_dir[col_name][:,i] - 3), np.max(all_data_dir[col_name][:,i] + 3), NUM_BINS) 


    # make the flow samples evaluation
    print()
    print("Building masked datasets...", flush=True)
    flow_samples_masked, _ = build_masked_datasets(all_data_dir, all_samples_dir, ALL_COLLECTIONS, NUM_COND_INPUTS, FEATURE_INDICES_DICT, stratify = False)




    print()
    print("Running evaluation suite with BDTs (one collection at a time)...", flush=True)

    single_bdt_results, single_bdt_scores, single_bdt_plot_data = {}, {}, {}


    print(f"Analyzing {col_name}...")

    # If NUM_SAMPLES_TO_COMPARE is -1, use all samples

    if args.NUM_SAMPLES_TO_COMPARE == -1:   
        N = min(all_data_dir[col_name].shape[0], flow_samples_masked[col_name].shape[0])
    else:
        N = min(args.NUM_SAMPLES_TO_COMPARE, all_data_dir[col_name].shape[0], flow_samples_masked[col_name].shape[0])

    
    indices_data = np.random.choice(all_data_dir[col_name].shape[0], size=N, replace=False)
    indices_flow = np.random.choice(flow_samples_masked[col_name].shape[0], size=N, replace=False)

    loc_bdt_results, loc_scores, loc_plot_data = run_eval_suite_BDTs(
                            all_data_dir[col_name][indices_data][:,:-NUM_COND_INPUTS],
                            {
                                #  "unmasked":all_samples_dir[col_name][indices_flow][:,:-NUM_COND_INPUTS],
                                "masked":flow_samples_masked[col_name][indices_flow][:,:-NUM_COND_INPUTS],
                            },
                            bins_dict[col_name],
                            n_cond=0,
                            device=device,
                            num_BDTs=args.num_BDTs,
                            plot_suffix=f"_{col_name}",
                            feature_labels=feature_labels_dict[col_name][:-NUM_COND_INPUTS],
                            run_single_feature_BDTs=run_single_feature_BDTs,
                            plot_dir=args.PLOTS_DIR,
                            )

    single_bdt_results[col_name] = loc_bdt_results
    single_bdt_scores[col_name] = loc_scores
    single_bdt_plot_data[col_name] = loc_plot_data

    # ----------------------------------------------------------------
    # Anomaly visualization: top 5% BDT score = anomalous (red),
    # remainder = regular (green).
    # Features in samples_val ([:,:-NUM_COND_INPUTS]) are ordered:
    #   0: log(E), 1: t, 2: r, 3: phi, 4: z
    # ----------------------------------------------------------------
    ANOMALY_PERCENTILE = 95
    SAMPLE_KEY = "masked"

    scores  = single_bdt_scores[col_name][SAMPLE_KEY]       # (n,)
    samples = single_bdt_plot_data[col_name][SAMPLE_KEY]    # (n, 5)
    feat_labels = feature_labels_dict[col_name][:-NUM_COND_INPUTS]

    threshold    = np.percentile(scores, ANOMALY_PERCENTILE)
    is_anomalous = scores >= threshold

    R_IDX, PHI_IDX, Z_IDX = 2, 3, 4
    r_reg, phi_reg, z_reg = samples[~is_anomalous, R_IDX], samples[~is_anomalous, PHI_IDX], samples[~is_anomalous, Z_IDX]
    r_ano, phi_ano, z_ano = samples[ is_anomalous, R_IDX], samples[ is_anomalous, PHI_IDX], samples[ is_anomalous, Z_IDX]

    x_reg, y_reg = r_reg * np.cos(phi_reg), r_reg * np.sin(phi_reg)
    x_ano, y_ano = r_ano * np.cos(phi_ano), r_ano * np.sin(phi_ano)

    # Plot 1: x-y scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_reg, y_reg, s=0.1, c="green", alpha=0.15, label=f"regular ({(~is_anomalous).sum()})", rasterized=True)
    ax.scatter(x_ano, y_ano, s=0.3, c="red",   alpha=0.5,  label=f"anomalous top {100 - ANOMALY_PERCENTILE}% ({is_anomalous.sum()})", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [mm]", fontsize=14)
    ax.set_ylabel("$y$ [mm]", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(markerscale=10, loc="upper right", frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{args.PLOTS_DIR}/anomaly_xy_{col_name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # Plot 2: r histogram (density, log y)
    from matplotlib.ticker import NullFormatter
    lo_r = min(r_reg.min(), r_ano.min())
    hi_r = max(r_reg.max(), r_ano.max())
    #bins_r = np.linspace(1151, 1151.3, 200)
    bins_r = np.linspace(553.9, 556.8, 1000)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.hist(r_reg, bins=bins_r, density=True, histtype="step", linewidth=1.5, color="green", label="regular")
    ax.hist(r_ano, bins=bins_r, density=True, histtype="step", linewidth=1.5, color="red",   label=f"anomalous (top {100 - ANOMALY_PERCENTILE}%)")
    ax.set_yscale("log")
    ax.set_xlabel("$r$ [mm]", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_xlim(553.9, 556.8)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{args.PLOTS_DIR}/anomaly_r_{col_name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # Plot 3: all 5 feature histograms
    n_feats = samples.shape[1]
    fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4))
    for fi in range(n_feats):
        vals_reg = samples[~is_anomalous, fi]
        vals_ano = samples[ is_anomalous, fi]
        lo_f = min(vals_reg.min(), vals_ano.min())
        hi_f = max(vals_reg.max(), vals_ano.max())
        bins_fi = np.linspace(lo_f, hi_f, 80)
        axes[fi].hist(vals_reg, bins=bins_fi, density=True, histtype="step", linewidth=1.5, color="green", label="regular")
        axes[fi].hist(vals_ano, bins=bins_fi, density=True, histtype="step", linewidth=1.5, color="red",   label="anomalous")
        axes[fi].set_yscale("log")
        label = feat_labels[fi] if fi < len(feat_labels) else f"feat {fi}"
        axes[fi].set_xlabel(label, fontsize=14)
        axes[fi].tick_params(labelsize=12)
        if fi == 0:
            axes[fi].set_ylabel("Density", fontsize=14)
        else:
            axes[fi].yaxis.set_tick_params(labelleft=False)
            axes[fi].yaxis.set_ticklabels([])
            axes[fi].yaxis.set_major_formatter(NullFormatter())
            axes[fi].yaxis.set_minor_formatter(NullFormatter())
        if fi == n_feats - 1:
            axes[fi].legend(frameon=False, fontsize=12)
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(f"{args.PLOTS_DIR}/anomaly_all_feats_{col_name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Anomaly plots saved for {col_name}.")

