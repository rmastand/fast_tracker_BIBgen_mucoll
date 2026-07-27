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
parser.add_argument("--num_BDTs", type=int, default=10, help="Number of BDTs to train for evaluation")
parser.add_argument("--MODEL", choices=["flow", "tabddpm"],default="flow", help="Generative_model")
parser.add_argument("--BASIS", choices=["xy", "rphi", "local_phi"], default="rphi", help="Coordinate basis used for training")
parser.add_argument("--NO_SNAP_Z", action="store_true", help="If set, do not snap z to detector")
parser.add_argument("--TRAIN_SINGLE_BDTS", action="store_true", help="If set, train BDTs on each collection separately")
parser.add_argument("--TRAIN_ALL_BDTS", action="store_true", help="If set, train BDTs on all collections combined")
parser.add_argument("--PLOTS_DIR", default="plots", help="Directory to save plots")
parser.add_argument("--SAVE_OUT_SAMPLES", action="store_true", help="If set, save out samples to disk")
parser.add_argument("--NUM_SAMPLES_TO_COMPARE", type=int, default=1_000_000, help="Number of samples to compare for BDT evaluation")

args = parser.parse_args()

if not os.path.exists(args.PLOTS_DIR):
    os.makedirs(args.PLOTS_DIR)



with open("configs.yaml", "r") as f:
    configs = yaml.safe_load(f)

BIN_BOUND = configs["BIN_BOUND"]
NUM_BINS = configs["NUM_BINS"]
ALL_COLLECTIONS = configs["ALL_COLLECTIONS"]
FEATURE_ORDER = configs["FEATURE_ORDER"]
FEATURE_INDICES_DICT = configs["FEATURE_INDICES_DICT"]
PATH_TO_DATA_DIR = configs["PATH_TO_DATA_DIR"]
NUM_COND_INPUTS = configs["NUM_COND_INPUTS"]
log_vars = []



# load in samples
if args.MODEL == "flow":

    if args.BASIS == "rphi":
        PATHS_TO_SAMPLES = {
            "OuterTrackerBarrelCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_OTBC_cond4/",
            "OuterTrackerEndcapCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_OTEC_cond4/",
            "InnerTrackerBarrelCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_ITBC_cond4/",
            "InnerTrackerEndcapCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_ITEC_cond4/",
            "VertexBarrelCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_VBC_cond4/",
            "VertexEndcapCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_VEC_cond4/",
                            }
    elif args.BASIS == "local_phi":
        PATHS_TO_SAMPLES = {
            "OuterTrackerBarrelCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_OTBC_cond4_local/",
            "OuterTrackerEndcapCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_OTBC_cond4_local/",
            "InnerTrackerBarrelCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_OTBC_cond4_local/",
            "InnerTrackerEndcapCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_OTBC_cond4_local/",
            "VertexBarrelCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_OTBC_cond4_local/",
            "VertexEndcapCollection": "/scratch/midway3/rmastand/muon_collider/zuko_outputs/flow_OTBC_cond4_local/",
                            }

elif args.MODEL == "tabddpm":

    if args.BASIS == "rphi":
        PATHS_TO_SAMPLES = {
            "OuterTrackerBarrelCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/diff_OTBC_cond4/",
            "OuterTrackerEndcapCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/diff_OTEC_cond4/",
            "InnerTrackerBarrelCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/diff_ITBC_cond4/",
            "InnerTrackerEndcapCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/diff_ITEC_cond4/",
            "VertexBarrelCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/diff_VBC_cond4/",
            "VertexEndcapCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/diff_VEC_cond4/",
                            }
    elif args.BASIS == "local_phi":
        PATHS_TO_SAMPLES = {
            "OuterTrackerBarrelCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/tabddpm_OTBC_cond4_local/",
            "OuterTrackerEndcapCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/tabddpm_OTBC_cond4_local/",
            "InnerTrackerBarrelCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/tabddpm_OTBC_cond4_local/",
            "InnerTrackerEndcapCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/tabddpm_OTBC_cond4_local/",
            "VertexBarrelCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/tabddpm_OTBC_cond4_local/",
            "VertexEndcapCollection": "/scratch/midway3/rmastand/muon_collider/ddpm_outputs/tabddpm_OTBC_cond4_local/",
                            }

print("Loading in data and samples...", flush=True)
all_data_dir, all_samples_dir = {}, {}
bins_dict = {}
feature_labels_dict = {}

for col_name in ALL_COLLECTIONS:

    X, feature_labels = load_in_data([col_name], "rphi", PATH_TO_DATA_DIR, 1, NUM_COND_INPUTS, FEATURE_ORDER)
    
    all_data_dir[col_name] = X

   
    all_samples_dir[col_name] = np.load(f"{PATHS_TO_SAMPLES[col_name]}/generated_samples.npy")
    feature_labels_dict[col_name] = feature_labels
    
    
    bins_dict[col_name] = {}

    for i in range(all_data_dir[col_name].shape[1]):
        if i in log_vars:
            bins_dict[col_name][i] = np.logspace(np.log10(0.9*np.min(all_data_dir[col_name][:,i])), np.log10(1.1*np.max(all_data_dir[col_name][:,i])), NUM_BINS) 
        else:
            bins_dict[col_name][i] = np.linspace(np.min(all_data_dir[col_name][:,i] - 3), np.max(all_data_dir[col_name][:,i] + 3), NUM_BINS) 


print()
if not args.NO_SNAP_Z:

    print("Snapping z to detector...", flush=True)

    for col_name in ALL_COLLECTIONS:

        if "Endcap" in col_name:
            plt.figure()
        
            z_idx = FEATURE_INDICES_DICT["z"]
            side_idx = FEATURE_INDICES_DICT["side"]
            layer_idx = FEATURE_INDICES_DICT["layer"]
        
            # original samples
            z_samples = all_samples_dir[col_name][:, z_idx]
            sides = all_samples_dir[col_name][:, side_idx]
            layers = all_samples_dir[col_name][:, layer_idx]
        
        

            z_lookup = build_xy_z_lookup(
                all_data_dir[col_name],
                FEATURE_INDICES_DICT["side"],
                FEATURE_INDICES_DICT["layer"],
                FEATURE_INDICES_DICT["r"],
                FEATURE_INDICES_DICT["phi"],
                FEATURE_INDICES_DICT["z"],
            )


            plt.hist(
                all_data_dir[col_name][:, z_idx],
                bins=np.linspace(-2000, 2000, 1000),
                histtype="step",
                label="Sim BIB"
            )

            plt.hist(
                all_samples_dir[col_name][:,FEATURE_INDICES_DICT["z"]],
                bins=np.linspace(-2000, 2000, 1000),
                histtype="step",
                label="ML BIB (before snapping)"
            )

                

            z_samples_snapped =  snap_z_to_detector_xy(
                all_samples_dir[col_name][:, FEATURE_INDICES_DICT["r"]],
                all_samples_dir[col_name][:, FEATURE_INDICES_DICT["phi"]],
                sides,
                layers,
                z_lookup,
            )

            all_samples_dir[col_name][:,FEATURE_INDICES_DICT["z"]] = z_samples_snapped
        
            # plot truth vs snapped samples
            
        
            plt.hist(
                z_samples_snapped,
                bins=np.linspace(-2000, 2000, 1000),
                histtype="step",
                label="ML BIB (after snapping)"
            )

            
            plt.title(col_name)
            plt.yscale("log")
            plt.legend()
            plt.xlabel("$z$ [mm]")
            plt.ylabel("Counts")
            plt.legend(loc = (1,0))
            plt.savefig(f"{args.PLOTS_DIR}/{col_name}_z_distribution.png")
            plt.close()


# make the flow samples evaluation
print()
print("Building masked datasets...", flush=True)
flow_samples_masked, flow_samples_masked_stratified = build_masked_datasets(all_data_dir, all_samples_dir, ALL_COLLECTIONS, NUM_COND_INPUTS, FEATURE_INDICES_DICT, stratify = True)




for i, col_name in enumerate(ALL_COLLECTIONS):

    plot_hists_1d({
        "Sim BIB": all_data_dir[col_name],
        "ML BIB (unmasked)": all_samples_dir[col_name],
        "ML BIB (masked)": flow_samples_masked[col_name],
        "ML BIB (masked stratified)": flow_samples_masked_stratified[col_name],
    }, bins_dict[col_name], log_dims=log_vars, labels=feature_labels)
    plt.savefig(f"{args.PLOTS_DIR}/{col_name}_1d_histograms.png")
    plt.close()


    loc_array = {
        "Sim BIB": (
            all_data_dir[col_name][:, FEATURE_INDICES_DICT["r"]] *
            np.cos(all_data_dir[col_name][:, FEATURE_INDICES_DICT["phi"]]),
            all_data_dir[col_name][:, FEATURE_INDICES_DICT["r"]] *
            np.sin(all_data_dir[col_name][:, FEATURE_INDICES_DICT["phi"]])
        ),
        "ML BIB (unmasked)": (
            all_samples_dir[col_name][:, FEATURE_INDICES_DICT["r"]] *
            np.cos(all_samples_dir[col_name][:, FEATURE_INDICES_DICT["phi"]]),
            all_samples_dir[col_name][:, FEATURE_INDICES_DICT["r"]] *
            np.sin(all_samples_dir[col_name][:, FEATURE_INDICES_DICT["phi"]])
        ),
        "ML BIB (masked)": (
            flow_samples_masked[col_name][:, FEATURE_INDICES_DICT["r"]] *
            np.cos(flow_samples_masked[col_name][:, FEATURE_INDICES_DICT["phi"]]),
            flow_samples_masked[col_name][:, FEATURE_INDICES_DICT["r"]] *
            np.sin(flow_samples_masked[col_name][:, FEATURE_INDICES_DICT["phi"]])
        ),
        "ML BIB (masked stratified)": (
            flow_samples_masked_stratified[col_name][:, FEATURE_INDICES_DICT["r"]] *
            np.cos(flow_samples_masked_stratified[col_name][:, FEATURE_INDICES_DICT["phi"]]),
            flow_samples_masked_stratified[col_name][:, FEATURE_INDICES_DICT["r"]] *
            np.sin(flow_samples_masked_stratified[col_name][:, FEATURE_INDICES_DICT["phi"]])
        ),
    }

    make_2d_plots(loc_array, ["$x$ [mm]", "$y$ [mm]"], col_name)
    plt.savefig(f"{args.PLOTS_DIR}/{col_name}_2d_histograms.png")
    plt.close()

    # loc_array = {
    #     "data": (
    #         all_data_dir[col_name][:, feature_indices_dict[col_name]["r"]],
    #         all_data_dir[col_name][:, feature_indices_dict[col_name]["z"]]
    #     ),
    #     "samples": (
    #         all_samples_dir[col_name][:, feature_indices_dict[col_name]["r"]],
    #         all_samples_dir[col_name][:, feature_indices_dict[col_name]["z"]]
    #     ),
    #     "masked samples": (
    #         all_samples_dir[col_name][mask][:, feature_indices_dict[col_name]["r"]],
    #         all_samples_dir[col_name][mask][:, feature_indices_dict[col_name]["z"]]
    #     )
    # }

    # make_2d_plots(loc_array, ["r", "z"])



if args.TRAIN_SINGLE_BDTS:

    print()
    print("Running evaluation suite with BDTs (one collection at a time)...", flush=True)

    single_bdt_results, single_bdt_scores, single_bdt_plot_data = {}, {}, {}

    for col_name in ALL_COLLECTIONS:

        print(f"Analyzing {col_name}...")

        
        N = np.min([args.NUM_SAMPLES_TO_COMPARE, len(flow_samples_masked[col_name])])

        
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
                                run_single_feature_BDTs=False,
                                plot_dir=args.PLOTS_DIR,
                                )

        single_bdt_results[col_name] = loc_bdt_results
        single_bdt_scores[col_name] = loc_scores
        single_bdt_plot_data[col_name] = loc_plot_data




if args.TRAIN_ALL_BDTS:

    print()
    print("Running evaluation suite with BDTs (all collections at once)...", flush=True)
    # all 6 systems at once

    all_bdt_results, all_scores, all_plot_data = {}, {}, {}

    total_data = []
    total_samples = []
            
    for col_name in ALL_COLLECTIONS:

        total_data.append(all_data_dir[col_name])
        total_samples.append(all_samples_dir[col_name])

        
    total_data = np.concatenate(total_data, axis = 0)
    total_samples = np.concatenate(total_samples, axis = 0)
    print(total_data.shape, total_samples.shape)

    

    indices_data = np.random.choice(total_data.shape[0], size=args.NUM_SAMPLES_TO_COMPARE, replace=False)
    indices_flow = np.random.choice(total_samples.shape[0], size=args.NUM_SAMPLES_TO_COMPARE, replace=False)

    loc_bdt_results, loc_scores, loc_plot_data = run_eval_suite_BDTs(
                            total_data[indices_data][:,:-NUM_COND_INPUTS],
                            {
                                #"unmasked":all_samples_dir[col_name][indices_2],
                                "masked":total_samples[indices_flow][:,:-NUM_COND_INPUTS],
                            },
                            bins_dict[col_name],
                            n_cond=0,
                            device=device,
                            num_BDTs=args.num_BDTs,
                            plot_suffix=f"_{col_name}",
                            feature_labels=feature_labels_dict[col_name],
                            run_single_feature_BDTs=False,
                            )

    all_bdt_results[col_name] = loc_bdt_results
    all_scores[col_name] = loc_scores
    all_plot_data[col_name] = loc_plot_data



if args.SAVE_OUT_SAMPLES:

    print()
    print("Saving out samples...", flush=True)
    # make the samples dir if it doesn't exist
    PATH_TO_SAVE_SAMPLES = configs["PATH_TO_SAVE_SAMPLES"] + f"{args.MODEL}_{args.BASIS}/"


    if not os.path.exists(PATH_TO_SAVE_SAMPLES):
        os.makedirs(PATH_TO_SAVE_SAMPLES)


    hits_dict_inside_bounds = {
        "OuterTrackerBarrelCollection":3_280_678,
        "OuterTrackerEndcapCollection":1_465_825,
        "InnerTrackerBarrelCollection":3_926_196,
        "InnerTrackerEndcapCollection":1_614_679,
        "VertexBarrelCollection":2_129_166,
        "VertexEndcapCollection":4_096_371,
    }

    # hits_dict_no_condition = {
    #     "OuterTrackerBarrelCollection":6_787_500,
    #     "OuterTrackerEndcapCollection":2_825_825,
    #     "InnerTrackerBarrelCollection":6_639_439,
    #     "InnerTrackerEndcapCollection":2_360_215,
    #     "VertexBarrelCollection":2_641_857,
    #     "VertexEndcapCollection":5_325_807,
    # }

    for col in ALL_COLLECTIONS:

        print(col)

        n_samples = hits_dict_inside_bounds[col]

        idx = np.random.choice(len(flow_samples_masked[col]), size=n_samples, replace=False)

        loc_subset = flow_samples_masked[col][idx]

        print(f"flow samples have shape {loc_subset.shape} ({100*loc_subset.shape[0]/n_samples}% of target)")
        np.save(f"{PATH_TO_SAVE_SAMPLES}/{col}.npy", loc_subset)

        #print(loc.shape,)