# Paste pre-Pyrfected Python
import numpy as np
import xgboost as xgb
import yaml
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from helpers.plotting import plot_hists_1d, plot_corner_hist_2d



def get_kl_dist(samp0, samp1):

    divs = []
    for i in range(samp0.shape[1]):
        divs.append(ks_2samp(samp0[:, i], samp1[:, i])[0])
    return divs


def get_wasserstein_dist(samp0, samp1):

    divs = []
    for i in range(samp0.shape[1]):
        divs.append(wasserstein_distance(samp0[:, i], samp1[:, i]))
    return divs



def get_median_percentiles(x_array):

    x_median = np.median(x_array, axis=1)
    x_lower = np.percentile(x_array, 16, axis=1)
    x_upper = np.percentile(x_array, 84, axis=1)

    return x_median, x_lower, x_upper






class SimpleDNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.0):
        super(SimpleDNN, self).__init__()
        layers = []
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def discriminate_data_from_samples(
    data,
    samples,
    n_runs,
    config_file,
    model_type="bdt",
    plot_losses=False,
    device=None,
    val_size=0.3,
    subsample_frac=None,  # optional subsample for large datasets,
    plot_dir=None,
    evaluation_name="",
    verbose=False,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    losses_name = f"losses_{evaluation_name}" if evaluation_name else "losses"

    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)

    # ======================
    # Preprocessing only for DNN
    # ======================
    scaler = None
    if model_type.lower() == "dnn":
        scaler = StandardScaler()
        X_all = np.vstack([data, samples])
        scaler.fit(X_all)
        data = scaler.transform(data)
        samples = scaler.transform(samples)

    if subsample_frac is not None:
        n_data = int(data.shape[0] * subsample_frac)
        n_samples = int(samples.shape[0] * subsample_frac)
        idx_data = np.random.choice(data.shape[0], n_data, replace=False)
        idx_samples = np.random.choice(samples.shape[0], n_samples, replace=False)
        data = data[idx_data]
        samples = samples[idx_samples]

    if verbose:
        print(data.shape, samples.shape)

    # ======================
    # Split data
    # ======================
    data_train, data_val = train_test_split(data, test_size=val_size, random_state=42)
    samples_train, samples_val = train_test_split(samples, test_size=val_size, random_state=42)


    X_train = np.vstack([data_train, samples_train])
    Y_train = np.vstack([np.zeros((data_train.shape[0], 1)), np.ones((samples_train.shape[0], 1))])
    X_val = np.vstack([data_val, samples_val])
    Y_val = np.vstack([np.zeros((data_val.shape[0], 1)), np.ones((samples_val.shape[0], 1))])

    auc_list = []
    best_epoch_list = []
    model_list = []

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32, device=device)

    # ======================
    # Training loop
    # ======================
    for i in range(n_runs):

        if verbose:
            print(f"On {model_type.upper()} run {i+1} of {n_runs}...")

        if model_type.lower() == "bdt":
            bdt_params = config_dict["bdt_hyperparameters"]
            model = xgb.XGBClassifier(
                n_estimators=bdt_params["n_estimators"],
                max_depth=bdt_params["max_depth"],
                learning_rate=bdt_params["learning_rate"],
                subsample=bdt_params["subsample"],
                early_stopping_rounds=bdt_params["early_stopping_rounds"],
                objective="binary:logistic",
                random_state=i,
                eval_metric="logloss",
            )
            eval_set = [(X_train, Y_train), (X_val, Y_val)]
            model.fit(X_train, Y_train, eval_set=eval_set, verbose=False)
            best_epoch = model.best_iteration

            if plot_losses:
                results = model.evals_result()
                plt.figure()
                plt.plot(results["validation_0"]["logloss"], label="Train logloss")
                plt.plot(results["validation_1"]["logloss"], label="Val logloss")
                plt.xlabel("Iteration")
                plt.ylabel("Logloss")
                plt.title(f"BDT Run {i+1} Logloss")
                plt.legend()
                plt.savefig(f"{plot_dir}/{losses_name}_{i}.png")
                plt.show()
                plt.close()

        elif model_type.lower() == "dnn":
            dnn_params = config_dict["dnn_hyperparameters"]
            model = SimpleDNN(
                input_dim=X_train.shape[1],
                hidden_layers=dnn_params["hidden_layers"],
                dropout=dnn_params.get("dropout", 0.0)
            ).to(device)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=dnn_params["learning_rate"])

            train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=dnn_params["batch_size"], shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=dnn_params["batch_size"], shuffle=False)

            train_losses, val_losses = [], []
            best_val_loss = float("inf")
            best_epoch = 0
            best_model_state = None

            for epoch in tqdm(range(dnn_params["epochs"])):
                model.train()
                batch_losses = []
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                train_loss = np.mean(batch_losses)
                train_losses.append(train_loss)

                # Validation
                model.eval()
                batch_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        preds = model(xb)
                        loss = criterion(preds, yb)
                        batch_losses.append(loss.item())
                    val_loss = np.mean(batch_losses)
                    val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Restore model to best epoch
            model.load_state_dict(best_model_state)
            model.eval()

            if plot_losses:
                plt.figure()
                plt.plot(train_losses, label="Train loss")
                plt.plot(val_losses, label="Val loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"DNN Run {i+1} Losses (Best epoch {best_epoch})")
                plt.legend()
                plt.savefig(f"{plot_dir}/{losses_name}_{i}.png")
                plt.show()
                plt.close()

        else:
            raise ValueError("model_type must be 'bdt' or 'dnn'")

        # Compute scores for this model
        if model_type.lower() == "bdt":
            loc_scores = model.predict_proba(X_val)[:, 1]
        else:
            with torch.no_grad():
                loc_scores = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy().flatten()

        auc = roc_auc_score(Y_val, loc_scores)
        #print(f"   auc={auc:.4f}")

        

        auc_list.append(auc)
        best_epoch_list.append(best_epoch)
        model_list.append(model)

    # ======================
    # Average scores over all models (ensemble)
    # ======================
    scores_list = []
    for model in model_list:
        if model_type.lower() == "bdt":
            scores_list.append(model.predict_proba(samples_val)[:, 1].reshape(-1, 1))
        else:
            with torch.no_grad():
                scores_list.append(model(torch.tensor(samples_val, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1, 1))

    loc_scores = np.mean(np.concatenate(scores_list, axis=1), axis=1)

    n_epochs_or_estimators = (
        config_dict["bdt_hyperparameters"]["n_estimators"]
        if model_type.lower() == "bdt"
        else config_dict["dnn_hyperparameters"]["epochs"]
    )

    if model_type.lower() == "dnn":
        samples_val = scaler.inverse_transform(samples_val)

    return (
        np.mean(auc_list),
        np.std(auc_list),
        best_epoch_list,
        n_epochs_or_estimators,
        model_list,
        samples_val,  # unpreprocessed tensor for DNN
        loc_scores
    )


def compute_feature_bdt_scores(data, samples, device="cuda", num_BDTs=3, feature_labels=None):
    """
    Train a 1D BDT per feature to discriminate data vs samples.
    Returns dict: feature_index -> (auc_mean, auc_std)
    """
    results = {}

    for i in range(data.shape[1]):
        data_feat = data[:, i].reshape(-1, 1)
        samples_feat = samples[:, i].reshape(-1, 1)

        auc_mean, auc_std, *_ = discriminate_data_from_samples(
            data_feat,
            samples_feat,
            num_BDTs,
            "configs/bdt.yml",
            model_type="bdt",
            plot_losses=False,
            device=device,
            val_size=0.25,
            plot_dir="."
        )

        results[i] = (auc_mean, auc_std)
        if feature_labels is not None:
            print(f"Feature {feature_labels[i]} BDT AUC: {auc_mean:.4f} ± {auc_std:.4f}")
        else:
            print(f"Feature {i} BDT AUC: {auc_mean:.4f} ± {auc_std:.4f}")

    return results

def compute_pairwise_bdt_scores(data, samples, device="cuda", num_BDTs=3, n_cond=1):
    """
    Train BDTs on all feature pairs, excluding cond-cond pairs.
    Returns:
        auc_matrix: (n_features, n_features)
    """
    n_features = data.shape[1]
    auc_matrix = np.zeros((n_features, n_features))

    cond_start = n_features - n_cond

    # count only valid pairs
    total_pairs = 0
    for i in range(n_features):
        for j in range(i, n_features):
            if not (i >= cond_start and j >= cond_start):
                total_pairs += 1

    pbar = tqdm(total=total_pairs, desc="Pairwise BDTs (no cond-cond)")

    for i in range(n_features):
        for j in range(i, n_features):

            # ❌ skip conditioning-conditioning pairs
            if i >= cond_start and j >= cond_start:
                continue

            data_pair = data[:, [i, j]]
            samples_pair = samples[:, [i, j]]

            auc_mean, auc_std, *_ = discriminate_data_from_samples(
                data_pair,
                samples_pair,
                num_BDTs,
                "configs/bdt.yml",
                model_type="bdt",
                plot_losses=False,
                device=device,
                val_size=0.25,
                plot_dir="."
            )

            auc_matrix[i, j] = auc_mean
            auc_matrix[j, i] = auc_mean

            pbar.update(1)

    pbar.close()
    return auc_matrix

def plot_pairwise_auc_matrix(auc_matrix, feature_labels=None):
    plt.figure(figsize=(8, 7))

    im = plt.imshow(auc_matrix, vmin=0.5, vmax=np.max(auc_matrix), cmap="viridis")

    n = auc_matrix.shape[0]
    plt.xticks(range(n), feature_labels if feature_labels else range(n), rotation=90)
    plt.yticks(range(n), feature_labels if feature_labels else range(n))

    plt.title("Pairwise BDT AUC")
    plt.colorbar(im, label="AUC")

    plt.tight_layout()
    plt.show()



def run_eval_suite_BDTs(
        data, 
        samples_dict, 
        bins,
        n_cond,
        device,
        num_BDTs=3, 
        run_single_feature_BDTs=True, 
        plot_suffix="",
        log_vars=[],
        feature_labels=None,
        ):

    print(f"Len data: {len(data)}")

    loc_scores_list = []
    single_bdt_results = {key:{} for key in samples_dict.keys()}

    scores_results = {key:{} for key in samples_dict.keys()}
    X_plot_results = {key:{} for key in samples_dict.keys()}

    for sample_key in samples_dict.keys():
        loc_samples = samples_dict[sample_key]
        print(f"Len samples {sample_key}: {len(loc_samples)}")


        if run_single_feature_BDTs:
            # Per-feature BDT evaluation
            print("\nPer-feature BDT performance:")
            bdt_feature_results = compute_feature_bdt_scores(data, loc_samples, device=device, num_BDTs=num_BDTs, feature_labels=feature_labels)
            single_bdt_results[sample_key] = bdt_feature_results


            print("\nPairwise BDT performance:")
            pairwise_auc = compute_pairwise_bdt_scores(data, loc_samples, device=device, num_BDTs=num_BDTs, n_cond=n_cond)
            
            plot_pairwise_auc_matrix(pairwise_auc, feature_labels)
            
            # store if you want
            single_bdt_results[sample_key]["pairwise_auc"] = pairwise_auc
            

        # BDT
        auc_mean, auc_std, best_epoch_list, max_epochs, bdt_list, samples_test, loc_scores = discriminate_data_from_samples(
                                                data,
                                                loc_samples,
                                                num_BDTs,
                                                "configs/bdt.yml",
                                                model_type="bdt",
                                                plot_losses=False,
                                                device="cuda",
                                                val_size = 0.2, 
                                                plot_dir="."
                                            )

        print("\nFeature importances (full BDT):")

        importances = np.array([bdt.feature_importances_ for bdt in bdt_list])
        mean_importance = importances.mean(axis=0)
        std_importance = importances.std(axis=0)
        
        for i, (mean, std) in enumerate(zip(mean_importance, std_importance)):
            label = feature_labels[i] if feature_labels else f"Feature {i}"
            print(f"{label}: {mean:.4f} ± {std:.4f}")

    
    
        print(f"auc {auc_mean} \pm {auc_std}. best epoch {best_epoch_list} of {max_epochs}.\n")
        loc_scores_list.append(loc_scores)
        scores_results[sample_key] = loc_scores
        X_plot_results[sample_key] = samples_test

        X_plot = {"samples":  samples_test}
        percentiles = [70, 80, 90, 99]
        for p in percentiles:
            X_plot[f"samples, top {p}%"] =  samples_test[loc_scores >= np.percentile(loc_scores, p)]
    
        plot_hists_1d({"data":data, **X_plot}, bins, log_dims=log_vars, labels=feature_labels)

    
        
    plt.figure()
    for i, sample_key in enumerate(samples_dict.keys()):
              
        plt.hist(loc_scores_list[i], bins = np.linspace(0, 1, 100), histtype = "step", density = True, label = sample_key)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("scores")
    plt.ylabel("Density")
    plt.show()


    return single_bdt_results, scores_results, X_plot_results
        


def run_eval_suite(reference, generated_samples, save_dir, evaluation_name, num_bins, num_bdts, device, subsample_frac, feature_labels=None, log_vars=()):
    """Run plots, KS tests, and BDT discrimination for one coordinate representation."""
    bins_dict = {}

    for i in range(reference.shape[1]):
        if i in log_vars:
            bins_dict[i] = np.logspace(np.log10(0.9*np.min(reference[:,i])), np.log10(1.1*np.max(reference[:,i])), num_bins)
        else:
            bins_dict[i] = np.linspace(np.min(reference[:,i] - 1), np.max(reference[:,i] + 1), num_bins)

    plot_hists_1d({"data":reference, "generated":generated_samples}, bins_dict, log_dims=log_vars, labels=feature_labels)
    plt.savefig(f"{save_dir}/plots/hists_{evaluation_name}.png")
    plt.close()

    fig_samp, axes_samp = plot_corner_hist_2d(
        generated_samples,
        feature_labels=feature_labels,
        bins_dict=bins_dict,
        log_dims=log_vars,
        title="generated",
    )
    plt.savefig(f"{save_dir}/plots/corner_generated_{evaluation_name}.png")
    plt.close()

    ks_dists_samples = get_kl_dist(reference, generated_samples)
    ks_dists_gaussians = get_kl_dist(np.random.normal(size=reference.shape), np.random.normal(size=generated_samples.shape))

    auc_mean, auc_std, best_epoch_list, max_epochs, _, _, _ = discriminate_data_from_samples(
        reference,
        generated_samples,
        num_bdts,
        "configs/bdt.yml",
        model_type="bdt",
        plot_losses=True,
        device=device,
        val_size=0.2,
        subsample_frac=subsample_frac,
        plot_dir=f"{save_dir}/plots",
        evaluation_name=evaluation_name,
    )

    with open(f"{save_dir}/results_{evaluation_name}.txt", "w") as ofile:
        for i, ks_dist in enumerate(ks_dists_samples):
            ofile.write("Feature {i} KL div: {ks_dist} (for gaussian: {ks_gauss})".format(i=i, ks_dist=ks_dist, ks_gauss=ks_dists_gaussians[i]))
            ofile.write("\n")

        ofile.write(f"auc {auc_mean} pm {auc_std}. best epoch {best_epoch_list} of {max_epochs}.\n")

    return auc_mean, auc_std, best_epoch_list


def evaluate_samples(samples, samples_global, basis, X, X_global, feature_labels, global_feature_labels, save_dir, NUM_BINS, NUM_BDTS, BDT_SUBSAMPLE_FRAC, device, log_vars):
    evaluations = [(basis, X, samples, feature_labels)]

    if basis in ["local_phi", "local_rphi"]:
        evaluations.append(("global", X_global, samples_global, global_feature_labels))

    results_dir = {}

    for evaluation_name, reference, generated, labels in evaluations:
        auc_mean, auc_std, best_epoch_list = run_eval_suite(reference, generated, save_dir, evaluation_name, NUM_BINS, NUM_BDTS, device, BDT_SUBSAMPLE_FRAC, labels, log_vars)
        prefix = "global_" if evaluation_name == "global" else ""

        results_dir[evaluation_name] = {
            f"{prefix}auc_mean": auc_mean,
            f"{prefix}auc_std": auc_std,
            f"{prefix}bdt_best_epoch": np.mean(best_epoch_list),
        }

    return results_dir