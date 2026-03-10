# Paste pre-Pyrfected Python
import numpy as np
import xgboost as xgb
import yaml
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


def get_kl_dist(samp0, samp1):

    divs = []
    for i in range(samp0.shape[1]):
        divs.append(ks_2samp(samp0[:, i], samp1[:, i])[0])
    return divs


def get_median_percentiles(x_array):

    x_median = np.median(x_array, axis=1)
    x_lower = np.percentile(x_array, 16, axis=1)
    x_upper = np.percentile(x_array, 84, axis=1)

    return x_median, x_lower, x_upper


def discriminate_data_from_samples(data, samples, n_runs, bdt_config):
    # unweighted! try to have data and samples be the same size

    with open(bdt_config, "r") as file:
        bdt_hyperparams_dict = yaml.safe_load(file)["bdt_hyperparameters"]

    data_train, data_test = train_test_split(data, test_size=0.1, random_state=42)
    samples_train, samples_test = train_test_split(
        samples, test_size=0.1, random_state=42
    )

    X_train = np.vstack([data_train, samples_train])
    Y_train = np.vstack(
        [np.ones((data_train.shape[0], 1)), np.zeros((samples_train.shape[0], 1))]
    )
    X_val = np.vstack([data_test, samples_test])
    Y_val = np.vstack(
        [np.ones((data_test.shape[0], 1)), np.zeros((samples_test.shape[0], 1))]
    )

    auc_list = []
    best_epoch_list = []
    bdt_list = []

    for i in range(n_runs):

        print(f"On BDT {i+1} of {n_runs}...")

        eval_set = [(X_train, Y_train), (X_val, Y_val)]

        bst_i = xgb.XGBClassifier(
            n_estimators=bdt_hyperparams_dict["n_estimators"],
            max_depth=bdt_hyperparams_dict["max_depth"],
            learning_rate=bdt_hyperparams_dict["learning_rate"],
            subsample=bdt_hyperparams_dict["subsample"],
            early_stopping_rounds=bdt_hyperparams_dict["early_stopping_rounds"],
            objective="binary:logistic",
            random_state=i,
            eval_metric="logloss",
        )

        bst_i.fit(X_train, Y_train, eval_set=eval_set, verbose=False)
        results_f = bst_i.evals_result()
        losses = results_f["validation_0"]["logloss"]
        losses_val = results_f["validation_1"]["logloss"]
        best_epoch = bst_i.best_iteration

        loc_scores = bst_i.predict_proba(
            X_val, iteration_range=(0, bst_i.best_iteration)
        )[:, 1]
        loc_auc = roc_auc_score(Y_val, loc_scores)
        print(f"   auc={loc_auc}")

        auc_list.append(loc_auc)
        best_epoch_list.append(best_epoch)
        bdt_list.append(bst_i)

    return (
        np.mean(auc_list),
        np.std(auc_list),
        best_epoch_list,
        bdt_hyperparams_dict["n_estimators"],
        bdt_list
    )

